"""真实 C# 队列和真实 Transformer 的串行／合批等价验收。"""

from types import SimpleNamespace

import pytest
import torch

from common.policy.data import DataSpec, SkillVocab
from common.policy.model import CausalPolicyModel
from common.policy.config import ModelConfig
from common.policy.data.context_encoding import ContextEncoder
from common.torch_runtime import move_batch
from scripts.autoregressive_replay.parallel import ParallelRollouts, TrainingPolicyBackend
from scripts.autoregressive_replay.ppg import evaluate_validation_ppg
from tests.training._common_fixtures import make_dataset, make_demo_pt
from training import TrainingCollator


@pytest.fixture
def dataset(tmp_path):
    paths = [
        make_demo_pt(tmp_path, ["fire_iii", "fire_iv", "blizzard_iii"][:length], fight_id=f"parallel_{length}")
        for length in (1, 2, 3)
    ]
    return make_dataset(paths)


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("precision", ["float32", "bf16"])
def test_real_model_variable_history_matches_serial(dataset, use_cache, precision):
    if precision == "bf16" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        pytest.skip("native CUDA BF16 is unavailable")
    device = torch.device("cpu" if precision == "float32" else "cuda")
    spec = DataSpec.from_dataset(dataset)
    torch.manual_seed(81)
    model = CausalPolicyModel(
        spec, ModelConfig(d_model=16, n_heads=2, n_layers=1, ff_dim=32, dropout=0),
        vocab_size=SkillVocab.build_from_job_tag(spec.job_tag).size(),
    ).eval().to(device=device, dtype=torch.float32 if precision == "float32" else torch.bfloat16)
    policy = TrainingPolicyBackend(model, data_spec=spec, device=device, precision=precision)
    encoder = ContextEncoder(dataset.normalizer, dataset.schema, model.config).to(device=device)
    samples = [encoder.encode(move_batch(TrainingCollator()([dataset[index]]), device))
               for index in range(min(3, len(dataset)))]
    # 标签与训练监督字段不属于在线模型输入。
    from scripts.onnx_export import TENSOR_INPUT_NAMES
    live_keys = {*TENSOR_INPUT_NAMES, "action_legal_mask"}
    samples = [{key: value for key, value in sample.items() if key in live_keys} for sample in samples]
    samples = [{key: value.to(device) for key, value in sample.items()} for sample in samples]
    with torch.inference_mode():
        reference = []
        for batch in samples:
            policy.configure_cache(use_cache)
            reference.append(policy.raw_logits(batch, spec.action_keys).clone())

    def worker(item, client, engine):
        client.configure_cache(use_cache)
        # 轨迹在不同步数退出，剩余 batch 变化时不能误用已退出轨迹的 KV。
        for _ in range(item + 1):
            result = client.raw_logits(samples[item], spec.action_keys)
        return result

    with ParallelRollouts(policy, job_tag=spec.job_tag, workers=3) as pool:
        actual = list(pool.map(worker, range(len(samples))))
    for expected, result in zip(reference, actual, strict=True):
        tolerance = 2e-6 if precision == "float32" else 0.01
        torch.testing.assert_close(result, expected, atol=tolerance, rtol=tolerance * 10)


@pytest.mark.parametrize("precision", ["float32", "float16", "bf16"])
def test_same_canonical_compact_training_live_and_onnx_host_encode_identically(tmp_path, precision):
    """同一真实状态机上下文经三条消费链路，只允许部署目标 dtype 引入舍入。"""
    from dataclasses import asdict

    from common.policy.data import ModelInputContract
    from scripts.autoregressive_replay.backends import build_fixed_ort_inputs
    from scripts.autoregressive_replay.context import LiveBatchBuilder, SceneTemplateProvider
    from scripts.onnx_export import TENSOR_INPUT_NAMES
    from scripts.onnx_export.contracts.contract import CapacityContract
    from scripts.onnx_export.contracts.deployment_contract import DeploymentContract
    from scripts.onnx_export.policy.policy import _build_batch
    from tests.training._common_fixtures import _TEST_TRAINING_PAYLOADS

    source = make_demo_pt(
        tmp_path,
        ["fire_iii", "fire_iv", "blizzard_iii", "blizzard_iv", "fire_iii", "fire_iv"],
        fight_id="same_canonical",
    )
    dataset = make_dataset(
        [source], max_history=4, history_reset_keep=2,
        float_dtype=torch.float32, int_dtype=torch.int32,
    )
    payload = _TEST_TRAINING_PAYLOADS[source.resolve()]
    spec = DataSpec.from_dataset(dataset)
    vocab = SkillVocab.build_from_job_tag(spec.job_tag)
    config = ModelConfig(
        d_model=16, n_layers=1, n_heads=2, num_kv_heads=1, ff_dim=32,
        dropout=0.0, history_capacity=4, history_reset_keep=2, scene_capacity=8,
    )
    encoder = ContextEncoder(dataset.normalizer, dataset.schema, config)
    scene_provider = SceneTemplateProvider(
        next(dataset.iter_source_readers()), normalizer=dataset.normalizer,
    )
    builder = LiveBatchBuilder(
        backend=None, vocab=vocab, normalizer=dataset.normalizer,
        schema=dataset.schema, skill_feature_names=spec.skill_feature_names,
        scene_provider=scene_provider, device=torch.device("cpu"), max_history=4,
        model_config=config, action_keys=spec.action_keys, action_is_gcd=spec.action_is_gcd,
    )
    contract = DeploymentContract.create(
        precision=precision, capacity=CapacityContract(scene_capacity=8, history_capacity=4),
        data_spec=spec,
        input_contract=ModelInputContract.from_training(
            skill_vocab=vocab, data_spec=spec, schema=dataset.schema, normalizer=dataset.normalizer,
        ),
        model_config=asdict(config), repetition_config={}, vocab_entries=tuple(vocab),
        capacity_report={
            "semantic_sha256": "0" * 64, "scene_capacity": 8, "history_capacity": 4,
            "evidence": {"fixture": "same_real_canonical"},
        },
        embedding_vocab_size=vocab.size(),
    )
    saw_overflow = False
    for index, row in enumerate(payload["samples"]):
        raw = TrainingCollator()([dataset[index]])
        assert "history_bank_state_abs_values" in raw
        assert "history_skill_ids" not in raw
        training = encoder.encode(raw)
        canonical = row["context"]
        saw_overflow |= canonical["history_cursor"] > config.history_capacity
        live, keys = builder.build_from_canonical(canonical, gcd_phase=True, max_history=4)
        assert tuple(keys) == spec.action_keys
        for name in TENSOR_INPUT_NAMES:
            torch.testing.assert_close(training[name], live[name], rtol=0, atol=0, msg=name)
        assert live["history_cursor"].item() == canonical["history_cursor"]
        assert live["history_window_length"].item() == int(training["history_mask"].sum())
        assert live["history_window_start"].item() == canonical["history_cursor"] - int(training["history_mask"].sum())

        # 宿主仅补 padding 和转目标精度；图适配器直接接收 prepared tensor，不再次差分。
        fixed = build_fixed_ort_inputs(live, contract)
        adapted = _build_batch(*fixed)
        for name, tensor in zip(TENSOR_INPUT_NAMES, fixed, strict=True):
            assert adapted[name] is tensor
            expected = training[name].to(dtype=tensor.dtype)
            actual = tensor[:, :expected.shape[1]] if name.startswith(("scene_", "history_")) else tensor
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
        history_length = training["history_mask"].shape[1]
        assert adapted["history_state_null_mask"][:, history_length:].all()
        assert not adapted["history_state_reset_mask"][:, history_length:].any()
    assert saw_overflow


@pytest.mark.parametrize("workers", [1, 3, 16])
def test_validation_real_queues_match_serial_and_release(dataset, monkeypatch, workers):
    spec = DataSpec.from_dataset(dataset)
    vocab = SkillVocab.build_from_job_tag(spec.job_tag)

    class FixedPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []
            self.config = ModelConfig(history_capacity=4, history_reset_keep=4)

        def forward(self, batch):
            self.batch_sizes.append(batch["action_legal_mask"].shape[0])
            values = torch.full_like(batch["action_legal_mask"], -10, dtype=torch.float32)
            for key, score in (("fire_iii", 4), ("blizzard_iii", 3), ("ogcd_wait", 2)):
                values[:, spec.action_keys.index(key)] = score
            return {"logits": values}

    config = SimpleNamespace(
        model=ModelConfig(history_capacity=4, history_reset_keep=4),
        ppg=SimpleNamespace(enabled=True, normalization=1000, use_kv_cache=False),
        precision="float32",
    )
    model = FixedPolicy()
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_WORKERS", "1")
    kwargs = dict(model=model, dataset=dataset, data_spec=spec, vocab=vocab, config=config, device=torch.device("cpu"))
    expected = evaluate_validation_ppg(**kwargs)
    model.batch_sizes.clear()
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_WORKERS", str(workers))
    actual = evaluate_validation_ppg(**kwargs)
    assert actual == expected
    assert actual["val_ppg_output_gcds"] > 0
    assert max(model.batch_sizes) == min(workers, 3)


def test_grpo_sampling_is_independent_of_worker_count(dataset, tmp_path):
    from common.policy.data import ModelInputContract
    from common.policy.model.repetition import RepetitionConfig
    from common.policy.replay import AutoregressiveReplayConfig
    from grpo.trainer import _run_scene_rollout
    from scripts.autoregressive_replay.replay import AutoregressiveReplaySession, ReplayCacheStore

    spec = DataSpec.from_dataset(dataset)
    vocab = SkillVocab.build_from_job_tag(spec.job_tag)

    class Policy:
        supports_batch_inference = True
        data_spec = spec
        input_device = torch.device("cpu")
        model_config = ModelConfig(history_capacity=4, history_reset_keep=4)
        input_contract = ModelInputContract.from_training(
            skill_vocab=vocab,
            data_spec=spec, schema=dataset.schema, normalizer=dataset.normalizer,
        )
        vocab_entries = tuple(vocab)
        repetition = RepetitionConfig()
        name = "test"
        source_path = tmp_path / "checkpoint.pt"
        execution_provider = "cpu"

        def configure_cache(self, enabled):
            assert not enabled

        def metrics(self):
            return SimpleNamespace(to_dict=lambda: {})

        def raw_logits(self, batch, keys):
            values = torch.full_like(batch["action_legal_mask"], -20, dtype=torch.float32)
            for key, score in (("fire_iii", 1.5), ("blizzard_iii", 1.2), ("ogcd_wait", 0.8)):
                values[:, keys.index(key)] = score
            return values

    paths = sorted((tmp_path / "raw").glob("*.json"))
    config = AutoregressiveReplayConfig(
        checkpoint_path=Policy.source_path, output_path=tmp_path / "unused.md",
        scene_json_path=paths[0], cache_dir=tmp_path / ".cache",
        model_history_capacity=4, cache_shard_size=512, cache_max_shards=2,
        max_steps=None, max_duration_seconds=10, max_history=4, initial_action=None,
        temperature=1, top_p=0.95, use_kv_cache=False, device="cpu", job_tag=spec.job_tag,
    )
    cache = ReplayCacheStore(max_shards=2, max_readers=2)
    policy = Policy()

    def worker(index, client, engine):
        with AutoregressiveReplaySession(config, backend=client, cache_store=cache, engine=engine) as session:
            result, decisions = _run_scene_rollout(
                config, scene_json_path=paths[index % len(paths)],
                temperature=1, record_decisions=True, session=session, sampling_seed=120 + index,
            )
            assert decisions
            return tuple(row.action_key for row in result.rows), result.ppg, tuple(d.old_logprob for d in decisions)

    with ParallelRollouts(policy, job_tag=spec.job_tag, workers=1) as pool:
        expected = list(pool.map(worker, range(6)))
        assert pool.engine.active_count == 0
    with ParallelRollouts(policy, job_tag=spec.job_tag, workers=3) as pool:
        actual = list(pool.map(worker, range(6)))
        assert pool.engine.active_count == 0
    assert actual == expected
    assert len(cache._readers) <= 2
    assert len(cache._shard_cache) <= 2
    # 普通回放入口也实际走共享队列，独立 seed 与轨迹结果不受调度顺序影响。
    from scripts.autoregressive_replay.batch_replay import run_replays
    serial = list(run_replays([config] * 4, backend=policy, workers=1))
    parallel = list(run_replays([config] * 4, backend=policy, workers=3))
    assert [result.rows for result in parallel] == [result.rows for result in serial]
    assert [result.ppg for result in parallel] == [result.ppg for result in serial]


def test_sixteen_real_queues_keep_bounded_history_for_180_seconds():
    from scripts.autoregressive_replay.ppg import _run_rollout_until_time
    from scripts.autoregressive_replay.scheduler import gcd_request_delay

    keys = ("heated_split_shot", "ogcd_wait")
    batch_sizes = []
    class FixedPolicy(torch.nn.Module):
        config = ModelConfig(history_capacity=4, history_reset_keep=4)
        def forward(self, batch):
            size = batch["action_legal_mask"].shape[0]
            batch_sizes.append(size)
            return {"logits": torch.tensor([[2.0, 1.0]]).expand(size, -1)}
    policy = TrainingPolicyBackend(
        FixedPolicy(), data_spec=SimpleNamespace(action_keys=keys),
        device=torch.device("cpu"), precision="float32",
    )

    def worker(index, client, engine):
        with engine.create_backend(max_history=4, fight_remaining=180) as backend:
            class Batcher:
                def build(self, state):
                    legal = backend.validate_at(state.time, keys[0]).legal
                    wait = gcd_request_delay(state) > 1e-6
                    return {"action_legal_mask": torch.tensor([[legal, wait]])}, keys
            result = _run_rollout_until_time(
                client, backend, Batcher(), scene_provider=None, end_time=180,
                normalization=1000, precision="float32", device=torch.device("cpu"),
                expected_action_keys=keys, source_label=str(index),
            )
            stats = backend.statistics()
            assert stats["timestamp"] == pytest.approx(180)
            assert stats["action_history_count"] <= 4
            assert stats["policy_history_count"] <= 4
            assert stats["pending_event_count"] <= 8
            return result

    with ParallelRollouts(policy, job_tag="machinist", workers=16) as pool:
        results = list(pool.map(worker, range(17)))
        assert pool.engine.active_count == 0
    assert max(batch_sizes) == 16
    assert all(result == results[0] for result in results)
    assert results[0].output_gcds > 40

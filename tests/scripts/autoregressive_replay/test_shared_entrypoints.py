"""冷缓存、历史消融和 ONNX 验收的共享引擎集成回归。"""

from dataclasses import replace
import json
from threading import Barrier, get_ident
from types import SimpleNamespace

import pytest
import torch

from common.policy.data import DataSpec, ModelInputContract, SkillVocab
from common.policy.model.repetition import RepetitionConfig
from common.policy.replay import AutoregressiveReplayConfig
from scripts.autoregressive_replay import parallel, parity
from scripts.autoregressive_replay.batch_replay import run_replays
from scripts.common.inprocess_backend import InProcessEngine
from scripts.common.json_io import atomic_write_json
from scripts.convert_fflogs.cache import cache_compile
from tests.training._common_fixtures import make_dataset, make_demo_pt


@pytest.fixture
def policy_config(tmp_path):
    source = make_demo_pt(tmp_path, ["fire_iii", "blizzard_iii"], fight_id="shared")
    dataset = make_dataset([source])
    spec = DataSpec.from_dataset(dataset)
    vocab = SkillVocab.build_from_job_tag(spec.job_tag)

    class Policy:
        supports_batch_inference = True
        data_spec = spec
        input_device = torch.device("cpu")
        input_contract = ModelInputContract.from_training(
            data_spec=spec, schema=dataset.schema, normalizer=dataset.normalizer,
        )
        repetition = RepetitionConfig()
        vocab_entries = tuple(vocab)
        name = "fixed-reference"
        source_path = tmp_path / "model.pt"
        execution_provider = "cpu"

        def configure_cache(self, enabled):
            assert not enabled

        def raw_logits(self, batch, keys):
            logits = torch.full_like(batch["candidate_legal_mask"], -10, dtype=torch.float32)
            for key, value in (("fire_iii", 4), ("blizzard_iii", 3), ("ogcd_wait", 2)):
                logits[:, keys.index(key)] = value
            return logits

        def metrics(self):
            return SimpleNamespace(to_dict=lambda: {})

    config = AutoregressiveReplayConfig(
        checkpoint_path=Policy.source_path, output_path=tmp_path / "unused.md",
        scene_json_path=source, cache_dir=tmp_path / ".cache",
        model_history_capacity=4, cache_shard_size=512, cache_max_shards=2,
        max_steps=None, max_duration_seconds=8, max_history=4, initial_action=None,
        temperature=0, use_kv_cache=False, device="cpu", job_tag="black_mage",
    )
    return Policy(), config


def track_engine(monkeypatch):
    engines = []
    class TrackedEngine(InProcessEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            engines.append(self)

        def close(self):
            assert self.active_count == 0
            super().close()

    monkeypatch.setattr(parallel, "InProcessEngine", TrackedEngine)
    monkeypatch.setattr(cache_compile, "InProcessEngine", lambda *_a, **_k: pytest.fail("补编译不得另建引擎"))
    return engines


def test_cold_cache_compiles_parallel_before_replays_and_ablation_reuses_engine(policy_config, monkeypatch, tmp_path):
    policy, config = policy_config
    configs = []
    for index in range(3):
        source = tmp_path / "cold_raw" / f"scene{index}.json.br"
        atomic_write_json(source, {
            "source_id": 10, "report_code": f"COLD{index}", "fight_id": index + 1,
            "events": [{"type": "cast", "sourceID": 10, "timestamp": 1000 + step * 5000,
                        "abilityGameID": 152 if step % 2 == 0 else 154} for step in range(6)],
        })
        configs.append(replace(config, scene_json_path=source, cache_dir=tmp_path / "cold_cache"))
    engines = track_engine(monkeypatch)
    barrier = Barrier(3)
    converted = []
    actual_build = InProcessEngine.create_backend
    def build(self, **kwargs):
        backend = actual_build(self, **kwargs)
        if kwargs["max_history"] is None:
            assert self is engines[0]
            converted.append(backend)
            try:
                barrier.wait(timeout=10)
            except BaseException:
                backend.close()
                raise
        return backend
    monkeypatch.setattr(InProcessEngine, "create_backend", build)
    original_logits = policy.raw_logits
    def infer(batch, keys):
        assert len(converted) == 3
        assert all(queue._closed for queue in converted)
        return original_logits(batch, keys)
    policy.raw_logits = infer
    results = list(run_replays(configs, backend=policy, workers=3))
    assert len(engines) == 1 and engines[0]._engine is None
    assert all(result.output_gcds > 0 for result in results)
    def existing_only(self, **kwargs):
        assert kwargs["max_history"] is not None, "有效缓存不应重新编译"
        return actual_build(self, **kwargs)
    monkeypatch.setattr(InProcessEngine, "create_backend", existing_only)
    ablations = list(run_replays(configs, backend=policy, workers=3, history_limits=(0, 2)))
    assert len(engines) == 2 and engines[1]._engine is None
    assert len(ablations) == 3
    for original, variants in zip(results, ablations, strict=True):
        assert len(variants) == 3
        assert variants[0].rows == original.rows
        assert [row.reference_action_key for row in variants[1].rows] == [row.action_key for row in original.rows]


def test_parallel_parity_isolates_reports_and_registers_all_results_on_owner(policy_config, monkeypatch, tmp_path):
    reference, config = policy_config
    candidate = type(reference)()
    candidate.name = "fixed-candidate"
    candidate.contract = SimpleNamespace(precision="float32")
    candidate.package_dir = tmp_path / "deployment"
    candidate.manifest = SimpleNamespace(payload={})
    engines = track_engine(monkeypatch)
    active_counts = []
    original_logits = candidate.raw_logits
    def infer(batch, keys):
        active_counts.append(engines[0].active_count)
        # 只有真实场景验收失败，空场景仍必须完成并单独登记成功结果。
        return original_logits(batch, keys) + (0.1 if batch["scene_mask"].any() else 0)
    candidate.raw_logits = infer
    monkeypatch.setattr(parity, "PyTorchPolicyBackend", lambda *_a, **_k: reference)
    monkeypatch.setattr(parity, "OrtPolicyBackend", lambda *_a, **_k: candidate)
    monkeypatch.setattr(parity, "parity_artifact_bindings", lambda **_k: {})
    monkeypatch.setattr(parity, "package_runtime_targets", lambda **_k: {})
    owner = get_ident()
    registered = []
    def register(**kwargs):
        assert get_ident() == owner
        registered.append(kwargs["parity_report_path"])
    monkeypatch.setattr(parity, "record_parity_result", register)
    paths = [tmp_path / "empty.json", tmp_path / "scene.json"]
    with pytest.raises(AssertionError, match="audit report written"):
        parity.run_rollout_parities(
            [replace(config, scene_mode="empty"), config], output_paths=paths,
            onnx_package_path=candidate.package_dir, provider="CPUExecutionProvider",
            release_gate=True, workers=2,
        )
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert [report["status"] for report in reports] == ["passed", "failed"]
    assert registered == paths
    assert len(engines) == 1 and engines[0]._engine is None
    assert max(active_counts) == 2
    for report in reports:
        rows = report["parity"]["decisions"]
        assert len(rows) > 1
        assert [row["decision_index"] for row in rows] == list(range(len(rows)))


def test_queue_and_replay_require_explicit_shared_owners(policy_config):
    from scripts.common.inprocess_backend import InProcessBackend
    from scripts.autoregressive_replay.replay import AutoregressiveReplay, AutoregressiveReplaySession

    policy, config = policy_config
    with pytest.raises(TypeError, match="engine"):
        InProcessBackend("black_mage")
    with pytest.raises(ValueError, match="shared engine"):
        InProcessBackend("black_mage", engine=None)
    with pytest.raises(TypeError, match="engine"):
        AutoregressiveReplaySession(config, backend=policy)
    with pytest.raises(ValueError, match="shared engine"):
        AutoregressiveReplaySession(config, backend=policy, engine=None)
    with pytest.raises(TypeError, match="session"):
        AutoregressiveReplay(config)
    with pytest.raises(ValueError, match="shared session"):
        AutoregressiveReplay(config, session=None)
    with InProcessEngine("black_mage", capacity=2) as engine:
        with engine.create_backend(max_history=4) as other:
            with AutoregressiveReplaySession(config, backend=policy, engine=engine) as session:
                assert session.state_machine.submit_action(0, "fire_iii").accepted
                assert engine.active_count == 2
            assert engine.active_count == 1
            assert other.submit_action(0, "blizzard_iii").accepted
        assert engine.active_count == 0


def test_self_checks_share_one_engine_and_release_after_failure(cs_backend, monkeypatch):
    import main

    engines = track_engine(monkeypatch)
    monkeypatch.setattr(main, "InProcessEngine", parallel.InProcessEngine)
    outputs = main.run_checks("smoke", "black_mage", queues=16)
    assert len(outputs) == 16
    assert all(output == outputs[0] for output in outputs)
    assert len(engines) == 1 and engines[0]._engine is None
    def fail(_backend):
        raise RuntimeError("self check failure")
    monkeypatch.setattr(main, "_check_smoke", fail)
    with pytest.raises(RuntimeError, match="self check failure"):
        main.run_checks("smoke", "black_mage", queues=4)
    assert len(engines) == 2 and engines[1]._engine is None


@pytest.mark.parametrize("mode", ["parity", "history"])
def test_multi_scene_cli_uses_one_batch_and_distinct_reports(policy_config, monkeypatch, tmp_path, mode):
    from scripts.autoregressive_replay import main

    _, config = policy_config
    monkeypatch.setattr(main, "load_replay_config", lambda **_kwargs: config)
    outputs = []
    calls = []
    def run(configs, **kwargs):
        calls.append((configs, kwargs))
        if mode == "parity":
            outputs.extend(kwargs["output_paths"])
            return kwargs["output_paths"]
        return [(SimpleNamespace(history_limit=4), SimpleNamespace(history_limit=0)) for _ in configs]
    monkeypatch.setattr(main, "run_rollout_parities" if mode == "parity" else "run_replays", run)
    monkeypatch.setattr(main, "write_markdown", lambda _result, path: outputs.append(path) or path)
    args = ["replay", "--scenes", str(config.scene_json_path), str(config.scene_json_path), "--workers", "2"]
    args += ["--parity-onnx-package", str(tmp_path / "deployment")] if mode == "parity" else ["--history-ablation", "0"]
    monkeypatch.setattr("sys.argv", args)
    main.main()
    assert len(calls) == 1
    configs, kwargs = calls[0]
    assert len(configs) == 2 and kwargs["workers"] == 2
    assert len(outputs) == len(set(outputs)) == (2 if mode == "parity" else 4)
    if mode == "history":
        assert kwargs["history_limits"] == (0,)

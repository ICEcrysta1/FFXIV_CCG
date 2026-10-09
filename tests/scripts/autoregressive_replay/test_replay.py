"""自回归回放核心逻辑与 CLI 测试。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from common.torch_runtime import autocast_context, model_dtype
from common.policy.data import SkillVocab
from common.policy.model import RepetitionConfig
from common.policy.replay import AutoregressiveReplayConfig
from scripts.autoregressive_replay import main as replay_main_module
from scripts.autoregressive_replay.outputs.markdown import SKILL_NAMES, write_markdown
from scripts.autoregressive_replay.replay import (
    AutoregressiveReplay,
    AutoregressiveReplaySession,
    NoLegalActionError,
    ReplayCacheStore,
    ReplayResult,
    ReplayRow,
    ReplaySnapshot,
    _resolve_device,
)
from scripts.autoregressive_replay import replay as replay_module


_FAKE_GCD_SKILL = SimpleNamespace(key="fire", kind=SimpleNamespace(value="gcd"))
_FAKE_OGCD_SKILL = SimpleNamespace(key="ogcd_wait", kind=SimpleNamespace(value="ogcd"))


@pytest.mark.parametrize("scene_mode,initial_time", [("cache", None), ("none", None), ("cache", 0.0)])
def test_default_precast_replay_accepts_negative_scene_facts(tmp_path, scene_mode, initial_time):
    """真实会话、场景同步和预读首个技能贯通，不需要训练或模型推理。"""
    from dataclasses import asdict

    from common.policy.config import ModelConfig
    from common.policy.data import DataSpec, ModelInputContract
    from common.policy.model import CausalPolicyModel
    from scripts.autoregressive_replay.backends import PyTorchPolicyBackend
    from scripts.common.inprocess_backend import InProcessEngine
    from tests.training._common_fixtures import make_dataset, make_demo_pt

    source = make_demo_pt(tmp_path, ["fire_iii"], fight_id="precast")
    dataset = make_dataset([source], max_history=4, int_dtype=torch.int32, float_dtype=torch.float32)
    spec = DataSpec.from_dataset(dataset)
    vocab = SkillVocab.build_from_job_tag(spec.job_tag)
    contract = ModelInputContract.from_training(
        data_spec=spec, schema=dataset.schema, normalizer=dataset.normalizer, skill_vocab=vocab,
    )
    model_config = ModelConfig(
        d_model=8, n_layers=1, n_heads=2, num_kv_heads=1, ff_dim=16,
        dropout=0.0, scene_capacity=8, history_capacity=4, history_reset_keep=4,
    )
    model = CausalPolicyModel(spec, model_config, vocab_size=vocab.size()).eval()
    checkpoint = tmp_path / "model.pt"
    torch.save({
        "data_spec": asdict(spec), "model_config": asdict(model_config),
        "input_contract": contract.to_dict(), "model_state_dict": model.state_dict(),
    }, checkpoint)
    backend = PyTorchPolicyBackend(checkpoint, device="cpu", use_kv_cache=False)
    config = AutoregressiveReplayConfig(
        checkpoint_path=checkpoint, output_path=tmp_path / "replay.md", scene_json_path=source,
        cache_dir=tmp_path / ".cache", cache_shard_size=512, cache_max_shards=2,
        model_history_capacity=4, max_history=4, max_steps=1, scene_duration_seconds=10.0,
        scene_mode=scene_mode, initial_time_seconds=initial_time, device="cpu", use_kv_cache=False,
    )
    assert config.initial_action == "fire_iii"
    with InProcessEngine(spec.job_tag, capacity=1) as engine, AutoregressiveReplaySession(
        config, backend=backend, engine=engine,
    ) as session:
        replay = AutoregressiveReplay(config, session=session)
        expected_start = -session.skill_book.get("fire_iii").cast_time if initial_time is None else initial_time
        # 复用同一会话再跑一遍，确保事实游标随负起点重置。
        for _ in range(2):
            result = replay.run()
            assert len(result.rows) == result.output_gcds == 1
            assert result.rows[0].forced and result.rows[0].action_key == "fire_iii"
            assert result.ppg > 0
            stats = session.state_machine.statistics()
            assert stats["action_history_count"] == 1
            canonical = session.state_machine.observe_at(
                stats["timestamp"], format="vector", next_observation_timestamp=stats["timestamp"],
            ).context
            history = canonical["state_history_context"]
            request_time = history["player_state_feature_keys"].index("request_state.time_seconds")
            assert history["tokens"][0]["player_state"][request_time] == pytest.approx(expected_start)
        assert backend.metrics().calls == 0


@pytest.mark.parametrize("drift", ["reorder_disabled", "add_disabled", "remove_disabled"])
@pytest.mark.parametrize("use_kv_cache", [False, True])
def test_checkpoint_restores_history_embedding_and_logits_after_yaml_vocab_drift(tmp_path, monkeypatch, drift, use_kv_cache):
    """真实模型、回放会话和 batcher 使用训练时词表，当前 YAML 改变不影响行身份。"""
    from dataclasses import asdict
    from common.config import load_project_config
    from common.policy.config import ModelConfig
    from common.policy.data import ActionSpace, DataSpec, ModelInputContract, Normalizer
    from common.policy.data import skill_vocab as vocab_module
    from common.policy.data.schema import SceneWindowSchema, StateFeatureGroup, TrainingSchema, TRAINING_SAMPLE_SCHEMA_VERSION
    from common.output_context_schema import CANONICAL_CONTEXT_SCHEMA_VERSION
    from common.policy.model import CausalPolicyModel
    from scripts.autoregressive_replay.backends import PyTorchPolicyBackend
    from scripts.autoregressive_replay.context import LiveBatchBuilder

    project = load_project_config(job_tag="black_mage")
    disabled = replace(project.job.skills[0], key="zzzz_disabled", game_id=900001, enabled=False)
    original = replace(project, job=replace(project.job, skills=(*project.job.skills, disabled)))
    vocab = SkillVocab.build_from_config(original)
    actions = ActionSpace.from_config(original, skill_vocab=vocab)
    keys = ("previous_action_after.time_seconds", "previous_action_after.mp",
            "request_state.time_seconds", "request_state.mp")
    keys += tuple(f"{prefix}.{field}" for prefix in ("previous_action_after", "request_state")
                  for field in ("next_untargetable_in_seconds", "downtime_remaining_seconds"))
    availability_keys = tuple(f"{prefix}.{key}" for prefix in ("previous_action_after", "request_state")
                              for key in actions.action_keys)
    schema = TrainingSchema(
        serialization_format="test", sample_schema_version=TRAINING_SAMPLE_SCHEMA_VERSION,
        context_schema_version=CANONICAL_CONTEXT_SCHEMA_VERSION, scene_context_mode="absolute",
        scene_windows=(SceneWindowSchema.from_feature_keys(
            context_key="targetable_window_context", scene_type_id=0,
            feature_keys=("start_offset_seconds", "end_offset_seconds", "duration_seconds"),
        ),), state_groups=(
            StateFeatureGroup("player_state", "player_state_feature_keys", keys, "anchored_delta"),
            StateFeatureGroup("skill_availability", "skill_availability_feature_keys", availability_keys, "absolute_binary"),
        ), state_snapshots=("previous_action_after", "request_state"),
        skill_history_fields=("kind", "potency"),
    )
    layout = schema.state_layout(actions.action_keys)
    spec = DataSpec(job_tag="black_mage", num_actions=len(actions.action_keys),
                    state_dim=layout.state_dim, base_state_dim=layout.base_state_dim,
                    scene_dim=3, skill_feature_dim=2, num_scene_types=1,
                    action_keys=actions.action_keys, skill_feature_names=("kind", "potency"),
                    action_to_vocab_id=actions.action_to_vocab_id, action_is_gcd=actions.action_is_gcd)
    normalizer = Normalizer()
    normalizer.ensure_job_resources("black_mage")
    contract = ModelInputContract.from_training(data_spec=spec, schema=schema, normalizer=normalizer, skill_vocab=vocab)
    config = ModelConfig(d_model=8, n_layers=1, n_heads=2, num_kv_heads=1, ff_dim=16,
                         dropout=0.0, scene_capacity=1, history_capacity=4, history_reset_keep=4)
    with torch.random.fork_rng():
        torch.manual_seed(20261004)
        model = CausalPolicyModel(spec, config, vocab_size=vocab.size()).eval()
    path = tmp_path / "model.pt"
    torch.save({"data_spec": asdict(spec), "model_config": asdict(config),
                "input_contract": contract.to_dict(), "model_state_dict": model.state_dict()}, path)
    canonical = {
        "schema_version": CANONICAL_CONTEXT_SCHEMA_VERSION,
        "history_cursor": 3,
        "action_keys": actions.action_keys, "action_legal_mask": [True] * len(actions.action_keys),
        "skill_history_context": [
            {"skill_id": 3577, "skill_key": "fire_iv", "kind": 1, "potency": 310},
            {"skill_id": 900001, "skill_key": "zzzz_disabled", "kind": 1, "potency": 0},
            {"skill_id": 0, "skill_key": "ogcd_wait", "kind": 0, "potency": 0},
        ],
        "state_history_context": {"player_state_feature_keys": keys, "tokens": [
            {"player_state": [0.0, 10000.0, 1.0, 8000.0]},
            {"player_state": [2.0, 8000.0, 3.0, 6000.0]},
            {"player_state": [3.0, 6000.0, 4.0, 6000.0]},
        ]},
        "current_state_context": {"player_state_feature_keys": keys, "tokens": [
            {"player_state": [4.0, 6000.0, 5.0, 6200.0]},
        ]},
    }

    for name in ("state_history_context", "current_state_context"):
        canonical[name]["skill_availability_feature_keys"] = availability_keys
        for token in canonical[name]["tokens"]:
            token["player_state"].extend([0.0] * 4)
            token["skill_availability"] = [1] * layout.availability_dim

    def build(saved_vocab, saved_normalizer):
        return LiveBatchBuilder(
            backend=None, vocab=saved_vocab, normalizer=saved_normalizer, schema=schema,
            skill_feature_names=spec.skill_feature_names, device=torch.device("cpu"), max_history=4,
            model_config=config,
            action_keys=spec.action_keys, action_is_gcd=spec.action_is_gcd,
            scene_provider=SimpleNamespace(at_time=lambda _: (torch.tensor([[0.0, 1.0, 1.0]]), torch.zeros(1, dtype=torch.long)),
                                           state_at=lambda _: SimpleNamespace(next_downtime_eta=0.0, downtime_remaining=0.0)),
        ).build_from_canonical(canonical, gcd_phase=True, max_history=4)[0]

    before = build(vocab, contract.create_normalizer())
    reference = PyTorchPolicyBackend(path, device="cpu", use_kv_cache=use_kv_cache)
    expected = reference.raw_logits(before, spec.action_keys)
    if drift == "reorder_disabled":
        modified_skills = (*project.job.skills, replace(disabled, key="aaa_disabled"))
    elif drift == "add_disabled":
        modified_skills = (*original.job.skills, replace(disabled, key="aaa_new_disabled", game_id=900002))
    else:
        modified_skills = project.job.skills
    changed = replace(project, job=replace(project.job, skills=modified_skills))
    assert dict(SkillVocab.build_from_config(changed)) != dict(vocab)
    monkeypatch.setattr(vocab_module, "load_project_config", lambda *_a, **_kw: changed)
    monkeypatch.setattr(replay_module, "load_project_config", lambda *_a, **_kw: changed)
    backend = PyTorchPolicyBackend(path, device="cpu", use_kv_cache=use_kv_cache)
    replay_config = SimpleNamespace(job_tag="black_mage", cache_max_shards=1)
    with AutoregressiveReplaySession(replay_config, backend=backend, engine=SimpleNamespace(job_tag="black_mage")) as session:
        after = build(session.vocab, session.normalizer)
        assert session.vocab.to_dict() == vocab.to_dict()
        assert after["history_skill_ids"].tolist() == [[vocab.lookup(3577), vocab.lookup(900001), vocab.lookup(0)]]
        assert after["history_skill_ids"][0, -1] > 0
        for name in before:
            if isinstance(before[name], torch.Tensor):
                torch.testing.assert_close(after[name], before[name], atol=0, rtol=0)
        torch.testing.assert_close(backend.raw_logits(after, spec.action_keys), expected, atol=0, rtol=0)


class _FakeBackend:
    """最小绝对时间后端桩：驱动回放的 observe/submit/advance 接口。"""

    def __init__(self, state, skill):
        self.state = SimpleNamespace(**vars(state))
        self.state.ogcd_wait_boundary_pending = getattr(
            state,
            "ogcd_wait_boundary_pending",
            False,
        )
        self._skill = skill
        self.history = []

    def advance_to(self, timestamp):
        state = self.state
        seconds = float(timestamp) - float(state.time)
        self.state = SimpleNamespace(
            time=float(timestamp),
            gcd_remaining=max(0.0, state.gcd_remaining - seconds),
            downtime_remaining=0.0,
            natural_mp_tick_progress=0.0,
            cooldowns={},
            statuses={},
            dots={},
            fight_remaining=getattr(state, "fight_remaining", 600.0),
            ogcd_wait_boundary_pending=state.ogcd_wait_boundary_pending,
            gcd_index=getattr(state, "gcd_index", 0),
            current_gcd=2.5,
        )
        return SimpleNamespace(timestamp=float(timestamp), next_scheduled_event_time=None)

    def observe_at(self, timestamp, *, format="seconds", next_observation_timestamp=None):
        del next_observation_timestamp
        self.state.time = float(timestamp)
        if format == "vector":
            context = {"skill_history_context": list(self.history)}
        else:
            context = {
                "time_seconds": self.state.time,
                "gcd_index": getattr(self.state, "gcd_index", 0),
                "current_gcd_seconds": 2.5,
                "gcd_remaining_seconds": self.state.gcd_remaining,
                "fight_remaining_seconds": self.state.fight_remaining,
            }
        return SimpleNamespace(timestamp=self.state.time, context=context, next_scheduled_event_time=None)

    def submit_action(self, timestamp, action, **kwargs):
        del kwargs
        self.state.time = float(timestamp)
        if action == "fire":
            self.state.gcd_index = getattr(self.state, "gcd_index", 0) + 1
        self.history.append({"skill_key": action})
        return SimpleNamespace(
            accepted=True,
            queued=False,
            reason="",
            accepted_timestamp=float(timestamp),
            effect_timestamp=float(timestamp),
        )

    def record_policy_action(self, timestamp, action, next_observation_timestamp):
        del next_observation_timestamp
        self.state.time = float(timestamp)
        self.history.append({"skill_key": action})


def test_fake_backend_does_not_mutate_initial_state():
    initial_state = SimpleNamespace(time=0.0)

    backend = _FakeBackend(initial_state, _FAKE_GCD_SKILL)

    assert not hasattr(initial_state, "ogcd_wait_boundary_pending")
    assert backend.state.ogcd_wait_boundary_pending is False


def test_replay_device_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _resolve_device("cuda")


def test_replay_compiles_missing_cache_and_retries(monkeypatch, tmp_path):
    config = SimpleNamespace(
        scene_json_path=tmp_path / "scene.json",
        cache_dir=tmp_path / "cache",
        model_history_capacity=240,
        cache_shard_size=768,
        cache_max_shards=24,
    )
    normalizer = SimpleNamespace(
        ensure_job_resources=lambda _job_tag: None,
        normalization_contract={"version": 1},
    )
    reader = object()
    load_calls = []
    compile_calls = []

    def fake_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        return None if len(load_calls) == 1 else reader

    monkeypatch.setattr(replay_module, "load_raw_compiled_cache", fake_load)
    monkeypatch.setattr(
        replay_module,
        "precompile_raw_training_caches",
        lambda *args, **kwargs: compile_calls.append((args, kwargs)),
    )

    engine = object()
    action_space = replay_module.ActionSpace.from_job_tag("black_mage")
    vocab = SkillVocab.build_from_job_tag("black_mage")
    assert replay_module._load_replay_cache(
        config, "black_mage", normalizer, engine=engine, expected_action_space=action_space,
        expected_skill_vocab=vocab,
    ) is reader
    assert len(load_calls) == 2
    assert all(call[1]["expected_action_space"] == action_space for call in load_calls)
    assert compile_calls == [
        (
            ([config.scene_json_path],),
            {
                "job_tag": "black_mage",
                "normalizer": normalizer,
                "expected_action_space": action_space,
                "expected_skill_vocab": vocab,
                "int_dtype": torch.int32,
                "float_dtype": torch.float32,
                "cache_dir": config.cache_dir,
                "shard_size": 768,
                "max_workers": 1,
                "max_shards": 24,
                "engine": engine,
            },
        )
    ]


def test_replay_cache_store_reuses_reader_for_unchanged_scene(monkeypatch, tmp_path):
    source_path = tmp_path / "scene.json"
    source_path.write_text("{}", encoding="utf-8", newline="\n")
    config = SimpleNamespace(
        scene_json_path=source_path,
        cache_dir=tmp_path / "cache",
        model_history_capacity=240,
        cache_shard_size=768,
        cache_max_shards=24,
    )
    normalizer = SimpleNamespace(cache_signature={"version": 1})
    reader = object()
    load_calls = []

    monkeypatch.setattr(
        replay_module,
        "_load_replay_cache",
        lambda *args, **kwargs: load_calls.append((args, kwargs)) or reader,
    )

    store = ReplayCacheStore(max_shards=2)
    engine = object()
    action_space = replay_module.ActionSpace.from_job_tag("black_mage")
    vocab = SkillVocab.build_from_job_tag("black_mage")
    kwargs = dict(job_tag="black_mage", normalizer=normalizer, engine=engine,
                  expected_action_space=action_space, expected_skill_vocab=vocab)
    assert store.load(config, **kwargs) is reader
    assert store.load(config, **kwargs) is reader
    assert len(load_calls) == 1
    assert load_calls[0][1]["shard_cache"] is store._shard_cache
    # 同一 source 的 reader 不能跨模型动作契约复用。
    changed = replace(action_space, action_is_gcd=(not action_space.action_is_gcd[0], *action_space.action_is_gcd[1:]))
    assert store.load(config, **{**kwargs, "expected_action_space": changed}) is reader
    assert len(load_calls) == 2
    assert load_calls[-1][1]["expected_action_space"] == changed
    # 完整词表发生变化时，即使输出动作未变，也必须重新校验 reader。
    entries = list(vocab)
    entries[0], entries[1] = (entries[0][0], entries[1][1]), (entries[1][0], entries[0][1])
    changed_vocab = SkillVocab.from_entries(entries)
    assert store.load(config, **{**kwargs, "expected_skill_vocab": changed_vocab}) is reader
    assert len(load_calls) == 3


def test_replay_session_reset_reinitializes_backend_and_state_machine():
    class FakeBackend:
        model_config = SimpleNamespace(history_capacity=8, history_reset_keep=8, time_delta_scale=120.0)
        input_device = torch.device("cpu")

        def __init__(self):
            self.cache_calls = []

        def configure_cache(self, enabled):
            self.cache_calls.append(bool(enabled))

    class FakeStateMachine:
        def __init__(self):
            self.init_calls = []
            self.close_calls = 0

        def init(self, **kwargs):
            self.init_calls.append(kwargs)

        def close(self):
            self.close_calls += 1

    session = object.__new__(AutoregressiveReplaySession)
    session.backend = FakeBackend()
    state_machine = session._state_machine = FakeStateMachine()
    session.data_spec = SimpleNamespace(job_tag="black_mage")
    session._closed = False
    config = SimpleNamespace(
        job_tag="black_mage",
        use_kv_cache=False,
        base_gcd=2.5,
        max_history=768,
    )

    session.reset(config)
    session.reset(config)

    assert session.backend.cache_calls == [False, False]
    assert session.state_machine.init_calls == [
        {"actual_base_gcd": 2.5, "max_history": 768, "initial_timestamp": 0.0},
        {"actual_base_gcd": 2.5, "max_history": 768, "initial_timestamp": 0.0},
    ]
    session.close()
    session.close()
    assert state_machine.close_calls == 1
    assert session._state_machine is None


def test_replay_reset_for_trajectory_resets_scene_provider_before_session():
    replay = object.__new__(AutoregressiveReplay)
    calls = []

    class FakeSceneProvider:
        def reset(self):
            calls.append("scene")

    class FakeSession:
        def reset(self, config, *, initial_timestamp):
            del config, initial_timestamp
            calls.append("session")

    replay.scene_provider = FakeSceneProvider()
    replay._session = FakeSession()
    replay.config = SimpleNamespace(use_kv_cache=True)
    replay._observe_state = lambda timestamp: calls.append(("observe", timestamp)) or timestamp

    assert replay._reset_for_trajectory(initial_timestamp=3.5) == 3.5
    assert calls == ["scene", "session", ("observe", 3.5)]


@pytest.fixture
def constructor_config(tmp_path):
    return AutoregressiveReplayConfig(
        checkpoint_path=None,
        output_path=tmp_path / "rollout.md",
        scene_json_path=tmp_path / "scene.json",
        cache_dir=tmp_path / "cache",
        model_history_capacity=8,
        cache_shard_size=4,
        cache_max_shards=2,
        scene_duration_seconds=10.0,
        job_tag="black_mage",
        use_kv_cache=False,
        scene_mode="empty",
        max_history=8,
    )


def test_replay_requires_session_before_loading_scene(constructor_config):
    config = replace(constructor_config, scene_duration_seconds=None)
    assert not config.scene_json_path.exists()
    with pytest.raises(ValueError, match="requires a shared session"):
        AutoregressiveReplay(config, session=None)


def test_replay_uses_session_normalizer_for_batch_builder(monkeypatch, constructor_config):
    session_normalizer = object()
    schema = SimpleNamespace()
    reader = SimpleNamespace(
        job_tag="black_mage",
        schema=schema,
        skill_feature_names=("kind",),
    )

    class FakeInputSchema:
        def assert_compatible_with(self, other):
            assert other is schema

    input_contract = SimpleNamespace(
        schema=FakeInputSchema(),
    )

    class FakeBackend:
        input_device = torch.device("cpu")
        model_config = SimpleNamespace(history_capacity=8, history_reset_keep=8, time_delta_scale=120.0)

        def __init__(self):
            self.cache_calls = []

        def configure_cache(self, enabled):
            self.cache_calls.append(bool(enabled))

    backend = FakeBackend()
    session = SimpleNamespace(
        backend=backend,
        device=torch.device("cpu"),
        data_spec=SimpleNamespace(
            job_tag="black_mage",
            skill_feature_names=("kind",),
            action_keys=("fire",),
            action_is_gcd=(True,),
        ),
        input_contract=input_contract,
        vocab=object(),
        normalizer=session_normalizer,
        skill_book=object(),
        mp_tick_interval_seconds=None,
        state_machine=object(),
        load_reader=lambda _config: reader,
    )
    config = constructor_config
    captured = {}

    def fake_batch_builder(**kwargs):
        captured["batch_normalizer"] = kwargs["normalizer"]
        return object()

    monkeypatch.setattr(replay_module, "SceneTemplateProvider", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(replay_module, "LiveBatchBuilder", fake_batch_builder)

    AutoregressiveReplay(config, session=session)

    assert captured["batch_normalizer"] is session_normalizer
    assert backend.cache_calls == [False]


def test_replay_constructor_failure_keeps_session_ownership_with_caller(constructor_config):
    class FakeBackend:
        input_device = torch.device("cpu")

        def configure_cache(self, _enabled):
            return None

    class FakeSession:
        def __init__(self):
            self.backend = FakeBackend()
            self.device = torch.device("cpu")
            self.data_spec = SimpleNamespace(
                job_tag="black_mage",
                skill_feature_names=("kind",),
            )
            self.input_contract = SimpleNamespace()
            self.vocab = object()
            self.close_calls = 0

        def load_reader(self, _config):
            raise RuntimeError("cache load failed")

        def close(self):
            self.close_calls += 1

    config = constructor_config
    session = FakeSession()
    with pytest.raises(RuntimeError, match="cache load failed"):
        AutoregressiveReplay(config, session=session)

    assert session.close_calls == 0
    session.close()
    assert session.close_calls == 1


def test_replay_prediction_modes_and_result_helpers(monkeypatch, tmp_path):
    replay = object.__new__(AutoregressiveReplay)
    replay.device = torch.device("cpu")
    replay.precision = "float32"
    replay.config = SimpleNamespace(
        temperature=0.0,
        top_k=2,
        top_p=1.0,
        use_kv_cache=False,
        checkpoint_path=Path("checkpoint.pt"),
        scene_json_path=Path("scene.json"),
        max_history=8,
    )

    class FakeBatcher:
        @staticmethod
        def build(_state, *, max_history=None):
            return (
                {
                    "action_legal_mask": torch.tensor([[True, False, True]]),
                },
                ["first", "illegal", "third"],
            )

    class FakeBackend:
        name = "pytorch"
        source_path = Path("checkpoint.pt")
        execution_provider = "PyTorch:cpu"
        repetition = RepetitionConfig()

        @staticmethod
        def raw_logits(_batch, _action_keys):
            return torch.tensor([[1.0, 100.0, 2.0]])

        @staticmethod
        def metrics():
            return SimpleNamespace(to_dict=lambda: {})

    replay.batcher = FakeBatcher()
    replay.backend = FakeBackend()
    row = replay._predict_row(SimpleNamespace(), gcd_step=2)
    assert row.action_key == "third"
    assert row.top_actions[0][0] == "third"
    assert row.top_actions[1][0] == "first"

    replay.config = SimpleNamespace(
        temperature=1.0,
        top_k=1,
        top_p=0.7,
        use_kv_cache=False,
        checkpoint_path=Path("checkpoint.pt"),
        scene_json_path=Path("scene.json"),
        max_history=8,
    )
    sampled_probabilities = []

    def fake_multinomial(probabilities, count, *, generator=None):
        sampled_probabilities.append(probabilities.clone())
        return torch.tensor([2])

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    sampled = replay._predict_row(SimpleNamespace(), gcd_step=3, max_history=0)
    assert sampled.action_key == "third"
    assert torch.equal(sampled_probabilities[0], torch.tensor([0.0, 0.0, 1.0]))
    replay.batcher = SimpleNamespace(
        build=lambda _state, *, max_history=None: (
            {"action_legal_mask": torch.zeros((1, 3), dtype=torch.bool)},
            ["a", "b", "c"],
        )
    )
    with pytest.raises(RuntimeError, match="no legal action"):
        replay._predict_row(SimpleNamespace(), gcd_step=0)

    assert model_dtype("float32") is torch.float32
    assert model_dtype("float16") is torch.float16
    assert model_dtype("bf16") is torch.bfloat16
    with pytest.raises(ValueError, match="unsupported precision"):
        model_dtype("int8")
    with pytest.raises(ValueError, match="requires a CUDA device"):
        autocast_context(torch.device("cpu"), "bf16")

    result = replay._build_result([row], history_limit=2, is_history_ablation=True)
    assert isinstance(result, ReplayResult)
    assert result.history_limit == 2
    assert result.is_history_ablation is True


def test_replay_respects_shared_action_mask():
    replay = object.__new__(AutoregressiveReplay)
    replay.config = SimpleNamespace(
        temperature=0.0,
        top_k=3,
        top_p=1.0,
    )
    replay.skill_book = SimpleNamespace(
        get=lambda action_key: SimpleNamespace(
            kind=SimpleNamespace(
                value="gcd" if action_key == "fire" else "ogcd",
            ),
        ),
    )
    replay.backend = SimpleNamespace(
        repetition=RepetitionConfig(),
        raw_logits=lambda _batch, _action_keys: torch.tensor(
            [[100.0, 2.0, 90.0]],
        ),
    )

    row = replay._score_row(
        {"action_legal_mask": torch.tensor([[False, True, False]])},
        ["swiftcast", "fire", "potion"],
        gcd_step=1,
    )

    assert row.action_key == "fire"
    assert row.top_actions[0] == ("fire", 2.0, 1.0, True)
    assert all(
        legal is False
        for action_key, _logit, _probability, legal in row.top_actions
        if action_key in {"swiftcast", "potion"}
    )


def test_replay_waits_until_early_gcd_decision():
    replay = object.__new__(AutoregressiveReplay)
    replay._session = SimpleNamespace(reset=lambda _config, **_kwargs: None)
    replay.config = SimpleNamespace(
        initial_action=None,
        max_steps=2,
        max_gcds=1,
        use_kv_cache=False,
    )
    replay.scene_provider = None
    replay._mp_tick_interval_seconds = None
    replay.backend = SimpleNamespace(configure_cache=lambda _enabled: None)
    replay.skill_book = SimpleNamespace(
        get=lambda action_key: {
            "ogcd_wait": _FAKE_OGCD_SKILL,
            "fire": _FAKE_GCD_SKILL,
        }[action_key],
    )

    state = SimpleNamespace(
        time=0.0,
        gcd_remaining=1.0,
        fight_remaining=600.0,
        ogcd_wait_boundary_pending=False,
    )

    class ScriptedBackend:
        def __init__(self, initial_state):
            self.state = initial_state
            self.actions = []
            self.history = []

        def observe_at(self, timestamp, *, format="seconds", next_observation_timestamp=None):
            del next_observation_timestamp
            self.state.time = float(timestamp)
            context = (
                {"skill_history_context": list(self.history)}
                if format == "vector"
                else {
                    "time_seconds": self.state.time,
                    "gcd_index": 0,
                    "gcd_remaining_seconds": self.state.gcd_remaining,
                    "fight_remaining_seconds": self.state.fight_remaining,
                    "current_gcd_seconds": 2.5,
                }
            )
            return SimpleNamespace(
                timestamp=self.state.time,
                context=context,
                next_scheduled_event_time=None,
            )

        def submit_action(self, timestamp, action_key):
            self.actions.append((action_key, "submit"))
            self.state.time = float(timestamp)
            self.history.append({"skill_key": action_key})
            return SimpleNamespace(
                accepted=True,
                queued=False,
                reason="",
                accepted_timestamp=float(timestamp),
                effect_timestamp=float(timestamp),
            )

        def record_policy_action(self, timestamp, action_key, next_observation_timestamp):
            del next_observation_timestamp
            self.actions.append((action_key, "policy"))
            self.state.time = float(timestamp)
            self.history.append({"skill_key": action_key})

        def advance_to(self, timestamp):
            delta = float(timestamp) - float(self.state.time)
            self.state.time = float(timestamp)
            self.state.gcd_remaining = max(0.0, self.state.gcd_remaining - delta)
            return SimpleNamespace(timestamp=self.state.time, next_scheduled_event_time=None)

    replay._state_machine = ScriptedBackend(state)
    seen_restrictions = []

    def predict(_state, *, gcd_step):
        gcd_phase = _state.gcd_remaining <= 0.050001
        seen_restrictions.append(gcd_phase)
        if gcd_phase:
            return ReplayRow(gcd_step, "fire", 1.0, (("fire", 1.0, 1.0, True),))
        return ReplayRow(
            gcd_step,
            "ogcd_wait",
            1.0,
            (("ogcd_wait", 1.0, 1.0, True),),
        )

    replay._predict_row = predict
    rows, _snapshots, _final_state = replay._generate_full_trajectory()

    assert [row.action_key for row in rows] == ["ogcd_wait", "fire"]
    assert seen_restrictions == [False, True]


def test_top_p_keeps_minimum_probability_nucleus():
    probabilities = torch.tensor([0.5, 0.3, 0.2, 0.0])
    descending_order = torch.argsort(probabilities, descending=True)

    filtered = replay_module._apply_top_p(probabilities, descending_order, 0.75)

    assert torch.allclose(filtered, torch.tensor([0.625, 0.375, 0.0, 0.0]))
    assert replay_module._apply_top_p(probabilities, descending_order, 1.0) is probabilities


def test_replay_delegates_kv_cache_toggle_to_backend():
    replay = object.__new__(AutoregressiveReplay)
    replay.device = torch.device("cpu")
    replay.precision = "float32"
    replay.config = SimpleNamespace(
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        checkpoint_path=Path("checkpoint.pt"),
        scene_json_path=Path("scene.json"),
        max_history=8,
    )

    class FakeBackend:
        repetition = RepetitionConfig()

        def __init__(self):
            self.kv_cache_enabled = False
            self.calls = 0

        def configure_cache(self, enabled):
            self.kv_cache_enabled = bool(enabled)

        def raw_logits(self, _batch, _action_keys):
            assert self.kv_cache_enabled is True
            self.calls += 1
            return torch.tensor([[1.0, 2.0]])

    class FakeBatcher:
        @staticmethod
        def build(_state, *, max_history=None):
            return (
                {"action_legal_mask": torch.tensor([[True, True]])},
                ["first", "second"],
            )

    backend = FakeBackend()
    replay.backend = backend
    replay.batcher = FakeBatcher()
    replay._configure_kv_cache(True)

    row = replay._predict_row(SimpleNamespace(), gcd_step=0)

    assert row.action_key == "second"
    assert backend.calls == 1


def test_replay_history_ablation_guards(monkeypatch, tmp_path):
    replay = object.__new__(AutoregressiveReplay)
    replay.config = SimpleNamespace(
        max_history=8,
        checkpoint_path=Path("checkpoint.pt"),
        scene_json_path=Path("scene.json"),
    )

    reference = ReplayRow(1, "a", 1.0, (("a", 0.0, 1.0, True),))
    replay._generate_full_trajectory = lambda: (
        [reference],
        (ReplaySnapshot({"canonical": "x"}, reference, gcd_phase=False),),
        SimpleNamespace(),
    )
    seen_forbid_flags = []

    def predict_from_canonical(*_args, **kwargs):
        assert kwargs["gcd_phase"] is False
        seen_forbid_flags.append(kwargs["max_history"])
        return ReplayRow(1, "b", 0.5, (("b", 0.0, 0.5, True),))

    replay._predict_from_canonical = predict_from_canonical
    replay._build_result = lambda rows, **kwargs: (tuple(rows), kwargs)
    with pytest.raises(ValueError, match="at least one"):
        replay.run_history_ablation(())
    with pytest.raises(ValueError, match=">= 0"):
        replay.run_history_ablation((-1,))
    with pytest.raises(ValueError, match="unique"):
        replay.run_history_ablation((1, 1))
    results = replay.run_history_ablation((0, 2))
    assert len(results) == 3
    assert results[1][1]["history_limit"] == 0
    assert seen_forbid_flags == [0, 2]


def test_replay_event_timeline_advances_action_completion_and_gcd_window(cs_backend):
    replay = object.__new__(AutoregressiveReplay)
    replay.config = SimpleNamespace()
    replay.scene_provider = None
    replay._state_machine = cs_backend
    replay.skill_book = SimpleNamespace(
        get=lambda key: SimpleNamespace(
            key=key,
            kind=SimpleNamespace(value="gcd" if key == "fire_iii" else "ogcd"),
        )
    )
    cs_backend.init(initial_timestamp=0.0)
    initial = replay._observe_state(0.0)
    swiftcast_result = cs_backend.submit_action(0.0, "swiftcast")
    after_swiftcast = replay._advance_submitted_action(initial, "swiftcast", swiftcast_result)
    assert after_swiftcast.time == pytest.approx(0.1)

    gcd_result = cs_backend.submit_action(after_swiftcast.time, "fire_iii")
    after_gcd = replay._advance_submitted_action(
        replay._observe_state(after_swiftcast.time), "fire_iii", gcd_result
    )
    assert after_gcd.time == pytest.approx(0.15)
    assert after_gcd.gcd_remaining == pytest.approx(2.45)

    ogcd_result = cs_backend.submit_action(after_gcd.time, "amplifier")
    after_ogcd = replay._advance_submitted_action(
        replay._observe_state(after_gcd.time), "amplifier", ogcd_result
    )
    assert after_ogcd.time == pytest.approx(0.25)
    assert after_ogcd.gcd_remaining == pytest.approx(2.35)

    cs_backend.record_policy_action(after_ogcd.time, "ogcd_wait", 2.9)
    after_window = replay._advance_event_time(
        replay._observe_state(after_ogcd.time),
        after_ogcd.gcd_remaining,
        interrupt_on_scene_event=True,
    )
    assert after_window.time == pytest.approx(2.6)
    assert after_window.gcd_remaining == pytest.approx(0.0)



def test_replay_no_legal_candidate_waits_until_next_event_and_continues():
    replay = object.__new__(AutoregressiveReplay)
    replay._session = SimpleNamespace(reset=lambda _config, **_kwargs: None)
    replay.config = SimpleNamespace(
        initial_action=None,
        initial_time_seconds=None,
        max_steps=2,
        max_gcds=1,
        use_kv_cache=False,
    )
    replay.scene_provider = None
    replay._mp_tick_interval_seconds = 3.0
    replay.skill_book = SimpleNamespace(get=lambda _key: _FAKE_GCD_SKILL)

    initial_state = SimpleNamespace(
        time=0.0,
        gcd_remaining=0.5,
        downtime_remaining=0.0,
        natural_mp_tick_progress=0.0,
        cooldowns={},
        statuses={},
        dots={},
        fight_remaining=600.0,
    )
    replay._state_machine = _FakeBackend(initial_state, _FAKE_GCD_SKILL)
    replay.backend = SimpleNamespace(configure_cache=lambda _enabled: None)

    def predict(state, *, gcd_step):
        if state.gcd_remaining > 0.0:
            raise NoLegalActionError("locked")
        return ReplayRow(gcd_step, "fire", 1.0, (("fire", 1.0, 1.0, True),))

    replay._predict_row = predict
    rows, _snapshots, final_state = replay._generate_full_trajectory()

    assert [row.action_key for row in rows] == ["fire"]
    assert final_state.time == pytest.approx(0.55)


def test_replay_no_legal_candidate_reports_finished_fight():
    replay = object.__new__(AutoregressiveReplay)
    replay._session = SimpleNamespace(reset=lambda _config, **_kwargs: None)
    replay.config = SimpleNamespace(
        initial_action=None,
        max_steps=1,
        max_gcds=1,
        use_kv_cache=False,
    )
    replay.scene_provider = None
    replay._mp_tick_interval_seconds = 3.0

    state = SimpleNamespace(
        time=600.0,
        fight_remaining=0.0,
        gcd_remaining=0.0,
        downtime_remaining=0.0,
        natural_mp_tick_progress=3.0,  # mp tick 已完成，避免产生等待事件
        cooldowns={},
        statuses={},
        dots={},
    )
    replay._state_machine = _FakeBackend(state, _FAKE_GCD_SKILL)
    replay.backend = SimpleNamespace(configure_cache=lambda _enabled: None)
    replay._predict_row = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        NoLegalActionError("finished")
    )

    with pytest.raises(RuntimeError, match="no future time event"):
        replay._generate_full_trajectory()


def test_replay_rejects_unreached_max_gcd_target():
    replay = object.__new__(AutoregressiveReplay)
    replay._session = SimpleNamespace(reset=lambda _config, **_kwargs: None)
    replay.config = SimpleNamespace(
        initial_action=None,
        initial_time_seconds=None,
        max_steps=2,
        max_gcds=1,
        use_kv_cache=False,
    )
    replay.scene_provider = None
    replay._mp_tick_interval_seconds = 3.0
    replay.skill_book = SimpleNamespace(get=lambda _key: _FAKE_OGCD_SKILL)

    state = SimpleNamespace(
        time=0.0,
        fight_remaining=600.0,
        gcd_remaining=0.0,
        downtime_remaining=0.0,
        natural_mp_tick_progress=0.0,
        cooldowns={},
        statuses={},
        dots={},
    )
    replay._state_machine = _FakeBackend(state, _FAKE_OGCD_SKILL)
    replay.backend = SimpleNamespace(configure_cache=lambda _enabled: None)
    replay._predict_row = lambda *_args, **_kwargs: ReplayRow(
        0,
        "amplifier",
        1.0,
        (("amplifier", 0.0, 1.0, True),),
    )
    with pytest.raises(RuntimeError, match="did not reach requested GCD target"):
        replay._generate_full_trajectory()


def test_markdown_output_and_cli_history_ablation(monkeypatch, tmp_path):
    assert SKILL_NAMES["ogcd_wait"] == "-"
    rows = (
        ReplayRow(0, "fire_iii", 1.0, (("fire_iii", 0.0, 1.0, True),), forced=True),
        ReplayRow(
            1,
            "fire_iv",
            0.7,
            (("fire_iv", 1.0, 0.7, True), ("blizzard_iii", -1.0, 0.2, False)),
            reference_action_key="fire_iii",
        ),
    )
    output = tmp_path / "nested" / "rollout.md"
    path = write_markdown(
        ReplayResult(
            rows, Path("checkpoint.pt"), Path("scene.pt"), "cpu",
            backend_metrics={"scope": "backend_lifetime"},
        ),
        output,
    )
    text = path.read_text(encoding="utf-8")
    assert "爆炎（强制）" in text
    assert "炽炎 0.700" in text
    assert "冰封 0.200" in text
    assert "共享后端累计；内存为进程/PyTorch 设备峰值" in text

    ort_output = tmp_path / "ort.md"
    write_markdown(
        ReplayResult(
            rows,
            None,
            Path("scene.pt"),
            "cpu",
            backend_name="onnxruntime",
            model_source_path=Path("model.onnx"),
        ),
        ort_output,
    )
    assert "checkpoint: `None`" not in ort_output.read_text(encoding="utf-8")

    ablation_path = tmp_path / "ablation.md"
    write_markdown(
        ReplayResult(
            rows,
            Path("checkpoint.pt"),
            Path("scene.pt"),
            "cpu",
            history_limit=2,
            is_history_ablation=True,
        ),
        ablation_path,
    )
    assert "截断历史后的技能" in ablation_path.read_text(encoding="utf-8")

    calls = []
    config_kwargs = {}

    def fake_load_replay_config(**kwargs):
        config_kwargs.update(kwargs)
        return SimpleNamespace(output_path=tmp_path / "rollout.md")

    monkeypatch.setattr(
        replay_main_module,
        "load_replay_config",
        fake_load_replay_config,
    )
    monkeypatch.setattr(replay_main_module, "write_markdown", lambda result, path: calls.append((result, path)) or path)
    monkeypatch.setattr(
        replay_main_module,
        "argparse",
        replay_main_module.argparse,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "autoregressive_replay",
            "--history-ablation",
            "2",
            "4",
            "--temperature",
            "0.3",
            "--top-p",
            "0.85",
            "--backend",
            "onnxruntime",
            "--onnx-package",
            "deployment",
            "--ort-provider",
            "CPUExecutionProvider",
            "--no-use-kv-cache",
        ],
    )
    def fake_replays(configs, *, workers, history_limits):
        assert len(configs) == 1
        calls.append(("ablation", history_limits))
        yield (
            ReplayResult((), Path("checkpoint.pt"), Path("scene.pt"), "cpu"),
            ReplayResult((), Path("checkpoint.pt"), Path("scene.pt"), "cpu", history_limit=2),
            ReplayResult((), Path("checkpoint.pt"), Path("scene.pt"), "cpu", history_limit=4),
        )

    monkeypatch.setattr(replay_main_module, "run_replays", fake_replays)
    replay_main_module.main()
    assert ("ablation", (2, 4)) in calls
    assert config_kwargs["top_p"] == 0.85
    assert config_kwargs["backend"] == "onnxruntime"
    assert config_kwargs["onnx_package"] == Path("deployment")
    assert config_kwargs["ort_provider"] == "CPUExecutionProvider"
    assert config_kwargs["use_kv_cache"] is False

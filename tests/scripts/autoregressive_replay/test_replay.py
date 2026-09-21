"""自回归回放核心逻辑与 CLI 测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from common.torch_runtime import autocast_context, model_dtype
from common.policy.model import RepetitionConfig
from scripts.autoregressive_replay import main as replay_main_module
from scripts.autoregressive_replay.outputs.markdown import SKILL_NAMES, write_markdown
from scripts.autoregressive_replay.replay import (
    AutoregressiveReplay,
    AutoregressiveReplaySession,
    NoLegalCandidateError,
    ReplayCacheStore,
    ReplayResult,
    ReplayRow,
    ReplaySnapshot,
    _resolve_device,
)
from scripts.autoregressive_replay import replay as replay_module


_FAKE_GCD_SKILL = SimpleNamespace(key="fire", kind=SimpleNamespace(value="gcd"))
_FAKE_OGCD_SKILL = SimpleNamespace(key="ogcd_wait", kind=SimpleNamespace(value="ogcd"))


class _FakeSidecar:
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

    def apply_external_event(self, *args, **kwargs):
        del args, kwargs
        return None


def test_fake_sidecar_does_not_mutate_initial_state():
    initial_state = SimpleNamespace(time=0.0)

    sidecar = _FakeSidecar(initial_state, _FAKE_GCD_SKILL)

    assert not hasattr(initial_state, "ogcd_wait_boundary_pending")
    assert sidecar.state.ogcd_wait_boundary_pending is False


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

    assert replay_module._load_replay_cache(config, "black_mage", normalizer) is reader
    assert len(load_calls) == 2
    assert compile_calls == [
        (
            ([config.scene_json_path],),
            {
                "job_tag": "black_mage",
                "normalizer": normalizer,
                "int_dtype": torch.int32,
                "float_dtype": torch.float32,
                "cache_dir": config.cache_dir,
                "shard_size": 768,
                "max_workers": 1,
                "max_shards": 24,
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
    assert store.load(config, job_tag="black_mage", normalizer=normalizer) is reader
    assert store.load(config, job_tag="black_mage", normalizer=normalizer) is reader
    assert len(load_calls) == 1
    assert load_calls[0][1]["shard_cache"] is store._shard_cache


def test_replay_session_reset_reinitializes_backend_and_sidecar():
    class FakeBackend:
        input_device = torch.device("cpu")

        def __init__(self):
            self.cache_calls = []

        def configure_cache(self, enabled):
            self.cache_calls.append(bool(enabled))

    class FakeSidecar:
        def __init__(self):
            self.init_calls = []
            self.close_calls = 0

        def init(self, **kwargs):
            self.init_calls.append(kwargs)

        def close(self):
            self.close_calls += 1

    session = object.__new__(AutoregressiveReplaySession)
    session.backend = FakeBackend()
    session.sidecar = FakeSidecar()
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
    assert session.sidecar.init_calls == [
        {"actual_base_gcd": 2.5, "max_history": 768, "initial_timestamp": 0.0},
        {"actual_base_gcd": 2.5, "max_history": 768, "initial_timestamp": 0.0},
    ]
    session.close()
    session.close()
    assert session.sidecar.close_calls == 1


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


def test_replay_uses_session_normalizer_for_context_builders(monkeypatch):
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
            candidate_action_keys=("fire",),
        ),
        input_contract=input_contract,
        vocab=object(),
        normalizer=session_normalizer,
        skill_book=object(),
        mp_tick_interval_seconds=None,
        sidecar=object(),
        load_reader=lambda _config: reader,
    )
    config = SimpleNamespace(
        job_tag="black_mage",
        use_kv_cache=False,
        scene_mode="empty",
        scene_sample_index=0,
        max_history=8,
    )
    captured = {}

    def fake_scene_provider(*_args, **kwargs):
        captured["scene_normalizer"] = kwargs["normalizer"]
        return object()

    def fake_batch_builder(**kwargs):
        captured["batch_normalizer"] = kwargs["normalizer"]
        return object()

    monkeypatch.setattr(replay_module, "SceneTemplateProvider", fake_scene_provider)
    monkeypatch.setattr(replay_module, "LiveBatchBuilder", fake_batch_builder)

    AutoregressiveReplay(config, session=session)

    assert captured["scene_normalizer"] is session_normalizer
    assert captured["batch_normalizer"] is session_normalizer
    assert backend.cache_calls == [False]


def test_replay_closes_owned_session_when_constructor_fails(monkeypatch):
    class FakeBackend:
        input_device = torch.device("cpu")

        def configure_cache(self, _enabled):
            return None

    class FakeSession:
        instance = None

        def __init__(self, _config, *, backend=None):
            self.backend = FakeBackend() if backend is None else backend
            self.device = torch.device("cpu")
            self.data_spec = SimpleNamespace(
                job_tag="black_mage",
                skill_feature_names=("kind",),
            )
            self.input_contract = SimpleNamespace()
            self.vocab = object()
            self.close_calls = 0
            FakeSession.instance = self

        def load_reader(self, _config):
            raise RuntimeError("cache load failed")

        def close(self):
            self.close_calls += 1

    monkeypatch.setattr(replay_module, "AutoregressiveReplaySession", FakeSession)
    config = SimpleNamespace(
        scene_json_path=None,
        job_tag="black_mage",
        use_kv_cache=False,
        scene_mode="empty",
        scene_sample_index=0,
        max_history=8,
    )

    with pytest.raises(RuntimeError, match="cache load failed"):
        AutoregressiveReplay(config)

    assert FakeSession.instance is not None
    assert FakeSession.instance.close_calls == 1


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
                    "candidate_legal_mask": torch.tensor([[True, False, True]]),
                },
                ["first", "illegal", "third"],
            )

    class FakeBackend:
        name = "pytorch"
        source_path = Path("checkpoint.pt")
        execution_provider = "PyTorch:cpu"
        repetition = RepetitionConfig()

        @staticmethod
        def raw_logits(_batch, _candidate_keys):
            return torch.tensor([[1.0, 100.0, 2.0]])

        @staticmethod
        def metrics():
            return SimpleNamespace(to_dict=lambda: {})

    replay.batcher = FakeBatcher()
    replay.backend = FakeBackend()
    row = replay._predict_row(SimpleNamespace(), gcd_step=2)
    assert row.action_key == "third"
    assert row.top_candidates[0][0] == "third"
    assert row.top_candidates[1][0] == "first"

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

    def fake_multinomial(probabilities, count):
        sampled_probabilities.append(probabilities.clone())
        return torch.tensor([2])

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    sampled = replay._predict_row(SimpleNamespace(), gcd_step=3, max_history=0)
    assert sampled.action_key == "third"
    assert torch.equal(sampled_probabilities[0], torch.tensor([0.0, 0.0, 1.0]))
    replay.batcher = SimpleNamespace(
        build=lambda _state, *, max_history=None: (
            {"candidate_legal_mask": torch.zeros((1, 3), dtype=torch.bool)},
            ["a", "b", "c"],
        )
    )
    with pytest.raises(RuntimeError, match="no legal candidate"):
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


def test_replay_respects_shared_candidate_mask():
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
        raw_logits=lambda _batch, _candidate_keys: torch.tensor(
            [[100.0, 2.0, 90.0]],
        ),
    )

    row = replay._score_row(
        {"candidate_legal_mask": torch.tensor([[False, True, False]])},
        ["swiftcast", "fire", "potion"],
        gcd_step=1,
    )

    assert row.action_key == "fire"
    assert row.top_candidates[0] == ("fire", 2.0, 1.0, True)
    assert all(
        legal is False
        for action_key, _logit, _probability, legal in row.top_candidates
        if action_key in {"swiftcast", "potion"}
    )


def test_replay_waits_until_early_gcd_decision():
    replay = object.__new__(AutoregressiveReplay)
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

    class ScriptedSidecar:
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

    replay._sidecar = ScriptedSidecar(state)
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

        def raw_logits(self, _batch, _candidate_keys):
            assert self.kv_cache_enabled is True
            self.calls += 1
            return torch.tensor([[1.0, 2.0]])

    class FakeBatcher:
        @staticmethod
        def build(_state, *, max_history=None):
            return (
                {"candidate_legal_mask": torch.tensor([[True, True]])},
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
        (ReplaySnapshot({"canonical": "x"}, reference),),
        SimpleNamespace(),
    )
    seen_forbid_flags = []

    def predict_from_canonical(*_args, **kwargs):
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
    replay._sidecar = cs_backend
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
    replay._sidecar = _FakeSidecar(initial_state, _FAKE_GCD_SKILL)
    replay.backend = SimpleNamespace(configure_cache=lambda _enabled: None)

    def predict(state, *, gcd_step):
        if state.gcd_remaining > 0.0:
            raise NoLegalCandidateError("locked")
        return ReplayRow(gcd_step, "fire", 1.0, (("fire", 1.0, 1.0, True),))

    replay._predict_row = predict
    rows, _snapshots, final_state = replay._generate_full_trajectory()

    assert [row.action_key for row in rows] == ["fire"]
    assert final_state.time == pytest.approx(0.55)


def test_replay_no_legal_candidate_reports_finished_fight():
    replay = object.__new__(AutoregressiveReplay)
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
    replay._sidecar = _FakeSidecar(state, _FAKE_GCD_SKILL)
    replay.backend = SimpleNamespace(configure_cache=lambda _enabled: None)
    replay._predict_row = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        NoLegalCandidateError("finished")
    )

    with pytest.raises(RuntimeError, match="no future time event"):
        replay._generate_full_trajectory()


def test_replay_rejects_unreached_max_gcd_target():
    replay = object.__new__(AutoregressiveReplay)
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
    replay._sidecar = _FakeSidecar(state, _FAKE_OGCD_SKILL)
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
        ReplayResult(rows, Path("checkpoint.pt"), Path("scene.pt"), "cpu"),
        output,
    )
    text = path.read_text(encoding="utf-8")
    assert "爆炎（强制）" in text
    assert "炽炎 0.700" in text
    assert "冰封 0.200" in text

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
    monkeypatch.setattr(replay_main_module, "AutoregressiveReplay", lambda config: calls.append(config) or "replay")
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
    class FakeReplay:
        def run_history_ablation(self, limits):
            calls.append(("ablation", limits))
            return (
                ReplayResult((), Path("checkpoint.pt"), Path("scene.pt"), "cpu"),
                ReplayResult((), Path("checkpoint.pt"), Path("scene.pt"), "cpu", history_limit=2),
                ReplayResult((), Path("checkpoint.pt"), Path("scene.pt"), "cpu", history_limit=4),
            )

    calls.clear()
    monkeypatch.setattr(replay_main_module, "AutoregressiveReplay", lambda config: FakeReplay())
    replay_main_module.main()
    assert ("ablation", (2, 4)) in calls
    assert config_kwargs["top_p"] == 0.85
    assert config_kwargs["backend"] == "onnxruntime"
    assert config_kwargs["onnx_package"] == Path("deployment")
    assert config_kwargs["ort_provider"] == "CPUExecutionProvider"
    assert config_kwargs["use_kv_cache"] is False

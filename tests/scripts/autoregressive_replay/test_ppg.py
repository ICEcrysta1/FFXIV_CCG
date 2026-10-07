"""自回归 PPG 指标测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import scripts.autoregressive_replay.ppg as ppg_module
from scripts.autoregressive_replay.ppg import (
    PpgResult,
    _infer_initial_base_gcd,
    _run_rollout,
    _run_rollout_until_time,
    _select_legal_action,
)
from common.output_context_schema import CANONICAL_CONTEXT_SCHEMA_VERSION
from common.policy.data import Normalizer
from common.policy.config import ModelConfig
from common.policy.data.schema import TRAINING_SAMPLE_SCHEMA_VERSION, TrainingSchema, SceneWindowSchema
from scripts.autoregressive_replay.context import LiveBatchBuilder


class _FakeState:
    def __init__(self):
        self.cumulative_potency = 0.0
        self.cumulative_dot_potency = 0.0
        self.action_index = 0
        self.gcd_index = 0
        self.time = 0.0
        self.gcd_remaining = 0.0
        self.fight_remaining = 600.0


class _FakeBackend:
    """最小绝对时间后端桩：驱动 PPG 的 observe/submit/advance 接口。"""

    def __init__(self):
        self.state = _FakeState()

    def observe_at(self, timestamp, *, format="seconds", next_observation_timestamp=None):
        del next_observation_timestamp
        self.state.time = float(timestamp)
        if format == "vector":
            context = {
                "current_state_context": {
                    "target_buff_state_feature_keys": [
                        "request_state.target.cumulative_potency",
                        "request_state.target.cumulative_dot_potency",
                    ],
                    "tokens": [
                        {
                            "target_buff_state": [
                                self.state.cumulative_potency,
                                self.state.cumulative_dot_potency,
                            ]
                        }
                    ],
                }
            }
        else:
            context = {
                "time_seconds": self.state.time,
                "gcd_index": self.state.gcd_index,
                "current_gcd_seconds": 2.5,
                "gcd_remaining_seconds": self.state.gcd_remaining,
                "cast_remaining_seconds": getattr(self.state, "cast_remaining", 0.0),
                "fight_remaining_seconds": self.state.fight_remaining,
            }
        return SimpleNamespace(
            timestamp=self.state.time,
            next_scheduled_event_time=None,
            context=context,
        )

    def submit_action(self, _timestamp, _action_key, **kwargs):
        del kwargs
        state = self.state
        state.action_index += 1
        is_gcd = state.action_index in {2, 3, 5}
        if is_gcd:
            state.gcd_index += 1
            state.cumulative_potency += 100.0
            state.cumulative_dot_potency += 20.0
        state.gcd_remaining = 1.0 if is_gcd else 0.0
        return SimpleNamespace(
            accepted=True,
            queued=False,
            reason="",
            accepted_timestamp=state.time,
            effect_timestamp=state.time,
        )

    def advance_to(self, timestamp):
        self.state.time = float(timestamp)
        self.state.gcd_remaining = max(0.0, self.state.gcd_remaining - 1.0)
        return SimpleNamespace(timestamp=self.state.time, next_scheduled_event_time=None)

    def record_policy_action(self, timestamp, action, next_observation_timestamp):
        del action, next_observation_timestamp
        self.state.time = float(timestamp)


class _FakeBatcher:
    @staticmethod
    def build(_state):
        return (
            {"action_legal_mask": torch.tensor([[True, False]])},
            ["gcd", "illegal"],
        )


class _FakeModel:
    @staticmethod
    def eval():
        return None

    @staticmethod
    def __call__(_batch):
        return {"logits": torch.tensor([[2.0, 100.0]])}


def test_real_ppg_counts_queued_gcds_and_masks_policy_wait(cs_backend, monkeypatch):
    # 连续瞬发制造完整插入窗口，真实 Sidecar 负责排队与瞬发资源消耗。
    cs_backend.submit_action(0.0, "swiftcast")
    cs_backend.submit_action(0.0, "triplecast")
    canonical = cs_backend.observe_at(0.0, format="vector", next_observation_timestamp=0.0).context
    state_context = canonical["current_state_context"]
    groups = {group: state_context[f"{group}_feature_keys"]
              for group in ("player_state", "buff_state", "target_buff_state", "resource_state")}
    keys = tuple(canonical["action_keys"])
    from common.policy.data.action_space import ActionSpace
    action_space = ActionSpace.from_job_tag("black_mage")
    schema = TrainingSchema(
        serialization_format="test", sample_schema_version=TRAINING_SAMPLE_SCHEMA_VERSION,
        context_schema_version=CANONICAL_CONTEXT_SCHEMA_VERSION, scene_context_mode="absolute",
        state_group_feature_keys=groups, skill_history_fields=("kind",),
        scene_windows=(SceneWindowSchema.from_feature_keys(
            context_key="targetable_window_context", scene_type_id=0,
            feature_keys=("start_offset_seconds", "end_offset_seconds", "duration_seconds"),
        ),),
    )
    normalizer = Normalizer()
    normalizer.ensure_job_resources("black_mage")
    builder = LiveBatchBuilder(
        backend=cs_backend,
        vocab=SimpleNamespace(require_lookup=lambda value, **kwargs: value),
        normalizer=normalizer, schema=schema,
        skill_feature_names=("kind",),
        scene_provider=SimpleNamespace(at_time=lambda _: (torch.zeros((0, 3)), torch.zeros(0, dtype=torch.int32))),
        device=torch.device("cpu"), max_history=20,
        model_config=ModelConfig(history_capacity=20),
        action_keys=keys, action_is_gcd=action_space.action_is_gcd,
    )

    class Model:
        def eval(self):
            pass

        def __call__(self, batch):
            return {"logits": torch.tensor([[100.0 if key == "blizzard_iii" else
                                             90.0 if key == "ogcd_wait" else -1000.0 for key in keys]])}

    submissions = []
    submit = cs_backend.submit_action

    def tracked_submit(timestamp, key):
        result = submit(timestamp, key)
        submissions.append(result)
        return result

    monkeypatch.setattr(cs_backend, "submit_action", tracked_submit)
    result = _run_rollout(Model(), cs_backend, builder, gcd_count=3, normalization=1000.0,
                          precision="float32", device=torch.device("cpu"), expected_action_keys=keys)
    assert result.output_gcds == 3
    assert result.cumulative_potency > 0
    assert len(submissions) == 3
    assert [item.request_timestamp for item in submissions] == pytest.approx([0.0, 2.45, 4.9])
    assert [item.queued for item in submissions] == [False, True, True]


@pytest.mark.parametrize(
    ("logits", "legal", "expected_index", "expected_has_legal"),
    (
        ([1.0, 10.0, 2.0], [True, False, True], 2, True),
        ([1.0, 10.0], [False, False], 0, False),
    ),
)
def test_select_legal_action_returns_index_and_legality(
    logits,
    legal,
    expected_index,
    expected_has_legal,
):
    selected_index, has_legal = _select_legal_action(
        torch.tensor(logits),
        torch.tensor(legal),
    )

    assert selected_index == expected_index
    assert has_legal is expected_has_legal


def test_ppg_rollout_counts_output_gcds_and_includes_dot_potency():
    result = _run_rollout(
        _FakeModel(),
        _FakeBackend(),
        _FakeBatcher(),
        gcd_count=3,
        normalization=1000.0,
        precision="float32",
        device=torch.device("cpu"),
        expected_action_keys=("gcd", "illegal"),
    )

    assert result.output_gcds == 3
    assert result.cumulative_potency == pytest.approx(300.0)
    assert result.cumulative_dot_potency == pytest.approx(60.0)
    assert result.ppg == pytest.approx(120.0)
    assert result.normalized_ppg == pytest.approx(0.12)


def test_ppg_rollout_rejects_action_order_mismatch():
    with pytest.raises(ValueError, match="action order mismatch"):
        _run_rollout(
            _FakeModel(),
            _FakeBackend(),
            _FakeBatcher(),
            gcd_count=1,
            normalization=1000.0,
            precision="float32",
            device=torch.device("cpu"),
            expected_action_keys=("illegal", "gcd"),
        )


def test_validation_rollout_stops_at_fight_end_and_uses_executed_history():
    class FakeState(_FakeState):
        def __init__(self):
            super().__init__()
            self.time = 0.0

    class FakeBackend(_FakeBackend):
        def __init__(self):
            self.state = FakeState()

        def submit_action(self, timestamp, _action_key, **kwargs):
            del kwargs
            self.state.time = float(timestamp)
            self.state.cast_remaining = 1.0
            self.state.action_index += 1
            self.state.gcd_index += 1
            self.state.cumulative_potency += 100.0
            self.state.cumulative_dot_potency += 20.0
            return SimpleNamespace(
                accepted=True,
                queued=False,
                reason="",
                accepted_timestamp=self.state.time,
                effect_timestamp=self.state.time + 0.5,
            )

        def advance_to(self, timestamp):
            self.state.time = float(timestamp)
            return SimpleNamespace(timestamp=self.state.time, next_scheduled_event_time=None)

    class FakeSceneProvider:
        @staticmethod
        def next_state_event_after(_time_seconds):
            return None

        @staticmethod
        def targetable_windows():
            return []

    result = _run_rollout_until_time(
        _FakeModel(),
        FakeBackend(),
        _FakeBatcher(),
        scene_provider=FakeSceneProvider(),
        end_time=3.5,
        normalization=1000.0,
        precision="float32",
        device=torch.device("cpu"),
        expected_action_keys=("gcd", "illegal"),
        source_label="test-fight",
    )

    # 每次动作占用 1 秒，最后一次只推进到副本结束；所有伤害都来自已执行历史。
    assert result.output_gcds == 4
    assert result.cumulative_potency == pytest.approx(400.0)
    assert result.cumulative_dot_potency == pytest.approx(80.0)
    assert result.ppg == pytest.approx(120.0)
    assert result.normalized_ppg == pytest.approx(0.12)


def test_validation_rollout_skips_fight_when_all_actions_are_illegal(caplog):
    class NoLegalBatcher:
        @staticmethod
        def build(_state):
            return (
                {"action_legal_mask": torch.tensor([[False, False]])},
                ["gcd", "illegal"],
            )

    class FakeSceneProvider:
        @staticmethod
        def next_state_event_after(_time_seconds):
            return None

        @staticmethod
        def targetable_windows():
            return []

    class FakeBackend(_FakeBackend):
        def __init__(self):
            self.state = _FakeState()

    with caplog.at_level("WARNING"):
        result = _run_rollout_until_time(
            _FakeModel(),
            FakeBackend(),
            NoLegalBatcher(),
            scene_provider=FakeSceneProvider(),
            end_time=3.5,
            normalization=1000.0,
            precision="float32",
            device=torch.device("cpu"),
            expected_action_keys=("gcd", "illegal"),
            source_label="skipped-fight",
        )

    assert result.output_gcds == 0
    assert result.cumulative_potency == pytest.approx(0.0)
    assert result.cumulative_dot_potency == pytest.approx(0.0)
    assert result.ppg == pytest.approx(0.0)
    assert result.normalized_ppg == pytest.approx(0.0)
    assert "跳过本副本并记 PPG=0" in caplog.text


@pytest.mark.parametrize("base_gcd,haste", ((2.4, False), (2.07, True)))
def test_validation_ppg_recovers_initial_base_gcd_from_saved_current_state(base_gcd, haste):
    normalizer = Normalizer()
    from common.policy.data.schema import TrainingSchema
    schema = TrainingSchema(
        serialization_format="raw_training_source_v1", sample_schema_version=TRAINING_SAMPLE_SCHEMA_VERSION,
        context_schema_version=CANONICAL_CONTEXT_SCHEMA_VERSION, scene_context_mode="absolute_from_fight_scene_context", scene_windows=(),
        state_group_feature_keys={
            "player_state": ("previous_action_after.current_gcd_seconds", "request_state.current_gcd_seconds"),
            "buff_state": ("previous_action_after.job.ley_lines.active", "request_state.job.ley_lines.active"),
        }, skill_history_fields=(),
    )
    normalizer.register_schema(schema)
    actual_gcd = base_gcd * (0.85 if haste else 1.0)
    previous_gcd = 2.9
    request_gcd = actual_gcd

    class Reader:
        @staticmethod
        def sample(_index):
            return {
                "current_state_abs_values": torch.tensor([previous_gcd, request_gcd, float(not haste), float(haste)]),
                "current_state_null_mask": torch.zeros(4, dtype=torch.bool),
            }
    Reader.schema = schema

    assert _infer_initial_base_gcd(
        Reader(),
        normalizer=normalizer,
        job_tag="black_mage",
    ) == pytest.approx(base_gcd)


@pytest.mark.parametrize("missing_state,null_gcd,gcd", ((True, False, 2.5), (False, True, 2.5), (False, False, 0.0)))
def test_initial_ppg_state_cannot_silently_fall_back_to_local_gcd(missing_state, null_gcd, gcd):
    normalizer = Normalizer()
    keys = ("previous_action_after.current_gcd_seconds", "request_state.current_gcd_seconds")
    normalizer.register_feature_keys("player_state", list(keys))
    schema = SimpleNamespace(
        state_group_feature_keys={"player_state": keys}, state_vector_dim=lambda: 2,
        state_group_slices=lambda: {"player_state": slice(0, 2)},
    )
    values = {} if missing_state else {
        "current_state_abs_values": torch.tensor([gcd, gcd]),
        "current_state_null_mask": torch.tensor([False, null_gcd]),
    }
    reader = SimpleNamespace(schema=schema, sample=lambda _index: values)
    with pytest.raises(ValueError, match="missing initial current state|has null|invalid initial base GCD"):
        _infer_initial_base_gcd(reader, normalizer=normalizer, job_tag="machinist")


def test_real_validation_runs_to_end_with_a_small_history_window():
    """真实队列保留四条记录也必须跑完整场，最终 PPG 与完整记录相同。"""
    from scripts.common.inprocess_backend import InProcessEngine
    from scripts.autoregressive_replay.scheduler import gcd_request_delay
    from tests.scripts.conftest import _require_inprocess_backend

    _require_inprocess_backend()
    results = []
    with InProcessEngine("machinist", capacity=2) as engine:
        for history_limit in (None, 4):
            with engine.create_backend(max_history=history_limit, fight_remaining=180) as backend:
                class Batcher:
                    def build(self, state):
                        legal = backend.validate_at(state.time, "heated_split_shot").legal
                        wait = gcd_request_delay(state) > 1e-6
                        return ({"action_legal_mask": torch.tensor([[legal, wait]])},
                                ("heated_split_shot", "ogcd_wait"))

                result = _run_rollout_until_time(
                    _FakeModel(), backend, Batcher(), scene_provider=None,
                    end_time=180, normalization=1000, precision="float32",
                    device=torch.device("cpu"),
                    expected_action_keys=("heated_split_shot", "ogcd_wait"),
                    source_label="full-duration-regression",
                )
                assert backend.statistics()["timestamp"] == pytest.approx(180)
                assert result.output_gcds > 40
                if history_limit is not None:
                    assert backend.statistics()["action_history_count"] <= history_limit
                    assert backend.statistics()["policy_history_count"] <= history_limit
                results.append(result)
    assert results[0] == results[1]


@pytest.mark.parametrize("cache_was_enabled", [False, True])
@pytest.mark.parametrize("use_kv_cache", [False, True])
def test_validation_ppg_reads_history_capacity_from_model_config(
    monkeypatch, cache_was_enabled, use_kv_cache
):
    captured = {}

    class FakeNormalizer:
        def configure_job_resources(self, _job_tag):
            return None

        def register_schema(self, _schema):
            return None

    class FakeBackend:
        def __init__(self, *, job_tag, max_history, engine):
            captured["backend_max_history"] = max_history
            self.state = _FakeState()

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return None

        def init(self, **_kwargs):
            return None

        def observe_at(self, timestamp, *, format="seconds", next_observation_timestamp=None):
            del next_observation_timestamp
            self.state.time = float(timestamp)
            return SimpleNamespace(
                timestamp=self.state.time,
                next_scheduled_event_time=None,
                context=(
                    {"current_state_context": {"target_buff_state_feature_keys": ["request_state.target.cumulative_potency", "request_state.target.cumulative_dot_potency"], "tokens": [{"target_buff_state": [0.0, 0.0]}]}}
                    if format == "vector"
                    else {
                        "time_seconds": self.state.time,
                        "gcd_index": self.state.gcd_index,
                        "current_gcd_seconds": 2.5,
                        "fight_remaining_seconds": self.state.fight_remaining,
                        "gcd_remaining_seconds": self.state.gcd_remaining,
                        "cast_remaining_seconds": getattr(self.state, "cast_remaining", 0.0),
                    }
                ),
            )

    class FakeReader:
        fight_id = "test-fight"

        @staticmethod
        def step_metadata(_index):
            return {"time_offset": 0.0}

    class FakeSceneProvider:
        def __init__(self, *_args, **_kwargs):
            return None

        @staticmethod
        def last_targetable_end():
            return 1.0

        @staticmethod
        def sync_state(state):
            return state

    class FakeBatcher:
        def __init__(self, **kwargs):
            captured["batcher_max_history"] = kwargs["max_history"]

    monkeypatch.setattr(ppg_module, "_dataset_normalizer", lambda _dataset: FakeNormalizer())
    monkeypatch.setattr(
        "scripts.common.inprocess_backend.InProcessEngine.create_backend",
        lambda engine, **kwargs: FakeBackend(job_tag=engine.job_tag, engine=engine, **kwargs),
    )
    monkeypatch.setattr(ppg_module, "SceneTemplateProvider", FakeSceneProvider)
    monkeypatch.setattr(ppg_module, "LiveBatchBuilder", FakeBatcher)
    monkeypatch.setattr(ppg_module, "_infer_initial_base_gcd", lambda *args, **kwargs: 2.5)
    def fake_rollout(model, *args, **kwargs):
        assert model._kv_cache_enabled is use_kv_cache
        return PpgResult(
            output_gcds=1,
            cumulative_potency=100.0,
            cumulative_dot_potency=0.0,
            ppg=100.0,
            normalized_ppg=0.1,
        )

    monkeypatch.setattr(ppg_module, "_run_rollout_until_time", fake_rollout)

    config = SimpleNamespace(
        model=SimpleNamespace(history_capacity=37),
        ppg=SimpleNamespace(
            enabled=True, normalization=1000.0, use_kv_cache=use_kv_cache
        ),
        precision="float32",
    )
    dataset = SimpleNamespace(
        schema=object(),
        skill_feature_names=(),
        iter_source_readers=lambda: (FakeReader(),),
    )
    data_spec = SimpleNamespace(
        job_tag="black_mage",
        action_keys=("fire",),
        action_is_gcd=(True,),
    )
    cache_events = []

    class FakeModel:
        _kv_cache_enabled = False

        @staticmethod
        def eval():
            return None

        def enable_kv_cache(self, enabled):
            self._kv_cache_enabled = bool(enabled)
            cache_events.append(("enable", self._kv_cache_enabled))

        def reset_kv_cache(self):
            cache_events.append(("reset", self._kv_cache_enabled))

    model = FakeModel()
    model.config = config.model
    model._kv_cache_enabled = cache_was_enabled

    result = ppg_module.evaluate_validation_ppg(
        model=model,
        dataset=dataset,
        data_spec=data_spec,
        vocab=None,
        config=config,
        device=torch.device("cpu"),
    )

    assert captured == {
        "backend_max_history": 37,
        "batcher_max_history": 37,
    }
    assert result["val_ppg"] == pytest.approx(100.0)
    assert result["val_ppg_normalized"] == pytest.approx(0.1)
    assert model._kv_cache_enabled is cache_was_enabled
    assert cache_events == [
        ("enable", use_kv_cache),
        ("reset", use_kv_cache),
        ("enable", cache_was_enabled),
    ]

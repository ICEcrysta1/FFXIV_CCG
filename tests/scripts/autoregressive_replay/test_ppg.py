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
)
from common.policy.data import Normalizer
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
                "state_history_context": {
                    "target_buff_state_feature_keys": [
                        "after.target.cumulative_potency",
                        "after.target.cumulative_dot_potency",
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
            {"candidate_legal_mask": torch.tensor([[True, False]])},
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
    state_context = canonical["candidate_state_context"]
    groups = {group: state_context[f"{group}_feature_keys"]
              for group in ("player_state", "buff_state", "target_buff_state", "resource_state")}
    keys = tuple(token["skill_key"] for token in canonical["candidate_skill_context"])
    builder = LiveBatchBuilder(
        backend=cs_backend,
        vocab=SimpleNamespace(require_lookup=lambda value, **kwargs: value),
        normalizer=SimpleNamespace(normalize=lambda values, *args, **kwargs: values,
                                   normalize_skill_features=lambda values, *args: values),
        schema=SimpleNamespace(state_group_feature_keys=groups,
                               state_vector_dim=lambda: sum(map(len, groups.values())),
                               scene_feature_dim=lambda: 1),
        skill_feature_names=("kind",),
        scene_provider=SimpleNamespace(at_time=lambda _: (torch.zeros((0, 1)), torch.zeros(0, dtype=torch.int32))),
        device=torch.device("cpu"), max_history=20,
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
                          precision="float32", device=torch.device("cpu"), expected_candidate_keys=keys)
    assert result.output_gcds == 3
    assert result.cumulative_potency > 0
    assert len(submissions) == 3
    assert [item.request_timestamp for item in submissions] == pytest.approx([0.0, 2.45, 4.9])
    assert [item.queued for item in submissions] == [False, True, True]


def test_ppg_rollout_counts_output_gcds_and_includes_dot_potency():
    result = _run_rollout(
        _FakeModel(),
        _FakeBackend(),
        _FakeBatcher(),
        gcd_count=3,
        normalization=1000.0,
        precision="float32",
        device=torch.device("cpu"),
        expected_candidate_keys=("gcd", "illegal"),
    )

    assert result.output_gcds == 3
    assert result.cumulative_potency == pytest.approx(300.0)
    assert result.cumulative_dot_potency == pytest.approx(60.0)
    assert result.ppg == pytest.approx(120.0)
    assert result.normalized_ppg == pytest.approx(0.12)


def test_ppg_rollout_rejects_candidate_order_mismatch():
    with pytest.raises(ValueError, match="candidate action order mismatch"):
        _run_rollout(
            _FakeModel(),
            _FakeBackend(),
            _FakeBatcher(),
            gcd_count=1,
            normalization=1000.0,
            precision="float32",
            device=torch.device("cpu"),
            expected_candidate_keys=("illegal", "gcd"),
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
        expected_candidate_keys=("gcd", "illegal"),
        source_label="test-fight",
    )

    # 每次动作占用 1 秒，最后一次只推进到副本结束；所有伤害都来自已执行历史。
    assert result.output_gcds == 4
    assert result.cumulative_potency == pytest.approx(400.0)
    assert result.cumulative_dot_potency == pytest.approx(80.0)
    assert result.ppg == pytest.approx(120.0)
    assert result.normalized_ppg == pytest.approx(0.12)


def test_validation_rollout_skips_fight_when_all_candidates_are_illegal(caplog):
    class NoLegalBatcher:
        @staticmethod
        def build(_state):
            return (
                {"candidate_legal_mask": torch.tensor([[False, False]])},
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
            expected_candidate_keys=("gcd", "illegal"),
            source_label="skipped-fight",
        )

    assert result.output_gcds == 0
    assert result.cumulative_potency == pytest.approx(0.0)
    assert result.cumulative_dot_potency == pytest.approx(0.0)
    assert result.ppg == pytest.approx(0.0)
    assert result.normalized_ppg == pytest.approx(0.0)
    assert "跳过本副本并记 PPG=0" in caplog.text


def test_validation_ppg_recovers_initial_base_gcd_from_cached_candidate_token():
    normalizer = Normalizer()

    class Reader:
        @staticmethod
        def sample(_index):
            return {
                "candidate_skill_features": torch.tensor(
                    [
                        [0.0, 0.0, 2.5 / 120.0],
                        [0.0, 1.0, 2.4 / 120.0],
                        [1.0, 1.0, 2.4 / 120.0],
                    ]
                ),
                "candidate_legal_mask": torch.tensor([False, True, True]),
            }

    assert _infer_initial_base_gcd(
        Reader(),
        normalizer=normalizer,
        skill_feature_names=("kind", "is_legal", "gcd_window.seconds"),
    ) == pytest.approx(2.4)


def test_validation_ppg_reads_history_capacity_from_model_config(monkeypatch):
    captured = {}

    class FakeNormalizer:
        def configure_job_resources(self, _job_tag):
            return None

        def register_schema(self, _schema):
            return None

    class FakeBackend:
        def __init__(self, *, job_tag, max_history):
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
                    {"state_history_context": {"tokens": []}}
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

    monkeypatch.setattr(ppg_module, "Normalizer", FakeNormalizer)
    monkeypatch.setattr(ppg_module, "InProcessBackend", FakeBackend)
    monkeypatch.setattr(ppg_module, "SceneTemplateProvider", FakeSceneProvider)
    monkeypatch.setattr(ppg_module, "LiveBatchBuilder", FakeBatcher)
    monkeypatch.setattr(ppg_module, "_infer_initial_base_gcd", lambda *args, **kwargs: 2.5)
    monkeypatch.setattr(
        ppg_module,
        "_run_rollout_until_time",
        lambda *args, **kwargs: PpgResult(
            output_gcds=1,
            cumulative_potency=100.0,
            cumulative_dot_potency=0.0,
            ppg=100.0,
            normalized_ppg=0.1,
        ),
    )

    config = SimpleNamespace(
        model=SimpleNamespace(history_capacity=37),
        ppg=SimpleNamespace(enabled=True, normalization=1000.0),
        precision="float32",
    )
    dataset = SimpleNamespace(
        schema=object(),
        skill_feature_names=(),
        iter_source_readers=lambda: (FakeReader(),),
    )
    data_spec = SimpleNamespace(
        job_tag="black_mage",
        candidate_action_keys=("fire",),
    )
    model = SimpleNamespace(eval=lambda: None)

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

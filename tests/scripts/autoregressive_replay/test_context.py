"""自回归回放 scene 上下文与 live batch 测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from common.contracts import SLIDECAST_WINDOW_SECONDS
from common.policy.data import Normalizer
from common.policy.config import ModelConfig
from common.policy.data.schema import (
    SCENE_TYPE_MOVEMENT,
    SCENE_TYPE_RAID_BUFF,
    SCENE_TYPE_TARGET_COUNT,
    SCENE_TYPE_TARGETABLE,
    SceneWindowSchema,
    TrainingSchema,
    StateFeatureGroup,
    TRAINING_SAMPLE_SCHEMA_VERSION,
)
from common.output_context_schema import CANONICAL_CONTEXT_SCHEMA_VERSION
from scripts.autoregressive_replay.context import (
    LiveBatchBuilder,
    SceneTemplateProvider,
    _tail_history,
)


def test_tail_history_keeps_only_recent_entries_and_supports_empty_history():
    values = [1, 2, 3, 4]

    assert _tail_history(values, 2) == [3, 4]
    assert _tail_history(values, 10) == values
    assert _tail_history(values, 0) == []


class _SceneWindow:
    scene_type_id = SCENE_TYPE_TARGET_COUNT
    context_key = "target_count_window_context"
    feature_keys = (
        "start_offset_seconds",
        "end_offset_seconds",
        "duration_seconds",
        "target_count",
    )


class _SceneSchema:
    scene_windows = (_SceneWindow(),)

    @staticmethod
    def scene_feature_dim():
        return 4

class _SceneReader:
    num_samples = 3
    schema = _SceneSchema()

    @staticmethod
    def step_metadata(index):
        return {"time_offset": float(index)}

    @staticmethod
    def scene_tokens(index, *, float_dtype, int_dtype):
        if index == 0:
            return torch.zeros((0, 4), dtype=float_dtype), torch.zeros((0,), dtype=int_dtype)
        return (
            torch.tensor(
                [
                    [0.0, 1.0, 1.0, 2.0],
                    [1.0, 3.0, 2.0, 3.0],
                ],
                dtype=float_dtype,
            ),
            torch.tensor([SCENE_TYPE_TARGET_COUNT, SCENE_TYPE_TARGET_COUNT], dtype=int_dtype),
        )


class _TargetableSceneWindow:
    scene_type_id = SCENE_TYPE_TARGETABLE
    context_key = "targetable_window_context"
    feature_keys = (
        "start_offset_seconds",
        "end_offset_seconds",
        "duration_seconds",
        "targetable",
        "segment_kind.combat",
        "segment_kind.downtime",
        "segment_kind.combat_final",
    )
    start_offset_index = 0
    end_offset_index = 1


class _TargetableSceneSchema:
    scene_windows = (_TargetableSceneWindow(),)

    @staticmethod
    def scene_feature_dim():
        return 7


class _TargetableSceneReader:
    num_samples = 1
    schema = _TargetableSceneSchema()

    @staticmethod
    def step_metadata(index):
        return {"time_offset": float(index)}

    @staticmethod
    def scene_tokens(index, *, float_dtype, int_dtype):
        del index
        scale = 1.0
        return (
            torch.tensor(
                [
                    [0.0, 22.0 / scale, 22.0 / scale, 1.0, 1.0, 0.0, 0.0],
                    [22.0 / scale, 82.0 / scale, 60.0 / scale, 0.0, 0.0, 1.0, 0.0],
                    [82.0 / scale, 600.0 / scale, 518.0 / scale, 1.0, 0.0, 0.0, 1.0],
                ],
                dtype=float_dtype,
            ),
            torch.tensor([SCENE_TYPE_TARGETABLE] * 3, dtype=int_dtype),
        )


def test_scene_template_provider_uses_raw_scene_and_resolves_target_count():
    provider = SceneTemplateProvider(_SceneReader(), initial_sample_index=1)
    vectors, types = provider.at_time(0.0)
    assert vectors.shape == (2, 4)
    assert int(types[0]) == SCENE_TYPE_TARGET_COUNT
    assert provider.state_at(0.0).target_count == 2
    assert provider.state_at(1.8).target_count == 3
    assert provider.state_at(3.0).target_count == 1
    later_vectors, later_types = provider.at_time(2.0)
    assert later_vectors.equal(vectors)
    assert later_types.equal(types)

    disabled = SceneTemplateProvider(_SceneReader(), enabled=False)
    empty_vectors, empty_types = disabled.at_time(2.0)
    assert empty_vectors.shape == (0, 4)
    assert empty_types.shape == (0,)
    assert disabled.state_at(2.0).target_count == 1

    with pytest.raises(ValueError, match="out of range"):
        SceneTemplateProvider(_SceneReader(), initial_sample_index=3)

    class EmptyReader(_SceneReader):
        num_samples = 1

        @staticmethod
        def scene_tokens(index, *, float_dtype, int_dtype):
            return torch.zeros((0, 4), dtype=float_dtype), torch.zeros((0,), dtype=int_dtype)

    with pytest.raises(ValueError, match="no usable scene tokens"):
        SceneTemplateProvider(EmptyReader())

    class RawSceneReader(_SceneReader):
        @staticmethod
        def scene_tokens(index, *, float_dtype, int_dtype):
            vectors, types = _SceneReader.scene_tokens(
                index,
                float_dtype=float_dtype,
                int_dtype=int_dtype,
            )
            if vectors.shape[0]:
                vectors = vectors.clone()
                vectors[:, 0] *= 1800.0
                vectors[:, 1] *= 1800.0
                vectors[:, 2] *= 1800.0
            return vectors, types

    long_scene = SceneTemplateProvider(RawSceneReader(), initial_sample_index=1)
    assert long_scene.at_time(0)[0][1, 1].item() == 5400.0
    assert long_scene.state_at(1801.0).target_count == 3


@pytest.mark.parametrize("target_count", [0.0, -1.0, 0.5])
def test_scene_template_provider_keeps_zero_targets_and_rejects_invalid_counts(target_count):
    class TargetCountReader(_SceneReader):
        @staticmethod
        def scene_tokens(index, *, float_dtype, int_dtype):
            vectors, types = _SceneReader.scene_tokens(
                index, float_dtype=float_dtype, int_dtype=int_dtype,
            )
            vectors[0, 3] = target_count
            return vectors, types

    if target_count == 0:
        provider = SceneTemplateProvider(TargetCountReader(), initial_sample_index=1)
        assert provider.state_at(0.5).target_count == 0
        assert provider.state_at(1.8).target_count == 3
        assert provider.state_at(3.0).target_count == 1
    else:
        with pytest.raises(ValueError, match="non-negative integer"):
            SceneTemplateProvider(TargetCountReader(), initial_sample_index=1)


def test_scene_template_provider_syncs_targetable_timeline_into_live_state():
    class FakeBackend:
        def __init__(self):
            self.events = []

        def apply_external_events(self, events):
            for event in events:
                self.events.append((event["timestamp"], event["event_kind"], {
                    key: value for key, value in event.items()
                    if key not in {"timestamp", "event_kind"} and value is not None
                }))
            return SimpleNamespace(accepted=True)


    provider = SceneTemplateProvider(
        _TargetableSceneReader(),
        backend=FakeBackend(),
    )
    state = SimpleNamespace(time=10.0, fight_remaining=100.0)
    backend = provider._backend
    provider.sync_state(state)
    assert provider.last_targetable_end() == pytest.approx(600.0)
    assert provider.next_state_event_after(10.0) == pytest.approx(22.0)

    state.time = 30.0
    provider.sync_state(state)
    assert backend.events[-1] == (22.0, "boss_targetable_changed", {"value": False})
    assert provider.state_at(81.9).boss_targetable is False
    assert provider.next_state_event_after(30.0) == pytest.approx(82.0)

    state.time = 82.0
    provider.sync_state(state)
    assert backend.events[-1] == (82.0, "boss_targetable_changed", {"value": True})


def test_scene_template_provider_reset_resyncs_same_signature():
    class FakeBackend:
        def __init__(self):
            self.events = []

        def apply_external_events(self, events):
            for event in events:
                self.events.append((event["timestamp"], event["event_kind"], {
                    key: value for key, value in event.items()
                    if key not in {"timestamp", "event_kind"} and value is not None
                }))
            return SimpleNamespace(accepted=True)


    backend = FakeBackend()
    provider = SceneTemplateProvider(
        _TargetableSceneReader(),
        backend=backend,
    )
    state = SimpleNamespace(time=10.0, fight_remaining=100.0)

    provider.sync_state(state)
    assert len(backend.events) == 4
    provider.sync_state(state)
    assert len(backend.events) == 4

    provider.reset()
    provider.sync_state(state)
    assert len(backend.events) == 8
    assert backend.events[:4] == backend.events[4:]


def test_scene_template_provider_syncs_movement_with_slidecast_boundary():
    class Window:
        def __init__(self, scene_type_id, feature_keys):
            self.scene_type_id = scene_type_id
            self.context_key = {SCENE_TYPE_TARGETABLE: "targetable_window_context",
                                SCENE_TYPE_MOVEMENT: "forced_movement_context",
                                SCENE_TYPE_RAID_BUFF: "raid_buff_window_context",
                                SCENE_TYPE_TARGET_COUNT: "target_count_window_context"}[scene_type_id]
            self.feature_keys = feature_keys
            self.start_offset_index = 0
            self.end_offset_index = 1

    schema = SimpleNamespace(
        scene_windows=(
            Window(
                SCENE_TYPE_TARGETABLE,
                (
                    "start_offset_seconds",
                    "end_offset_seconds",
                    "duration_seconds",
                    "targetable",
                ),
            ),
            Window(
                SCENE_TYPE_MOVEMENT,
                (
                    "start_offset_seconds",
                    "end_offset_seconds",
                    "duration_seconds",
                ),
            ),
            Window(
                SCENE_TYPE_RAID_BUFF,
                (
                    "start_offset_seconds",
                    "end_offset_seconds",
                    "duration_seconds",
                    "source.amplifier",
                ),
            ),
            Window(
                SCENE_TYPE_TARGET_COUNT,
                (
                    "start_offset_seconds",
                    "end_offset_seconds",
                    "duration_seconds",
                    "target_count",
                ),
            ),
        ),
        scene_feature_dim=lambda: 4,
    )
    scale = 1.0

    class Reader:
        num_samples = 1

        @staticmethod
        def step_metadata(index):
            return {"time_offset": float(index)}

        @staticmethod
        def scene_tokens(index, *, float_dtype, int_dtype):
            del index
            return (
                torch.tensor(
                    [
                        [0.0, 100.0 / scale, 100.0 / scale, 1.0],
                        [10.0 / scale, 20.0 / scale, 10.0 / scale, 0.0],
                        [30.0 / scale, 50.0 / scale, 20.0 / scale, 1.0],
                        [40.0 / scale, 60.0 / scale, 20.0 / scale, 2.0],
                    ],
                    dtype=float_dtype,
                ),
                torch.tensor(
                    [
                        SCENE_TYPE_TARGETABLE,
                        SCENE_TYPE_MOVEMENT,
                        SCENE_TYPE_RAID_BUFF,
                        SCENE_TYPE_TARGET_COUNT,
                    ],
                    dtype=int_dtype,
                ),
            )

    Reader.schema = schema

    class FakeBackend:
        def __init__(self):
            self.events = []

        def apply_external_events(self, events):
            for event in events:
                self.events.append((event["timestamp"], event["event_kind"], {
                    key: value for key, value in event.items()
                    if key not in {"timestamp", "event_kind"} and value is not None
                }))
            return SimpleNamespace(accepted=True)


    backend = FakeBackend()
    provider = SceneTemplateProvider(
        Reader(),
        backend=backend,
    )

    state = SimpleNamespace(time=12.0, fight_remaining=100.0)
    provider.sync_state(state)
    assert backend.events == [
        (12.0, "boss_targetable_changed", {"value": True}),
        (12.0, "movement_changed", {"value": True}),
        (12.0, "target_count_changed", {"target_count": 1}),
        (12.0, "raid_buff_window_changed", {"value": False}),
    ]
    assert provider.state_at(12.0).is_moving is True
    assert provider.state_at(19.5).is_moving is False
    assert provider.next_state_event_after(10.0) == pytest.approx(
        20.0 - SLIDECAST_WINDOW_SECONDS
    )

    state.time = 19.6
    provider.sync_state(state)
    assert backend.events[-1] == (19.5, "movement_changed", {"value": False})

    state.time = 45.0
    provider.sync_state(state)
    assert (40.0, "target_count_changed", {"target_count": 2}) in backend.events
    assert (30.0, "raid_buff_window_changed", {"value": True, "remaining_seconds": 20.0}) in backend.events
    state.time = 55.0
    provider.sync_state(state)
    assert backend.events[-1] == (50.0, "raid_buff_window_changed", {"value": False})
    state.time = 61.0
    provider.sync_state(state)
    assert backend.events[-1] == (60.0, "target_count_changed", {"target_count": 1})


def _live_builder_fixture(*, max_history=2, reorder_state_fields=False):
    from copy import deepcopy

    player_keys = ("previous_action_after.time_seconds", "previous_action_after.mp",
                   "request_state.time_seconds", "request_state.mp")
    player_keys += tuple(f"{prefix}.{field}" for prefix in ("previous_action_after", "request_state")
                         for field in ("next_untargetable_in_seconds", "downtime_remaining_seconds"))
    availability_keys = tuple(f"{prefix}.{key}" for prefix in ("previous_action_after", "request_state")
                              for key in ("fire", "ogcd_wait"))
    if reorder_state_fields:
        player_keys = tuple(reversed(player_keys))
    initial_values = {"previous_action_after.time_seconds": 0.0, "previous_action_after.mp": 200.0,
                      "request_state.time_seconds": 0.0, "request_state.mp": 200.0}
    initial_values.update({key: 0.0 for key in player_keys if key not in initial_values})
    canonical = {
        "schema_version": CANONICAL_CONTEXT_SCHEMA_VERSION,
        "history_cursor": 0,
        "action_keys": ["fire", "ogcd_wait"],
        "action_legal_mask": [True, True],
        "skill_history_context": [],
        "state_history_context": {"player_state_feature_keys": player_keys, "skill_availability_feature_keys": availability_keys, "tokens": []},
        "current_state_context": {"player_state_feature_keys": player_keys, "skill_availability_feature_keys": availability_keys,
                                  "tokens": [{"player_state": [initial_values[key] for key in player_keys], "skill_availability": [1, 1, 1, 1]}]},
    }

    class Backend:
        def observe_at(self, timestamp, *, format, next_observation_timestamp):
            assert format == "vector"
            return SimpleNamespace(context=deepcopy(canonical))

    schema = TrainingSchema(
        serialization_format="test", sample_schema_version=TRAINING_SAMPLE_SCHEMA_VERSION,
        context_schema_version=CANONICAL_CONTEXT_SCHEMA_VERSION, scene_context_mode="absolute",
        scene_windows=(SceneWindowSchema.from_feature_keys(
            context_key="targetable_window_context", scene_type_id=0,
            feature_keys=("start_offset_seconds", "end_offset_seconds", "duration_seconds"),
        ),),
        state_groups=(StateFeatureGroup("player_state", "player_state_feature_keys", player_keys, "anchored_delta"),
                      StateFeatureGroup("skill_availability", "skill_availability_feature_keys", availability_keys, "absolute_binary")),
        state_snapshots=("previous_action_after", "request_state"),
        skill_history_fields=("kind", "potency"),
    )
    normalizer = Normalizer()
    normalizer.ensure_job_resources("black_mage")
    builder = LiveBatchBuilder(
        backend=Backend(), vocab=SimpleNamespace(require_lookup=lambda value, **kwargs: int(value)),
        normalizer=normalizer, schema=schema, skill_feature_names=("kind", "potency"),
        scene_provider=SimpleNamespace(at_time=lambda _: (torch.empty((0, 3)), torch.empty(0, dtype=torch.int32)),
                                       state_at=lambda _: SimpleNamespace(next_downtime_eta=0.0, downtime_remaining=0.0)),
        device=torch.device("cpu"), max_history=max_history,
        model_config=ModelConfig(history_capacity=max_history, history_reset_keep=max_history),
        action_keys=("fire", "ogcd_wait"), action_is_gcd=(True, False),
    )
    return builder, canonical


def _append_live_history(canonical, index, *, after=100.0, request_time=None, skill_id=None):
    canonical["history_cursor"] += 1
    canonical["skill_history_context"].append({
        "skill_id": index if skill_id is None else skill_id, "skill_key": "fire", "kind": 1, "potency": 50.0,
    })
    state_values = {"previous_action_after.time_seconds": float(index - 1), "previous_action_after.mp": 200.0,
                    "request_state.time_seconds": float(index) if request_time is None else request_time,
                    "request_state.mp": after}
    canonical["state_history_context"]["tokens"].append({
        "player_state": [state_values.get(key, 0.0) for key in canonical["state_history_context"]["player_state_feature_keys"]],
        "skill_availability": [1, 1, 1, 1],
    })


@pytest.mark.parametrize("remaining,expected", ((0.0, [True, False]), (1.0, [False, True])))
def test_live_current_state_is_independent_of_action_legality_and_preserves_phase(remaining, expected):
    builder, canonical = _live_builder_fixture()
    batch, keys = builder.build(SimpleNamespace(time=1.0, gcd_remaining=remaining))
    assert keys == ["fire", "ogcd_wait"]
    assert batch["current_state_vectors"].shape == (1, 12)
    torch.testing.assert_close(batch["current_state_vectors"][..., :4], torch.tensor([[0.0, 0.02, 0.0, 0.02]]))
    assert batch["current_state_vectors"][..., -4:].tolist() == [[1.0] * 4]
    assert batch["current_state_reset_mask"].all()
    assert batch["current_state_null_mask"].tolist() == [[False] * 8]
    assert batch["history_skill_ids"].shape == (1, 0)
    assert batch["action_legal_mask"].tolist() == [expected]
    canonical["action_legal_mask"] = [False, False]
    new_batch, _ = builder.build(SimpleNamespace(time=2.0, gcd_remaining=remaining))
    torch.testing.assert_close(new_batch["current_state_vectors"], batch["current_state_vectors"])
    assert not new_batch["action_legal_mask"].any()


def test_live_history_window_keeps_matching_skill_and_state_rows():
    builder, canonical = _live_builder_fixture(max_history=2)
    for index in (1, 2, 3):
        _append_live_history(canonical, index, after=float(index))
    batch, _ = builder.build(SimpleNamespace(time=4.0, gcd_remaining=0.0))
    assert batch["history_skill_ids"].tolist() == [[2, 3]]
    torch.testing.assert_close(batch["history_state_vectors"][..., :4], torch.tensor([[[-1/120, .02, 0, .0002], [1/120, 0, 1/120, .0001]]]))
    assert batch["history_state_reset_mask"][0, 0].all()
    assert not batch["history_state_reset_mask"][0, 1].any()
    assert batch["history_skill_features"][0, :, 0].tolist() == [1.0, 1.0]
    assert batch["history_action_keys"] == [["fire", "fire"]]
    empty, _ = builder.build(SimpleNamespace(time=4.0, gcd_remaining=0.0), max_history=0)
    assert empty["history_skill_ids"].shape == (1, 0)
    assert empty["current_state_vectors"].shape == (1, 12)


@pytest.mark.parametrize("context_key", ["state_history_context", "current_state_context"])
@pytest.mark.parametrize("change", ["missing", "reordered", "renamed"])
def test_live_batch_rejects_same_width_state_field_drift(context_key, change):
    builder, canonical = _live_builder_fixture()
    _append_live_history(canonical, 1)
    context = canonical[context_key]
    if change == "missing":
        context.pop("player_state_feature_keys")
    elif change == "reordered":
        context["player_state_feature_keys"] = tuple(reversed(context["player_state_feature_keys"]))
        for token in context["tokens"]:
            token["player_state"].reverse()
    else:
        context["player_state_feature_keys"] = ("wrong.time_seconds", *context["player_state_feature_keys"][1:])
    with pytest.raises(ValueError, match="feature keys mismatch|lacks"):
        builder.build_from_canonical(canonical, gcd_phase=True, max_history=2)


@pytest.mark.parametrize("raw_id", [None, 900001])
def test_live_history_does_not_map_missing_or_unknown_skill_to_padding(raw_id):
    from common.policy.data import SkillVocab

    builder, canonical = _live_builder_fixture()
    builder._vocab = SkillVocab.from_entries([(3577, 1), (0, 2)])
    _append_live_history(canonical, 1)
    canonical["skill_history_context"][0]["skill_id"] = raw_id
    with pytest.raises(ValueError, match="live replay raw_skill_id"):
        builder.build_from_canonical(canonical, gcd_phase=True, max_history=2)


def test_live_history_cache_reuses_unchanged_rows_and_refreshes_mutated_rows(monkeypatch):
    builder, canonical = _live_builder_fixture(max_history=3)
    calls = []
    real_build = builder._build_cached_history_row

    def track(*args, **kwargs):
        calls.append(args[0]["skill_id"])
        return real_build(*args, **kwargs)

    monkeypatch.setattr(builder, "_build_cached_history_row", track)
    _append_live_history(canonical, 1)
    _append_live_history(canonical, 2)
    state = SimpleNamespace(time=3.0, gcd_remaining=0.0)
    first, _ = builder.build(state)
    repeated, _ = builder.build(state)
    assert calls == [1, 2]
    torch.testing.assert_close(repeated["history_state_vectors"], first["history_state_vectors"])
    canonical["current_state_context"]["tokens"][0]["player_state"] = [0.0, 150.0, 0.0, 150.0, 0.0, 0.0, 0.0, 0.0]
    refreshed, _ = builder.build(state)
    assert calls == [1, 2]
    torch.testing.assert_close(refreshed["current_state_vectors"][..., :4], torch.tensor([[-1/120, -.005, -2/120, .005]]))
    _append_live_history(canonical, 3)
    builder.build(state)
    assert calls == [1, 2, 3]
    _append_live_history(canonical, 4)
    slid, _ = builder.build(state)
    assert calls == [1, 2, 3, 4]
    assert slid["history_skill_ids"].tolist() == [[2, 3, 4]]
    canonical["state_history_context"]["tokens"][-1]["player_state"] = [3.0, 10.0, 4.0, None, 0.0, 0.0, 0.0, 0.0]
    changed, _ = builder.build(state)
    assert calls[-1] == 4
    assert changed["history_state_null_mask"][0, -1].tolist() == [False, False, False, True, False, False, False, False]


@pytest.mark.parametrize("mutation,error", (
    (lambda c: c.update(action_keys=[]), "nonempty"),
    (lambda c: c.update(action_keys=["ogcd_wait", "fire"]), "order differs"),
    (lambda c: c.update(action_legal_mask=[True]), "legality width"),
    (lambda c: c["current_state_context"].update(tokens=[]), "exactly one"),
    (lambda c: c["current_state_context"].update(tokens=[{"player_state": [1.0]}]), "width"),
))
def test_live_builder_rejects_incomplete_or_reordered_action_contract(mutation, error):
    builder, canonical = _live_builder_fixture()
    mutation(canonical)
    with pytest.raises(ValueError, match=error):
        builder.build(SimpleNamespace(time=0.0, gcd_remaining=0.0))


def test_live_and_cached_decisions_use_identical_caller_phase():
    builder, canonical = _live_builder_fixture()
    state = SimpleNamespace(time=1.0, gcd_remaining=0.03)
    batch, _ = builder.build(state)
    cached, _ = builder.build_from_canonical(canonical, gcd_phase=True, max_history=2)
    torch.testing.assert_close(batch["action_legal_mask"], cached["action_legal_mask"])
    torch.testing.assert_close(batch["current_state_vectors"], cached["current_state_vectors"])

@pytest.mark.parametrize("reordered_fields", [False, True])
@pytest.mark.parametrize("identical_snapshots", [False, True])
def test_live_history_cache_preserves_same_skill_same_time_rows_and_sliding_window(
    monkeypatch, reordered_fields, identical_snapshots,
):
    from copy import deepcopy

    builder, canonical = _live_builder_fixture(max_history=2, reorder_state_fields=reordered_fields)
    calls = []
    real_build = builder._build_cached_history_row

    def track(*args, **kwargs):
        calls.append(args[0]["skill_id"])
        return real_build(*args, **kwargs)

    monkeypatch.setattr(builder, "_build_cached_history_row", track)
    _append_live_history(canonical, 1, after=100.0, request_time=10.0, skill_id=7)
    _append_live_history(canonical, 2, after=120.0, request_time=10.0, skill_id=7)
    if identical_snapshots:
        canonical["state_history_context"]["tokens"][1] = deepcopy(
            canonical["state_history_context"]["tokens"][0]
        )
    state = SimpleNamespace(time=12.0, gcd_remaining=0.0)
    first, _ = builder.build(state)
    assert first["history_skill_ids"].tolist() == [[7, 7]]
    assert len(builder._cached_history_rows) == 2
    assert builder._cached_history_rows[0].identity == builder._cached_history_rows[1].identity
    assert builder._cached_history_rows[0].identity[-1] == 10.0
    repeated, _ = builder.build(state)
    assert calls == [7, 7]
    torch.testing.assert_close(repeated["history_state_vectors"], first["history_state_vectors"])

    _append_live_history(canonical, 3, after=140.0, request_time=10.0, skill_id=7)
    slid, _ = builder.build(state)
    mp_index = canonical["state_history_context"]["player_state_feature_keys"].index("request_state.mp")
    assert slid["history_skill_ids"].tolist() == [[7, 7]]
    first_mp = 100.0 if identical_snapshots else 120.0
    torch.testing.assert_close(slid["history_state_vectors"][0, :, mp_index],
                               torch.tensor([first_mp / 10000, (140.0 - first_mp) / 10000]))
    assert calls == [7, 7, 7]

    canonical["skill_history_context"][-1]["potency"] = 99.0
    changed, _ = builder.build(state)
    assert changed["history_skill_features"][0, -1, 0].item() == 1.0
    assert changed["history_skill_features"][0, -1, 1] > slid["history_skill_features"][0, -1, 1]
    assert calls == [7, 7, 7, 7]


@pytest.mark.parametrize("invalid_time", [None, True, float("nan"), float("inf")])
def test_live_history_rejects_invalid_request_timestamp(invalid_time):
    builder, canonical = _live_builder_fixture()
    _append_live_history(canonical, 1)
    request_index = canonical["state_history_context"]["player_state_feature_keys"].index("request_state.time_seconds")
    canonical["state_history_context"]["tokens"][0]["player_state"][request_index] = invalid_time
    with pytest.raises(ValueError, match="request_state.time_seconds must be finite numeric"):
        builder.build_from_canonical(canonical, gcd_phase=True, max_history=2)


def test_live_history_rejects_old_skill_timestamp_even_with_new_state_schema():
    builder, canonical = _live_builder_fixture()
    _append_live_history(canonical, 1)
    canonical["skill_history_context"][0]["time_seconds"] = 1.0
    with pytest.raises(ValueError, match="live skill token must not include time_seconds"):
        builder.build_from_canonical(canonical, gcd_phase=True, max_history=2)


def test_live_window_uses_accumulated_cursor_after_canonical_retention():
    """留存一直是 300 行，分段窗口仍由累计 301/593/594 游标决定。"""
    from dataclasses import replace

    builder, canonical = _live_builder_fixture(max_history=300)
    builder._model_config = replace(builder._model_config, history_reset_keep=8)
    expected_lengths = {300: 300, 301: 8, 593: 300, 594: 8}
    for index in range(1, 595):
        _append_live_history(canonical, index, after=100.0)
        canonical["skill_history_context"] = canonical["skill_history_context"][-300:]
        canonical["state_history_context"]["tokens"] = canonical["state_history_context"]["tokens"][-300:]
        if index not in expected_lengths:
            continue
        actual, _ = builder.build_from_canonical(canonical, gcd_phase=True, max_history=300)
        expected_length = expected_lengths[index]
        assert actual["history_skill_ids"].shape[1] == expected_length
        assert builder.context_metadata == {"history_cursor": index}
        assert int(actual["history_mask"].sum()) == expected_length
        assert not {"history_cursor", "history_window_start", "history_window_length"} & actual.keys()
        assert actual["history_skill_ids"][0, 0].item() == index - expected_length + 1
        assert actual["history_state_reset_mask"][0, 0].all()
        repeated, _ = builder.build_from_canonical(canonical, gcd_phase=True, max_history=300)
        torch.testing.assert_close(repeated["history_state_vectors"], actual["history_state_vectors"])
        assert canonical["history_cursor"] == index


@pytest.mark.parametrize("cursor", [None, True, -1, 1.0])
def test_live_rejects_missing_or_invalid_accumulated_cursor(cursor):
    builder, canonical = _live_builder_fixture()
    canonical["history_cursor"] = cursor
    with pytest.raises(ValueError, match="history_cursor"):
        builder.build_from_canonical(canonical, gcd_phase=True, max_history=2)


def test_live_history_limit_can_be_smaller_than_saved_reset_keep():
    builder, canonical = _live_builder_fixture(max_history=8)
    for index in range(1, 10):
        _append_live_history(canonical, index)
    limited, _ = builder.build_from_canonical(canonical, gcd_phase=True, max_history=4)
    assert limited["history_skill_ids"].tolist() == [[6, 7, 8, 9]]
    no_history, _ = builder.build_from_canonical(canonical, gcd_phase=True, max_history=0)
    assert no_history["current_state_reset_mask"].all()
    assert builder.context_metadata == {"history_cursor": 9}


@pytest.mark.parametrize("capacity,total,limit,expected", [
    (300, 33, 32, 32), (300, 33, 0, 0), (300, 301, 32, 8),
    (300, 593, 32, 32), (300, 594, 32, 8), (32, 33, 32, 8),
])
def test_live_history_limit_does_not_change_model_reset_cycle(capacity, total, limit, expected):
    """真实 builder 同时覆盖读取消融与正式容量重置，不改变完整游标。"""
    from dataclasses import replace

    builder, canonical = _live_builder_fixture(max_history=capacity)
    builder._model_config = replace(builder._model_config, history_reset_keep=8)
    for index in range(1, total + 1):
        _append_live_history(canonical, index)
    limited, _ = builder.build_from_canonical(canonical, gcd_phase=True, max_history=limit)
    assert limited["history_skill_ids"].shape[1] == expected
    assert limited["history_skill_ids"].tolist() == [list(range(total - expected + 1, total + 1))]
    assert builder.context_metadata == {"history_cursor": total}
    if expected:
        assert limited["history_state_reset_mask"][0, 0].all()
    else:
        assert limited["current_state_reset_mask"].all()
    metadata = builder.context_metadata
    metadata["history_cursor"] = -1
    assert builder.context_metadata == {"history_cursor": total}


@pytest.mark.parametrize("limit", [True, -1, 1.0, None])
def test_live_rejects_invalid_history_limit(limit):
    builder, canonical = _live_builder_fixture()
    with pytest.raises(ValueError, match="max_history"):
        builder.build_from_canonical(canonical, gcd_phase=True, max_history=limit)


def test_live_unknown_recovery_marks_absolute_fields_and_checks_current_time():
    builder, canonical = _live_builder_fixture()
    _append_live_history(canonical, 1, after=None)
    _append_live_history(canonical, 2, after=150.0)
    actual, _ = builder.build_from_canonical(canonical, gcd_phase=True, max_history=2)
    assert actual["history_state_null_mask"][0, 0, 3]
    assert not actual["history_state_reset_mask"][0, 0, 3]
    assert actual["history_state_vectors"][0, 0, 3].item() == -1.0
    assert actual["history_state_reset_mask"][0, 1, 3]
    assert actual["history_state_vectors"][0, 1, 3].item() == pytest.approx(.015)
    canonical["current_state_context"]["tokens"][0]["player_state"][2] = None
    with pytest.raises(ValueError, match="request_state.time_seconds.*finite"):
        builder.build_from_canonical(canonical, gcd_phase=True, max_history=2)


def test_live_scene_eta_is_shared_by_both_build_entries_and_preserves_other_tokens():
    from copy import deepcopy
    from scripts.common.scene_state import SceneStateLookup
    from tests.helpers import build_test_scene_context, targetable_window_token
    builder, canonical = _live_builder_fixture()
    scene = build_test_scene_context(targetable_tokens=[
        targetable_window_token(10, 20, targetable=False, segment_kind="downtime"),
    ])
    builder._scene_provider.state_at = SceneStateLookup(scene).state_at
    context = canonical["current_state_context"]
    keys, vector = context["player_state_feature_keys"], context["tokens"][0]["player_state"]
    vector[keys.index("previous_action_after.time_seconds")] = 5.0
    vector[keys.index("request_state.time_seconds")] = 15.0
    baseline = deepcopy(canonical)
    live, _ = builder.build(SimpleNamespace(time=15.0, gcd_remaining=0.0))
    cached, _ = builder.build_from_canonical(canonical, gcd_phase=True, max_history=2)
    assert canonical == baseline
    for key in live:
        if isinstance(live[key], torch.Tensor):
            torch.testing.assert_close(live[key], cached[key])
    # ETA/剩余沿用 Normalizer 的秒数上限，两段使用各自冻结时间。
    expected_seconds = 5 / builder._normalizer.remaining_seconds_max
    assert live["current_state_vectors"][0, keys.index("previous_action_after.next_untargetable_in_seconds")] == pytest.approx(expected_seconds)
    assert live["current_state_vectors"][0, keys.index("request_state.downtime_remaining_seconds")] == pytest.approx(expected_seconds)
    before_non_state = {key: value.clone() for key, value in live.items()
                        if isinstance(value, torch.Tensor) and not key.startswith("current_state")}
    context["tokens"][0]["skill_availability"][0] = 0
    changed, _ = builder.build_from_canonical(canonical, gcd_phase=True, max_history=2)
    for key, value in before_non_state.items():
        torch.testing.assert_close(changed[key], value)
    assert changed["current_state_vectors"][0, -4] == 0.0


def test_live_history_cache_refreshes_absolute_availability_without_delta(monkeypatch):
    builder, canonical = _live_builder_fixture()
    _append_live_history(canonical, 1)
    _append_live_history(canonical, 2)
    first, _ = builder.build_from_canonical(canonical, gcd_phase=True, max_history=2)
    canonical["state_history_context"]["tokens"][-1]["skill_availability"] = [0, 1, 0, 1]
    changed, _ = builder.build_from_canonical(canonical, gcd_phase=True, max_history=2)
    torch.testing.assert_close(changed["history_state_vectors"][..., :8], first["history_state_vectors"][..., :8])
    assert changed["history_state_vectors"][0, -1, -4:].tolist() == [0.0, 1.0, 0.0, 1.0]


@pytest.mark.parametrize("version", [None, CANONICAL_CONTEXT_SCHEMA_VERSION - 1, CANONICAL_CONTEXT_SCHEMA_VERSION + 1])
def test_live_rejects_missing_old_or_future_canonical_version(version):
    builder, canonical = _live_builder_fixture()
    if version is None:
        canonical.pop("schema_version")
    else:
        canonical["schema_version"] = version
    with pytest.raises(ValueError, match="canonical context schema version"):
        builder.build_from_canonical(canonical, gcd_phase=True, max_history=2)
    with pytest.raises(ValueError, match="canonical context schema version"):
        builder.build(SimpleNamespace(time=0.0, gcd_remaining=0.0))

"""自回归回放 scene 上下文与 live batch 测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from common.contracts import SLIDECAST_WINDOW_SECONDS
from common.numeric import flatten_numeric_mapping
from common.policy.data import Normalizer
from common.policy.data.schema import (
    SCENE_TYPE_MOVEMENT,
    SCENE_TYPE_RAID_BUFF,
    SCENE_TYPE_TARGET_COUNT,
    SCENE_TYPE_TARGETABLE,
)
from scripts.autoregressive_replay import context as replay_context_module
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
    feature_keys = (
        "start_offset_seconds",
        "end_offset_seconds",
        "duration_seconds",
        "target_count",
    )


class _SceneSchema:
    scene_windows = (_SceneWindow(),)
    state_group_feature_keys = {"player_state": ("previous_action_after.mp",)}

    @staticmethod
    def scene_feature_dim():
        return 4

    @staticmethod
    def state_vector_dim():
        return 1


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
                    [0.0, 1.0 / 1800.0, 1.0 / 1800.0, 2.0 / 3.0],
                    [1.0 / 1800.0, 3.0 / 1800.0, 2.0 / 1800.0, 1.0],
                ],
                dtype=float_dtype,
            ),
            torch.tensor([SCENE_TYPE_TARGET_COUNT, SCENE_TYPE_TARGET_COUNT], dtype=int_dtype),
        )


class _TargetableSceneWindow:
    scene_type_id = SCENE_TYPE_TARGETABLE
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
        scale = 1800.0
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


def test_scene_template_provider_uses_normalized_scene_and_resolves_target_count():
    provider = SceneTemplateProvider(_SceneReader(), normalizer=Normalizer(), initial_sample_index=1)
    vectors, types = provider.at_time(0.0)
    assert vectors.shape == (2, 4)
    assert int(types[0]) == SCENE_TYPE_TARGET_COUNT
    assert provider.target_count_at(0.0) == 2
    assert provider.target_count_at(1.8) == 3
    assert provider.target_count_at(3.0) == 1
    later_vectors, later_types = provider.at_time(2.0)
    assert later_vectors.equal(vectors)
    assert later_types.equal(types)

    disabled = SceneTemplateProvider(_SceneReader(), normalizer=Normalizer(), enabled=False)
    empty_vectors, empty_types = disabled.at_time(2.0)
    assert empty_vectors.shape == (0, 4)
    assert empty_types.shape == (0,)
    assert disabled.target_count_at(2.0) == 1

    with pytest.raises(ValueError, match="out of range"):
        SceneTemplateProvider(_SceneReader(), normalizer=Normalizer(), initial_sample_index=3)

    class EmptyReader(_SceneReader):
        num_samples = 1

        @staticmethod
        def scene_tokens(index, *, float_dtype, int_dtype):
            return torch.zeros((0, 4), dtype=float_dtype), torch.zeros((0,), dtype=int_dtype)

    with pytest.raises(ValueError, match="no usable scene tokens"):
        SceneTemplateProvider(EmptyReader(), normalizer=Normalizer())

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

    with pytest.raises(ValueError, match="must be normalized"):
        SceneTemplateProvider(RawSceneReader(), normalizer=Normalizer(), initial_sample_index=1)


def test_scene_template_provider_syncs_targetable_timeline_into_live_state():
    class FakeBackend:
        def __init__(self):
            self.events = []

        def apply_external_event(self, timestamp, event_kind, **payload):
            self.events.append((timestamp, event_kind, payload))

    provider = SceneTemplateProvider(
        _TargetableSceneReader(),
        normalizer=Normalizer(),
        backend=FakeBackend(),
    )
    state = SimpleNamespace(time=10.0, fight_remaining=100.0)
    backend = provider._backend
    provider.sync_state(state)
    assert provider.last_targetable_end() == pytest.approx(600.0)
    assert provider.next_targetable_event_after(10.0) == pytest.approx(22.0)

    state.time = 30.0
    provider.sync_state(state)
    assert backend.events[-4:] == [
        (30.0, "boss_targetable_changed", {"value": False}),
        (30.0, "movement_changed", {"value": False}),
        (30.0, "target_count_changed", {"target_count": 1}),
        (30.0, "raid_buff_window_changed", {"value": False}),
    ]
    assert provider.targetable_at(81.9) is False
    assert provider.next_targetable_event_after(30.0) == pytest.approx(82.0)

    state.time = 82.0
    provider.sync_state(state)
    assert backend.events[-4:] == [
        (82.0, "boss_targetable_changed", {"value": True}),
        (82.0, "movement_changed", {"value": False}),
        (82.0, "target_count_changed", {"target_count": 1}),
        (82.0, "raid_buff_window_changed", {"value": False}),
    ]


def test_scene_template_provider_reset_resyncs_same_signature():
    class FakeBackend:
        def __init__(self):
            self.events = []

        def apply_external_event(self, timestamp, event_kind, **payload):
            self.events.append((timestamp, event_kind, payload))

    backend = FakeBackend()
    provider = SceneTemplateProvider(
        _TargetableSceneReader(),
        normalizer=Normalizer(),
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
    scale = 1800.0

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
                        [40.0 / scale, 60.0 / scale, 20.0 / scale, 2.0 / 3.0],
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

        def apply_external_event(self, timestamp, event_kind, **payload):
            self.events.append((timestamp, event_kind, payload))

    backend = FakeBackend()
    provider = SceneTemplateProvider(
        Reader(),
        normalizer=Normalizer(),
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
    assert provider.is_moving_at(12.0) is True
    assert provider.is_moving_at(19.5) is False
    assert provider.next_state_event_after(10.0) == pytest.approx(
        20.0 - SLIDECAST_WINDOW_SECONDS
    )

    state.time = 19.6
    provider.sync_state(state)
    assert backend.events[-4:] == [
        (19.6, "boss_targetable_changed", {"value": True}),
        (19.6, "movement_changed", {"value": False}),
        (19.6, "target_count_changed", {"target_count": 1}),
        (19.6, "raid_buff_window_changed", {"value": False}),
    ]

    state.time = 45.0
    provider.sync_state(state)
    assert backend.events[-2] == (45.0, "target_count_changed", {"target_count": 2})
    event = backend.events[-1]
    assert event[:2] == (45.0, "raid_buff_window_changed")
    assert event[2]["value"] is True
    assert event[2]["remaining_seconds"] == pytest.approx(5.0)

    state.time = 55.0
    provider.sync_state(state)
    assert backend.events[-1] == (55.0, "raid_buff_window_changed", {"value": False})

    state.time = 61.0
    provider.sync_state(state)
    assert (61.0, "target_count_changed", {"target_count": 1}) in backend.events


def _live_builder_fixture(*, max_history=2, reorder_state_fields=False):
    from copy import deepcopy

    player_keys = ("previous_action_after.time_seconds", "previous_action_after.mp",
                   "request_state.time_seconds", "request_state.mp")
    if reorder_state_fields:
        player_keys = tuple(reversed(player_keys))
    initial_values = {"previous_action_after.time_seconds": 0.0, "previous_action_after.mp": 200.0,
                      "request_state.time_seconds": 0.0, "request_state.mp": 200.0}
    canonical = {
        "action_keys": ["fire", "ogcd_wait"],
        "action_legal_mask": [True, True],
        "skill_history_context": [],
        "state_history_context": {"player_state_feature_keys": player_keys, "tokens": []},
        "current_state_context": {"tokens": [{"player_state": [initial_values[key] for key in player_keys]}]},
    }

    class Backend:
        def observe_at(self, timestamp, *, format, next_observation_timestamp):
            assert format == "vector"
            return SimpleNamespace(context=deepcopy(canonical))

    schema = SimpleNamespace(
        state_group_feature_keys={"player_state": player_keys},
        state_vector_dim=lambda: 4, scene_feature_dim=lambda: 2,
    )
    builder = LiveBatchBuilder(
        backend=Backend(), vocab=SimpleNamespace(require_lookup=lambda value, **kwargs: int(value)),
        normalizer=SimpleNamespace(
            normalize=lambda value, *args, **kwargs: value,
            normalize_skill_features=lambda value, *args: value,
        ), schema=schema, skill_feature_names=("kind", "potency"),
        scene_provider=SimpleNamespace(at_time=lambda _: (torch.empty((0, 2)), torch.empty(0, dtype=torch.int32))),
        device=torch.device("cpu"), max_history=max_history,
        action_keys=("fire", "ogcd_wait"), action_is_gcd=(True, False),
    )
    return builder, canonical


def _append_live_history(canonical, index, *, after=100.0, request_time=None, skill_id=None):
    canonical["skill_history_context"].append({
        "skill_id": index if skill_id is None else skill_id, "skill_key": "fire", "kind": 1, "potency": 50.0,
    })
    state_values = {"previous_action_after.time_seconds": float(index - 1), "previous_action_after.mp": 200.0,
                    "request_state.time_seconds": float(index) if request_time is None else request_time,
                    "request_state.mp": after}
    canonical["state_history_context"]["tokens"].append({
        "player_state": [state_values[key] for key in canonical["state_history_context"]["player_state_feature_keys"]],
    })


@pytest.mark.parametrize("remaining,expected", ((0.0, [True, False]), (1.0, [False, True])))
def test_live_current_state_is_independent_of_action_legality_and_preserves_phase(remaining, expected):
    builder, canonical = _live_builder_fixture()
    batch, keys = builder.build(SimpleNamespace(time=1.0, gcd_remaining=remaining))
    assert keys == ["fire", "ogcd_wait"]
    assert batch["current_state_vectors"].shape == (1, 4)
    assert batch["current_state_vectors"].tolist() == [[0.0, 200.0, 0.0, 200.0]]
    assert batch["current_state_null_mask"].tolist() == [[False, False, False, False]]
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
    assert batch["history_state_vectors"].tolist() == [[[1.0, 200.0, 2.0, 2.0], [2.0, 200.0, 3.0, 3.0]]]
    assert batch["history_skill_features"].tolist() == [[[1.0, 50.0], [1.0, 50.0]]]
    assert batch["history_action_keys"] == [["fire", "fire"]]
    empty, _ = builder.build(SimpleNamespace(time=4.0, gcd_remaining=0.0), max_history=0)
    assert empty["history_skill_ids"].shape == (1, 0)
    assert empty["current_state_vectors"].shape == (1, 4)


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
    assert repeated["history_state_vectors"].data_ptr() == first["history_state_vectors"].data_ptr()
    canonical["current_state_context"]["tokens"][0]["player_state"] = [0.0, 150.0, 0.0, 150.0]
    refreshed, _ = builder.build(state)
    assert calls == [1, 2]
    assert refreshed["current_state_vectors"].tolist() == [[0.0, 150.0, 0.0, 150.0]]
    _append_live_history(canonical, 3)
    builder.build(state)
    assert calls == [1, 2, 3]
    _append_live_history(canonical, 4)
    slid, _ = builder.build(state)
    assert calls == [1, 2, 3, 4]
    assert slid["history_skill_ids"].tolist() == [[2, 3, 4]]
    canonical["state_history_context"]["tokens"][-1]["player_state"] = [3.0, 10.0, 4.0, None]
    changed, _ = builder.build(state)
    assert calls[-1] == 4
    assert changed["history_state_null_mask"][0, -1].tolist() == [False, False, False, True]


@pytest.mark.parametrize("mutation,error", (
    (lambda c: c.update(action_keys=[]), "nonempty"),
    (lambda c: c.update(action_keys=["ogcd_wait", "fire"]), "order differs"),
    (lambda c: c.update(action_legal_mask=[True]), "legality width"),
    (lambda c: c["current_state_context"].update(tokens=[]), "exactly one"),
    (lambda c: c["current_state_context"].update(tokens=[{"player_state": [1.0]}]), "vector width"),
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
    assert repeated["history_state_vectors"].data_ptr() == first["history_state_vectors"].data_ptr()

    _append_live_history(canonical, 3, after=140.0, request_time=10.0, skill_id=7)
    slid, _ = builder.build(state)
    mp_index = canonical["state_history_context"]["player_state_feature_keys"].index("request_state.mp")
    assert slid["history_skill_ids"].tolist() == [[7, 7]]
    assert slid["history_state_vectors"][0, :, mp_index].tolist() == [
        100.0 if identical_snapshots else 120.0, 140.0,
    ]
    assert calls == [7, 7, 7]

    canonical["skill_history_context"][-1]["potency"] = 99.0
    changed, _ = builder.build(state)
    assert changed["history_skill_features"][0, -1].tolist() == [1.0, 99.0]
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

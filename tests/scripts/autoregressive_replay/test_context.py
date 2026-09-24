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
    state_group_feature_keys = {"player_state": ("before.mp",)}

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


def test_live_batch_builder_builds_padded_history_and_state_vectors():
    class FakeNormalizer:
        @staticmethod
        def normalize_skill_features(values, _names):
            normalized = values + 1.0
            normalized[..., 0] = values[..., 0]
            return normalized

        @staticmethod
        def normalize(values, _group, *, null_mask):
            return values + 2.0

    class FakeVocab:
        @staticmethod
        def require_lookup(value, *, context):
            assert context == "live replay"
            return int(value) + 100

    class FakeProvider:
        def target_count_at(self, time):
            return 2

        def at_time(self, time):
            return torch.tensor([[2.0, 0.0]]), torch.tensor([SCENE_TYPE_TARGET_COUNT], dtype=torch.int32)

    class FakeBackend:
        @staticmethod
        def observe_at(_timestamp, *, format, next_observation_timestamp):
            del format, next_observation_timestamp
            return SimpleNamespace(
                timestamp=float(_timestamp),
                next_scheduled_event_time=None,
                context={
                    "candidate_skill_context": [
                        {
                            "skill_id": 10,
                            "skill_key": "a",
                            "kind": 1,
                            "potency": 100,
                            "cast_time": {"seconds": 2.5},
                            "is_legal": True,
                        },
                        {"skill_id": 11, "skill_key": "b", "kind": 0, "is_legal": False},
                    ],
                    "candidate_state_context": {
                        "player_state_feature_keys": ["before.gcd_remaining_seconds"],
                        "tokens": [
                            {"player_state": [0.0], "resource_state": [None]},
                            {"player_state": [0.2], "resource_state": [True]},
                        ]
                    },
                    "skill_history_context": [
                        {"skill_id": 12, "skill_key": "old", "kind": 1, "potency": 10},
                        {"skill_id": 13, "skill_key": "new", "kind": 0, "potency": 20},
                    ],
                    "state_history_context": {
                        "tokens": [
                            {"player_state": [0.0], "resource_state": [False]},
                            {"player_state": [0.0], "resource_state": [None]},
                        ]
                    },
                },
            )

    schema = SimpleNamespace(
        state_group_feature_keys={"player_state": ("mp",), "resource_state": ("af",)},
        state_vector_dim=lambda: 2,
        scene_feature_dim=lambda: 2,
    )
    builder = LiveBatchBuilder(
        backend=FakeBackend(),
        vocab=FakeVocab(),
        normalizer=FakeNormalizer(),
        schema=schema,
        skill_feature_names=("kind", "potency", "cast_time.seconds"),
        scene_provider=FakeProvider(),
        device=torch.device("cpu"),
        max_history=1,
        candidate_action_keys=("b", "a"),
    )
    state = SimpleNamespace(time=1.0, target_count=1, gcd_remaining=0.0)
    batch, keys = builder.build(state)
    assert keys == ["b", "a"]
    assert batch["history_action_keys"] == [["new"]]
    assert batch["history_skill_ids"].tolist() == [[113]]
    assert batch["history_skill_features"].tolist() == [[[0.0, 21.0, 1.0]]]
    assert batch["candidate_skill_ids"].tolist() == [[111, 110]]
    assert batch["candidate_skill_features"].tolist() == [[[0.0, 1.0, 1.0], [1.0, 101.0, 3.5]]]
    assert batch["candidate_legal_mask"].tolist() == [[False, True]]
    assert batch["history_state_vectors"].tolist() == [[[2.0, 2.0]]]
    assert batch["history_state_null_mask"].tolist() == [[[False, True]]]
    torch.testing.assert_close(
        batch["candidate_state_vectors"],
        torch.tensor([[[2.2, 3.0], [2.0, 2.0]]]),
    )
    assert batch["candidate_state_null_mask"].tolist() == [[[False, False], [False, True]]]
    assert batch["scene_vectors"].shape == (1, 1, 2)

    empty_state_values, empty_state_null_mask = builder._build_state_tensors([])
    assert empty_state_values.shape == (0, 2)
    assert empty_state_null_mask.shape == (0, 2)
    assert builder._build_skill_features([]).shape == (0, 3)
    with pytest.raises(ValueError, match="player_state vector width mismatch"):
        builder._build_state_tensors(
            [{"player_state": [0.0, 1.0], "resource_state": [None]}]
        )

    with pytest.raises(ValueError, match="max_history"):
        replay_context_module._tail_history([], -1)
    assert flatten_numeric_mapping(
        {"flag": True, "kind": "x", "nested": {"keep": 2, "drop": 3}},
        ignored_keys={"kind", "drop"},
    ) == {"flag": 1.0, "nested.keep": 2.0}
    with pytest.raises(ValueError, match="must be a list"):
        replay_context_module._extract_nullable_vector(None)
    with pytest.raises(ValueError, match="unsupported"):
        replay_context_module._extract_nullable_vector([object()])

    class EmptyBackend:
        @staticmethod
        def observe_at(_timestamp, *, format, next_observation_timestamp):
            del format, next_observation_timestamp
            return SimpleNamespace(
                timestamp=1.0,
                next_scheduled_event_time=None,
                context={
                    "candidate_skill_context": [],
                    "candidate_state_context": {"tokens": []},
                    "skill_history_context": [],
                    "state_history_context": {"tokens": []},
                },
            )

    empty_builder = LiveBatchBuilder(
        backend=EmptyBackend(),
        vocab=FakeVocab(),
        normalizer=FakeNormalizer(),
        schema=schema,
        skill_feature_names=(),
        scene_provider=FakeProvider(),
        device=torch.device("cpu"),
        max_history=1,
        candidate_action_keys=("a",),
    )
    with pytest.raises(RuntimeError, match="no candidate actions"):
        empty_builder.build(state)


def test_live_batch_builder_reuses_history_rows_and_refreshes_changed_rows():
    class FakeNormalizer:
        @staticmethod
        def normalize_skill_features(values, _names):
            normalized = values + 1.0
            normalized[..., 0] = values[..., 0]
            return normalized

        @staticmethod
        def normalize(values, _group, *, null_mask):
            del null_mask
            return values + 2.0

    class CountingBuilder(LiveBatchBuilder):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.built_history_keys = []

        def _build_cached_history_row(self, skill_token, state_token, identity):
            self.built_history_keys.append(str(skill_token.get("skill_key", "")))
            return super()._build_cached_history_row(skill_token, state_token, identity)

    def make_builder():
        schema = SimpleNamespace(
            state_group_feature_keys={
                "player_state": ("mp",),
                "resource_state": ("af",),
            },
            state_vector_dim=lambda: 2,
            scene_feature_dim=lambda: 1,
        )
        return CountingBuilder(
            backend=SimpleNamespace(),
            vocab=SimpleNamespace(
                require_lookup=lambda value, **_kwargs: int(value) + 100
            ),
            normalizer=FakeNormalizer(),
            schema=schema,
            skill_feature_names=("kind", "potency"),
            scene_provider=SimpleNamespace(
                at_time=lambda _time: (
                    torch.zeros((0, 1), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.int32),
                )
            ),
            device=torch.device("cpu"),
            max_history=2,
            candidate_action_keys=("candidate",),
        )

    def make_context(history, *, candidate_legal=True, candidate_remaining=0.0):
        skill_history = [
            {
                "skill_id": skill_id,
                "skill_key": key,
                "kind": 1,
                "potency": potency,
                "time_seconds": float(skill_id),
                "gcd_index": skill_id,
            }
            for skill_id, key, potency in history
        ]
        state_history = [
            {
                "player_state": [float(skill_id)],
                "resource_state": [bool(skill_id % 2)],
            }
            for skill_id, _key, _potency in history
        ]
        return {
            "candidate_skill_context": [
                {
                    "skill_id": 99,
                    "skill_key": "candidate",
                    "kind": 1,
                    "potency": 50,
                    "is_legal": candidate_legal,
                }
            ],
            "candidate_state_context": {
                "player_state_feature_keys": ["before.gcd_remaining_seconds"],
                "tokens": [
                    {
                        "player_state": [candidate_remaining],
                        "resource_state": [True],
                    }
                ],
            },
            "skill_history_context": skill_history,
            "state_history_context": {"tokens": state_history},
        }

    def assert_batches_equal(actual, expected):
        actual_batch, actual_keys = actual
        expected_batch, expected_keys = expected
        assert actual_keys == expected_keys
        assert actual_batch.keys() == expected_batch.keys()
        for name, actual_value in actual_batch.items():
            expected_value = expected_batch[name]
            if isinstance(actual_value, torch.Tensor):
                assert torch.equal(actual_value, expected_value), name
            else:
                assert actual_value == expected_value, name

    builder = make_builder()
    first = make_context([(1, "first", 10), (2, "second", 20)])
    builder.build_from_canonical(first, max_history=2)
    assert builder.built_history_keys == ["first", "second"]

    # 历史窗口滚动时复用仍在窗口内的行，只转换新追加的一行。
    appended = make_context([(1, "first", 10), (2, "second", 20), (3, "third", 30)])
    cached_appended = builder.build_from_canonical(appended, max_history=2)
    assert builder.built_history_keys == ["first", "second", "third"]
    fresh_appended = make_builder().build_from_canonical(appended, max_history=2)
    assert_batches_equal(cached_appended, fresh_appended)

    # 同一事件标识的历史内容若变化，必须重算该行，不能命中旧特征。
    changed = make_context([(2, "second", 20), (3, "third", 300)])
    changed["state_history_context"]["tokens"][1]["player_state"] = [300.0]
    cached_changed = builder.build_from_canonical(changed, max_history=2)
    assert builder.built_history_keys == ["first", "second", "third", "third"]
    assert cached_changed[0]["history_skill_features"][0, -1, 1].item() == 301.0
    fresh_changed = make_builder().build_from_canonical(changed, max_history=2)
    assert_batches_equal(cached_changed, fresh_changed)

    # 候选状态每步重建；候选变化不能被历史缓存遮蔽。
    dynamic = make_context(
        [(2, "second", 20), (3, "third", 300)],
        candidate_legal=False,
        candidate_remaining=0.75,
    )
    dynamic["state_history_context"]["tokens"][1]["player_state"] = [300.0]
    cached_dynamic = builder.build_from_canonical(dynamic, max_history=2)
    assert builder.built_history_keys == ["first", "second", "third", "third"]
    assert cached_dynamic[0]["candidate_legal_mask"].tolist() == [[False]]
    fresh_dynamic = make_builder().build_from_canonical(dynamic, max_history=2)
    assert_batches_equal(cached_dynamic, fresh_dynamic)


@pytest.mark.parametrize("remaining,expected", [
    (2.45, [False, True, True]),
    (0.05, [True, False, False]),
    (0.0, [True, False, False]),
])
def test_live_and_cached_decisions_share_phase_mask(remaining, expected):
    canonical = {
        "candidate_skill_context": [
            {"skill_id": i, "skill_key": key, "kind": kind, "is_legal": True}
            for i, (key, kind) in enumerate([("fire", 1), ("swiftcast", 0), ("ogcd_wait", 0)])
        ],
        "candidate_state_context": {
            "player_state_feature_keys": ["before.gcd_remaining_seconds"],
            "tokens": [{"player_state": [remaining]} for _ in range(3)],
        },
        "skill_history_context": [],
        "state_history_context": {"tokens": []},
    }
    builder = LiveBatchBuilder(
        backend=SimpleNamespace(observe_at=lambda *args, **kwargs: SimpleNamespace(context=canonical)),
        vocab=SimpleNamespace(require_lookup=lambda value, **kwargs: value),
        normalizer=SimpleNamespace(normalize=lambda values, *args, **kwargs: values,
                                   normalize_skill_features=lambda values, *args: values),
        schema=SimpleNamespace(state_group_feature_keys={"player_state": ["before.gcd_remaining_seconds"]},
                               state_vector_dim=lambda: 1, scene_feature_dim=lambda: 1),
        skill_feature_names=("kind",),
        scene_provider=SimpleNamespace(at_time=lambda _: (torch.zeros((0, 1)), torch.zeros(0, dtype=torch.int32))),
        device=torch.device("cpu"), max_history=4,
    )
    live, _ = builder.build(SimpleNamespace(time=0, gcd_remaining=remaining))
    cached, _ = builder.build_from_canonical(canonical, max_history=0)
    assert live["candidate_legal_mask"].tolist() == [expected]
    assert torch.equal(live["candidate_legal_mask"], cached["candidate_legal_mask"])

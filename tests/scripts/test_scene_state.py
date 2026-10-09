"""共享场景执行视图、同刻事实批次与纯状态合成回归。"""

from __future__ import annotations

import pytest

from scripts.common.scene_state import (
    BOSS_TARGETABLE_CHANGED,
    RAID_BUFF_WINDOW_CHANGED,
    TARGET_COUNT_CHANGED,
    SceneFactScheduler,
    SceneStateLookup,
)
from scripts.convert_fflogs.scene.scene_context import (
    build_raid_buff_window_token,
    build_target_count_window_token,
    build_targetable_window_token,
)
from tests.helpers import build_test_scene_context


def test_scene_lookup_cache_preserves_boundaries_is_bounded_and_releases_owner():
    import weakref
    from scripts.common import scene_state as module
    from common.contracts import FORCED_MOVEMENT_CONTEXT_KEY, SCENE_EPSILON

    scene = build_test_scene_context(targetable_tokens=[
        build_targetable_window_token(0, 10, targetable=True, segment_kind="combat"),
        build_targetable_window_token(10, 20, targetable=False, segment_kind="downtime"),
        build_targetable_window_token(20, 30, targetable=True, segment_kind="combat_final"),
    ])
    movement = scene[FORCED_MOVEMENT_CONTEXT_KEY]
    keys = movement["feature_keys"]
    movement["tokens"] = [[3.0 if key == "start_offset_seconds" else
                            7.0 if key == "end_offset_seconds" else 0.0 for key in keys]]
    lookup = module.SceneStateLookup(scene)
    timestamps = [boundary + epsilon for boundary in (0, 3, 6.5, 7, 10, 20, 30)
                  for epsilon in (-2 * SCENE_EPSILON, -SCENE_EPSILON, 0, SCENE_EPSILON)]
    # 端点判断与事实时刻严格一致，历史查询顺序可以回退。
    for timestamp in timestamps + list(reversed(timestamps)):
        state = lookup.state_at(timestamp)
        assert state.is_moving == (3 <= timestamp < 6.5)
        assert state.boss_targetable == (not 10 <= timestamp < 20)
    assert lookup._cached_state_at.cache_info().hits >= len(timestamps)
    for index in range(10000):
        lookup.state_at(index / 1000)
    assert lookup._cached_state_at.cache_info().currsize <= 4096
    reference = weakref.ref(lookup)
    del lookup
    assert reference() is None


def test_target_count_facts_restore_single_target_after_window():
    """多目标区间结束后必须显式归位到单目标。

    状态机的 `CombatState.TargetCount` 是持久字段；只发"进入多目标"不发
    "离开多目标"会让 AoE 衰减倍率从第一次多目标之后永久生效。
    """
    scene_context = build_test_scene_context(
        target_count_tokens=[build_target_count_window_token(10.0, 20.0, 2)],
    )
    facts = _target_count_facts(scene_context)

    assert [(fact.timestamp, fact.target_count) for fact in facts] == [(10.0, 2), (20.0, 1)]


def test_target_count_facts_do_not_flip_between_adjacent_windows():
    """首尾相接的同目标数区间不应产生 2→1→2 的来回翻转。"""
    scene_context = build_test_scene_context(
        target_count_tokens=[
            build_target_count_window_token(10.0, 20.0, 2),
            build_target_count_window_token(20.0, 30.0, 2),
        ],
    )
    facts = _target_count_facts(scene_context)

    assert [(fact.timestamp, fact.target_count) for fact in facts] == [(10.0, 2), (30.0, 1)]


def test_target_count_facts_change_between_distinct_counts():
    """相邻区间的目标数不同时，两个端点都要发事实。"""
    scene_context = build_test_scene_context(
        target_count_tokens=[
            build_target_count_window_token(10.0, 20.0, 2),
            build_target_count_window_token(20.0, 30.0, 4),
        ],
    )
    facts = _target_count_facts(scene_context)

    assert [(fact.timestamp, fact.target_count) for fact in facts] == [
        (10.0, 2),
        (20.0, 4),
        (30.0, 1),
    ]


@pytest.mark.parametrize(
    "tokens",
    [
        [],
        [build_target_count_window_token(0.0, 100.0, 1)],
    ],
)
def test_target_count_facts_stay_silent_without_multi_target(tokens):
    """没有多目标区间时不发事实，状态机保持默认单目标。"""
    scene_context = build_test_scene_context(target_count_tokens=tokens)

    assert _target_count_facts(scene_context) == []


def test_target_count_fact_replay_matches_lookup():
    """事实流重放出来的目标数必须等于按时刻查表的结果。"""
    scene_context = build_test_scene_context(
        target_count_tokens=[
            build_target_count_window_token(10.0, 20.0, 2),
            build_target_count_window_token(40.0, 50.0, 3),
        ],
    )
    facts = _target_count_facts(scene_context)
    lookup = SceneStateLookup(scene_context)

    replayed = 1
    cursor = 0
    for step in range(0, 601):
        timestamp = step / 10.0
        while cursor < len(facts) and facts[cursor].timestamp <= timestamp:
            replayed = facts[cursor].target_count
            cursor += 1
        assert replayed == lookup.state_at(timestamp).target_count, timestamp


def test_scene_facts_only_cover_injected_kinds():
    """没有移动窗口时只生成其余三类变化事实。"""
    scene_context = build_test_scene_context(
        targetable_tokens=[
            build_targetable_window_token(0.0, 30.0, targetable=True, segment_kind="combat"),
            build_targetable_window_token(30.0, 40.0, targetable=False, segment_kind="downtime"),
            build_targetable_window_token(40.0, 60.0, targetable=True, segment_kind="combat_final"),
        ],
        target_count_tokens=[build_target_count_window_token(10.0, 20.0, 2)],
        raid_buff_tokens=[build_raid_buff_window_token(5.0, 25.0)],
    )
    facts = SceneFactScheduler(scene_context).pop_facts_through(1_000.0)

    assert {fact.event_kind for fact in facts} == {
        BOSS_TARGETABLE_CHANGED,
        TARGET_COUNT_CHANGED,
        RAID_BUFF_WINDOW_CHANGED,
    }
    assert [fact.timestamp for fact in facts] == sorted(fact.timestamp for fact in facts)
    assert [
        (fact.timestamp, fact.value)
        for fact in facts
        if fact.event_kind == BOSS_TARGETABLE_CHANGED
    ] == [(30.0, False), (40.0, True)]


def _target_count_facts(scene_context: dict[str, object]):
    facts = SceneFactScheduler(scene_context).pop_facts_through(1_000.0)
    return [fact for fact in facts if fact.event_kind == TARGET_COUNT_CHANGED]


def _state_rewrite_context(previous_time=9.0, request_time=12.0):
    from common.schema_config import load_schema_config
    fields = load_schema_config()["state_vector_fields"]["player_state"]
    keys = [f"{prefix}.{field}" for prefix in ("previous_action_after", "request_state") for field in fields]
    vector = [0.0] * len(keys)
    vector[keys.index("previous_action_after.time_seconds")] = previous_time
    vector[keys.index("request_state.time_seconds")] = request_time
    return {"player_state_feature_keys": keys, "tokens": [{"player_state": vector}]}


def test_scene_rewrite_reads_each_snapshot_time_without_skill_timestamp():
    from types import SimpleNamespace
    from scripts.common.scene_state import rewrite_scene_player_state
    canonical = {
        "current_state_context": _state_rewrite_context(9.0, 12.0),
        "state_history_context": _state_rewrite_context(8.0, 11.0),
        "skill_history_context": [{"skill_key": "fire_iv", "cast_time": {"seconds": 200.0}}],
    }
    queried = []
    def lookup(timestamp):
        queried.append(timestamp)
        return SimpleNamespace(is_moving=timestamp >= 10.0, next_downtime_eta=30.0-timestamp, downtime_remaining=0.0)
    rewritten = rewrite_scene_player_state(canonical, scene_state_at=lookup)
    assert queried == [8.0, 11.0, 9.0, 12.0]
    for key, etas in (("state_history_context", [22.0, 19.0]), ("current_state_context", [21.0, 18.0])):
        context = rewritten[key]
        keys = context["player_state_feature_keys"]
        vector = context["tokens"][0]["player_state"]
        assert [vector[keys.index(f"{prefix}.next_untargetable_in_seconds")] for prefix in ("previous_action_after", "request_state")] == etas


def test_scene_rewrite_preserves_raw_boundary_precision_and_uses_names_with_reordered_fields():
    from types import SimpleNamespace
    from scripts.common.scene_state import rewrite_scene_player_state
    context = _state_rewrite_context(9.99996, 10.0)
    context["player_state_feature_keys"].reverse()
    context["tokens"][0]["player_state"].reverse()
    from copy import deepcopy
    canonical = {"state_history_context": context, "current_state_context": deepcopy(context)}
    rewritten = rewrite_scene_player_state(canonical,
                               scene_state_at=lambda t: SimpleNamespace(is_moving=t>=10.0, next_downtime_eta=0.0, downtime_remaining=0.0))
    keys = context["player_state_feature_keys"]
    vector = context["tokens"][0]["player_state"]
    assert vector[keys.index("previous_action_after.is_moving")] == 0.0
    assert vector[keys.index("request_state.is_moving")] == 0.0
    assert rewritten is canonical


@pytest.mark.parametrize("invalid", [None, True, float("nan"), float("inf")])
def test_scene_rewrite_rejects_missing_or_invalid_time(invalid):
    from scripts.common.scene_state import rewrite_scene_player_state
    from copy import deepcopy
    context = _state_rewrite_context(invalid, 12.0)
    canonical = {"state_history_context": context, "current_state_context": deepcopy(context)}
    with pytest.raises(ValueError, match="time_seconds must be finite numeric"):
        rewrite_scene_player_state(canonical, scene_state_at=lambda _: None)


def test_scene_rewrite_rejects_missing_time_key():
    from scripts.common.scene_state import rewrite_scene_player_state
    context = _state_rewrite_context()
    context["player_state_feature_keys"][context["player_state_feature_keys"].index("previous_action_after.time_seconds")] = "unknown"
    with pytest.raises(ValueError, match="lacks previous_action_after"):
        rewrite_scene_player_state({"state_history_context": context}, scene_state_at=lambda _: None)


def test_scene_execution_fp32_projection_and_facts_share_exact_boundaries():
    """原端点与 FP32 端点之间的首个差异只修正事实，不改变 scene 输出。"""
    from copy import deepcopy
    import math
    import struct
    from scripts.common.scene_state import MOVEMENT_CHANGED, SceneStateLookup
    from tests.helpers import forced_movement_window_token

    scene = build_test_scene_context(forced_movement_tokens=[
        forced_movement_window_token(600.004, 604.007),
        forced_movement_window_token(603.0, 608.0),
        forced_movement_window_token(610.0, 610.2),
    ])
    before = deepcopy(scene)
    lookup = SceneStateLookup(scene)
    start = struct.unpack("f", struct.pack("f", 600.004))[0]
    facts = [fact for fact in lookup.facts() if fact.event_kind == MOVEMENT_CHANGED]
    assert [(fact.timestamp, fact.value) for fact in facts] == [(start, True), (607.5, False)]
    assert lookup.state_at(600.004).is_moving is False
    assert lookup.state_at(math.nextafter(start, -math.inf)).is_moving is False
    assert lookup.state_at(start).is_moving is True
    assert lookup.state_at(math.nextafter(start, math.inf)).is_moving is True
    assert lookup.state_at(603.6).is_moving is True
    assert lookup.state_at(math.nextafter(607.5, -math.inf)).is_moving is True
    assert lookup.state_at(607.5).is_moving is False
    assert scene == before
    restored = deepcopy(scene)
    for window in restored.values():
        if isinstance(window, dict) and "tokens" in window:
            window["tokens"] = [[struct.unpack("f", struct.pack("f", value))[0] for value in row]
                                for row in window["tokens"]]
    assert SceneStateLookup(restored).facts() == lookup.facts()


def test_scene_sync_batches_every_same_time_fact_and_resets_nonzero_origin():
    from types import SimpleNamespace
    from scripts.common.scene_state import MOVEMENT_CHANGED
    from tests.helpers import forced_movement_window_token

    scene = build_test_scene_context(
        targetable_tokens=[build_targetable_window_token(10, 20, targetable=False, segment_kind="downtime")],
        forced_movement_tokens=[forced_movement_window_token(10, 20)],
        target_count_tokens=[build_target_count_window_token(10, 20, 3)],
        raid_buff_tokens=[build_raid_buff_window_token(10, 20)],
    )
    batches = []
    backend = SimpleNamespace(apply_external_events=lambda batch: (
        batches.append(batch) or SimpleNamespace(accepted=True)))
    scheduler = SceneFactScheduler(scene)
    scheduler.sync_through(backend, 5.0)
    scheduler.sync_through(backend, 10.0)
    assert [event["event_kind"] for event in batches[-1]] == [
        BOSS_TARGETABLE_CHANGED, MOVEMENT_CHANGED, TARGET_COUNT_CHANGED, RAID_BUFF_WINDOW_CHANGED,
    ]
    assert {event["timestamp"] for event in batches[-1]} == {10.0}
    scheduler.sync_through(backend, 10.0)
    assert len(batches) == 2
    scheduler.reset()
    scheduler.sync_through(backend, 15.0)
    assert len(batches[-1]) == 4
    assert {event["timestamp"] for event in batches[-1]} == {15.0}
    assert batches[-1][-1]["remaining_seconds"] == 5.0


def test_scene_rewrite_only_copies_changed_state_fields_and_is_idempotent():
    from copy import deepcopy
    from scripts.common.scene_state import SceneStateLookup, rewrite_scene_player_state
    scene = build_test_scene_context(targetable_tokens=[
        build_targetable_window_token(10, 20, targetable=False, segment_kind="downtime"),
    ])
    canonical = {
        "current_state_context": _state_rewrite_context(5.0, 15.0),
        "state_history_context": _state_rewrite_context(4.0, 14.0),
        "skill_history_context": [{"skill_key": "ogcd_wait", "kind": 0}],
        "scene_context": scene, "action_keys": ["fire", "ogcd_wait"],
        "execution": {"potency": 123.0},
    }
    before = deepcopy(canonical)
    lookup = SceneStateLookup(scene)
    result = rewrite_scene_player_state(canonical, scene_state_at=lookup.state_at)
    assert canonical == before
    for key in ("skill_history_context", "scene_context", "action_keys", "execution"):
        assert result[key] is canonical[key]
        assert result[key] == before[key]
    context = result["current_state_context"]
    keys, vector = context["player_state_feature_keys"], context["tokens"][0]["player_state"]
    assert vector[keys.index("previous_action_after.next_untargetable_in_seconds")] == 5.0
    assert vector[keys.index("request_state.downtime_remaining_seconds")] == 5.0
    assert rewrite_scene_player_state(result, scene_state_at=lookup.state_at) is result


@pytest.mark.parametrize("start,end,error", [
    (float("nan"), 10.0, "finite numeric"), (1e40, 2e40, "finite FP32"),
    (20.0, 10.0, "start <= end"),
])
def test_scene_execution_rejects_invalid_bounds_before_scheduling(start, end, error):
    from common.contracts import FORCED_MOVEMENT_CONTEXT_KEY
    from scripts.common.scene_state import SceneStateLookup
    scene = {FORCED_MOVEMENT_CONTEXT_KEY: {
        "feature_keys": ["start_offset_seconds", "end_offset_seconds"], "tokens": [[start, end]],
    }}
    with pytest.raises(ValueError, match=error):
        SceneStateLookup(scene)

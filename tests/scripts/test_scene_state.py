"""scene 事实流测试：三类事实的端点语义与目标数归位。"""

from __future__ import annotations

import pytest

from scripts.common.scene_state import (
    BOSS_TARGETABLE_CHANGED,
    RAID_BUFF_WINDOW_CHANGED,
    TARGET_COUNT_CHANGED,
    SceneFactScheduler,
    resolve_target_count_at,
)
from scripts.convert_fflogs.scene_context import (
    build_raid_buff_window_token,
    build_target_count_window_token,
    build_targetable_window_token,
)
from tests.helpers import build_test_scene_context


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

    replayed = 1
    cursor = 0
    for step in range(0, 601):
        timestamp = step / 10.0
        while cursor < len(facts) and facts[cursor].timestamp <= timestamp:
            replayed = facts[cursor].target_count
            cursor += 1
        assert replayed == resolve_target_count_at(scene_context, timestamp), timestamp


def test_scene_facts_only_cover_injected_kinds():
    """只发三类进入状态机的事实，移动状态不注入。"""
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

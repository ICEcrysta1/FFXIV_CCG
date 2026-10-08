from types import SimpleNamespace

import pytest

from scripts.autoregressive_replay.scheduler import DecisionScheduler, is_gcd_decision


class Timeline:
    def __init__(self, *, gcd_ready=2.5, cast_end=0.0):
        self.gcd_ready = gcd_ready
        self.cast_end = cast_end
        self.advances = []

    def observe(self, timestamp):
        return SimpleNamespace(time=timestamp,
                               gcd_remaining=max(0, self.gcd_ready - timestamp),
                               cast_remaining=max(0, self.cast_end - timestamp),
                               next_scheduled_event_time=None)

    def advance_to(self, timestamp):
        self.advances.append(timestamp)
        return SimpleNamespace(timestamp=timestamp)


@pytest.mark.parametrize("cast_end,gcd_ready,expected,weave", [
    (0.0, 2.5, 0.05, True),
    (1.0, 2.5, 1.05, True),
    (2.0, 2.5, 2.05, True),
    (2.5, 2.5, 2.55, False),
    (3.5, 2.5, 3.55, False),
    (1.65, 2.5, 1.70, True),
])
def test_gcd_completion_reserves_two_intervals(cast_end, gcd_ready, expected, weave):
    timeline = Timeline(gcd_ready=gcd_ready, cast_end=cast_end)
    scheduler = DecisionScheduler(timeline, timeline.observe)
    state = scheduler.advance_submitted_action(
        timeline.observe(0), SimpleNamespace(accepted_timestamp=0), action_kind="gcd",
    )
    assert state.time == pytest.approx(expected)
    assert is_gcd_decision(state.gcd_remaining) is not weave


def test_ogcd_waits_one_interval_then_checks_room_for_next():
    timeline = Timeline(gcd_ready=1.0)
    scheduler = DecisionScheduler(timeline, timeline.observe)
    first = scheduler.advance_submitted_action(
        timeline.observe(0), SimpleNamespace(accepted_timestamp=0), action_kind="ogcd",
    )
    assert first.time == pytest.approx(0.1)
    assert not is_gcd_decision(first.gcd_remaining)
    second = scheduler.advance_submitted_action(
        first, SimpleNamespace(accepted_timestamp=0.1), action_kind="ogcd",
    )
    assert second.time == pytest.approx(0.2)
    assert not is_gcd_decision(second.gcd_remaining)


def test_queued_gcd_observes_cast_after_acceptance_and_respects_end_time():
    timeline = Timeline(gcd_ready=4.0, cast_end=3.0)
    scheduler = DecisionScheduler(timeline, timeline.observe)
    state = scheduler.advance_submitted_action(
        timeline.observe(0.95), SimpleNamespace(accepted_timestamp=1.0),
        action_kind="gcd", end_time=2.0,
    )
    assert state.time == 2.0
    assert timeline.advances == [1.0, 2.0]


def test_scene_facts_are_synchronized_before_advancing_across_boundary():
    timeline = Timeline()
    calls = []

    class Scene:
        def next_state_event_after(self, timestamp):
            return 1.0 if timestamp < 1.0 else None

        def sync_state(self, state):
            calls.append((state.time, list(timeline.advances)))

    scheduler = DecisionScheduler(timeline, timeline.observe, Scene())
    assert scheduler.advance_by(timeline.observe(0), 2.0).time == 2.0
    assert timeline.advances == [1.0, 2.0]
    assert calls[0] == (0.0, [])
    assert calls[1] == (1.0, [])
    assert calls[2] == (1.0, [1.0])


def test_no_weave_candidate_advances_to_gcd_request_then_next_event():
    timeline = Timeline()
    scheduler = DecisionScheduler(timeline, timeline.observe)
    state = scheduler.advance_to_next_decision(timeline.observe(0.05))
    assert state.time == pytest.approx(2.45)
    state = scheduler.advance_to_next_decision(state)
    assert state.time == pytest.approx(2.5)
    assert scheduler.advance_to_next_decision(state) is None


@pytest.mark.parametrize("request_time", [0.1, 1.7])
def test_scene_fp32_boundary_below_decision_epsilon_still_advances(request_time):
    import struct
    from common.contracts import FORCED_MOVEMENT_CONTEXT_KEY
    from scripts.common.scene_state import SceneFactScheduler

    boundary = struct.unpack("f", struct.pack("f", request_time))[0]
    assert 0.0 < boundary - request_time < 1e-6
    facts = SceneFactScheduler({FORCED_MOVEMENT_CONTEXT_KEY: {
        "feature_keys": ["start_offset_seconds", "end_offset_seconds"],
        "tokens": [[request_time, request_time + 3.0]],
    }})
    timeline = Timeline(gcd_ready=0.0)
    batches = []

    def apply_batch(events):
        batches.append((events, list(timeline.advances)))
        return SimpleNamespace(accepted=True)

    backend = SimpleNamespace(apply_external_events=apply_batch)

    class Scene:
        def next_state_event_after(self, timestamp):
            return facts.next_event_after(timestamp)

        def sync_state(self, state):
            facts.sync_through(backend, state.time)

    scheduler = DecisionScheduler(timeline, timeline.observe, Scene())
    state = timeline.observe(request_time)
    result = scheduler.advance_to_next_decision(state)
    assert result is not None
    assert result.time == boundary
    assert batches[-1][0][0]["event_kind"] == "movement_changed"
    assert batches[-1][0][0]["value"] is True
    assert batches[-1][1] == []
    assert timeline.advances == [boundary]
    # 再推进完整区间，不会反复卡在同一未消费场景边界。
    result = scheduler.advance_by(result, 1.0)
    assert result.time == boundary + 1.0
    assert len(batches) == 2

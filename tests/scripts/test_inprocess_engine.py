"""统一多队列引擎的 Python.NET 并发、生命周期和转换接入回归。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from scripts.common.inprocess_backend import InProcessEngine
from scripts.convert_fflogs import build_training_samples
from tests.helpers import build_test_scene_context, targetable_window_token
from tests.scripts.conftest import _require_inprocess_backend


@pytest.fixture(autouse=True)
def require_engine():
    _require_inprocess_backend()


@pytest.mark.parametrize("job,action", [("black_mage", "blizzard_iii"), ("machinist", "heated_split_shot")])
def test_sixteen_python_threads_share_one_engine_without_context_leaks(job, action):
    def replay(backend, index):
        contexts = []
        for step in range(5):
            timestamp = index * 100 + step * 6.0
            backend.apply_external_events([{
                "timestamp": timestamp, "event_kind": "target_count_changed", "target_count": 1 + index % 3,
            }])
            assert backend.submit_action(timestamp, action).accepted
            backend.advance_to(timestamp + 4)
            backend.record_policy_action(timestamp + 4, "ogcd_wait", timestamp + 6)
            context = backend.observe_at(timestamp + 4, format="vector", next_observation_timestamp=timestamp + 6)
            assert isinstance(context.context, dict)
            assert len(context.context["skill_history_context"]) <= 4
            contexts.append(context)
        return contexts

    expected = []
    for index in range(16):
        with InProcessEngine(job) as engine, engine.create_backend(max_history=4, actual_base_gcd=2.0 + index * 0.03, initial_timestamp=index * 100) as backend:
            expected.append(replay(backend, index))

    barrier = Barrier(16)
    with InProcessEngine(job, capacity=16) as engine:
        queues = [engine.create_backend(max_history=4, actual_base_gcd=2.0 + index * 0.03,
                                        initial_timestamp=index * 100) for index in range(16)]
        assert engine.active_count == 16
        assert len({queue.queue_id for queue in queues}) == 16

        def run(index):
            with queues[index] as backend:
                barrier.wait(timeout=20)
                return replay(backend, index)

        with ThreadPoolExecutor(max_workers=16) as workers:
            actual = list(workers.map(run, range(16)))
        assert actual == expected
        assert engine.active_count == 0


def test_capacity_reset_close_and_failed_creation_are_isolated():
    with InProcessEngine("black_mage", capacity=2) as engine:
        first = engine.create_backend(max_history=4)
        second = engine.create_backend(max_history=None)
        old_id = first.queue_id
        with pytest.raises(Exception, match="capacity reached"):
            engine.create_backend(max_history=4)
        assert engine.active_count == 2
        assert first.submit_action(0, "fire_iii").accepted
        first.record_policy_action(0, "ogcd_wait", 4)
        with pytest.raises(Exception):
            first.init(actual_base_gcd=float("nan"))
        assert first.statistics()["policy_history_count"] == 1
        assert first.init(actual_base_gcd=2.1, initial_timestamp=100, fight_remaining=80) == 100
        assert first.queue_id == old_id
        assert first.statistics()["policy_history_count"] == 0
        assert first.statistics()["action_history_count"] == 0
        assert second.statistics()["timestamp"] == 0
        first.close()
        first.close()
        with pytest.raises(Exception):
            engine.create_backend(max_history=-1)
        assert engine.active_count == 1
        with engine.create_backend(max_history=4) as replacement:
            assert replacement.queue_id != old_id
            with pytest.raises(RuntimeError, match="closed"):
                first.advance_to(100)
            assert replacement.advance_to(0).timestamp == 0
        assert second.submit_action(0, "blizzard_iii").accepted
    with pytest.raises(RuntimeError, match="closed"):
        second.advance_to(4)
    second.close()


def test_six_conversion_workers_share_engine_and_preserve_full_history(cs_skill_book):
    payload = {
        "fight_id": "shared_conversion", "job_tag": "black_mage", "duration": 30.0,
        "gcd_time": 2.46,
        "scene_context": build_test_scene_context(targetable_tokens=[
            targetable_window_token(0, 30, targetable=True, segment_kind="combat"),
        ]),
        "actions": [
            {"time_offset": time, "request_time_offset": time, "time_gap": 6.0,
             "action_key": key, "fight_remaining": 30.0 - time, "anchor": "combat"}
            for time, key in [(0.0, "fire_iii"), (6.0, "high_thunder"),
                              (12.0, "blizzard_iii"), (18.0, "fire_iii")]
        ],
    }
    with InProcessEngine("black_mage") as engine, engine.create_backend(actual_base_gcd=2.46, max_history=None) as backend:
        expected = build_training_samples(backend, cs_skill_book, payload)

    barrier = Barrier(6)
    with InProcessEngine("black_mage", capacity=6) as engine:
        def convert(_):
            with engine.create_backend(max_history=None, actual_base_gcd=2.46) as backend:
                barrier.wait(timeout=20)
                result = build_training_samples(backend, cs_skill_book, payload)
                assert result["resolved_sequence"] == expected["resolved_sequence"]
                return result

        with ThreadPoolExecutor(max_workers=6) as workers:
            results = list(workers.map(convert, range(6)))
        assert all(result == expected for result in results)
        assert engine.active_count == 0


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_invalid_capacity_is_rejected_before_loading_runtime(capacity):
    with pytest.raises(ValueError, match="capacity"):
        InProcessEngine("black_mage", capacity=capacity)


def test_external_fact_batch_is_atomic_and_accepts_single_element():
    with InProcessEngine("black_mage", capacity=1) as engine, engine.create_backend(max_history=None) as backend:
        before = backend.statistics()
        with pytest.raises(Exception):
            backend.apply_external_events([
                {"timestamp": 1.0, "event_kind": "movement_changed", "value": True},
                {"timestamp": 1.0, "event_kind": "target_count_changed", "target_count": -1},
            ])
        assert backend.statistics() == before
        applied = backend.apply_external_events([
            {"timestamp": 1.0, "event_kind": "movement_changed", "value": True},
            {"timestamp": 1.0, "event_kind": "target_count_changed", "target_count": 2},
        ])
        assert applied.accepted
        assert applied.timestamp == 1.0
        canonical = backend.observe_at(1.0, format="vector", next_observation_timestamp=1.0).context
        current = canonical["current_state_context"]
        index = current["player_state_feature_keys"].index("request_state.is_moving")
        assert current["tokens"][0]["player_state"][index] == 1.0
        assert backend.apply_external_events([
            {"timestamp": 1.0, "event_kind": "movement_changed", "value": False},
        ]).accepted
        assert backend.validate_at(1.0, "fire_iii").legal

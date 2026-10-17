"""统一多队列引擎的 Python.NET 并发、生命周期和转换接入回归。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from scripts.common import inprocess_backend as backend_module
from scripts.common.inprocess_backend import InProcessEngine, MAX_QUEUE_CAPACITY
from scripts.convert_fflogs import build_training_samples
from tests.helpers import build_test_scene_context, targetable_window_token
from tests.scripts.conftest import _require_inprocess_backend


@pytest.fixture
def require_engine():
    _require_inprocess_backend()


@pytest.mark.parametrize("job,action", [("black_mage", "blizzard_iii"), ("machinist", "heated_split_shot")])
def test_sixteen_python_threads_share_one_engine_without_context_leaks(job, action, require_engine):
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


def test_capacity_reset_close_and_failed_creation_are_isolated(require_engine):
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


def test_six_conversion_workers_share_engine_and_preserve_full_history(cs_skill_book, require_engine):
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


@pytest.mark.parametrize("capacity", [0, -1, 101, 1.5, True, False])
def test_invalid_capacity_is_rejected_before_loading_runtime(capacity, monkeypatch):
    monkeypatch.setattr(
        backend_module, "_load_dotnet_types",
        lambda: pytest.fail("非法队列容量必须在加载 .NET runtime 前拒绝"),
    )
    with pytest.raises(ValueError, match="capacity"):
        InProcessEngine("black_mage", capacity=capacity)


def test_python_queue_capacity_matches_csharp_limit(require_engine):
    engine_type, _, _ = backend_module._load_dotnet_types()
    assert MAX_QUEUE_CAPACITY == int(engine_type.MaxCapacity) == 100


def test_default_engine_holds_one_hundred_independent_queues_and_reuses_capacity(require_engine):
    """默认容量不预建队列；满载后的拒绝和回收不影响其他队列的状态。"""
    with InProcessEngine("black_mage") as engine:
        assert engine.capacity == 100
        assert engine.active_count == 0
        queues = [
            engine.create_backend(
                max_history=None, actual_base_gcd=2.0 + index * 0.005,
                initial_timestamp=index * 100.0,
            )
            for index in range(100)
        ]
        assert engine.active_count == 100
        assert len({queue.queue_id for queue in queues}) == 100
        with pytest.raises(Exception, match="capacity reached"):
            engine.create_backend(max_history=None)
        assert engine.active_count == 100

        # 先分别提交不同时间和动作，再回读所有队列，捕获后续操作造成的串状态。
        actions = ["fire_iii" if index % 2 == 0 else "blizzard_iii" for index in range(100)]
        for index, (backend, action) in enumerate(zip(queues, actions, strict=True)):
            assert backend.statistics()["timestamp"] == index * 100.0
            assert backend.statistics()["action_history_count"] == 0
            assert backend.submit_action(index * 100.0, action, actual_cast_seconds=0.0).accepted
        for index, (backend, action) in enumerate(zip(queues, actions, strict=True)):
            timestamp = index * 100.0
            scalar = backend.observe_at(timestamp, format="seconds").context
            assert scalar["time_seconds"] == timestamp
            assert scalar["gcd_remaining_seconds"] == pytest.approx(2.0 + index * 0.005)
            canonical = backend.observe_at(
                timestamp, format="vector", next_observation_timestamp=timestamp,
            ).context
            assert [row["skill_key"] for row in canonical["skill_history_context"]] == [action]

        released = queues[37]
        released_id = released.queue_id
        released.close()
        assert engine.active_count == 99
        with engine.create_backend(
            max_history=None, actual_base_gcd=2.37, initial_timestamp=12345.0,
        ) as replacement:
            assert engine.active_count == 100
            assert replacement.queue_id not in {queue.queue_id for queue in queues if queue is not released}
            assert replacement.queue_id != released_id
            assert replacement.statistics()["timestamp"] == 12345.0
            assert replacement.statistics()["action_history_count"] == 0
            assert replacement.submit_action(12345.0, "blizzard_iii", actual_cast_seconds=0.0).accepted
            assert queues[38].statistics()["timestamp"] == 3800.0
            assert queues[38].statistics()["action_history_count"] == 1
            with pytest.raises(RuntimeError, match="closed"):
                released.advance_to(3700.0)
        assert engine.active_count == 99


def test_external_fact_batch_is_atomic_and_accepts_single_element(require_engine):
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


def test_death_returns_special_token_without_changing_model_inputs(require_engine):
    with InProcessEngine("black_mage", capacity=1) as engine, engine.create_backend(max_history=None) as backend:
        ordinary = backend.apply_external_events([
            {"timestamp": 0.0, "event_kind": "target_count_changed", "target_count": 1},
        ])
        assert ordinary.token is None
        assert ordinary.history_cursor is None
        assert backend.submit_action(0.0, "fire_iii").accepted
        assert backend.submit_action(4.0, "high_thunder").accepted

        death = backend.apply_external_events([
            {"timestamp": 6.0, "event_kind": "player_death"},
        ])

        assert death.accepted
        assert death.token == "<dead/>"
        assert death.history_cursor == 2
        context = backend.observe_at(6.0, format="vector", next_observation_timestamp=6.0).context
        current = context["current_state_context"]
        token = current["tokens"][0]
        assert sum(len(values) for values in token.values()) == 132
        assert len(context["action_keys"]) == 25
        assert [row["skill_key"] for row in context["skill_history_context"]] == ["fire_iii", "high_thunder"]
        fields = {
            group: dict(zip(current[f"{group}_feature_keys"], values, strict=True))
            for group, values in token.items()
        }
        assert fields["player_state"]["request_state.mp"] == 0
        for group in ("buff_state", "resource_state"):
            assert all(value == 0 for key, value in fields[group].items() if key.startswith("request_state."))
        assert fields["target_buff_state"]["request_state.target.high_thunder.active"] == 1

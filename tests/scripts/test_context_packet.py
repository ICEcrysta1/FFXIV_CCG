"""真实 Rust 回放数据包及观测容器隔离。"""

import gc
import json
import pickle

import pytest

from scripts.common.inprocess_backend import InProcessEngine
from tests.scripts.conftest import _require_inprocess_backend


@pytest.fixture(autouse=True)
def require_bridge():
    _require_inprocess_backend()


@pytest.mark.parametrize("job,actions", [
    ("black_mage", ("fire_iii", "blizzard_iii")),
    ("machinist", ("heated_split_shot", "heated_slug_shot")),
])
def test_packet_matches_public_replay_and_decodes_after_engine_close(job, actions):
    requests = [{"timestamp": timestamp, "skill_key": action, "actual_cast_seconds": 0.0}
                for timestamp, action in zip((0.0, 3.0), actions, strict=True)]
    facts = [{"timestamp": 1.0, "event_kind": "player_death"},
             {"timestamp": 2.0, "event_kind": "player_revived", "mp": 3000}]
    options = {"fight_duration": 10.0, "policy_action_key": "ogcd_wait"}
    with InProcessEngine(job) as engine, engine.create_backend(max_history=None, actual_base_gcd=2.46) as backend:
        expected = backend.replay_policy_sequence(requests, facts, **options)
        backend.init(actual_base_gcd=2.46, initial_timestamp=0.0)
        packet = backend.replay_policy_sequence_packet(json.dumps(requests), json.dumps(facts), **options)
    assert type(packet) is bytes
    gc.collect()
    decoded = pickle.loads(packet)
    assert decoded["observations"] == expected
    decoded["observations"].clear()
    assert pickle.loads(packet)["observations"] == expected


@pytest.mark.parametrize("format", ["seconds", "gcd", "vector"])
def test_observations_are_independent_after_caller_mutation_and_reset(format):
    from copy import deepcopy

    with InProcessEngine("black_mage") as engine, engine.create_backend(max_history=2) as backend:
        assert backend.submit_action(0, "blizzard_iii").accepted
        backend.advance_to(4)
        backend.record_policy_action(4, "ogcd_wait", 6)
        first = backend.observe_at(4, format=format, next_observation_timestamp=6).context
        expected = deepcopy(first)
        if format == "vector":
            first["state_history_context"]["tokens"][0]["player_state"][0] = 123456
            first["skill_history_context"][0]["skill_key"] = "caller_mutation"
        else:
            first.clear()
        assert backend.observe_at(4, format=format, next_observation_timestamp=6).context == expected
        backend.init(initial_timestamp=0)
        fresh = backend.observe_at(0, format="vector", next_observation_timestamp=0).context
        assert fresh["skill_history_context"] == []

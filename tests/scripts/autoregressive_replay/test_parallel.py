"""并行回放的合批、容量、缓存和故障退出契约。"""

from threading import get_ident

import pytest
import torch

from scripts.autoregressive_replay import parallel


class FakeEngine:
    def __init__(self, job_tag, *, capacity):
        self.capacity = capacity
        self.closed = False

    def close(self):
        self.closed = True


class FakePolicy:
    supports_batch_inference = True

    def __init__(self):
        self.batches = []
        self.cache_changes = []
        self.owner = get_ident()

    def configure_cache(self, enabled):
        assert get_ident() == self.owner
        self.cache_changes.append(enabled)

    def raw_logits(self, batch, keys):
        assert get_ident() == self.owner
        assert keys == ("a", "b")
        self.batches.append(batch)
        return batch["candidate_skill_features"][:, :, 0]


def sample(value, length=1):
    return {
        "history_skill_ids": torch.full((1, length), value, dtype=torch.int64),
        "history_mask": torch.ones((1, length), dtype=torch.bool),
        "history_state_null_mask": torch.zeros((1, length, 1), dtype=torch.bool),
        "candidate_skill_features": torch.tensor([[[float(value)], [0.0]]]),
        "history_action_keys": [["a"] * length],
        "candidate_action_keys": [["a", "b"]],
    }


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch):
    monkeypatch.setattr(parallel, "InProcessEngine", FakeEngine)


def test_bounded_parallel_batches_preserve_order_and_cache_layout():
    policy = FakePolicy()
    seen = []

    def inputs():
        for value in range(7):
            seen.append(value)
            yield value

    def worker(value, client, engine):
        assert get_ident() != policy.owner
        assert engine.capacity == 3
        client.configure_cache(True)
        for step in range(1 + value % 2):
            logits = client.raw_logits(sample(value, step + 1), ("a", "b"))
        return int(logits[0, 0])

    with parallel.ParallelRollouts(policy, job_tag="test", workers=3) as pool:
        results = pool.map(worker, inputs())
        assert next(results) == 0
        assert seen == [0, 1, 2]
        assert list(results) == list(range(1, 7))
        assert max(batch["history_mask"].shape[0] for batch in policy.batches) == 3
        assert all(policy.cache_changes)
    assert pool.engine.closed


def test_collation_preserves_masks_and_variable_lengths():
    batch = parallel.collate_live_batches([sample(2, 0), sample(7, 3)])
    assert batch["history_mask"].tolist() == [[False] * 3, [True] * 3]
    assert batch["history_state_null_mask"][0].all()
    assert not batch["history_state_null_mask"][1].any()
    assert batch["history_action_keys"] == [[], ["a"] * 3]


@pytest.mark.parametrize("failure", ["worker", "model"])
def test_failure_wakes_other_queues_and_releases_engine(failure):
    policy = FakePolicy()
    if failure == "model":
        def fail(*args):
            raise ValueError("model failure")
        policy.raw_logits = fail

    def worker(value, client, engine):
        if failure == "worker" and value == 1:
            raise ValueError("worker failure")
        return client.raw_logits(sample(value), ("a", "b"))

    with pytest.raises(ValueError, match=f"{failure} failure"):
        with parallel.ParallelRollouts(policy, job_tag="test", workers=4) as pool:
            list(pool.map(worker, range(10)))
    assert pool.engine.closed


def test_fixed_batch_backend_still_services_all_parallel_queues():
    policy = FakePolicy()
    policy.supports_batch_inference = False
    with parallel.ParallelRollouts(policy, job_tag="test", workers=3) as pool:
        result = list(pool.map(lambda item, client, engine: client.raw_logits(
            sample(item), ("a", "b")), range(3)))
    assert [int(row[0, 0]) for row in result] == [0, 1, 2]
    assert all(batch["history_mask"].shape[0] == 1 for batch in policy.batches)


def test_mixed_cache_modes_are_not_merged():
    policy = FakePolicy()
    def worker(item, client, engine):
        client.configure_cache(bool(item % 2))
        return client.raw_logits(sample(item), ("a", "b"))
    with parallel.ParallelRollouts(policy, job_tag="test", workers=2) as pool:
        assert len(list(pool.map(worker, range(2)))) == 2
    assert policy.cache_changes == [False, True]


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_capacity_is_rejected(workers):
    with pytest.raises(ValueError):
        parallel.ParallelRollouts(FakePolicy(), job_tag="test", workers=workers)


def test_shared_shard_cache_serializes_loading_and_survives_spawn_pickle():
    import pickle
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from common.policy.data.compiled_cache import CompiledShardCache

    cache = CompiledShardCache(max_shards=2)
    calls = []
    barrier = Barrier(8)
    def load():
        calls.append(1)
        return [{"value": 7}]
    def worker(_):
        barrier.wait(timeout=5)
        return cache.get(("scene", 0), load)
    with ThreadPoolExecutor(max_workers=8) as executor:
        values = list(executor.map(worker, range(8)))
    assert len(calls) == 1
    assert all(value is values[0] for value in values)
    restored = pickle.loads(pickle.dumps(cache))
    assert restored.get(("scene", 0), lambda: pytest.fail("cache lost during spawn")) == values[0]
    for index in range(1, 6):
        restored.get(("scene", index), load)
    assert len(restored) == 2


def test_backend_measurement_retains_bounded_samples(monkeypatch):
    from scripts.autoregressive_replay import backends

    monkeypatch.setattr(backends, "_process_peak_working_set_bytes", lambda: 123)
    monkeypatch.setattr(backends, "perf_counter", lambda: 1.0)
    backend = backends._MeasuredBackend()
    for index in range(5000):
        backend._finish_measurement(0 if index == 0 else 0.5)
    metrics = backend.metrics()
    assert len(backend._latencies_ms) == 4096
    assert metrics.calls == 5000
    assert metrics.latency_ms_max == 1000
    assert metrics.latency_ms_p50 == 500


def test_per_queue_policy_failure_does_not_cancel_other_audits():
    shared = FakePolicy()
    policies = []
    prepared = []
    def prepare(items, engine):
        assert not policies
        prepared.extend(items)
    def factory(item):
        policy = FakePolicy()
        if item == 1:
            def fail(*_args):
                raise ValueError("one audit failed")
            policy.raw_logits = fail
        policies.append(policy)
        return policy
    def run(item, policy, engine):
        assert prepared == [0, 1, 2]
        try:
            for _ in range(3):
                value = policy.raw_logits(sample(item), ("a", "b"))
            return int(value[0, 0])
        except RuntimeError as exc:
            assert "one audit failed" in str(exc)
            return "failed"
    with parallel.ParallelRollouts(shared, job_tag="test", workers=3) as pool:
        results = list(pool.map(run, range(3), prepare=prepare, policy_factory=factory, isolate_errors=True))
    assert results == [0, "failed", 2]
    assert len(policies[0].batches) == len(policies[2].batches) == 3
    assert pool.engine.closed

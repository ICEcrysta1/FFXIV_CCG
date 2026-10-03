"""普通回放的多场景入口，共用单模型、共享引擎及有界缓存。"""

from itertools import chain

import torch

from .parallel import ParallelRollouts
from .replay import AutoregressiveReplay, AutoregressiveReplaySession, ReplayCacheStore, _create_backend


def run_replays(configs, *, backend=None, workers=None, history_limits=None):
    """按输入顺序产生结果；只保留一个有界批次，允许调用方及时写出报告。"""
    iterator = iter(configs)
    first = next(iterator, None)
    if first is None:
        return
    policy = _create_backend(first) if backend is None else backend
    signature_fields = ("checkpoint_path", "backend", "onnx_package_path", "device", "policy_precision", "ort_provider")
    signature = tuple(getattr(first, field, None) for field in signature_fields)
    seed = torch.initial_seed()

    def run(item, client, engine):
        index, config = item
        if tuple(getattr(config, field, None) for field in signature_fields) != signature:
            raise ValueError("parallel replays must use the same policy backend and checkpoint")
        with AutoregressiveReplaySession(config, backend=client, cache_store=cache, engine=engine) as session:
            replay = AutoregressiveReplay(config, session=session)
            replay._sampling_generator = torch.Generator(device=client.input_device)
            replay._sampling_generator.manual_seed((seed + index) % (2**64))
            return replay.run() if history_limits is None else replay.run_history_ablation(history_limits)

    with ParallelRollouts(policy, job_tag=policy.data_spec.job_tag, workers=workers) as pool:
        cache = ReplayCacheStore(max_shards=first.cache_max_shards, max_readers=pool.workers)
        def prepare(items, engine):
            for _, config in items:
                if tuple(getattr(config, field, None) for field in signature_fields) != signature:
                    raise ValueError("parallel replays must use the same policy backend and checkpoint")
            cache.prepare(
                [config for _, config in items], job_tag=policy.data_spec.job_tag,
                normalizer=policy.input_contract.create_normalizer(), engine=engine, workers=pool.workers,
            )
        yield from pool.map(run, enumerate(chain((first,), iterator)), prepare=prepare)

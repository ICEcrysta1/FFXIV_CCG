"""有界多队列回放：工作线程推进 C#，调用线程统一执行模型合批。"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from threading import Condition

import torch

from common.project_config import resolve_positive_worker_count
from common.torch_runtime import autocast_context
from scripts.common.inprocess_backend import InProcessEngine


def replay_worker_count() -> int:
    """机器并发参数统一从根目录 .env 读取。"""
    return resolve_positive_worker_count(
        project_root=Path(__file__).resolve().parents[2],
        env_name="AUTOREGRESSIVE_REPLAY_WORKERS",
    )


def collate_live_batches(samples):
    """右侧补齐变长场景和历史；固定候选保持原始顺序。"""
    if not samples:
        raise ValueError("cannot collate empty live batches")
    if any(sample.keys() != samples[0].keys() for sample in samples[1:]):
        raise ValueError("live batch fields must match across replay queues")
    result = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        if isinstance(values[0], torch.Tensor):
            if any(value.shape[0] != 1 for value in values):
                raise ValueError("each replay request must contain exactly one row")
            if key.startswith(("scene_", "history_")):
                result[key] = torch.nn.utils.rnn.pad_sequence(
                    [value[0] for value in values], batch_first=True,
                    padding_value=key == "history_state_null_mask",
                )
            else:
                result[key] = torch.cat(values, dim=0)
        else:
            result[key] = [list(value[0]) for value in values]
    return result


class TrainingPolicyBackend:
    """训练中的现有模型适配器，不加载 checkpoint 或复制模型权重。"""

    supports_batch_inference = True

    def __init__(self, model, *, data_spec, device, precision):
        self.model = model
        self.data_spec = data_spec
        self.input_device = device
        self.precision = precision

    def configure_cache(self, enabled):
        configure = getattr(self.model, "enable_kv_cache", None)
        if callable(configure):
            configure(enabled)

    def raw_logits(self, batch, candidate_action_keys):
        with autocast_context(self.input_device, self.precision):
            return self.model(batch)["logits"].float()


class _PolicyClient:
    """单轨迹推理句柄；工作线程不直接访问模型可变缓存。"""

    def __init__(self, coordinator, slot, backend):
        self._coordinator = coordinator
        self._slot = slot
        self._backend = backend
        self._cache_enabled = False
        self._cache_generation = 0

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def configure_cache(self, enabled):
        self._cache_enabled = bool(enabled)
        self._cache_generation += 1

    @property
    def _kv_cache_enabled(self):
        return self._cache_enabled

    def raw_logits(self, batch, candidate_action_keys):
        return self._coordinator.request(self, batch, tuple(candidate_action_keys))

    def eval(self):
        # 模型只由调度器的调用线程操作；兼容 PPG 的单轨迹接口。
        return self

    def __call__(self, batch):
        return {"logits": self.raw_logits(batch, self.data_spec.candidate_action_keys)}


class _InferenceCoordinator:
    def __init__(self, backend, count, *, isolate_errors=False):
        self.backend = backend
        self.condition = Condition()
        self.active = set(range(count))
        self.pending = {}
        self.failure = None
        self.cache_layout = None
        self.isolate_errors = isolate_errors

    def request(self, client, batch, keys):
        future = Future()
        with self.condition:
            if self.failure is not None:
                raise RuntimeError("parallel replay cancelled") from self.failure
            if client._slot in self.pending:
                raise RuntimeError("a replay queue already has an inference request")
            self.pending[client._slot] = (client, batch, keys, future)
            self.condition.notify_all()
        return future.result()

    def finish(self, slot):
        with self.condition:
            self.active.remove(slot)
            self.condition.notify_all()

    def abort(self, exc):
        with self.condition:
            if self.failure is None:
                self.failure = exc
            for _, _, _, future in self.pending.values():
                if not future.done():
                    future.set_exception(RuntimeError("parallel replay cancelled"))
            self.pending.clear()
            self.condition.notify_all()

    def serve(self):
        """只有存活轨迹全部到达决策点才合批，结果不依赖线程到达顺序。"""
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.failure is not None or
                                        not self.active or self.active == self.pending.keys())
                if self.failure is not None:
                    raise self.failure
                if not self.active:
                    return
                requests = [self.pending.pop(slot) for slot in sorted(self.active)]
            try:
                self._infer(requests)
            except BaseException as exc:
                for _, _, _, future in requests:
                    if not future.done():
                        failure = RuntimeError("parallel inference failed")
                        failure.__cause__ = exc
                        future.set_exception(failure)
                self.abort(exc)
                raise

    def _infer(self, requests):
        # 固定 batch=1 的部署包保留契约，状态机仍可跨队列并行。
        shared = all(r[0]._backend is requests[0][0]._backend for r in requests)
        backend = requests[0][0]._backend
        groups = [requests] if shared and getattr(backend, "supports_batch_inference", False) else [[r] for r in requests]
        for group in groups:
            try:
                self._infer_group(group)
            except Exception as exc:
                if not self.isolate_errors:
                    raise
                # 验收任务各自写失败报告，不因一条轨迹失败跳过其他验收。
                for _, _, _, future in group:
                    if future.done():
                        continue
                    failure = RuntimeError(f"parallel inference failed: {exc}")
                    failure.__cause__ = exc
                    future.set_exception(failure)

    def _infer_group(self, group):
        backend = group[0][0]._backend
        flags = {client._cache_enabled for client, *_ in group}
        if len(flags) != 1:
            for request in group:
                self._infer([request])
            return
        layout = (id(backend), tuple((c._slot, c._cache_generation, c._cache_enabled) for c, *_ in group))
        if layout != self.cache_layout:
            backend.configure_cache(next(iter(flags)))
            self.cache_layout = layout
        keys = group[0][2]
        if any(request[2] != keys for request in group):
            raise ValueError("parallel replay candidate order mismatch")
        batch = collate_live_batches([request[1] for request in group])
        with torch.inference_mode():
            logits = backend.raw_logits(batch, keys)
        if logits.ndim != 2 or logits.shape[0] != len(group):
            raise ValueError("parallel backend returned an invalid logits batch")
        for index, (_, _, _, future) in enumerate(group):
            # 克隆单行，避免慢轨迹持有整个 batch 的输出 storage。
            future.set_result(logits[index:index + 1].clone())


class ParallelRollouts:
    """每批最多 workers 条轨迹、一个共享引擎和一个模型；失败时唤醒所有等待者。"""

    def __init__(self, backend, *, job_tag, workers=None):
        self.workers = replay_worker_count() if workers is None else workers
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError("workers must be a positive integer")
        self.backend = backend
        self.engine = InProcessEngine(job_tag, capacity=self.workers)
        self._closed = False

    def map(self, worker, items, *, prepare=None, policy_factory=None, isolate_errors=False):
        """worker(item, policy, engine)；输入与输出有序，在途及完成结果均有上限。"""
        if self._closed:
            raise RuntimeError("parallel replay is closed")
        iterator = iter(items)
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="replay") as executor:
            while chunk := list(islice(iterator, self.workers)):
                # 准备阶段也复用本引擎；回放队列尚未占用容量，可并行补齐缓存。
                if prepare is not None:
                    prepare(chunk, self.engine)
                policies = [self.backend if policy_factory is None else policy_factory(item) for item in chunk]
                coordinator = _InferenceCoordinator(self.backend, len(chunk), isolate_errors=isolate_errors)

                def run(slot, item):
                    try:
                        with torch.inference_mode():
                            return worker(item, _PolicyClient(coordinator, slot, policies[slot]), self.engine)
                    except BaseException as exc:
                        coordinator.abort(exc)
                        raise
                    finally:
                        coordinator.finish(slot)

                futures = [executor.submit(run, slot, item) for slot, item in enumerate(chunk)]
                try:
                    coordinator.serve()
                    while futures:
                        yield futures.pop(0).result()
                except BaseException as exc:
                    coordinator.abort(exc)
                    raise

    def close(self):
        if not self._closed:
            self._closed = True
            self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

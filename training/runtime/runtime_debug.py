"""训练运行时的可选显存与耗时调试。"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Iterator, Mapping

import torch


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeDebugConfig:
    """控制训练运行时显存调试，默认关闭以保持正常训练路径不变。"""

    enabled: bool = False
    max_steps: int = 1
    synchronize: bool = True
    output_filename: str = "runtime_memory.jsonl"


class RuntimeDebugRecorder:
    """按训练 step 记录阶段耗时、显存高水位和可复用的原始 JSONL。"""

    def __init__(
        self,
        *,
        device: torch.device,
        config: RuntimeDebugConfig,
        output_path: Path,
    ) -> None:
        self.device = device
        self.config = config
        self.output_path = Path(output_path)
        self.enabled = bool(config.enabled)
        self._recorded_steps = 0
        self._active = False
        self._step_started_at = 0.0
        self._step_peak_allocated = 0
        self._step_peak_reserved = 0
        self._events: list[dict[str, object]] = []
        self._step_context: dict[str, object] = {}
        self._output_initialized = False

    @property
    def active(self) -> bool:
        """当前是否正在采集一个调试 step。"""
        return self._active

    def begin_step(
        self,
        *,
        epoch: int,
        step: int,
        batch: Mapping[str, object],
    ) -> None:
        """开始采集一个 step；只采集前 max_steps 个 step。"""
        self._active = self.enabled and self._recorded_steps < self.config.max_steps
        if not self._active:
            return

        self._synchronize()
        if self._cuda_memory_supported:
            torch.cuda.reset_peak_memory_stats(self.device)
        self._step_started_at = time.perf_counter()
        self._step_peak_allocated = self._current_allocated_bytes()
        self._step_peak_reserved = self._current_reserved_bytes()
        self._events = []
        self._step_context = {
            "epoch": int(epoch),
            "step": int(step),
            "global_debug_step": self._recorded_steps + 1,
            "batch": summarize_batch(batch),
        }

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """记录一个阶段；未启用或未处于采集 step 时完全退化为空上下文。"""
        if not self._active:
            with nullcontext():
                yield
            return

        self._synchronize()
        started_at = time.perf_counter()
        allocated_before = self._current_allocated_bytes()
        reserved_before = self._current_reserved_bytes()
        try:
            yield
        finally:
            self._synchronize()
            finished_at = time.perf_counter()
            allocated_after = self._current_allocated_bytes()
            reserved_after = self._current_reserved_bytes()
            peak_allocated = self._peak_allocated_bytes()
            peak_reserved = self._peak_reserved_bytes()
            previous_peak_allocated = self._step_peak_allocated
            previous_peak_reserved = self._step_peak_reserved
            self._step_peak_allocated = max(previous_peak_allocated, peak_allocated)
            self._step_peak_reserved = max(previous_peak_reserved, peak_reserved)
            self._events.append(
                {
                    "name": str(name),
                    "elapsed_ms": (finished_at - started_at) * 1000.0,
                    "allocated_before_mib": _mib(allocated_before),
                    "allocated_after_mib": _mib(allocated_after),
                    "allocated_delta_mib": _mib(allocated_after - allocated_before),
                    "peak_allocated_mib": _mib(peak_allocated),
                    "new_peak_mib": _mib(max(0, peak_allocated - previous_peak_allocated)),
                    "reserved_after_mib": _mib(reserved_after),
                    "peak_reserved_mib": _mib(peak_reserved),
                    "new_reserved_peak_mib": _mib(max(0, peak_reserved - previous_peak_reserved)),
                }
            )

    def end_step(self) -> None:
        """结束当前 step，写入一条包含原始事件与聚合结果的 JSONL。"""
        if not self._active:
            return

        self._synchronize()
        final_allocated = self._current_allocated_bytes()
        final_reserved = self._current_reserved_bytes()
        report = {
            "schema_version": 1,
            "device": str(self.device),
            "cuda_memory_supported": self._cuda_memory_supported,
            "synchronize": bool(self.config.synchronize),
            **self._step_context,
            "step_elapsed_ms": (time.perf_counter() - self._step_started_at) * 1000.0,
            "step_peak_allocated_mib": _mib(self._peak_allocated_bytes()),
            "step_peak_reserved_mib": _mib(self._peak_reserved_bytes()),
            "final_allocated_mib": _mib(final_allocated),
            "final_reserved_mib": _mib(final_reserved),
            "events": self._events,
            "aggregates": aggregate_events(self._events),
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._output_initialized else "w"
        with self.output_path.open(mode, encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        self._output_initialized = True

        hotspot = max(
            self._events,
            key=lambda event: float(event["new_peak_mib"]),
            default=None,
        )
        if hotspot is None:
            hotspot_name = "none"
        else:
            hotspot_name = str(hotspot["name"])
        logger.info(
            "runtime_debug step=%s peak_allocated=%.1fMiB peak_reserved=%.1fMiB hotspot=%s output=%s",
            self._step_context.get("global_debug_step"),
            report["step_peak_allocated_mib"],
            report["step_peak_reserved_mib"],
            hotspot_name,
            self.output_path,
        )
        self._recorded_steps += 1
        self._active = False
        self._events = []
        self._step_context = {}

    @property
    def _cuda_memory_supported(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    def _synchronize(self) -> None:
        if self.config.synchronize and self._cuda_memory_supported:
            torch.cuda.synchronize(self.device)

    def _current_allocated_bytes(self) -> int:
        if not self._cuda_memory_supported:
            return 0
        return int(torch.cuda.memory_allocated(self.device))

    def _current_reserved_bytes(self) -> int:
        if not self._cuda_memory_supported:
            return 0
        return int(torch.cuda.memory_reserved(self.device))

    def _peak_allocated_bytes(self) -> int:
        if not self._cuda_memory_supported:
            return 0
        return int(torch.cuda.max_memory_allocated(self.device))

    def _peak_reserved_bytes(self) -> int:
        if not self._cuda_memory_supported:
            return 0
        return int(torch.cuda.max_memory_reserved(self.device))


def aggregate_events(events: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """按阶段名称聚合重复调用，保留耗时总和和显存最大值。"""
    grouped: dict[str, dict[str, object]] = {}
    for event in events:
        name = str(event["name"])
        aggregate = grouped.setdefault(
            name,
            {
                "calls": 0,
                "elapsed_ms": 0.0,
                "max_peak_allocated_mib": 0.0,
                "max_new_peak_mib": 0.0,
                "max_peak_reserved_mib": 0.0,
            },
        )
        aggregate["calls"] = int(aggregate["calls"]) + 1
        aggregate["elapsed_ms"] = float(aggregate["elapsed_ms"]) + float(event["elapsed_ms"])
        aggregate["max_peak_allocated_mib"] = max(
            float(aggregate["max_peak_allocated_mib"]),
            float(event["peak_allocated_mib"]),
        )
        aggregate["max_new_peak_mib"] = max(
            float(aggregate["max_new_peak_mib"]),
            float(event["new_peak_mib"]),
        )
        aggregate["max_peak_reserved_mib"] = max(
            float(aggregate["max_peak_reserved_mib"]),
            float(event["peak_reserved_mib"]),
        )
    return grouped


def summarize_batch(batch: Mapping[str, object]) -> dict[str, object]:
    """提取与显存缩放相关的 batch 形状，方便直接按 history 分析结果。"""
    shapes: dict[str, list[int]] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            shapes[str(key)] = [int(size) for size in value.shape]

    batch_size = _shape_value(batch, "label_index", 0)
    history_length = _shape_value(batch, "history_skill_ids", 1)
    scene_length = _shape_value(batch, "scene_vectors", 1)
    candidate_count = _shape_value(batch, "candidate_skill_ids", 1)
    sequence_length = None
    if scene_length is not None and history_length is not None and candidate_count is not None:
        sequence_length = scene_length + history_length + candidate_count + 1
    return {
        "batch_size": batch_size,
        "history_length": history_length,
        "scene_length": scene_length,
        "candidate_count": candidate_count,
        "sequence_length": sequence_length,
        "shapes": shapes,
    }


def _shape_value(batch: Mapping[str, object], key: str, axis: int) -> int | None:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim <= axis:
        return None
    return int(value.shape[axis])


def _mib(value: int) -> float:
    return float(value) / (1024.0 * 1024.0)

"""按累计指标自动生成单轮训练与验证的进度表。"""

from __future__ import annotations

import os
import shutil
import sys
import time
from collections import deque
from collections.abc import Mapping

import torch

from common.training.metrics import MetricAccumulator

_LABELS = {
    "cross_entropy_loss": "ce",
    "value_preference_loss": "value_aux",
    "top1_accuracy": "top1",
    "top3_accuracy": "top3",
    "val_ppg_normalized": "val_norm",
    "none_ppg_normalized": "none_norm",
    "top1_val_ppg_average": "score",
}


def _columns(values, widths) -> str:
    return " ".join(f"{value:>{width}}" for width, value in zip(widths, values))


def format_metrics(metrics: Mapping[str, float], *, epoch: str) -> str:
    """验证附加指标沿用批次进度的列名、列宽和有效数字。"""
    labels = ("Val", "GPU_mem", *(_LABELS.get(name, name) for name in metrics))
    widths = tuple(max(8, len(label)) for label in labels)
    widths = (max(widths[0], len(epoch)), *widths[1:])
    values = (epoch, "-", *(f"{value:.4g}" for value in metrics.values()))
    return _columns(labels, widths) + "\n" + _columns(values, widths)


class EpochProgress:
    """交互终端定时读取指标；重定向时只输出表头和最终结果。"""

    def __init__(
        self,
        totals: MetricAccumulator,
        device,
        *,
        epoch: int,
        max_epochs: int | None,
        phase: str,
        total: int,
    ):
        self.totals, self.device, self.total = totals, device, total
        self.epoch = f"{epoch}/{max_epochs}" if max_epochs is not None else str(epoch)
        self.labels = (phase, "GPU_mem", *(_LABELS.get(name, name) for name in totals.names))
        self.widths = tuple(max(8, len(label)) for label in self.labels)
        self.widths = (max(self.widths[0], len(self.epoch)), *self.widths[1:])
        self.stream = sys.stderr
        self.interactive = self.stream.isatty()
        self.n = 0

    def _terminal_width(self) -> int:
        try:
            columns = os.get_terminal_size(self.stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            columns = shutil.get_terminal_size(fallback=(120, 20)).columns
        return (columns or 120) - 1

    def __enter__(self):
        self.started = self.last_display = time.monotonic()
        self.rate_samples = deque([(self.started, 0)])
        header = _columns(self.labels, self.widths)
        self.display_width = self._terminal_width() if self.interactive else 0
        suffix = self._progress_suffix(0.0, len(header))
        column_width = max(0, self.display_width - len(suffix)) if self.interactive else len(header)
        self.stream.write(header[:column_width] + "\n")
        self.stream.flush()
        return self

    def update(self) -> None:
        self.n += 1
        now = time.monotonic()
        self.rate_samples.append((now, self.n))
        # 保留跨越窗口起点的锚点，非交互输出也持续更新近期速度。
        while len(self.rate_samples) > 1 and self.rate_samples[1][0] <= now - 5.0:
            self.rate_samples.popleft()
        if not self.interactive:
            return
        if now - self.last_display >= 0.2:
            self._display()

    def _progress_suffix(self, elapsed: float, column_width: int) -> str:
        now = self.started + elapsed
        window_start = max(self.started, now - 5.0)
        while len(self.rate_samples) > 1 and self.rate_samples[1][0] <= window_start:
            self.rate_samples.popleft()
        first_time, first_count = self.rate_samples[0]
        if len(self.rate_samples) > 1 and first_time < window_start:
            next_time, next_count = self.rate_samples[1]
            # 按跨界批次的时间占比折算计数，避免长停顿继续污染整个恢复窗口。
            first_count += (next_count - first_count) * (window_start - first_time) / (next_time - first_time)
        window_elapsed = now - window_start
        rate = (self.n - first_count) / window_elapsed if window_elapsed else 0.0
        speed = f"{1 / rate:.1f}s/it" if 0 < rate < 1 else f"{rate:.1f}it/s"
        fraction = min(1.0, self.n / self.total) if self.total else 0.0
        remaining = (self.total - self.n) / rate if rate and self.n < self.total else 0.0
        prefix = f": {int(fraction * 100):3d}% "
        tail = f" {self.n}/{self.total} {speed} {elapsed:.1f}s<{remaining:.1f}s"
        bar_width = (
            min(12, max(1, self.display_width - column_width - len(prefix) - len(tail)))
            if self.interactive else 12
        )
        filled = int(fraction * bar_width)
        bar = "━" * filled + "─" * (bar_width - filled)
        return prefix + bar + tail

    def _display(self, *, final: bool = False, interrupted: bool = False) -> None:
        # 异常退出不再次同步设备，以免覆盖原始异常。
        metrics = {} if interrupted else self.totals.mean()
        memory = (
            f"{torch.cuda.memory_reserved(self.device) / 2**30:.3g}G"
            if self.device.type == "cuda" and not interrupted else "-"
        )
        columns = _columns((
            self.epoch, memory,
            *(f"{metrics[name]:.4g}" if name in metrics else "-" for name in self.totals.names),
        ), self.widths)
        now = time.monotonic()
        suffix = self._progress_suffix(max(0.0, now - self.started), len(columns))
        if self.interactive:
            columns = columns[:max(0, self.display_width - len(suffix))]
        line = columns + suffix
        if self.interactive:
            line = "\r" + line.ljust(self.display_width)
        self.stream.write(line + ("\n" if final else ""))
        self.stream.flush()
        self.last_display = now

    def __exit__(self, exc_type, exc_value, traceback):
        self._display(final=True, interrupted=exc_type is not None)


__all__ = ["EpochProgress", "format_metrics"]

"""按累计指标自动生成单轮训练与验证的进度表。"""

from __future__ import annotations

import os
import shutil
import sys
import time
from collections.abc import Mapping

import torch

from common.training.metrics import MetricAccumulator

_LABELS = {
    "cross_entropy_loss": "ce",
    "value_preference_loss": "value_aux",
    "top1_accuracy": "top1",
    "top3_accuracy": "top3",
}


def format_metrics(metrics: Mapping[str, float]) -> str:
    """按指标原有顺序汇报全部键值，无需单独维护格式串。"""
    return " ".join(f"{name}={value:.4g}" for name, value in metrics.items())


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

    def _columns(self, values) -> str:
        return " ".join(f"{value:>{width}}" for width, value in zip(self.widths, values))

    def _terminal_width(self) -> int:
        try:
            columns = os.get_terminal_size(self.stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            columns = shutil.get_terminal_size(fallback=(120, 20)).columns
        return (columns or 120) - 1

    def __enter__(self):
        self.started = self.last_display = time.monotonic()
        header = self._columns(self.labels)
        self.display_width = self._terminal_width() if self.interactive else 0
        self.column_width = max(0, self.display_width - 50) if self.interactive else len(header)
        self.stream.write(header[:self.column_width] + "\n")
        self.stream.flush()
        return self

    def update(self) -> None:
        self.n += 1
        if not self.interactive:
            return
        now = time.monotonic()
        if now - self.last_display >= 0.2:
            self._display()

    def _display(self, *, final: bool = False, interrupted: bool = False) -> None:
        # 异常退出不再次同步设备，以免覆盖原始异常。
        metrics = {} if interrupted else self.totals.mean()
        memory = (
            f"{torch.cuda.memory_reserved(self.device) / 2**30:.3g}G"
            if self.device.type == "cuda" and not interrupted else "-"
        )
        columns = self._columns((
            self.epoch, memory,
            *(f"{metrics[name]:.4g}" if name in metrics else "-" for name in self.totals.names),
        ))[:self.column_width]
        now = time.monotonic()
        elapsed = max(0.0, now - self.started)
        rate = self.n / elapsed if elapsed else 0.0
        speed = f"{1 / rate:.1f}s/it" if 0 < rate < 1 else f"{rate:.1f}it/s"
        fraction = min(1.0, self.n / self.total) if self.total else 0.0
        remaining = (self.total - self.n) / rate if rate and self.n < self.total else 0.0
        bar_width = min(12, max(1, self.display_width - len(columns) - 40)) if self.interactive else 12
        filled = int(fraction * bar_width)
        bar = "━" * filled + "─" * (bar_width - filled)
        line = f"{columns}: {int(fraction * 100):3d}% {bar} {self.n}/{self.total} {speed} {elapsed:.1f}s<{remaining:.1f}s"
        if self.interactive:
            line = "\r" + line[:self.display_width].ljust(self.display_width)
        self.stream.write(line + ("\n" if final else ""))
        self.stream.flush()
        self.last_display = now

    def __exit__(self, exc_type, exc_value, traceback):
        self._display(final=True, interrupted=exc_type is not None)


__all__ = ["EpochProgress", "format_metrics"]

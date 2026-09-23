"""跨 BC 与 GRPO 训练流程共用的 TensorBoard 写入工具。"""

from __future__ import annotations

import atexit
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_ACTIVE_WRITER_CLOSE_CALLBACKS: dict[int, Callable[[], None]] = {}


@dataclass(frozen=True)
class TensorBoardConfig:
    """控制训练曲线写入频率与 SummaryWriter 刷新间隔。"""

    enabled: bool = False
    log_every_steps: int = 50
    flush_secs: int = 30

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("training.tensorboard.enabled must be a boolean")
        if (
            isinstance(self.log_every_steps, bool)
            or not isinstance(self.log_every_steps, int)
            or self.log_every_steps < 1
        ):
            raise ValueError("training.tensorboard.log_every_steps must be >= 1")
        if (
            isinstance(self.flush_secs, bool)
            or not isinstance(self.flush_secs, int)
            or self.flush_secs < 1
        ):
            raise ValueError("training.tensorboard.flush_secs must be >= 1")

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object] | None = None,
    ) -> TensorBoardConfig:
        """从 YAML mapping 构造配置并校验类型与范围。"""
        values = {} if raw is None else raw
        if not isinstance(values, Mapping):
            raise TypeError("training.tensorboard must be a mapping")

        enabled = values.get("enabled", cls.enabled)
        if not isinstance(enabled, bool):
            raise TypeError("training.tensorboard.enabled must be a boolean")
        log_every_steps = values.get("log_every_steps", cls.log_every_steps)
        if isinstance(log_every_steps, bool) or not isinstance(log_every_steps, int):
            raise TypeError("training.tensorboard.log_every_steps must be an integer")
        flush_secs = values.get("flush_secs", cls.flush_secs)
        if isinstance(flush_secs, bool) or not isinstance(flush_secs, int):
            raise TypeError("training.tensorboard.flush_secs must be an integer")
        return cls(
            enabled=enabled,
            log_every_steps=log_every_steps,
            flush_secs=flush_secs,
        )


def _resolve_summary_writer():
    """延迟导入可选依赖，使未启用 TensorBoard 的路径不受影响。"""
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter


def create_tensorboard_writer(
    config: TensorBoardConfig,
    output_dir: Path,
    *,
    run_name: str,
):
    """为一次训练创建独立事件目录；禁用时不导入 TensorBoard。"""
    if not config.enabled:
        return None

    try:
        summary_writer = _resolve_summary_writer()
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard is enabled but its dependency is missing; "
            "install it with `.venv/Scripts/python -m pip install -r requirements.txt`."
        ) from exc

    root = Path(output_dir) / "tensorboard"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    base_name = f"{run_name}-{timestamp}"
    log_dir = root / base_name
    suffix = 1
    while log_dir.exists():
        log_dir = root / f"{base_name}-{suffix:03d}"
        suffix += 1
    writer = summary_writer(log_dir=str(log_dir), flush_secs=config.flush_secs)

    def close_at_exit() -> None:
        _ACTIVE_WRITER_CLOSE_CALLBACKS.pop(id(writer), None)
        writer.close()

    _ACTIVE_WRITER_CLOSE_CALLBACKS[id(writer)] = close_at_exit
    atexit.register(close_at_exit)
    return writer


def write_scalar_metrics(
    writer,
    metrics: Mapping[str, object],
    *,
    prefix: str,
    global_step: int,
) -> None:
    """写入有限数值型指标，忽略非数值和 NaN/Inf。"""
    if writer is None:
        return
    for name, value in metrics.items():
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "item"):
                value = value.item()
            scalar = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(scalar):
            writer.add_scalar(f"{prefix}/{name}", scalar, global_step)


def close_tensorboard_writer(writer) -> None:
    """关闭并刷新 SummaryWriter。"""
    if writer is not None:
        callback = _ACTIVE_WRITER_CLOSE_CALLBACKS.pop(id(writer), None)
        if callback is not None:
            atexit.unregister(callback)
        writer.close()


__all__ = [
    "TensorBoardConfig",
    "close_tensorboard_writer",
    "create_tensorboard_writer",
    "write_scalar_metrics",
]

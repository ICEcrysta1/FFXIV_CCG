"""跨 BC 与 GRPO 训练流程共用的 TensorBoard 写入工具。"""

from __future__ import annotations

import atexit
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .tensorboard_labels import tensorboard_tag

_ACTIVE_WRITER_CLOSE_CALLBACKS: dict[int, Callable[[], None]] = {}


@dataclass(frozen=True)
class TensorBoardConfig:
    """控制训练标量、模型诊断与 SummaryWriter 刷新间隔。"""

    enabled: bool = False
    log_every_steps: int = 50
    flush_secs: int = 30
    graph: bool = True
    histogram_every_steps: int = 500
    embedding_every_epochs: int = 1
    max_histogram_values: int = 4096
    pr_curves: bool = True
    pr_curve_max_samples: int = 4096

    def __post_init__(self) -> None:
        for name in ("enabled", "graph", "pr_curves"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"training.tensorboard.{name} must be a boolean")
        for name in (
            "log_every_steps", "flush_secs", "histogram_every_steps",
            "embedding_every_epochs", "max_histogram_values", "pr_curve_max_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"training.tensorboard.{name} must be an integer")
            if value < 1:
                raise ValueError(f"training.tensorboard.{name} must be >= 1")

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object] | None = None,
    ) -> TensorBoardConfig:
        """从 YAML mapping 构造配置并校验类型与范围。"""
        values = {} if raw is None else raw
        if not isinstance(values, Mapping):
            raise TypeError("training.tensorboard must be a mapping")

        from dataclasses import fields

        return cls(**{
            field.name: values.get(field.name, field.default)
            for field in fields(cls)
        })


def _resolve_summary_writer():
    """延迟导入可选依赖，使未启用 TensorBoard 的路径不受影响。"""
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter


def create_tensorboard_writer(
    config: TensorBoardConfig,
    output_dir: Path,
    *,
    run_name: str,
    model_variant: str | None = None,
):
    """为一次训练创建独立事件目录；禁用时不导入 TensorBoard。"""
    if not config.enabled:
        return None

    root = resolve_tensorboard_root(output_dir, model_variant)

    try:
        summary_writer = _resolve_summary_writer()
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard is enabled but its dependency is missing; "
            "install it with `.venv/bin/python3 -m pip install -r requirements.txt`."
        ) from exc

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


def resolve_tensorboard_root(
    output_dir: Path,
    model_variant: str | None,
) -> Path:
    """从模型输出目录和变体名解析 BC/GRPO 共用的事件根目录。"""
    if not isinstance(model_variant, str) or not model_variant.strip():
        raise ValueError("TensorBoard requires a configured model_variant")
    variant = model_variant.strip()
    variant_path = Path(variant)
    if (
        variant in {".", ".."}
        or variant_path.is_absolute()
        or len(variant_path.parts) != 1
    ):
        raise ValueError(
            "TensorBoard model_variant must be a single directory name, "
            f"got {variant!r}"
        )
    return Path(output_dir).resolve().parent / f"{variant}_tensorboard"


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
            writer.add_scalar(tensorboard_tag(f"{prefix}/{name}"), scalar, global_step)


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
    "resolve_tensorboard_root",
    "write_scalar_metrics",
]

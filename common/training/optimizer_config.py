"""BC 与 GRPO 共用的优化器配置校验，各算法独立管理超参数。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace


@dataclass(frozen=True)
class MuonConfig:
    """线性矩阵的 Muon 更新参数。"""

    learning_rate: float | None = None
    weight_decay: float | None = None
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    adjust_lr_fn: str = "match_rms_adamw"
    eps: float = 1e-7
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315)

    def __post_init__(self) -> None:
        label = "optimizer.muon"
        _validate_optional_rates(self, label)
        _validate_momentum(self.momentum, self.nesterov, label)
        if isinstance(self.ns_steps, bool) or not isinstance(self.ns_steps, int):
            raise TypeError(f"{label}.ns_steps must be an integer within [1, 99]")
        if not 1 <= self.ns_steps <= 99:
            raise ValueError(f"{label}.ns_steps must be an integer within [1, 99]")
        if not isinstance(self.adjust_lr_fn, str) or self.adjust_lr_fn not in {
            "original", "match_rms_adamw",
        }:
            raise ValueError(f"{label}.adjust_lr_fn must be original or match_rms_adamw")
        if _finite_number(self.eps, f"{label}.eps") <= 0.0:
            raise ValueError(f"{label}.eps must be > 0")
        coefficients = _number_tuple(self.ns_coefficients, f"{label}.ns_coefficients", 3)
        object.__setattr__(self, "ns_coefficients", coefficients)


@dataclass(frozen=True)
class AdamWConfig:
    """归一化、偏置与缩放参数的 AdamW 更新参数。"""

    learning_rate: float | None = None
    weight_decay: float | None = None
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False

    def __post_init__(self) -> None:
        label = "optimizer.adamw"
        _validate_optional_rates(self, label)
        betas = _number_tuple(self.betas, f"{label}.betas", 2)
        if any(not 0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"{label}.betas must be within [0, 1)")
        object.__setattr__(self, "betas", betas)
        if _finite_number(self.eps, f"{label}.eps") <= 0.0:
            raise ValueError(f"{label}.eps must be > 0")
        if not isinstance(self.amsgrad, bool):
            raise TypeError(f"{label}.amsgrad must be boolean")


@dataclass(frozen=True)
class SinkhornConfig:
    """embedding 和输出头的无衰减行列平衡更新参数。"""

    learning_rate: float | None = None
    momentum: float = 0.95
    nesterov: bool = True
    iters: int = 11
    threshold: float = 1e-3
    eps: float = 1e-20
    lr_scale: float = 0.2

    def __post_init__(self) -> None:
        label = "optimizer.sinkhorn"
        if self.learning_rate is not None and _finite_number(self.learning_rate, f"{label}.learning_rate") <= 0.0:
            raise ValueError(f"{label}.learning_rate must be > 0")
        _validate_momentum(self.momentum, self.nesterov, label)
        if isinstance(self.iters, bool) or not isinstance(self.iters, int):
            raise TypeError(f"{label}.iters must be a positive odd integer")
        if self.iters < 1 or self.iters % 2 != 1:
            raise ValueError(f"{label}.iters must be a positive odd integer")
        for name in ("threshold", "eps", "lr_scale"):
            value = _finite_number(getattr(self, name), f"{label}.{name}")
            if value < 0.0 or (name != "threshold" and value == 0.0):
                bound = ">= 0" if name == "threshold" else "> 0"
                raise ValueError(f"{label}.{name} must be {bound}")


@dataclass(frozen=True)
class OptimizerConfig:
    """选择算法组合，各算法设置在独立分组中。"""

    name: str = "adamw"
    muon: MuonConfig = field(default_factory=MuonConfig)
    adamw: AdamWConfig = field(default_factory=AdamWConfig)
    sinkhorn: SinkhornConfig = field(default_factory=SinkhornConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name not in {"adamw", "muon"}:
            raise ValueError("optimizer.name must be adamw or muon")
        for name, config_type in (("muon", MuonConfig), ("adamw", AdamWConfig), ("sinkhorn", SinkhornConfig)):
            if not isinstance(getattr(self, name), config_type):
                raise TypeError(f"optimizer.{name} must be {config_type.__name__}")

    @classmethod
    def from_mapping(cls, raw: object) -> OptimizerConfig:
        """严格读取算法配置，避免拼错字段后静默使用默认值。"""
        values = _mapping_fields(cls, raw, "optimizer")
        for name, config_type in (("muon", MuonConfig), ("adamw", AdamWConfig), ("sinkhorn", SinkhornConfig)):
            if name in values:
                values[name] = config_type(**_mapping_fields(config_type, values[name], f"optimizer.{name}"))
        return cls(**values)


def parse_optimizer_settings(
    policy_config: Mapping[str, object],
    *,
    stage: str,
    legacy: Mapping[str, object],
    default_learning_rate: float,
    default_weight_decay: float,
    default_warmup_steps: int,
) -> dict[str, object]:
    """读取阶段专属配置，兼容旧单文件并拒绝同一参数的两份配置权威。"""
    if stage not in {"bc", "grpo"}:
        raise ValueError("optimizer stage must be bc or grpo")
    optimizers = policy_config.get("optimizers", {})
    if not isinstance(optimizers, Mapping):
        raise TypeError("optimizers must be a mapping")
    unknown_stages = set(optimizers) - {"bc", "grpo"}
    if unknown_stages:
        raise ValueError(
            "optimizers contains unknown stages: "
            + ", ".join(sorted(str(key) for key in unknown_stages))
        )
    shared_fields = {"learning_rate", "weight_decay", "warmup_steps"}
    if stage in optimizers:
        values = optimizers[stage]
        if not isinstance(values, Mapping):
            raise ValueError(f"optimizers.{stage} must be a mapping")
        duplicate = set(legacy) & (shared_fields | {"optimizer"})
        if duplicate:
            legacy_section = "training" if stage == "bc" else "grpo"
            raise ValueError(
                f"optimizers.{stage} conflicts with {legacy_section} optimizer settings: "
                + ", ".join(sorted(duplicate))
            )
        algorithm_values = {key: value for key, value in values.items() if key not in shared_fields}
        label = f"optimizers.{stage}"
    else:
        values = legacy
        algorithm_values = values.get("optimizer", {})
        label = "training" if stage == "bc" else "grpo"
    optimizer = OptimizerConfig.from_mapping(algorithm_values)
    primary = optimizer.muon if optimizer.name == "muon" else optimizer.adamw
    for name in ("learning_rate", "weight_decay"):
        if name in values and getattr(primary, name) is not None:
            raise ValueError(f"{label}.{name} conflicts with optimizer.{optimizer.name}.{name}")
    learning_rate = _finite_number(
        primary.learning_rate if primary.learning_rate is not None
        else values.get("learning_rate", default_learning_rate), f"{label}.learning_rate",
    )
    weight_decay = _finite_number(
        primary.weight_decay if primary.weight_decay is not None
        else values.get("weight_decay", default_weight_decay), f"{label}.weight_decay",
    )
    warmup_steps = values.get("warmup_steps", default_warmup_steps)
    if learning_rate <= 0.0:
        raise ValueError(f"{label}.learning_rate must be > 0")
    if weight_decay < 0.0:
        raise ValueError(f"{label}.weight_decay must be >= 0")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError(f"{label}.warmup_steps must be an integer >= 0")
    if optimizer.name == "muon" and primary.learning_rate is None and any(
        getattr(optimizer, name).learning_rate is not None
        for name in ("adamw", "sinkhorn")
    ):
        # 固定辅助组的相对学习率基准，保留后续阶段学习率覆盖时的同比缩放。
        optimizer = replace(optimizer, **{optimizer.name: replace(primary, learning_rate=learning_rate)})
    return {
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
    }


def _mapping_fields(config_type, raw: object, label: str) -> dict:
    """只接受配置类声明的字段。"""
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label} must be a mapping")
    unknown = set(raw) - {item.name for item in fields(config_type)}
    if unknown:
        raise ValueError(f"{label} contains unknown fields: " + ", ".join(sorted(str(key) for key in unknown)))
    return dict(raw)


def _validate_optional_rates(config: MuonConfig | AdamWConfig, label: str) -> None:
    if config.learning_rate is not None and _finite_number(config.learning_rate, f"{label}.learning_rate") <= 0.0:
        raise ValueError(f"{label}.learning_rate must be > 0")
    if config.weight_decay is not None and _finite_number(config.weight_decay, f"{label}.weight_decay") < 0.0:
        raise ValueError(f"{label}.weight_decay must be >= 0")


def _validate_momentum(momentum: object, nesterov: object, label: str) -> None:
    if not 0.0 <= _finite_number(momentum, f"{label}.momentum") < 1.0:
        raise ValueError(f"{label}.momentum must be within [0, 1)")
    if not isinstance(nesterov, bool):
        raise TypeError(f"{label}.nesterov must be boolean")


def _number_tuple(raw: object, label: str, length: int) -> tuple:
    if not isinstance(raw, (tuple, list)):
        raise TypeError(f"{label} must be a sequence of {length} numbers")
    if len(raw) != length:
        raise ValueError(f"{label} must contain {length} numbers")
    return tuple(_finite_number(value, label) for value in raw)


def _finite_number(value: object, label: str) -> float:
    """严格校验数值，拒绝布尔值和非有限数值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric and finite")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be numeric and finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be numeric and finite")
    return number

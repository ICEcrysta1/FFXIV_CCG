"""统一张量精度配置、别名解析与 PyTorch dtype 解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.configuration.yaml import load_yaml_mapping
from .dependencies import import_torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRECISION_CONFIG_PATH = PROJECT_ROOT / "config" / "precision.yaml"


@dataclass(frozen=True)
class PrecisionConfig:
    """统一 tensor 输出精度配置。"""

    int_dtype: str
    index_dtype: str
    float_dtype: str

    def resolve_int_dtype(self):
        return _resolve_torch_dtype(
            self.int_dtype,
            field_name="int_dtype",
            allowed_names=_INT_DTYPE_NAMES,
            category_name="integer",
        )

    def resolve_float_dtype(self):
        return _resolve_torch_dtype(
            self.float_dtype,
            field_name="float_dtype",
            allowed_names=_FLOAT_DTYPE_NAMES,
            category_name="floating-point",
        )

    def resolve_index_dtype(self):
        return _resolve_torch_dtype(
            self.index_dtype,
            field_name="index_dtype",
            allowed_names=_INDEX_DTYPE_NAMES,
            category_name="index",
        )


_INT_DTYPE_ALIASES = {
    "int32": "int32",
    "int64": "int64",
}

# 交叉熵目标和 gather 索引需要 int64；作为独立配置显式记录该算子约束。
_INDEX_DTYPE_ALIASES = {"int64": "int64"}

_FLOAT_DTYPE_ALIASES = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "fp16": "float16",
}

_INT_DTYPE_NAMES = frozenset(_INT_DTYPE_ALIASES.values())
_INDEX_DTYPE_NAMES = frozenset(_INDEX_DTYPE_ALIASES.values())
_FLOAT_DTYPE_NAMES = frozenset(_FLOAT_DTYPE_ALIASES.values())


def load_precision_config(path: Path | None = None) -> PrecisionConfig:
    """加载统一精度配置。"""
    config_path = path or PRECISION_CONFIG_PATH
    raw = load_yaml_mapping(config_path)
    payload = raw.get("precision", {})
    resolved_int = _resolve_precision_alias(
        payload.get("int_dtype", "int32"),
        field_name="int_dtype",
        aliases=_INT_DTYPE_ALIASES,
    )
    resolved_index = _resolve_precision_alias(
        payload["index_dtype"],
        field_name="index_dtype",
        aliases=_INDEX_DTYPE_ALIASES,
    )
    resolved_float = _resolve_precision_alias(
        payload.get("float_dtype", "float32"),
        field_name="float_dtype",
        aliases=_FLOAT_DTYPE_ALIASES,
    )
    return PrecisionConfig(
        int_dtype=resolved_int,
        index_dtype=resolved_index,
        float_dtype=resolved_float,
    )


def _resolve_precision_alias(raw_value: object, *, field_name: str, aliases: dict[str, str]) -> str:
    raw_text = str(raw_value).lower()
    resolved = aliases.get(raw_text)
    if resolved is None:
        supported = ", ".join(sorted(aliases))
        raise ValueError(
            f"precision.yaml: {field_name}={raw_value!r} is invalid; supported aliases: {supported}"
        )
    return resolved


def _resolve_torch_dtype(
    dtype_name: str,
    *,
    field_name: str,
    allowed_names: frozenset[str],
    category_name: str,
):
    torch = import_torch()

    try:
        dtype = getattr(torch, dtype_name)
    except AttributeError:
        raise ValueError(
            f"unsupported torch dtype for {field_name}: {dtype_name!r}; "
            f"allowed canonical names: {', '.join(sorted(allowed_names))}"
        ) from None

    if dtype_name not in allowed_names:
        raise ValueError(
            f"{field_name} must use {category_name} torch dtype, got {dtype_name!r}; "
            f"allowed canonical names: {', '.join(sorted(allowed_names))}"
        )
    return dtype

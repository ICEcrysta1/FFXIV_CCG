"""跨模块复用的嵌套数值映射辅助。"""

from __future__ import annotations

from collections.abc import Collection, Mapping


def flatten_numeric_mapping(
    value: object,
    *,
    ignored_keys: Collection[str] = (),
) -> dict[str, float]:
    """展开嵌套映射中的布尔和数值字段，并跳过调用方指定的字段。"""
    output: dict[str, float] = {}
    _flatten_numeric_mapping(
        value,
        prefix="",
        output=output,
        ignored_keys=frozenset(ignored_keys),
    )
    return output


def _flatten_numeric_mapping(
    value: object,
    *,
    prefix: str,
    output: dict[str, float],
    ignored_keys: frozenset[str],
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in ignored_keys:
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric_mapping(
                child,
                prefix=child_prefix,
                output=output,
                ignored_keys=ignored_keys,
            )
        return
    if isinstance(value, bool):
        output[prefix] = 1.0 if value else 0.0
    elif isinstance(value, (int, float)):
        output[prefix] = float(value)

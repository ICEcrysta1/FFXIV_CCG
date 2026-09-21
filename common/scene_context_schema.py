"""scene_context 统一 schema 定义。

字段模板与常量由 `config/schema.yaml` 提供（C# 与 Python 单一事实来源）。
"""

from __future__ import annotations

from .contracts import (
    FORCED_MOVEMENT_CONTEXT_KEY,
    RAID_BUFF_WINDOW_CONTEXT_KEY,
    TARGETABLE_WINDOW_CONTEXT_KEY,
    TARGET_COUNT_WINDOW_CONTEXT_KEY,
)
from .schema_config import load_schema_config

_schema = load_schema_config()
_scene = _schema["scene_context"]
_window_extra_fields = _scene["window_extra_fields"]

SCENE_CONTEXT_KEYS = (
    TARGETABLE_WINDOW_CONTEXT_KEY,
    FORCED_MOVEMENT_CONTEXT_KEY,
    RAID_BUFF_WINDOW_CONTEXT_KEY,
    TARGET_COUNT_WINDOW_CONTEXT_KEY,
)

TARGETABLE_SEGMENT_KINDS = tuple(_scene["targetable_segment_kinds"])
RAID_BUFF_WINDOW_SOURCE_KEYS = tuple(_scene["raid_buff_window_source_keys"])


def _window_feature_keys(
    extra_fields: tuple[str, ...],
    *,
    source_keys: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """按窗口的时间字段 + 额外字段模板构造 feature keys。

    YAML 模板中的 `{key}` 占位由 source_keys 逐个展开（团辅窗口）。
    """
    expanded: list[str] = []
    for field in extra_fields:
        if "{key}" in field:
            for source_key in (source_keys or ()):
                expanded.append(field.format(key=source_key))
        else:
            expanded.append(field)
    return (
        *tuple(_scene["window_time_fields"]),
        *expanded,
    )


def raid_buff_window_feature_keys(source_keys: tuple[str, ...]) -> tuple[str, ...]:
    """按配置中的团辅标记技能构造 scene token feature keys。"""
    if not source_keys:
        raise ValueError("raid buff window source keys must not be empty")
    return _window_feature_keys(
        tuple(_window_extra_fields["raid_buff"]),
        source_keys=source_keys,
    )


TARGETABLE_WINDOW_FEATURE_KEYS = _window_feature_keys(
    (
        *tuple(_window_extra_fields["targetable"]),
        *(f"segment_kind.{kind}" for kind in TARGETABLE_SEGMENT_KINDS),
    )
)

FORCED_MOVEMENT_FEATURE_KEYS = _window_feature_keys(
    tuple(_window_extra_fields["forced_movement"])
)

RAID_BUFF_WINDOW_FEATURE_KEYS = raid_buff_window_feature_keys(RAID_BUFF_WINDOW_SOURCE_KEYS)

TARGET_COUNT_WINDOW_FEATURE_KEYS = _window_feature_keys(
    tuple(_window_extra_fields["target_count"])
)


def build_window_context(
    feature_keys: tuple[str, ...],
    tokens: list[list[float]] | None = None,
) -> dict[str, object]:
    """构造统一的窗口上下文结构。"""
    return {
        "feature_keys": list(feature_keys),
        "tokens": [] if tokens is None else tokens,
    }


def build_empty_scene_context() -> dict[str, object]:
    """构造稳定空壳 scene_context。"""
    return {
        TARGETABLE_WINDOW_CONTEXT_KEY: build_window_context(TARGETABLE_WINDOW_FEATURE_KEYS),
        FORCED_MOVEMENT_CONTEXT_KEY: build_window_context(FORCED_MOVEMENT_FEATURE_KEYS),
        RAID_BUFF_WINDOW_CONTEXT_KEY: build_window_context(RAID_BUFF_WINDOW_FEATURE_KEYS),
        TARGET_COUNT_WINDOW_CONTEXT_KEY: build_window_context(TARGET_COUNT_WINDOW_FEATURE_KEYS),
    }

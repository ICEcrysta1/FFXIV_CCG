"""scene window 的索引解析与绝对时间窗口公共逻辑。"""

from __future__ import annotations

from functools import lru_cache


REQUIRED_SCENE_WINDOW_FEATURE_KEYS = (
    "start_offset_seconds",
    "end_offset_seconds",
    "duration_seconds",
)


@lru_cache(maxsize=32)
def feature_index_map(feature_keys: tuple[str, ...]) -> dict[str, int]:
    """缓存一个 scene feature key 到索引的映射。"""
    return {feature_key: index for index, feature_key in enumerate(feature_keys)}


def resolve_scene_window_indices(
    feature_keys: tuple[str, ...],
    *,
    context_key: str,
) -> tuple[int, int, int]:
    """校验并返回 scene 窗口 start/end/duration 三个字段的索引。"""
    indices = feature_index_map(feature_keys)
    missing_features = tuple(
        feature_key
        for feature_key in REQUIRED_SCENE_WINDOW_FEATURE_KEYS
        if feature_key not in indices
    )
    if missing_features:
        raise ValueError(
            f"scene window {context_key!r} is missing required feature keys: "
            f"{', '.join(missing_features)}"
        )
    return tuple(indices[feature_key] for feature_key in REQUIRED_SCENE_WINDOW_FEATURE_KEYS)

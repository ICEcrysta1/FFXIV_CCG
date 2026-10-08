"""黑魔模型分析专用的职业状态标签。"""

from __future__ import annotations

import numpy as np


def decision_state_labels(batch_index: int, *, batch, layout) -> tuple[str, float]:
    """从黑魔状态 token 的当前请求段提取 AF/UI 和实际 MP。"""
    values = batch["current_state_abs_values"][batch_index]
    null_mask = batch["current_state_null_mask"][batch_index]
    resource_slice = layout.group_slices.get("resource_state")
    player_slice = layout.group_slices.get("player_state")
    if resource_slice is None or player_slice is None:
        return "unknown", float("nan")

    groups = {group.group_key: group.feature_keys for group in layout.base_groups}
    resource_keys = groups["resource_state"]
    player_keys = groups["player_state"]
    resource_index = {key: index for index, key in enumerate(resource_keys)}
    player_index = {key: index for index, key in enumerate(player_keys)}
    resource_values = values[resource_slice]
    resource_null = null_mask[resource_slice]
    player_values = values[player_slice]
    player_null = null_mask[player_slice]

    af = _state_scalar(resource_values, resource_null, resource_index.get("request_state.astral_fire"))
    ui = _state_scalar(resource_values, resource_null, resource_index.get("request_state.umbral_ice"))
    mp = _state_scalar(player_values, player_null, player_index.get("request_state.mp"))
    if af >= 0.5:
        elemental_state = f"AF{int(round(af))}"
    elif ui >= 0.5:
        elemental_state = f"UI{int(round(ui))}"
    else:
        elemental_state = "neutral"
    if not np.isfinite(mp) or mp < 0.0:
        return elemental_state, float("nan")
    return elemental_state, mp


def _state_scalar(values, null_mask, index: int | None) -> float:
    if index is None or bool(null_mask[index].item()):
        return float("nan")
    return float(values[index].item())

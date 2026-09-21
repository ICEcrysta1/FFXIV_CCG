"""黑魔模型分析专用的职业状态标签。"""

from __future__ import annotations

import numpy as np


ANALYSIS_MP_MAX = 10000.0


def decision_state_labels(batch_index: int, *, batch, schema) -> tuple[str, float]:
    """从黑魔候选 state 的 before 视图提取 AF/UI 和实际 MP。"""
    values = batch["candidate_state_vectors"][batch_index]
    null_mask = batch["candidate_state_null_mask"][batch_index]
    legal_mask = batch["candidate_legal_mask"][batch_index]
    resource_slice = schema.state_group_slices().get("resource_state")
    player_slice = schema.state_group_slices().get("player_state")
    if resource_slice is None or player_slice is None:
        return "unknown", float("nan")

    resource_keys = schema.state_group_feature_keys["resource_state"]
    player_keys = schema.state_group_feature_keys["player_state"]
    resource_index = {key: index for index, key in enumerate(resource_keys)}
    player_index = {key: index for index, key in enumerate(player_keys)}
    legal_indices = legal_mask.nonzero(as_tuple=False).flatten().tolist()
    candidate_index = legal_indices[0] if legal_indices else 0
    resource_values = values[candidate_index, resource_slice]
    resource_null = null_mask[candidate_index, resource_slice]
    player_values = values[candidate_index, player_slice]
    player_null = null_mask[candidate_index, player_slice]

    af = _state_scalar(resource_values, resource_null, resource_index.get("before.astral_fire"))
    ui = _state_scalar(resource_values, resource_null, resource_index.get("before.umbral_ice"))
    mp = _state_scalar(player_values, player_null, player_index.get("before.mp"))
    if af >= 0.5:
        elemental_state = f"AF{int(round(af))}"
    elif ui >= 0.5:
        elemental_state = f"UI{int(round(ui))}"
    else:
        elemental_state = "neutral"
    if not np.isfinite(mp) or mp < 0.0:
        return elemental_state, float("nan")
    return elemental_state, float(np.clip(mp, 0.0, 1.0) * ANALYSIS_MP_MAX)


def _state_scalar(values, null_mask, index: int | None) -> float:
    if index is None or bool(null_mask[index].item()):
        return float("nan")
    return float(values[index].item())

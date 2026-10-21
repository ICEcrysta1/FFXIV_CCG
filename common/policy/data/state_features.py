"""状态 canonical 原料的唯一 reader，以及基础编码与冻结技能表的唯一装配。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from common.context.output import SKILL_STATE_FIELDS
from common.pytorch.dependencies import import_torch

from .schema import StateFeatureLayout


@dataclass(frozen=True)
class StateTokenTensors:
    base_values: object
    base_null_mask: object
    skill_state: object


def validate_state_feature_keys(state_context: Mapping[str, object], layout: StateFeatureLayout) -> None:
    """布局检查可独立复用；只核验历史元数据时不分配空 tensor。"""
    if not isinstance(state_context, Mapping):
        raise TypeError("state context must be a mapping")
    for group in layout.groups:
        keys = state_context.get(group.feature_keys_field)
        if not isinstance(keys, (list, tuple)) or tuple(keys) != group.feature_keys:
            raise ValueError(f"state feature keys mismatch: {group.group_key}")


def read_state_tokens(state_context: Mapping[str, object], layout: StateFeatureLayout) -> StateTokenTensors:
    """统一拒绝错序、非有限 FP32、未知请求时间及非法技能状态。"""
    torch = import_torch()
    validate_state_feature_keys(state_context, layout)
    tokens = state_context.get("tokens")
    if not isinstance(tokens, (list, tuple)):
        raise TypeError("state context tokens must be a list")
    values, nulls, skill_states = [], [], []
    for token in tokens:
        if not isinstance(token, Mapping):
            raise TypeError("state token must be a mapping")
        row, missing, skills = [], [], []
        for group in layout.groups:
            items = token.get(group.group_key)
            if not isinstance(items, (list, tuple)) or len(items) != len(group.feature_keys):
                raise ValueError(f"state group width mismatch: {group.group_key}")
            for index, item in enumerate(items):
                if group.encoding == "absolute_numeric":
                    if isinstance(item, bool) or not isinstance(item, (int, float)):
                        raise ValueError("skill state must contain finite non-negative numbers without nulls")
                    try:
                        numeric = float(item)
                    except (ValueError, OverflowError) as exc:
                        raise ValueError("skill state must contain finite non-negative numbers without nulls") from exc
                    if not math.isfinite(numeric) or numeric < 0:
                        raise ValueError("skill state must contain finite non-negative numbers without nulls")
                    field = SKILL_STATE_FIELDS[index % len(SKILL_STATE_FIELDS)]
                    if field == "available" and numeric not in (0.0, 1.0):
                        raise ValueError("skill availability must contain numeric 0/1")
                    if field == "current_charges" and not numeric.is_integer():
                        raise ValueError("skill current_charges must contain whole numbers")
                    skills.append(numeric)
                else:
                    if item is None:
                        numeric = 0.0
                    else:
                        if not isinstance(item, (bool, int, float)):
                            raise ValueError("state values must contain finite numbers or null")
                        try:
                            numeric = float(item)
                        except (ValueError, OverflowError) as exc:
                            raise ValueError("state values must contain finite numbers or null") from exc
                        if not math.isfinite(numeric):
                            raise ValueError("state values must contain finite numbers or null")
                    row.append(numeric)
                    missing.append(item is None)
        if missing[layout.request_time_index]:
            raise ValueError("request_state.time_seconds must be known and finite")
        values.append(row)
        nulls.append(missing)
        skill_states.append(skills)
    base = torch.tensor(values, dtype=torch.float32).reshape(len(tokens), layout.base_state_dim)
    if not bool(torch.isfinite(base).all()):
        raise ValueError("state values must remain finite after conversion to FP32")
    skill_state = torch.tensor(skill_states, dtype=torch.float32).reshape(len(tokens), layout.skill_state_dim)
    if not bool(torch.isfinite(skill_state).all()):
        raise ValueError("skill state must remain finite after conversion to FP32")
    return StateTokenTensors(
        base,
        torch.tensor(nulls, dtype=torch.bool).reshape(len(tokens), layout.base_state_dim),
        skill_state,
    )


def assemble_state_inputs(encoded_base_values, skill_state, history_mask, layout: StateFeatureLayout):
    """技能状态保持绝对数值，和已编码基础值拼入同一个 S-Emb。"""
    torch = import_torch()
    if encoded_base_values.dtype != torch.float32 or encoded_base_values.shape[-1] != layout.base_state_dim:
        raise ValueError("encoded base states must be FP32 with the layout base width")
    if (skill_state.dtype != torch.float32 or skill_state.device != encoded_base_values.device
            or skill_state.shape != (*encoded_base_values.shape[:-1], layout.skill_state_dim)):
        raise ValueError("skill state must be FP32 and aligned with base state rows")
    combined = torch.cat((encoded_base_values, skill_state), dim=-1)
    if history_mask is not None:
        if history_mask.dtype != torch.bool or history_mask.shape != combined.shape[:-1]:
            raise ValueError("history mask must match assembled state rows")
        combined = combined.masked_fill(~history_mask[..., None], 0.0)
    return combined

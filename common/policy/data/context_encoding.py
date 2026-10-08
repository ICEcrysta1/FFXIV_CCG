"""神经模型外的 FP32 上下文重锚与 ABS/DELTA 输入编码。"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from ..config import ModelConfig
from .normalizer import Normalizer
from .schema import StateFeatureLayout, TrainingSchema
from .context_fields import CURRENT_STATE_RAW_FIELDS, ENCODING_BANK_FIELDS, HISTORY_RAW_FIELDS, RAW_ONLY_FIELD_NAMES, tensor_dimensions
from .state_features import assemble_state_inputs


def raw_state_delta(
    values: torch.Tensor,
    nulls: torch.Tensor,
    previous_values: torch.Tensor | None = None,
    previous_nulls: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """已知前驱求差；没有前驱或前驱缺失时保留 ABS 并显式标记 reset。"""
    _validate_raw_state(values, nulls)
    if (previous_values is None) != (previous_nulls is None):
        raise ValueError("previous_values and previous_nulls must be provided together")
    known = ~nulls
    if previous_values is None:
        return values.masked_fill(nulls, 0.0), known
    _validate_raw_state(previous_values, previous_nulls)
    if previous_values.shape != values.shape or previous_values.device != values.device:
        raise ValueError("previous state must have the same shape and device")
    reset = known & previous_nulls
    delta = torch.where(reset, values, values - previous_values)
    return delta.masked_fill(nulls, 0.0), reset


def _validate_raw_state(values: torch.Tensor, nulls: torch.Tensor) -> None:
    if values.dtype != torch.float32:
        raise ValueError("raw state values must remain FP32 before context encoding")
    if nulls.dtype != torch.bool or nulls.shape != values.shape or nulls.device != values.device:
        raise ValueError("raw state null masks must be boolean with matching shape and device")


class ContextEncoder(nn.Module):
    """共享读侧编码：完整 raw bank 的窗口 gather、重锚和有符号归一化。"""

    def __init__(self, normalizer: Normalizer, schema: TrainingSchema, config: ModelConfig, *, layout: StateFeatureLayout):
        super().__init__()
        metadata = normalizer.state_encoding_metadata(schema)
        if layout.groups != schema.state_groups or layout.snapshots != schema.state_snapshots:
            raise ValueError("ContextEncoder layout must match its saved schema")
        self.layout = layout
        self.base_state_dim = layout.base_state_dim
        self.state_dim = layout.state_dim
        self.scene_dim = schema.scene_feature_dim()
        self.time_delta_scale = config.time_delta_scale
        if metadata.feature_keys != layout.base_feature_keys:
            raise ValueError("normalizer base field order must match the state layout")
        self.request_time_index = layout.request_time_index
        time_mask = [key.rsplit(".", 1)[-1] == "time_seconds" for key in metadata.feature_keys]
        divisors = [self.time_delta_scale if is_time else divisor
                    for is_time, divisor in zip(time_mask, metadata.divisors, strict=True)]
        lower = [-float("inf") if is_time else bound
                 for is_time, bound in zip(time_mask, metadata.absolute_lower_bounds, strict=True)]
        upper = [float("inf") if is_time else bound
                 for is_time, bound in zip(time_mask, metadata.absolute_upper_bounds, strict=True)]
        for name, values, dtype in (
            ("state_divisors", divisors, torch.float32),
            ("state_absolute_lower", lower, torch.float32),
            ("state_absolute_upper", upper, torch.float32),
            ("state_logarithmic", metadata.logarithmic, torch.bool),
            ("state_time_mask", time_mask, torch.bool),
        ):
            self.register_buffer(name, torch.tensor(values, dtype=dtype), persistent=False)

        windows = sorted(schema.scene_windows, key=lambda window: window.scene_type_id)
        if not windows or [window.scene_type_id for window in windows] != list(range(len(windows))):
            raise ValueError("context encoding requires contiguous scene type IDs")
        self.register_buffer("scene_start_indices", torch.tensor(
            [window.start_offset_index for window in windows], dtype=torch.long), persistent=False)
        self.register_buffer("scene_end_indices", torch.tensor(
            [window.end_offset_index for window in windows], dtype=torch.long), persistent=False)
        self.register_buffer("scene_duration_indices", torch.tensor(
            [window.duration_index for window in windows], dtype=torch.long), persistent=False)
        count_masks = [[key == "target_count" for key in window.feature_keys]
                       + [False] * (self.scene_dim - window.feature_dim) for window in windows]
        feature_masks = [[True] * window.feature_dim
                         + [False] * (self.scene_dim - window.feature_dim) for window in windows]
        self.register_buffer("scene_count_mask", torch.tensor(count_masks, dtype=torch.bool), persistent=False)
        self.register_buffer("scene_feature_mask", torch.tensor(feature_masks, dtype=torch.bool), persistent=False)
        self.target_count_max = normalizer.target_count_max

    def encode(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """只接受 raw 输入；编码完成后移除 raw 字段，防止重复编码或双路径消费。"""
        if "history_state_vectors" in batch or "current_state_vectors" in batch or "scene_vectors" in batch:
            raise ValueError("ContextEncoder requires raw states and scenes; an encoded batch cannot be encoded twice")
        current_abs = batch["current_state_abs_values"]
        current_null = batch["current_state_null_mask"]
        _validate_raw_state(current_abs, current_null)
        if current_abs.ndim != 2 or current_abs.shape[-1] != self.base_state_dim:
            raise ValueError("current raw states must have shape [batch, base_state_dim]")
        if current_abs.device != self.state_divisors.device or self.state_divisors.dtype != torch.float32:
            raise ValueError("ContextEncoder buffers must be FP32 on the raw batch device")
        history_mask = batch["history_mask"]
        if history_mask.dtype != torch.bool or history_mask.ndim != 2 or history_mask.shape[0] != current_abs.shape[0]:
            raise ValueError("history_mask must be boolean with shape [batch, history]")
        dense = self._gather_history(batch)
        dimensions = tensor_dimensions(self, batch=current_abs.shape[0], history=history_mask.shape[1],
                                       skill_feature_dim=dense["history_skill_features"].shape[-1])
        for field in CURRENT_STATE_RAW_FIELDS:
            field.validate_tensor(batch[field.name], dimensions, torch=torch)
        for field in HISTORY_RAW_FIELDS:
            field.validate_tensor(dense[field.name], dimensions, torch=torch,
                                  float_dtype=dense["history_skill_features"].dtype)
        history_abs = dense["history_state_abs_values"]
        history_null = dense["history_state_null_mask"]
        _validate_raw_state(history_abs, history_null)
        if history_abs.shape != (*history_mask.shape, self.base_state_dim):
            raise ValueError("history raw states must have shape [batch, history, base_state_dim]")
        has_history = history_mask.any(dim=1)
        first = history_mask & (history_mask.to(torch.long).cumsum(dim=1) == 1)
        anchor = current_abs[:, self.request_time_index]
        anchor_null = current_null[:, self.request_time_index]
        if history_mask.shape[1]:
            first_indices = history_mask.to(torch.long).argmax(dim=1)
            first_values = history_abs.gather(1, first_indices[:, None, None].expand(-1, 1, self.base_state_dim)).squeeze(1)
            anchor = torch.where(has_history, first_values[:, self.request_time_index], anchor)
            first_null = history_null[..., self.request_time_index].gather(1, first_indices[:, None]).squeeze(1)
            anchor_null = torch.where(has_history, first_null, anchor_null)
        # CPU 实时入口拒绝坏锚点；CUDA bank 已在 cache reader 验证已知时间和有限值，
        # 不在热路径读取设备标量，也不以失败后破坏 CUDA context 的设备断言做输入检查。
        if anchor.device.type == "cpu" and not bool((~anchor_null & torch.isfinite(anchor)).all()):
            raise ValueError("context anchor request time must be known and finite")
        history_reset = dense["history_state_delta_reset_mask"] | first.unsqueeze(-1)
        history_vectors, history_reset = self._encode_state(
            history_abs, dense["history_state_delta_values"], history_null,
            history_reset, anchor[:, None, None],
        )
        current_vectors, current_reset = self._encode_state(
            current_abs, batch["current_state_delta_values"], current_null,
            batch["current_state_delta_reset_mask"] | ~has_history[:, None], anchor[:, None],
        )
        prepared = {
            key: value for key, value in batch.items()
            if not key.startswith("history_bank_")
            and key not in RAW_ONLY_FIELD_NAMES
            and key not in {"history_lengths", "history_ends"}
        }
        prepared.update(
            history_skill_ids=dense["history_skill_ids"].masked_fill(~history_mask, 0),
            history_skill_features=dense["history_skill_features"].masked_fill(~history_mask[..., None], 0),
            history_state_vectors=assemble_state_inputs(history_vectors, dense["history_state_skill_availability"], history_mask, self.layout),
            history_state_null_mask=history_null | ~history_mask[..., None],
            history_state_reset_mask=history_reset & history_mask[..., None],
            current_state_vectors=assemble_state_inputs(current_vectors, batch["current_state_skill_availability"], None, self.layout),
            current_state_null_mask=current_null,
            current_state_reset_mask=current_reset,
        )
        prepared.update(self._encode_scene(batch, anchor))
        return prepared

    def _gather_history(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        names = tuple(field.name for field in ENCODING_BANK_FIELDS)
        if "history_bank_state_abs_values" not in batch:
            return {"history_" + name: batch["history_" + name] for name in names}
        lengths, ends = batch["history_lengths"], batch["history_ends"]
        width = batch["history_mask"].shape[1]
        if lengths.dtype != torch.long or ends.dtype != torch.long or lengths.shape != ends.shape:
            raise ValueError("compact history lengths and ends must be matching int64 tensors")
        relative = torch.arange(width, dtype=torch.long, device=ends.device)
        valid = relative[None, :] < lengths[:, None]
        indices = ends[:, None] - lengths[:, None] + relative[None, :]
        safe_indices = torch.where(valid, indices, 0)
        return {"history_" + name: batch["history_bank_" + name][safe_indices] for name in names}

    def _encode_state(self, absolute, delta, nulls, reset, anchor):
        _validate_raw_state(delta, nulls)
        if reset.dtype != torch.bool or reset.shape != absolute.shape or reset.device != absolute.device:
            raise ValueError("state reset masks must be boolean with matching shape and device")
        reset = reset & ~nulls
        absolute = torch.where(self.state_time_mask, absolute - anchor, absolute)
        bounded = torch.minimum(torch.maximum(absolute, self.state_absolute_lower), self.state_absolute_upper)
        values = torch.where(reset, bounded, delta)
        signed_log = values.sign() * torch.log1p(values.abs())
        values = torch.where(self.state_logarithmic, signed_log, values / self.state_divisors)
        return values.masked_fill(nulls, -1.0), reset

    def _encode_scene(self, batch, anchor):
        values, types, mask = batch["scene_abs_values"], batch["scene_types"], batch["scene_mask"]
        if values.dtype != torch.float32 or values.ndim != 3 or values.shape[-1] != self.scene_dim:
            raise ValueError("raw scenes must be FP32 with shape [batch, scene, scene_dim]")
        if types.dtype not in (torch.int32, torch.int64) or mask.dtype != torch.bool or types.shape != values.shape[:2] or mask.shape != types.shape:
            raise ValueError("scene types and masks must match raw scene shape")
        # compiled cache 使用配置中的 int32；神经 gather 的索引统一为 int64。
        types = types.to(dtype=torch.long)
        if values.shape[1] == 0:
            return {"scene_vectors": values, "scene_types": types, "scene_mask": mask}
        starts = values.gather(-1, self.scene_start_indices[types][..., None]).squeeze(-1)
        ends = values.gather(-1, self.scene_end_indices[types][..., None]).squeeze(-1)
        mask = mask & (ends > anchor[:, None]) & (ends > starts)
        starts = torch.maximum(starts, anchor[:, None])
        # stable 多键排序保留同类、同起止窗口的原始序号；无效行沉到右侧。
        order = torch.argsort(types, dim=1, stable=True)
        for key in (ends, starts):
            sortable = key.masked_fill(~mask, float("inf")).gather(1, order)
            order = order.gather(1, torch.argsort(sortable, dim=1, stable=True))
        values = values.gather(1, order[..., None].expand(-1, -1, self.scene_dim)).clone()
        types, mask = types.gather(1, order), mask.gather(1, order)
        starts, ends = starts.gather(1, order) - anchor[:, None], ends.gather(1, order) - anchor[:, None]
        previous_start = torch.cat((torch.zeros_like(starts[:, :1]), starts[:, :-1]), dim=1)
        previous_end = torch.cat((torch.zeros_like(ends[:, :1]), ends[:, :-1]), dim=1)
        for indices, encoded in (
            (self.scene_start_indices, (starts - previous_start) / self.time_delta_scale),
            (self.scene_end_indices, (ends - previous_end) / self.time_delta_scale),
            (self.scene_duration_indices, (ends - starts) / self.time_delta_scale),
        ):
            values.scatter_(-1, indices[types][..., None], encoded[..., None])
        counts = values.clamp(min=0.0, max=self.target_count_max) / self.target_count_max
        values = torch.where(self.scene_count_mask[types], counts, values)
        values = values.masked_fill(~self.scene_feature_mask[types] | ~mask[..., None], 0.0)
        types = types.masked_fill(~mask, 0)
        return {"scene_vectors": values, "scene_types": types, "scene_mask": mask}

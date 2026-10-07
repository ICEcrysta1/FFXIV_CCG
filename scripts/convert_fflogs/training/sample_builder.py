"""把内存 training source 样本编码为最终 compiled cache 记录。"""

from __future__ import annotations

from common.policy.data.context_encoding import raw_state_delta

from .history_bank import history_reference

QUALITY_SEVERITY_LEVELS = {"minor": 1, "medium": 2, "major": 3}


class TrainingSampleBuilder:
    """装配完整历史引用、原始状态和固定动作监督的 compiled 样本。"""

    def __init__(
        self,
        *,
        torch,
        int_dtype,
        float_dtype,
        num_actions: int,
        history_bank: dict[str, object],
        ranking: dict[str, object] | None = None,
        annotation_status: str = "unannotated",
    ):
        self._torch = torch
        self._int_dtype = int_dtype
        self._float_dtype = float_dtype
        self._num_actions = num_actions
        self._history_bank = history_bank
        self._ranking = ranking or {"percentile": None, "percentile_bucket": None}
        self._annotation_status = annotation_status

    def build(self, reader, sample_idx: int) -> dict[str, object]:
        history_end, history_length = history_reference(reader, sample_idx)
        current_state = reader.current_state_matrix(
            sample_idx,
            dtype=self._torch.float32,
        )
        current_abs = current_state.values[0]
        previous_abs = self._history_bank["state_abs_values"][history_end - 1] if history_length else None
        previous_null = self._history_bank["state_null_mask"][history_end - 1] if history_length else None
        current_delta, current_reset = raw_state_delta(
            current_abs, current_state.null_mask[0], previous_abs, previous_null,
        )
        scene_vectors, scene_types = reader.scene_tokens(
            sample_idx,
            float_dtype=self._torch.float32,
            int_dtype=self._int_dtype,
        )
        label = reader.label(sample_idx)
        label_index = int(label.get("action_index", -1))
        if not 0 <= label_index < self._num_actions:
            raise ValueError(
                f"training label action index out of range: {label_index} "
                f"for {self._num_actions} output actions"
            )
        if str(label.get("action_key", "")) != str(reader.action_keys[label_index]):
            raise ValueError(
                "training label action does not match output index: "
                f"index={label_index} label={label.get('action_key')!r} "
                f"action={reader.action_keys[label_index]!r}"
            )

        quality_labels = list(label.get("quality_labels", []))
        if quality_labels and self._annotation_status != "partial":
            raise ValueError("action quality labels require an annotated source")
        severity_levels = []
        for quality_label in quality_labels:
            if not isinstance(quality_label, dict):
                raise TypeError("action quality label must be a mapping")
            severity = quality_label.get("severity")
            if severity not in QUALITY_SEVERITY_LEVELS:
                raise ValueError(f"unknown action quality severity: {severity!r}")
            severity_levels.append(QUALITY_SEVERITY_LEVELS[severity])
        if label.get("raw_event_index") is None:
            quality_status = "synthetic"
        elif quality_labels:
            quality_status = "attributed_label"
        elif self._annotation_status == "partial":
            quality_status = "no_attributed_label"
        else:
            quality_status = "unannotated"

        sample = {
            "metadata": {
                **reader.step_metadata(sample_idx),
                **self._ranking,
                "quality_label_status": quality_status,
            },
            "action_keys": list(reader.action_keys),
            "action_values": reader.action_values(sample_idx, dtype=self._float_dtype),
            "current_state_abs_values": current_abs,
            "current_state_delta_values": current_delta,
            "current_state_null_mask": current_state.null_mask[0],
            "current_state_delta_reset_mask": current_reset,
            "action_legal_mask": reader.action_legal_mask(sample_idx),
            "scene_vectors": scene_vectors,
            "scene_types": scene_types,
            "label_index": label_index,
            "label_action_key": str(label.get("action_key", "")),
            # 实际读条只用于监督侧序列采样，不进入模型输入。
            "label_cast_time_seconds": label.get("cast_time_seconds"),
            "raw_event_index": label.get("raw_event_index"),
            "quality_labels": quality_labels,
            "quality_label_levels": self._torch.tensor(severity_levels, dtype=self._int_dtype),
        }
        sample.update({"history_end": history_end, "history_length": history_length})
        return sample

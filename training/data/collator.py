"""训练公共层 collator。"""

from __future__ import annotations

import random
from collections.abc import Mapping

from common.config import load_precision_config
from common.torch_dependencies import import_torch


class TrainingCollator:
    """按语义分组对 batch 做 padding。"""

    def __init__(
        self,
        *,
        history_truncation_enabled: bool = False,
        history_truncation_probability: float = 0.0,
        history_min_recent: int = 1,
        require_quality_percentile: bool = False,
        int_dtype=None,
        index_dtype=None,
        float_dtype=None,
        rng=None,
    ):
        if not 0.0 <= history_truncation_probability <= 1.0:
            raise ValueError("history_truncation_probability must be between 0 and 1")
        if history_min_recent < 1:
            raise ValueError("history_min_recent must be >= 1")
        self.history_truncation_enabled = bool(history_truncation_enabled)
        self.history_truncation_probability = float(history_truncation_probability)
        self.history_min_recent = int(history_min_recent)
        self._require_quality_percentile = bool(require_quality_percentile)
        self._rng = rng or random
        precision = load_precision_config() if (int_dtype is None or index_dtype is None or float_dtype is None) else None
        self._int_dtype = int_dtype if int_dtype is not None else precision.resolve_int_dtype()
        self._index_dtype = index_dtype if index_dtype is not None else precision.resolve_index_dtype()
        self._float_dtype = float_dtype if float_dtype is not None else precision.resolve_float_dtype()

    def __call__(self, samples: list[dict[str, object]]) -> dict[str, object]:
        torch = import_torch()
        if not samples:
            return {}

        samples = [self._truncate_early_history(sample) for sample in samples]
        action_keys = tuple(samples[0]["action_keys"])
        if not action_keys or any(tuple(sample["action_keys"]) != action_keys for sample in samples):
            raise ValueError("batch samples must share the same fixed action output space")
        for sample in samples:
            if sample["action_values"].shape != (len(action_keys),) or sample["action_legal_mask"].shape != (len(action_keys),):
                raise ValueError("action supervision width must match the fixed output action space")
            label_index = int(sample["label_index"])
            if not 0 <= label_index < len(action_keys) or str(sample["label_action_key"]) != action_keys[label_index]:
                raise ValueError("label must match the fixed output action index")

        history_lengths = [int(sample["history_length"]) for sample in samples]
        scene_lengths = [sample["scene_vectors"].shape[0] for sample in samples]
        history_action_batches = [_compact_history_action_keys(sample) for sample in samples]

        quality_levels = [
            sample.get("quality_label_levels", torch.empty(0, dtype=self._int_dtype))
            for sample in samples
        ]
        quality_lengths = [int(levels.numel()) for levels in quality_levels]
        quality_width = max(quality_lengths)
        padded_quality_levels = torch.zeros((len(samples), quality_width), dtype=self._int_dtype)
        quality_label_mask = torch.zeros((len(samples), quality_width), dtype=torch.bool)
        for index, levels in enumerate(quality_levels):
            length = quality_lengths[index]
            padded_quality_levels[index, :length] = levels
            quality_label_mask[index, :length] = True
        quality_statuses = [sample["metadata"].get("quality_label_status", "unannotated") for sample in samples]

        batch = {
            "metadata": [sample["metadata"] for sample in samples],
            "history_action_keys": history_action_batches,
            "action_keys": [sample["action_keys"] for sample in samples],
            "label_action_key": [sample["label_action_key"] for sample in samples],
            "quality_label_levels": padded_quality_levels,
            "quality_label_mask": quality_label_mask,
            "quality_annotation_available": torch.tensor(
                [status in {"attributed_label", "no_attributed_label"} for status in quality_statuses],
                dtype=torch.bool,
            ),
            "source_quality": torch.tensor(
                [
                    -1.0 if sample["metadata"].get("percentile") is None
                    else float(sample["metadata"]["percentile"]) / 100.0
                    for sample in samples
                ],
                dtype=self._float_dtype,
            ),
            "label_index": torch.tensor(
                [sample["label_index"] for sample in samples],
                dtype=self._index_dtype,
            ),
            "action_legal_mask": torch.stack(
                [sample["action_legal_mask"] for sample in samples]
            ),
            "action_values": torch.stack([sample["action_values"] for sample in samples]),
        }
        for field in ("abs_values", "delta_values", "null_mask", "delta_reset_mask"):
            key = f"current_state_{field}"
            batch[key] = torch.stack([sample[key] for sample in samples])
        _validate_quality_supervision(
            batch, torch=torch, require_percentile=self._require_quality_percentile,
        )
        bank, history_ends = _merge_history_banks(samples, torch=torch)
        batch["history_lengths"] = torch.tensor(history_lengths, dtype=self._index_dtype)
        batch["history_ends"] = torch.tensor(history_ends, dtype=self._index_dtype)
        for key, value in bank.items():
            batch[f"history_bank_{key}"] = value
        batch["history_mask"] = _build_length_mask(history_lengths, torch=torch)
        batch["scene_mask"] = _build_length_mask(scene_lengths, torch=torch)

        pad_sequence = torch.nn.utils.rnn.pad_sequence
        for key in ("scene_vectors", "scene_types"):
            batch[key] = pad_sequence([sample[key] for sample in samples], batch_first=True)
        return batch

    def _truncate_early_history(self, sample: dict[str, object]) -> dict[str, object]:
        """仅截掉早期历史，保留最近状态；scene 和当前请求状态保持原样。"""
        history_length = int(sample["history_length"])
        if (
            history_length <= self.history_min_recent
            or not self.history_truncation_enabled
            or self.history_truncation_probability <= 0.0
            or self._rng.random() >= self.history_truncation_probability
        ):
            return sample

        keep_length = self._rng.randint(self.history_min_recent, history_length - 1)
        truncated = dict(sample)
        truncated["history_length"] = keep_length
        return truncated

def _validate_quality_supervision(
    batch: Mapping[str, object], *, torch, require_percentile: bool,
) -> None:
    """在 CPU 数据入口校验静态监督字段，损失计算不再读取设备标量。"""
    levels = batch["quality_label_levels"]
    mask = batch["quality_label_mask"]
    available = batch["quality_annotation_available"]
    quality = batch["source_quality"]
    if any(value.device.type != "cpu" for value in (levels, mask, available, quality)):
        raise ValueError("quality supervision must be collated on CPU")
    if bool((mask & ((levels < 1) | (levels > 3))).any()):
        raise ValueError("quality label levels must be 1, 2 or 3")
    tagged = mask.any(dim=1)
    if bool((tagged & ~available).any()):
        raise ValueError("attributed quality labels require available annotation")
    # 精确排名只用于质量加权；验证或关闭质量损失时允许旧数据缺少排名。
    if not require_percentile:
        return
    bad_quality = tagged & (
        ~torch.isfinite(quality) | (quality < 0.0) | (quality > 1.0)
    )
    if bool(bad_quality.any()):
        raise ValueError("attributed quality labels require a percentile within [0, 100]")


def _build_length_mask(lengths: list[int], *, torch):
    """一次广播生成整个 batch 的右侧 padding mask，支持全空序列。"""
    return torch.arange(max(lengths)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)


def _merge_history_banks(samples: list[dict[str, object]], *, torch):
    """按显式 source bank ID 合并；同 source 即使样本被复制也只合并一次。"""
    groups: dict[str, tuple[dict[str, object], int]] = {}
    ordered: list[dict[str, object]] = []
    sample_offsets: list[int] = []
    total_rows = 0
    bank_keys = (
        "skill_ids", "skill_features", "state_abs_values", "state_delta_values",
        "state_null_mask", "state_delta_reset_mask",
    )
    for sample in samples:
        bank = {key: sample[f"history_bank_{key}"] for key in bank_keys}
        bank_id = sample.get("history_bank_id")
        if bank_id is None or not str(bank_id):
            raise ValueError("compact history sample is missing history_bank_id")
        existing = groups.get(str(bank_id))
        if existing is None:
            offset = total_rows
            groups[str(bank_id)] = (bank, offset)
            ordered.append(bank)
            total_rows += bank["skill_ids"].shape[0]
        else:
            offset = existing[1]
        sample_offsets.append(offset)

    if len(ordered) == 1:
        merged = ordered[0]
    else:
        merged = {
            key: torch.cat([bank[key] for bank in ordered], dim=0)
            for key in bank_keys
        }
    history_ends = [
        int(sample["history_end"]) + offset
        for sample, offset in zip(samples, sample_offsets)
    ]
    return merged, history_ends


def _compact_history_action_keys(sample: dict[str, object]) -> list[str]:
    bank_keys = sample["history_bank_action_keys"]
    end = int(sample["history_end"])
    length = int(sample["history_length"])
    return [str(key) for key in bank_keys[end - length : end]]

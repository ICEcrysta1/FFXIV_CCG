"""训练公共层 collator。"""

from __future__ import annotations

import random
from collections.abc import Mapping

from common.torch_dependencies import import_torch


class TrainingCollator:
    """按语义分组对 batch 做 padding。"""

    def __init__(
        self,
        *,
        history_truncation_enabled: bool = False,
        history_truncation_probability: float = 0.0,
        history_min_recent: int = 1,
        candidate_shuffle_enabled: bool = False,
        candidate_shuffle_probability: float = 0.0,
        skill_values: Mapping[str, float] | None = None,
        rng=None,
    ):
        if not 0.0 <= history_truncation_probability <= 1.0:
            raise ValueError("history_truncation_probability must be between 0 and 1")
        if history_min_recent < 1:
            raise ValueError("history_min_recent must be >= 1")
        if not 0.0 <= candidate_shuffle_probability <= 1.0:
            raise ValueError("candidate_shuffle_probability must be between 0 and 1")
        self.history_truncation_enabled = bool(history_truncation_enabled)
        self.history_truncation_probability = float(history_truncation_probability)
        self.history_min_recent = int(history_min_recent)
        self.candidate_shuffle_enabled = bool(candidate_shuffle_enabled)
        self.candidate_shuffle_probability = float(candidate_shuffle_probability)
        self._skill_values = (
            None
            if skill_values is None
            else {str(action_key): float(value) for action_key, value in skill_values.items()}
        )
        self._rng = rng or random

    def __call__(self, samples: list[dict[str, object]]) -> dict[str, object]:
        torch = import_torch()
        if not samples:
            return {}

        samples = [self._truncate_early_history(sample) for sample in samples]
        samples = [self._shuffle_candidates(sample, torch) for sample in samples]

        compact_flags = ["history_end" in sample for sample in samples]
        if any(compact_flags) and not all(compact_flags):
            raise ValueError("cannot mix compact and dense history samples in one batch")
        use_compact_history = all(compact_flags)
        history_lengths = (
            [int(sample["history_length"]) for sample in samples]
            if use_compact_history
            else [sample["history_skill_ids"].shape[0] for sample in samples]
        )
        scene_lengths = [sample["scene_vectors"].shape[0] for sample in samples]
        history_action_batches = (
            [_compact_history_action_keys(sample) for sample in samples]
            if use_compact_history
            else [sample["history_action_keys"] for sample in samples]
        )

        batch = {
            "metadata": [sample["metadata"] for sample in samples],
            "history_action_keys": history_action_batches,
            "candidate_action_keys": [sample["candidate_action_keys"] for sample in samples],
            "candidate_invalid_reasons": [
                sample.get("candidate_invalid_reasons", []) for sample in samples
            ],
            "label_action_key": [sample["label_action_key"] for sample in samples],
            "label_index": torch.tensor(
                [sample["label_index"] for sample in samples],
                dtype=torch.int64,
            ),
            "candidate_skill_ids": torch.stack([sample["candidate_skill_ids"] for sample in samples]),
            "candidate_skill_features": torch.stack(
                [sample["candidate_skill_features"] for sample in samples]
            ),
            "candidate_state_vectors": torch.stack(
                [sample["candidate_state_vectors"] for sample in samples]
            ),
            "candidate_state_null_mask": torch.stack(
                [sample["candidate_state_null_mask"] for sample in samples]
            ),
            "candidate_legal_mask": torch.stack(
                [sample["candidate_legal_mask"] for sample in samples]
            ),
        }
        if self._skill_values is not None:
            runtime_values = []
            for sample in samples:
                candidate_values = sample.get("candidate_values")
                if candidate_values is None:
                    candidate_values = torch.tensor(
                        [
                            self._resolve_skill_value(action_key)
                            for action_key in sample["candidate_action_keys"]
                        ],
                        dtype=samples[0]["candidate_skill_features"].dtype,
                    )
                runtime_values.append(candidate_values)
            batch["candidate_values"] = torch.stack(runtime_values).to(
                dtype=samples[0]["candidate_skill_features"].dtype,
            )

        if use_compact_history:
            bank, history_ends = _merge_history_banks(samples, torch=torch)
            batch["history_lengths"] = torch.tensor(
                history_lengths,
                dtype=torch.int64,
            )
            batch["history_ends"] = torch.tensor(history_ends, dtype=torch.int64)
            for key, value in bank.items():
                batch[f"history_bank_{key}"] = value
        batch["history_mask"] = _build_length_mask(history_lengths, torch=torch)
        batch["scene_mask"] = _build_length_mask(scene_lengths, torch=torch)

        pad_sequence = torch.nn.utils.rnn.pad_sequence
        if not use_compact_history:
            for key in (
                "history_skill_ids",
                "history_skill_features",
                "history_state_vectors",
                "history_state_null_mask",
            ):
                batch[key] = pad_sequence(
                    [sample[key] for sample in samples],
                    batch_first=True,
                    padding_value=True if key == "history_state_null_mask" else 0,
                )

        for key in ("scene_vectors", "scene_types"):
            batch[key] = pad_sequence([sample[key] for sample in samples], batch_first=True)
        return batch

    def _resolve_skill_value(self, action_key: str) -> float:
        assert self._skill_values is not None
        try:
            return self._skill_values[action_key]
        except KeyError as exc:
            raise ValueError(
                f"missing value for candidate action {action_key!r} in the job YAML"
            ) from exc

    def _truncate_early_history(self, sample: dict[str, object]) -> dict[str, object]:
        """仅截掉早期历史，保留最近状态；scene 和候选输入保持原样。"""
        compact_history = "history_end" in sample
        history_length = (
            int(sample["history_length"])
            if compact_history
            else int(sample["history_skill_ids"].shape[0])
        )
        if (
            history_length <= self.history_min_recent
            or not self.history_truncation_enabled
            or self.history_truncation_probability <= 0.0
            or self._rng.random() >= self.history_truncation_probability
        ):
            return sample

        keep_length = self._rng.randint(self.history_min_recent, history_length - 1)
        start = history_length - keep_length
        truncated = dict(sample)
        if compact_history:
            if "history_action_keys" in sample:
                truncated["history_action_keys"] = sample["history_action_keys"][start:]
            truncated["history_length"] = keep_length
        else:
            truncated["history_action_keys"] = sample["history_action_keys"][start:]
            for key in (
                "history_skill_ids",
                "history_skill_features",
                "history_state_vectors",
                "history_state_null_mask",
            ):
                truncated[key] = sample[key][start:]
        return truncated

    def _shuffle_candidates(self, sample: dict[str, object], torch) -> dict[str, object]:
        """同步打乱候选技能与对应预演状态，并重映射 label index。"""
        candidate_count = int(sample["candidate_skill_ids"].shape[0])
        if (
            candidate_count <= 1
            or not self.candidate_shuffle_enabled
            or self.candidate_shuffle_probability <= 0.0
            or self._rng.random() >= self.candidate_shuffle_probability
        ):
            return sample

        permutation = list(range(candidate_count))
        self._rng.shuffle(permutation)
        permutation_tensor = torch.tensor(permutation, dtype=torch.int64)
        shuffled = dict(sample)
        for key in (
            "candidate_skill_ids",
            "candidate_skill_features",
            "candidate_values",
            "candidate_state_vectors",
            "candidate_state_null_mask",
            "candidate_legal_mask",
        ):
            if key in sample:
                shuffled[key] = sample[key][permutation_tensor]
        shuffled["candidate_action_keys"] = [
            sample["candidate_action_keys"][index] for index in permutation
        ]
        if "candidate_invalid_reasons" in sample:
            shuffled["candidate_invalid_reasons"] = [
                sample["candidate_invalid_reasons"][index] for index in permutation
            ]
        shuffled["label_index"] = permutation.index(int(sample["label_index"]))
        return shuffled


def _build_length_mask(lengths: list[int], *, torch):
    """一次广播生成整个 batch 的右侧 padding mask，支持全空序列。"""
    return torch.arange(max(lengths)).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)


def _merge_history_banks(samples: list[dict[str, object]], *, torch):
    """按显式 source bank ID 合并；同 source 即使样本被复制也只合并一次。"""
    groups: dict[str, tuple[dict[str, object], int]] = {}
    ordered: list[dict[str, object]] = []
    sample_offsets: list[int] = []
    total_rows = 0
    bank_keys = ("skill_ids", "skill_features", "state_vectors", "state_null_mask")
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
    bank_keys = sample.get("history_bank_action_keys")
    if bank_keys is not None:
        end = int(sample["history_end"])
        length = int(sample["history_length"])
        return [str(key) for key in bank_keys[end - length : end]]
    legacy_keys = sample.get("history_action_keys")
    if legacy_keys is not None:
        return [str(key) for key in legacy_keys]
    raise ValueError("compact history sample is missing history bank action keys")

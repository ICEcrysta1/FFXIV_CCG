"""把内存 training source 样本编码为最终 compiled cache 记录。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..source.source_helpers import SKILL_ID_FIELD, build_skill_feature_matrix, to_optional_int
from .history_bank import history_reference

if TYPE_CHECKING:
    from common.policy.data.normalizer import Normalizer
    from common.policy.data.skill_vocab import SkillVocab


class TrainingSampleBuilder:
    """集中处理词表编码、归一化和compiled cache 样本契约拼装。"""

    def __init__(
        self,
        *,
        torch,
        normalizer: Normalizer | None,
        skill_vocab: SkillVocab,
        skill_feature_names: tuple[str, ...],
        int_dtype,
        float_dtype,
        num_candidates: int,
        history_bank: dict[str, object] | None = None,
    ):
        self._torch = torch
        self._normalizer = normalizer
        self._skill_vocab = skill_vocab
        self._skill_feature_names = skill_feature_names
        self._int_dtype = int_dtype
        self._float_dtype = float_dtype
        self._num_candidates = num_candidates
        self._history_bank = history_bank

    def build(self, reader, sample_idx: int) -> dict[str, object]:
        compact_history = self._history_bank is not None
        if compact_history:
            history_end, history_length = history_reference(reader, sample_idx)
        else:
            history_rows = reader.history_skill_rows(sample_idx)
            history_raw_skill_ids = [
                to_optional_int(row.get(SKILL_ID_FIELD))
                for row in history_rows
            ]
            history_skill_ids = self._encode_skill_ids(
                history_raw_skill_ids,
                context=f"history sample={sample_idx} fight={reader.fight_id}",
            )
            history_skill_features = build_skill_feature_matrix(
                history_rows,
                feature_names=reader.skill_feature_names,
                torch=self._torch,
                dtype=self._float_dtype,
            )
            if self._normalizer is not None:
                history_skill_features = self._normalizer.normalize_skill_features(
                    history_skill_features,
                    self._skill_feature_names,
                )
            history_skill_potencies, history_cumulative_dot_potencies = (
                reader.history_skill_metrics(
                    sample_idx,
                )
            )
            history_state = reader.history_state_matrix(
                sample_idx,
                dtype=self._float_dtype,
                normalizer=self._normalizer,
            )

        candidate_rows = reader.candidate_skill_rows(sample_idx)
        candidate_action_keys = [str(row.get("skill_key", "")) for row in candidate_rows]
        candidate_raw_skill_ids = [
            to_optional_int(row.get(SKILL_ID_FIELD))
            for row in candidate_rows
        ]
        candidate_invalid_reasons = [
            str(row.get("invalid_reason", ""))
            for row in candidate_rows
        ]
        candidate_skill_ids = self._encode_skill_ids(
            candidate_raw_skill_ids,
            context=f"candidate sample={sample_idx} fight={reader.fight_id}",
        )
        candidate_skill_features = build_skill_feature_matrix(
            candidate_rows,
            feature_names=reader.skill_feature_names,
            torch=self._torch,
            dtype=self._float_dtype,
        )
        if self._normalizer is not None:
            candidate_skill_features = self._normalizer.normalize_skill_features(
                candidate_skill_features,
                self._skill_feature_names,
            )
        candidate_state = reader.candidate_state_matrix(
            sample_idx,
            dtype=self._float_dtype,
            normalizer=self._normalizer,
        )
        candidate_legal_mask = self._torch.tensor(
            [bool(row.get("is_legal", True)) for row in candidate_rows],
            dtype=self._torch.bool,
        )
        candidate_values = self._torch.tensor(
            [float(row.get("value", 1.0)) for row in candidate_rows],
            dtype=self._float_dtype,
        )
        scene_vectors, scene_types = reader.scene_tokens(
            sample_idx,
            float_dtype=self._float_dtype,
            int_dtype=self._int_dtype,
        )
        if self._normalizer is not None:
            scene_vectors = self._normalizer.normalize_scene_tokens(
                scene_vectors,
                scene_types,
                reader.schema,
            )
        label = reader.label(sample_idx)
        label_index = int(label.get("candidate_index", -1))
        if not 0 <= label_index < self._num_candidates:
            raise ValueError(
                f"training label candidate index out of range: {label_index} "
                f"for {self._num_candidates} candidates"
            )
        if str(label.get("action_key", "")) != str(candidate_action_keys[label_index]):
            raise ValueError(
                "training label action does not match candidate index: "
                f"index={label_index} label={label.get('action_key')!r} "
                f"candidate={candidate_action_keys[label_index]!r}"
            )

        sample = {
            "metadata": reader.step_metadata(sample_idx),
            "candidate_action_keys": candidate_action_keys,
            "candidate_skill_ids": candidate_skill_ids,
            "candidate_invalid_reasons": candidate_invalid_reasons,
            "candidate_skill_features": candidate_skill_features,
            "candidate_values": candidate_values,
            "candidate_state_vectors": candidate_state.values,
            "candidate_state_null_mask": candidate_state.null_mask,
            "candidate_legal_mask": candidate_legal_mask,
            "scene_vectors": scene_vectors,
            "scene_types": scene_types,
            "label_index": label_index,
            "label_action_key": str(label.get("action_key", "")),
        }
        if compact_history:
            sample.update(
                {
                    "history_end": history_end,
                    "history_length": history_length,
                }
            )
        else:
            sample.update(
                {
                    "history_action_keys": reader.history_action_keys(
                        sample_idx,
                    ),
                    "history_skill_ids": history_skill_ids,
                    "history_skill_features": history_skill_features,
                    "history_skill_potencies": self._torch.tensor(
                        history_skill_potencies,
                        dtype=self._float_dtype,
                    ),
                    "history_cumulative_dot_potencies": self._torch.tensor(
                        history_cumulative_dot_potencies,
                        dtype=self._float_dtype,
                    ),
                    "history_state_vectors": history_state.values,
                    "history_state_null_mask": history_state.null_mask,
                }
            )
        return sample

    def _encode_skill_ids(self, raw_skill_ids: list[int | None], *, context: str):
        return self._torch.tensor(
            [
                self._skill_vocab.require_lookup(raw_skill_id, context=f"{context} index={index}")
                for index, raw_skill_id in enumerate(raw_skill_ids)
            ],
            dtype=self._int_dtype,
        )

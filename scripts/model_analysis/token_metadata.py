"""模型分析 token 元数据构造。"""

from __future__ import annotations

import numpy as np

from common.policy.model.input_encoder import ROLE_HISTORY

from .job_labels import decision_state_labels


ANALYSIS_MP_MAX = 10000.0

ANALYSIS_FEATURES = (
    "fight_id",
    "step_index",
    "candidate_index",
    "skill_id",
    "legal",
    "invalid_reason",
    "elemental_state",
    "mp_bucket",
    "label_rank",
    "model_logit",
)


def build_token_metadata(
    samples: list[dict[str, object]],
    *,
    batch: dict[str, object],
    encoded: dict[str, object],
    schema,
    job_tag: str,
    logits: np.ndarray,
) -> dict[str, np.ndarray]:
    """按编码器实际位置构造每个 token 的科研元数据。"""
    batch_size = len(samples)
    role_ids = encoded["role_ids"].detach().cpu().numpy()
    sequence_length = int(role_ids.shape[1])
    candidate_positions = encoded["candidate_positions"].detach().cpu().tolist()
    candidate_count = len(candidate_positions)
    metadata: dict[str, np.ndarray] = {
        "fight_id": np.full((batch_size, sequence_length), "unknown", dtype=object),
        "step_index": np.full((batch_size, sequence_length), np.nan, dtype=np.float64),
        "candidate_index": np.full((batch_size, sequence_length), -1, dtype=np.float64),
        "skill_id": np.full((batch_size, sequence_length), -1, dtype=np.float64),
        "legal": np.full((batch_size, sequence_length), "non_candidate", dtype=object),
        "invalid_reason": np.full((batch_size, sequence_length), "", dtype=object),
        "elemental_state": np.full((batch_size, sequence_length), "unknown", dtype=object),
        "mp_bucket": np.full((batch_size, sequence_length), np.nan, dtype=np.float64),
        "label_rank": np.full((batch_size, sequence_length), np.nan, dtype=np.float64),
        "model_logit": np.full((batch_size, sequence_length), np.nan, dtype=np.float64),
    }
    history_skill_ids = batch["history_skill_ids"].detach().cpu().numpy()
    candidate_skill_ids = batch["candidate_skill_ids"].detach().cpu().numpy()
    candidate_legal = batch["candidate_legal_mask"].detach().cpu().numpy()
    candidate_reasons = batch.get("candidate_invalid_reasons")
    if candidate_reasons is None:
        candidate_reasons = [sample.get("candidate_invalid_reasons", []) for sample in samples]

    for batch_index, sample in enumerate(samples):
        sample_metadata = sample["metadata"]
        metadata["fight_id"][batch_index, :] = str(sample_metadata.get("fight_id", "unknown"))
        metadata["step_index"][batch_index, :] = float(sample_metadata.get("step", np.nan))
        elemental_state, mp_bucket = decision_state_labels(
            job_tag,
            batch_index,
            batch=batch,
            schema=schema,
        )
        metadata["elemental_state"][batch_index, :] = elemental_state
        metadata["mp_bucket"][batch_index, :] = mp_bucket

        label_index = int(batch["label_index"][batch_index].item())
        label_logit = float(logits[batch_index, label_index])
        metadata["label_rank"][batch_index, :] = 1.0 + float(
            np.count_nonzero(logits[batch_index] > label_logit)
        )

        history_positions = np.flatnonzero(role_ids[batch_index] == ROLE_HISTORY)
        for position, skill_id in zip(history_positions, history_skill_ids[batch_index]):
            metadata["skill_id"][batch_index, position] = float(skill_id)

        reasons = candidate_reasons[batch_index]
        for candidate_index in range(candidate_count):
            skill_id = float(candidate_skill_ids[batch_index, candidate_index])
            legal = "legal" if bool(candidate_legal[batch_index, candidate_index]) else "illegal"
            reason = str(reasons[candidate_index]) if candidate_index < len(reasons) else ""
            position = candidate_positions[candidate_index]
            metadata["candidate_index"][batch_index, position] = float(candidate_index)
            metadata["skill_id"][batch_index, position] = skill_id
            metadata["legal"][batch_index, position] = legal
            metadata["invalid_reason"][batch_index, position] = reason
            metadata["model_logit"][batch_index, position] = float(
                logits[batch_index, candidate_index]
            )
    return metadata

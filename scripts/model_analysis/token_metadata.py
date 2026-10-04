"""模型分析 token 元数据构造。"""

from __future__ import annotations

import numpy as np

from common.policy.model.input_encoder import ROLE_HISTORY

from .job_labels import decision_state_labels


ANALYSIS_MP_MAX = 10000.0

ANALYSIS_FEATURES = (
    "fight_id",
    "step_index",
    "label_index",
    "prediction_index",
    "label_legal",
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
    """决策变量仅关联最新状态 hidden，历史技能身份仍按真实历史位置记录。"""
    batch_size = len(samples)
    role_ids = encoded["role_ids"].detach().cpu().numpy()
    sequence_length = int(role_ids.shape[1])
    position = int(encoded["current_state_position"])
    shape = (batch_size, sequence_length)
    metadata = {
        key: np.full(shape, np.nan, dtype=np.float64)
        for key in ANALYSIS_FEATURES
        if key not in {"fight_id", "label_legal", "elemental_state"}
    }
    metadata.update({
        "fight_id": np.full(shape, "unknown", dtype=object),
        "label_legal": np.full(shape, "not_decision", dtype=object),
        "elemental_state": np.full(shape, "unknown", dtype=object),
        "skill_id": np.full(shape, -1, dtype=np.float64),
    })
    history_skill_ids = batch["history_skill_ids"].detach().cpu().numpy()
    for batch_index, sample in enumerate(samples):
        sample_metadata = sample["metadata"]
        metadata["fight_id"][batch_index, :] = str(sample_metadata.get("fight_id", "unknown"))
        metadata["step_index"][batch_index, :] = float(sample_metadata.get("step", np.nan))
        elemental_state, mp_bucket = decision_state_labels(
            job_tag, batch_index, batch=batch, schema=schema,
        )
        metadata["elemental_state"][batch_index, position] = elemental_state
        metadata["mp_bucket"][batch_index, position] = mp_bucket
        label_index = int(batch["label_index"][batch_index].item())
        label_logit = float(logits[batch_index, label_index])
        metadata["label_index"][batch_index, position] = label_index
        metadata["prediction_index"][batch_index, position] = float(logits[batch_index].argmax())
        metadata["label_legal"][batch_index, position] = (
            "legal" if bool(batch["action_legal_mask"][batch_index, label_index].item()) else "illegal"
        )
        metadata["label_rank"][batch_index, position] = 1.0 + float(
            np.count_nonzero(logits[batch_index] > label_logit)
        )
        metadata["model_logit"][batch_index, position] = label_logit
        history_positions = np.flatnonzero(role_ids[batch_index] == ROLE_HISTORY)
        for history_position, skill_id in zip(history_positions, history_skill_ids[batch_index]):
            metadata["skill_id"][batch_index, history_position] = float(skill_id)
    return metadata

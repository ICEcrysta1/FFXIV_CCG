"""按职业路由模型分析所需的决策状态标签。"""

from __future__ import annotations

from .black_mage import decision_state_labels as _black_mage_decision_state_labels


_DECISION_STATE_LABELERS = {
    "black_mage": _black_mage_decision_state_labels,
}


def decision_state_labels(job_tag: str, batch_index: int, *, batch, schema) -> tuple[str, float]:
    """按职业提取 AF/UI、MP 等分析标签；未知职业不套用黑魔规则。"""
    labeler = _DECISION_STATE_LABELERS.get(str(job_tag))
    if labeler is None:
        return "unknown", float("nan")
    return labeler(batch_index, batch=batch, schema=schema)


__all__ = ["decision_state_labels"]

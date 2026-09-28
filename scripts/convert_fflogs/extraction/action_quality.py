"""把评估证据按原始 cast 事件索引关联到目标动作。"""

from __future__ import annotations

from collections import defaultdict

from ..config.config import load_action_quality_skill_policy
from .extraction import _get_ability_id


def attach_action_quality_labels(
    report: dict[str, object],
    actions: list[dict[str, object]],
    *,
    job_tag: str,
    source_id: int,
) -> None:
    """仅将明确准入且唯一匹配的成功施法标签写入动作；不按时间猜测。"""
    analysis = report.get("analysis")
    if analysis is None:
        return
    if not isinstance(analysis, dict) or analysis.get("schema_version") != 2:
        raise ValueError("annotated input requires action quality schema version 2")
    if analysis.get("actor", {}).get("id") != str(source_id):
        raise ValueError("analysis actor does not match conversion source")
    if analysis.get("job_tag") != job_tag:
        raise ValueError("analysis job does not match conversion job")
    if analysis.get("source", {}).get("fight_id") != report.get("fight_id"):
        raise ValueError("analysis fight does not match conversion fight")
    basis = analysis.get("time_basis", {})
    if basis.get("unit") != "ms" or basis.get("origin") != "pull_start":
        raise ValueError("analysis time basis is not pull-relative milliseconds")
    offset = basis.get("report_offset_ms")
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError("analysis report offset must be integer milliseconds")
    enabled = load_action_quality_skill_policy(job_tag)
    events = report.get("events")
    labels = analysis.get("fight_labels")
    if not isinstance(events, list) or not isinstance(labels, list):
        raise TypeError("annotated input must contain events and fight labels")

    by_index: dict[int, list[dict[str, object]]] = defaultdict(list)
    for label in labels:
        if not isinstance(label, dict):
            raise TypeError("fight label must be a mapping")
        content = label.get("content")
        props = content.get("props", {}) if isinstance(content, dict) else {}
        reason_id = props.get("id") if isinstance(props, dict) else None
        if reason_id not in enabled:
            continue
        if label.get("severity_kind") != "standard" or label.get("severity") not in {
            "minor", "medium", "major",
        }:
            continue
        references = label.get("actions")
        if not isinstance(references, list):
            raise TypeError(f"enabled suggestion lacks action references: {reason_id}")
        for reference in references:
            if not isinstance(reference, dict) or reference.get("match_status") != "exact":
                continue
            if "event_type" not in reference:
                raise ValueError(f"action reference lacks event_type; regenerate annotation: {reason_id}")
            if reference.get("event_type") != "cast":
                continue
            indices = reference.get("raw_event_indices")
            if not isinstance(indices, list) or len(indices) != 1:
                continue
            index = indices[0]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(events):
                raise ValueError(f"invalid raw event index in {reason_id}: {index!r}")
            event = events[index]
            if not isinstance(event, dict) or event.get("type") != "cast":
                raise ValueError(f"referenced event is not a cast: {reason_id} index={index}")
            action_id = _get_ability_id(event)
            if (
                event.get("sourceID") != source_id
                or action_id != reference.get("action_id")
                or event.get("timestamp") != reference.get("time_ms") + offset
            ):
                raise ValueError(f"action reference mismatch: {reason_id} index={index}")
            if any(item["reason_id"] == reason_id for item in by_index[index]):
                continue
            by_index[index].append({
                "reason_id": reason_id,
                "severity": label["severity"],
                "raw_event_index": index,
            })

    observed: set[int] = set()
    for action in actions:
        index = action.get("raw_event_index")
        if index in by_index:
            if action["skill_id"] != _get_ability_id(events[index]):
                raise ValueError(f"converted action ID mismatch at raw event {index}")
            action["quality_labels"] = list(by_index[index])
            observed.add(index)
    missing = set(by_index) - observed
    if missing:
        raise ValueError(f"quality-labelled casts were not converted: raw event indices={sorted(missing)}")

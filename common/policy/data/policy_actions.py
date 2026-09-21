"""模型策略控制动作配置。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from common.yaml_config import load_yaml_mapping


@dataclass(frozen=True)
class PolicyActionDefinition:
    """不进入游戏技能表、但会进入模型候选与词表的控制动作。"""

    key: str
    raw_id: int
    name: str
    candidate_kind: str
    behavior: str
    value: float
    tags: tuple[str, ...]


def load_policy_actions(path: Path | None = None) -> tuple[PolicyActionDefinition, ...]:
    """读取并严格校验 policy_actions.yaml。"""
    config_path = (
        Path(path).resolve()
        if path is not None
        else Path(__file__).resolve().parents[3] / "config" / "policy_actions.yaml"
    )
    payload = load_yaml_mapping(config_path, description="policy action config")
    raw_actions = payload.get("policy_actions")
    if not isinstance(raw_actions, Mapping):
        raise ValueError("policy action config must define a policy_actions mapping")

    actions: list[PolicyActionDefinition] = []
    raw_ids: set[int] = set()
    for raw_key, raw_definition in raw_actions.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("policy action key must not be empty")
        if not isinstance(raw_definition, Mapping):
            raise ValueError(f"policy action {key} must be a mapping")

        raw_id = _require_int(raw_definition, "raw_id", action_key=key)
        if raw_id in raw_ids:
            raise ValueError(f"duplicate policy action raw_id: {raw_id}")
        raw_ids.add(raw_id)

        candidate_kind = _require_text(
            raw_definition,
            "candidate_kind",
            action_key=key,
        )
        if candidate_kind not in {"gcd", "ogcd"}:
            raise ValueError(
                f"unsupported policy candidate kind for {key}: {candidate_kind}"
            )

        raw_tags = raw_definition.get("tags", ())
        if not isinstance(raw_tags, Sequence) or isinstance(raw_tags, (str, bytes)):
            raise ValueError(f"policy action {key}.tags must be a sequence")

        actions.append(
            PolicyActionDefinition(
                key=key,
                raw_id=raw_id,
                name=_require_text(raw_definition, "name", action_key=key),
                candidate_kind=candidate_kind,
                behavior=_require_text(raw_definition, "behavior", action_key=key),
                value=float(raw_definition.get("value", 1.0)),
                tags=tuple(str(tag) for tag in raw_tags),
            )
        )
    return tuple(actions)


def _require_text(
    definition: Mapping[object, object],
    field: str,
    *,
    action_key: str,
) -> str:
    value = definition.get(field)
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"policy action {action_key}.{field} must not be empty")
    return text


def _require_int(
    definition: Mapping[object, object],
    field: str,
    *,
    action_key: str,
) -> int:
    value = definition.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"policy action {action_key}.{field} must be an integer")
    return value

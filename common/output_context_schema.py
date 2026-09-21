"""canonical 输出 context schema 定义。

顶层 keys、状态向量分组与版本由 `config/schema.yaml` 提供
（C# 与 Python 单一事实来源）；`top_level_keys` 顺序即定义。
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema_config import load_schema_config

_output = load_schema_config()["canonical_output"]

CANONICAL_CONTEXT_SCHEMA_VERSION = int(_output["schema_version"])

CANONICAL_CONTEXT_TOP_LEVEL_KEYS = tuple(_output["top_level_keys"])

SCENE_CONTEXT_KEY = CANONICAL_CONTEXT_TOP_LEVEL_KEYS[2]
SKILL_HISTORY_CONTEXT_KEY = CANONICAL_CONTEXT_TOP_LEVEL_KEYS[3]
STATE_HISTORY_CONTEXT_KEY = CANONICAL_CONTEXT_TOP_LEVEL_KEYS[4]
CANDIDATE_SKILL_CONTEXT_KEY = CANONICAL_CONTEXT_TOP_LEVEL_KEYS[5]
CANDIDATE_STATE_CONTEXT_KEY = CANONICAL_CONTEXT_TOP_LEVEL_KEYS[6]
STATE_CONTEXT_TOKEN_KEY = str(_output["token_key"])


@dataclass(frozen=True)
class StateVectorGroupSchema:
    """状态向量分组 schema。"""

    group_key: str
    feature_keys_field: str
    context_key: str | None


STATE_VECTOR_GROUP_SCHEMAS = tuple(
    StateVectorGroupSchema(
        group_key=str(group_key),
        feature_keys_field=str(spec["feature_keys_field"]),
        context_key=None if spec["context_key"] is None else str(spec["context_key"]),
    )
    for group_key, spec in _output["state_vector_groups"].items()
)


def format_canonical_output_context(output_context: dict[str, object]) -> dict[str, object]:
    """按统一 schema 裁剪 canonical 输出。"""
    return {
        key: output_context[key]
        for key in CANONICAL_CONTEXT_TOP_LEVEL_KEYS
    }


def extract_state_feature_keys(state_context: dict[str, object]) -> dict[str, list[str]]:
    """提取状态上下文里每个向量分组的 feature key。"""
    return {
        group_schema.group_key: list(state_context[group_schema.feature_keys_field])
        for group_schema in STATE_VECTOR_GROUP_SCHEMAS
    }


def build_output_context_schema_metadata(output_context: dict[str, object]) -> dict[str, object]:
    """构造 canonical 输出的正式 schema 元数据。"""
    history_state_context = _expect_dict(output_context, STATE_HISTORY_CONTEXT_KEY)
    candidate_state_context = _expect_dict(output_context, CANDIDATE_STATE_CONTEXT_KEY)
    return {
        "schema_version": int(output_context["schema_version"]),
        "top_level_keys": list(CANONICAL_CONTEXT_TOP_LEVEL_KEYS),
        "scene_context_key": SCENE_CONTEXT_KEY,
        "skill_history_context_key": SKILL_HISTORY_CONTEXT_KEY,
        "state_history_context_key": STATE_HISTORY_CONTEXT_KEY,
        "candidate_skill_context_key": CANDIDATE_SKILL_CONTEXT_KEY,
        "candidate_state_context_key": CANDIDATE_STATE_CONTEXT_KEY,
        "state_context_token_key": STATE_CONTEXT_TOKEN_KEY,
        "state_vector_group_keys": [
            group_schema.group_key
            for group_schema in STATE_VECTOR_GROUP_SCHEMAS
        ],
        "state_vector_feature_key_fields": {
            group_schema.group_key: group_schema.feature_keys_field
            for group_schema in STATE_VECTOR_GROUP_SCHEMAS
        },
        "state_history_feature_keys": extract_state_feature_keys(history_state_context),
        "candidate_state_feature_keys": extract_state_feature_keys(candidate_state_context),
    }


def _expect_dict(container: dict[str, object], key: str) -> dict[str, object]:
    value = container[key]
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a dict, got {type(value).__name__}")
    return value

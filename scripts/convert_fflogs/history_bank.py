"""为 compiled cache 构建按动作递增的历史 bank。"""

from __future__ import annotations

from .source_helpers import (
    SKILL_ID_FIELD,
    build_skill_feature_matrix,
    extract_history_after_value,
    require_numeric_skill_kind,
    to_optional_int,
)


def build_history_bank(
    reader,
    *,
    torch,
    normalizer,
    skill_vocab,
    skill_feature_names: tuple[str, ...],
    int_dtype,
    float_dtype,
) -> dict[str, object]:
    """一次性编码一个 source 的历史行，供样本通过 end/length 引用。

    bank 的第 0 行是 padding sentinel，后续行按状态机实际生效历史追加。
    排队动作可能让相邻样本共享同一历史长度，也可能在两次观测间结算多行；
    每个样本只保存实际历史长度对应的 end-exclusive 下标。
    """
    skill_rows: list[dict[str, object]] = []
    state_tokens: list[dict[str, object]] = []
    action_keys = [""]
    skill_potencies = [0.0]
    cumulative_dot_potencies = [0.0]

    previous_length = 0
    for sample_idx in range(reader.num_samples):
        actual_length = reader.history_length(sample_idx)
        if actual_length < previous_length:
            raise ValueError(
                "training history length moved backwards: "
                f"sample={sample_idx} actual={actual_length} previous={previous_length}"
            )
        new_skill_rows, new_state_tokens = reader.history_delta(sample_idx, previous_length)
        for delta_index, (skill_row, state_token) in enumerate(
            zip(new_skill_rows, new_state_tokens, strict=True)
        ):
            context = (
                f"history bank sample={sample_idx} delta={delta_index} "
                f"fight={reader.fight_id}"
            )
            require_numeric_skill_kind(skill_row, context=context)
            skill_rows.append(skill_row)
            state_tokens.append(state_token)
            action_keys.append(str(skill_row.get("skill_key", "")))
            skill_potencies.append(float(skill_row.get("potency", 0.0)))
            cumulative_dot_potencies.append(
                extract_history_after_value(
                    state_token,
                    reader.schema,
                    feature_name="target.cumulative_dot_potency",
                    context=context,
                )
            )
        previous_length = actual_length

    feature_matrix = build_skill_feature_matrix(
        skill_rows,
        feature_names=reader.skill_feature_names,
        torch=torch,
        dtype=float_dtype,
    )
    if normalizer is not None:
        feature_matrix = normalizer.normalize_skill_features(
            feature_matrix,
            skill_feature_names,
        )
    skill_ids = torch.tensor(
        [
            0,
            *[
                skill_vocab.require_lookup(
                    to_optional_int(row.get(SKILL_ID_FIELD)),
                    context=f"history bank sample={sample_idx + 1} fight={reader.fight_id}",
                )
                for sample_idx, row in enumerate(skill_rows)
            ],
        ],
        dtype=int_dtype,
    )
    skill_features = torch.cat(
        (
            torch.zeros((1, feature_matrix.shape[-1]), dtype=float_dtype),
            feature_matrix,
        ),
        dim=0,
    )

    state_matrix = reader.state_matrix_from_tokens(
        state_tokens,
        dtype=float_dtype,
        normalizer=normalizer,
    )
    state_vectors = torch.cat(
        (
            torch.zeros((1, state_matrix.values.shape[-1]), dtype=float_dtype),
            state_matrix.values,
        ),
        dim=0,
    )
    state_null_mask = torch.cat(
        (
            torch.zeros((1, state_matrix.null_mask.shape[-1]), dtype=torch.bool),
            state_matrix.null_mask,
        ),
        dim=0,
    )

    return {
        "skill_ids": skill_ids,
        "skill_features": skill_features,
        "state_vectors": state_vectors,
        "state_null_mask": state_null_mask,
        "action_keys": tuple(action_keys),
        # PPG 使用原始历史威力；kind 已经在 skill_features 中以数值维度保存。
        "skill_potencies": torch.tensor(skill_potencies, dtype=float_dtype),
        "cumulative_dot_potencies": torch.tensor(
            cumulative_dot_potencies,
            dtype=float_dtype,
        ),
    }


def history_reference(reader, sample_idx: int) -> tuple[int, int]:
    """返回样本对应的 end-exclusive 引用和有效窗口长度。"""
    actual_length = reader.history_length(sample_idx)
    return actual_length + 1, actual_length

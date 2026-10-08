"""为 compiled cache 构建按动作递增的历史 bank。"""

from __future__ import annotations

from common.policy.data.context_encoding import raw_state_delta

from ..source.source_helpers import (
    SKILL_ID_FIELD,
    build_skill_feature_matrix,
    extract_execution_metric,
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
    execution_metrics: list[dict[str, object]] = []
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
        current_skill_rows, current_state_tokens = reader.history_delta(sample_idx, 0)
        current_metrics = reader.history_execution_metrics(sample_idx)
        # 长度递增不能证明历史只追加；相同长度的重排或旧行回填同样必须拒绝。
        for field, previous_rows, current_rows in (
            ("skill", skill_rows, current_skill_rows),
            ("state", state_tokens, current_state_tokens),
            ("execution_metrics", execution_metrics, current_metrics),
        ):
            for row_index, previous_row in enumerate(previous_rows):
                if current_rows[row_index] != previous_row:
                    raise ValueError(
                        "training history prefix changed: "
                        f"sample={sample_idx} row={row_index} field={field} fight={reader.fight_id}"
                    )
        for delta_index, (skill_row, state_token, metric) in enumerate(
            zip(current_skill_rows[previous_length:], current_state_tokens[previous_length:],
                current_metrics[previous_length:], strict=True)
        ):
            context = (
                f"history bank sample={sample_idx} delta={delta_index} "
                f"fight={reader.fight_id}"
            )
            require_numeric_skill_kind(skill_row, context=context)
            skill_rows.append(skill_row)
            state_tokens.append(state_token)
            execution_metrics.append(dict(metric))
            action_keys.append(str(skill_row.get("skill_key", "")))
            skill_potencies.append(float(skill_row.get("potency", 0.0)))
            cumulative_dot_potencies.append(
                extract_execution_metric(
                    metric,
                    feature_name="cumulative_dot_potency",
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
        dtype=torch.float32,
    )
    # 原始状态先保留完整物理值；缺失只由 mask 表达，不参与相邻差分。
    state_abs = state_matrix.base_values
    previous_abs = torch.cat((torch.zeros_like(state_abs[:1]), state_abs[:-1]), dim=0)
    previous_null = torch.cat(
        (torch.ones_like(state_matrix.base_null_mask[:1]), state_matrix.base_null_mask[:-1]), dim=0,
    )
    state_delta, state_reset = raw_state_delta(
        state_abs, state_matrix.base_null_mask, previous_abs, previous_null,
    )
    state_abs_values = torch.cat(
        (
            torch.zeros((1, state_abs.shape[-1]), dtype=torch.float32),
            state_abs,
        ),
        dim=0,
    )
    state_delta_values = torch.cat((torch.zeros_like(state_abs_values[:1]), state_delta), dim=0)
    state_null_mask = torch.cat(
        (
            torch.zeros((1, state_matrix.base_null_mask.shape[-1]), dtype=torch.bool),
            state_matrix.base_null_mask,
        ),
        dim=0,
    )
    state_delta_reset_mask = torch.cat((torch.zeros_like(state_null_mask[:1]), state_reset), dim=0)

    return {
        "skill_ids": skill_ids,
        "skill_features": skill_features,
        "state_abs_values": state_abs_values,
        "state_delta_values": state_delta_values,
        "state_null_mask": state_null_mask,
        "state_delta_reset_mask": state_delta_reset_mask,
        "state_skill_availability": torch.cat((torch.zeros((1, state_matrix.availability.shape[-1]), dtype=torch.bool), state_matrix.availability), dim=0),
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

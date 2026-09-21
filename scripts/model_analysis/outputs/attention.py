"""开场决策的候选动作注意力图。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from common.torch_runtime import move_batch
from training import TrainingCollator

from ..common import (
    ATTENTION_CMAP_NAME,
    HEATMAP_CELL_SIZE,
    AnalysisContext,
    ROLE_NAMES,
    create_figure,
    create_grid_figure,
    hide_empty_tiles,
    save_figure,
)


def plot_opener_attention(
    context: AnalysisContext,
    *,
    steps: int = 28,
    batch_size: int = 16,
) -> tuple[Path, Path]:
    """用 compiled cache 的真实开场样本，绘制 CLS 对全部候选动作的注意力。"""
    if steps <= 0:
        raise ValueError("attention steps must be positive")
    if batch_size <= 0:
        raise ValueError("attention batch size must be positive")
    sample_count = min(steps, len(context.dataset))
    if sample_count == 0:
        raise ValueError("dataset is empty, cannot plot opener attention")

    samples = [context.dataset[index] for index in range(sample_count)]
    candidate_keys = tuple(str(key) for key in samples[0]["candidate_action_keys"])
    collator = TrainingCollator()
    layer_rows: list[list[np.ndarray]] | None = None
    labels: list[str] = []
    label_indices: list[int] = []
    prediction_indices: list[int] = []

    with torch.no_grad():
        for start in range(0, sample_count, batch_size):
            sample_batch = samples[start : start + batch_size]
            batch = move_batch(collator(sample_batch), context.device)
            with context.autocast():
                trace = context.model.trace(batch)
                encoded = trace.encoded
                logits = context.model.score_hidden(encoded, trace.hidden, batch)
            attentions = trace.attentions
            candidate_positions = encoded["candidate_positions"]
            if layer_rows is None:
                layer_rows = [[] for _ in attentions]

            for layer_index, attention in enumerate(attentions):
                # attention: [batch, head, query, key]. CLS 是最后一个 query。
                cls_to_candidate = attention[:, :, -1, candidate_positions]
                candidate_attention = cls_to_candidate.mean(dim=1)
                layer_rows[layer_index].append(
                    candidate_attention.detach().float().cpu().numpy()
                )

            predictions = logits.argmax(dim=-1).detach().cpu().tolist()
            for row, sample in enumerate(sample_batch):
                label_action = str(sample["label_action_key"])
                label_index = int(sample["label_index"])
                labels.append(f"{start + row + 1}: {label_action}")
                label_indices.append(label_index)
                prediction_indices.append(int(predictions[row]))
            # 避免下一批前向开始时，上一批完整 trace 仍由局部变量引用。
            del batch, trace, encoded, logits, attentions, candidate_positions
            if context.device.type == "cuda":
                torch.cuda.empty_cache()

    assert layer_rows is not None
    layer_values = [np.concatenate(rows, axis=0) for rows in layer_rows]
    final_attention = layer_values[-1]
    candidate_focus = _normalize_candidate_attention(final_attention)
    layer_focus = np.stack(
        [_normalize_candidate_attention(values).mean(axis=0) for values in layer_values]
    )

    candidate_path = context.output_dir / "06_opener_attention_candidates.png"
    layer_path = context.output_dir / "07_opener_attention_by_layer.png"
    _plot_candidate_attention(
        candidate_focus,
        candidate_keys,
        labels,
        label_indices,
        prediction_indices,
        candidate_path,
    )
    _plot_layer_attention(layer_focus, candidate_keys, layer_path)
    return candidate_path, layer_path


def plot_standard_attention_outputs(
    context: AnalysisContext,
    *,
    steps: int = 28,
) -> tuple[Path, Path, Path]:
    """绘制完整 attention 矩阵，保留 query/key、mask 和 head 维度。"""
    if steps <= 0:
        raise ValueError("attention steps must be positive")
    sample_count = min(steps, len(context.dataset))
    if sample_count == 0:
        raise ValueError("dataset is empty, cannot plot standard attention")

    samples = [context.dataset[index] for index in range(sample_count)]
    # 选择上下文最长的样本，避免短样本把 history 的结构性区域截掉。
    representative = max(samples, key=_sample_token_count)
    collator = TrainingCollator()
    batch = move_batch(collator([representative]), context.device)
    with torch.no_grad():
        with context.autocast():
            trace = context.model.trace(batch)

    attention_arrays = tuple(_single_sample_attention(attention) for attention in trace.attentions)
    if not attention_arrays:
        raise ValueError("model trace did not return attention weights")
    token_count = attention_arrays[0].shape[-1]
    if any(attention.shape[-1] != token_count for attention in attention_arrays):
        raise ValueError("attention layers returned inconsistent token counts")

    attention_mask = _single_sample_matrix(trace.encoded, "attention_mask", token_count)
    role_ids = _single_sample_vector(trace.encoded, "role_ids", token_count)
    blocked = _to_block_mask(attention_mask)
    padding = trace.encoded.get("padding_mask")
    if padding is not None:
        padding_values = _single_sample_vector_value(padding, token_count).astype(bool)
        blocked = blocked | padding_values[:, None] | padding_values[None, :]
    del batch, trace, padding
    if context.device.type == "cuda":
        torch.cuda.empty_cache()

    matrix_path = context.output_dir / "08_attention_matrix_by_layer.png"
    heads_path = context.output_dir / "09_attention_matrix_last_layer_heads.png"
    blocks_path = context.output_dir / "10_attention_role_block_summary.png"
    _plot_attention_matrix_grid(attention_arrays, blocked, role_ids, matrix_path)
    _plot_attention_head_grid(attention_arrays[-1], blocked, role_ids, heads_path)
    _plot_attention_role_blocks(attention_arrays, blocked, role_ids, blocks_path)
    return matrix_path, heads_path, blocks_path


def _sample_token_count(sample: object) -> int:
    """按 compiled cache 字段估算样本 token 数，供代表样本选择使用。"""
    if not isinstance(sample, Mapping):
        raise TypeError(
            "expected mapping sample for representative selection, "
            f"got {type(sample).__name__}"
        )
    total = 0
    for key in ("scene_vectors", "candidate_skill_ids"):
        values = sample.get(key)
        if values is None:
            continue
        try:
            total += len(values)
        except TypeError as exc:
            raise TypeError(
                f"sample field {key!r} must be sized for representative selection"
            ) from exc

    history_values = sample.get("history_skill_ids")
    if history_values is not None:
        try:
            total += len(history_values)
        except TypeError as exc:
            raise TypeError(
                "sample field 'history_skill_ids' must be sized for representative selection"
            ) from exc
    else:
        # 增量 history bank 只保留 bank 游标和 history_length，模型装配前没有显式历史张量。
        history_length = sample.get("history_length")
        if history_length is not None:
            try:
                total += int(history_length)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError(
                    "sample field 'history_length' must be an integer for representative selection"
                ) from exc
    return total + 1


def _single_sample_attention(attention: torch.Tensor) -> np.ndarray:
    """取单个样本的逐 head attention，返回 [head, query, key]。"""
    if attention.ndim != 4 or attention.shape[0] == 0:
        raise ValueError("attention must have shape [batch, head, query, key]")
    values = attention[0].detach().float().cpu().numpy()
    if values.shape[-2] != values.shape[-1]:
        raise ValueError("attention query/key dimensions must match")
    return values


def _single_sample_matrix(
    encoded: dict[str, torch.Tensor],
    key: str,
    token_count: int,
) -> torch.Tensor:
    value = encoded.get(key)
    if value is None or not isinstance(value, torch.Tensor):
        raise ValueError(f"trace encoded data missing {key}")
    if value.ndim == 3:
        value = value[0]
    if value.ndim != 2 or tuple(value.shape) != (token_count, token_count):
        raise ValueError(f"{key} must have shape [{token_count}, {token_count}]")
    return value.detach().cpu()


def _single_sample_vector(
    encoded: dict[str, torch.Tensor],
    key: str,
    token_count: int,
) -> np.ndarray:
    value = encoded.get(key)
    if value is None or not isinstance(value, torch.Tensor):
        raise ValueError(f"trace encoded data missing {key}")
    return _single_sample_vector_value(value, token_count).astype(np.int64)


def _single_sample_vector_value(value: torch.Tensor, token_count: int) -> np.ndarray:
    if value.ndim == 2:
        value = value[0]
    if value.ndim != 1 or value.shape[0] != token_count:
        raise ValueError(f"encoded vector must have length {token_count}")
    return value.detach().cpu().numpy()


def _to_block_mask(attention_mask: torch.Tensor) -> np.ndarray:
    if attention_mask.dtype == torch.bool:
        return attention_mask.numpy().astype(bool)
    if torch.is_floating_point(attention_mask):
        return (~torch.isfinite(attention_mask)).numpy().astype(bool)
    return attention_mask.numpy().astype(bool)


def _attention_cmap():
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#d0d0d0")
    return cmap


def _attention_color_norm() -> Normalize:
    """使用固定 0~1 线性色阶映射相对注意力热图。"""
    return Normalize(vmin=0.0, vmax=1.0)


def _row_relative_attention(matrix: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    """按每个 query 行的最大有效权重归一化，供热图增强结构可读性。"""
    if matrix.shape != blocked.shape:
        raise ValueError(
            "attention matrix and blocked mask must have the same shape, "
            f"got {matrix.shape} != {blocked.shape}"
        )
    valid_values = np.where(blocked, 0.0, matrix)
    row_maximum = valid_values.max(axis=1, keepdims=True)
    normalized = np.zeros_like(valid_values, dtype=np.float32)
    np.divide(
        valid_values,
        row_maximum,
        out=normalized,
        where=row_maximum > np.finfo(np.float32).eps,
    )
    return normalized


def _role_spans(role_ids: np.ndarray) -> list[tuple[int, int, str]]:
    if role_ids.size == 0:
        return []
    spans: list[tuple[int, int, str]] = []
    start = 0
    for index in range(1, role_ids.size + 1):
        if index == role_ids.size or role_ids[index] != role_ids[start]:
            role_id = int(role_ids[start])
            spans.append((start, index, ROLE_NAMES.get(role_id, f"role_{role_id}")))
            start = index
    return spans


def _draw_attention_structure(ax, role_ids: np.ndarray) -> None:
    """在矩阵上标出 token role 边界和标准因果对角线参考线。"""
    spans = _role_spans(role_ids)
    centers = []
    labels = []
    for start, end, label in spans:
        centers.append((start + end - 1) / 2.0)
        labels.append(label)
        if end < role_ids.size:
            boundary = end - 0.5
            ax.axvline(boundary, color="white", linewidth=0.8, alpha=0.9)
            ax.axhline(boundary, color="white", linewidth=0.8, alpha=0.9)
    if centers:
        ax.set_xticks(centers, labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(centers, labels, fontsize=8)
    # 完全因果布局下对角线即 mask 边界：query 只能读取左上三角。
    diagonal_end = max(role_ids.size - 0.5, 0.5)
    ax.plot(
        [-0.5, diagonal_end],
        [-0.5, diagonal_end],
        linestyle="--",
        color="white",
        linewidth=0.8,
        alpha=0.75,
    )


def _plot_attention_matrix_grid(
    attentions: tuple[np.ndarray, ...],
    blocked: np.ndarray,
    role_ids: np.ndarray,
    path: Path,
) -> None:
    matrices = [
        _row_relative_attention(attention.mean(axis=0), blocked)
        for attention in attentions
    ]
    color_norm = _attention_color_norm()
    fig, axes = create_grid_figure(len(matrices), cell_size=HEATMAP_CELL_SIZE)
    cmap = _attention_cmap()
    image = None
    display_mask = blocked
    for index, matrix in enumerate(matrices):
        ax = axes.flat[index]
        display = np.ma.masked_where(display_mask, matrix)
        image = ax.imshow(display, cmap=cmap, norm=color_norm, aspect="auto")
        _draw_attention_structure(ax, role_ids)
        ax.set_title(f"Layer {index + 1} · head mean")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
    hide_empty_tiles(axes, len(matrices))
    assert image is not None
    fig.suptitle(
        "Full attention matrix by layer "
        "(gray = blocked by mask; each query row normalized independently)"
    )
    fig.colorbar(
        image,
        ax=list(axes.flat),
        label="row-relative attention (row maximum = 1.0)",
        shrink=0.82,
    )
    save_figure(fig, path)


def _plot_attention_head_grid(
    attention: np.ndarray,
    blocked: np.ndarray,
    role_ids: np.ndarray,
    path: Path,
) -> None:
    color_norm = _attention_color_norm()
    head_count = attention.shape[0]
    fig, axes = create_grid_figure(head_count, cell_size=HEATMAP_CELL_SIZE)
    cmap = _attention_cmap()
    image = None
    for index, matrix in enumerate(attention):
        ax = axes.flat[index]
        display = np.ma.masked_where(
            blocked,
            _row_relative_attention(matrix, blocked),
        )
        image = ax.imshow(display, cmap=cmap, norm=color_norm, aspect="auto")
        _draw_attention_structure(ax, role_ids)
        ax.set_title(f"Last layer · head {index + 1}")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
    hide_empty_tiles(axes, head_count)
    assert image is not None
    fig.suptitle(
        "Full attention matrix by head "
        "(gray = blocked by mask; each query row normalized independently)"
    )
    fig.colorbar(
        image,
        ax=list(axes.flat),
        label="row-relative attention (row maximum = 1.0)",
        shrink=0.82,
    )
    save_figure(fig, path)


def _role_block_means(
    matrices: list[np.ndarray],
    role_ids: np.ndarray,
    blocked: np.ndarray,
) -> np.ndarray:
    role_order = tuple(sorted(ROLE_NAMES))
    result = np.zeros((len(matrices), len(role_order), len(role_order)), dtype=np.float32)
    valid = ~blocked
    for layer_index, matrix in enumerate(matrices):
        for query_index, query_role in enumerate(role_order):
            query_positions = np.flatnonzero(role_ids == query_role)
            for key_index, key_role in enumerate(role_order):
                key_positions = np.flatnonzero(role_ids == key_role)
                if query_positions.size == 0 or key_positions.size == 0:
                    continue
                values = matrix[np.ix_(query_positions, key_positions)]
                allowed = valid[np.ix_(query_positions, key_positions)]
                result[layer_index, query_index, key_index] = float(
                    values[allowed].sum() / query_positions.size
                )
    return result


def _plot_attention_role_blocks(
    attentions: tuple[np.ndarray, ...],
    blocked: np.ndarray,
    role_ids: np.ndarray,
    path: Path,
) -> None:
    matrices = [attention.mean(axis=0) for attention in attentions]
    blocks = _role_block_means(matrices, role_ids, blocked)
    role_order = tuple(sorted(ROLE_NAMES))
    labels = [ROLE_NAMES[role] for role in role_order]
    fig, axes = create_grid_figure(len(matrices), cell_size=HEATMAP_CELL_SIZE)
    image = None
    for index, block in enumerate(blocks):
        ax = axes.flat[index]
        image = ax.imshow(block, cmap=ATTENTION_CMAP_NAME, vmin=0.0, vmax=1.0)
        ax.set_title(f"Layer {index + 1}")
        ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_xlabel("Key role")
        ax.set_ylabel("Query role")
        for query_index in range(block.shape[0]):
            for key_index in range(block.shape[1]):
                ax.text(
                    key_index,
                    query_index,
                    f"{block[query_index, key_index]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if block[query_index, key_index] < 0.5 else "black",
                    fontsize=8,
                )
    hide_empty_tiles(axes, len(matrices))
    assert image is not None
    fig.suptitle("Mean attention mass between token roles")
    fig.colorbar(image, ax=list(axes.flat), label="mean attention mass", shrink=0.82)
    save_figure(fig, path)


def _normalize_candidate_attention(values: np.ndarray) -> np.ndarray:
    """按每个开场决策行的候选最大注意力归一化到 0~1。"""
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("candidate attention must have shape [sample, candidate]")
    row_maximum = values.max(axis=1, keepdims=True)
    normalized = np.zeros_like(values, dtype=np.float32)
    np.divide(
        values,
        row_maximum,
        out=normalized,
        where=row_maximum > np.finfo(np.float32).eps,
    )
    return normalized


def _plot_candidate_attention(
    values: np.ndarray,
    candidate_keys: tuple[str, ...],
    labels: list[str],
    label_indices: list[int],
    prediction_indices: list[int],
    path: Path,
) -> None:
    fig_width = max(14.0, len(candidate_keys) * 0.48)
    fig_height = max(8.0, len(labels) * 0.32)
    fig, ax = create_figure((fig_width, fig_height))
    image = ax.imshow(
        values,
        cmap=_attention_cmap(),
        aspect="auto",
        norm=_attention_color_norm(),
    )
    ax.set_xticks(np.arange(len(candidate_keys)), candidate_keys, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Candidate action (CLS attention; each row normalized to max=1.0)")
    ax.set_ylabel("Opening decision step / recorded label")
    ax.set_title("Opener candidate attention — final Transformer layer")

    for row, (label_index, prediction_index) in enumerate(zip(label_indices, prediction_indices)):
        ax.add_patch(
            Rectangle(
                (label_index - 0.45, row - 0.45),
                0.9,
                0.9,
                fill=False,
                edgecolor="#ffffff",
                linewidth=1.4,
            )
        )
        ax.scatter(
            prediction_index,
            row,
            marker="x",
            s=42,
            color="#00bcd4",
            linewidths=1.5,
        )

    ax.scatter([], [], marker="s", facecolors="none", edgecolors="#ffffff", label="recorded label")
    ax.scatter([], [], marker="x", color="#00bcd4", label="model top-1")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        borderaxespad=0.0,
    )
    fig.colorbar(
        image,
        ax=ax,
        label="row-relative attention (row maximum = 1.0)",
        pad=0.04,
    )
    save_figure(fig, path)


def _plot_layer_attention(
    values: np.ndarray,
    candidate_keys: tuple[str, ...],
    path: Path,
) -> None:
    fig, ax = create_figure((max(14.0, len(candidate_keys) * 0.48), 5.5))
    image = ax.imshow(
        values,
        cmap=_attention_cmap(),
        aspect="auto",
        norm=_attention_color_norm(),
    )
    ax.set_xticks(np.arange(len(candidate_keys)), candidate_keys, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(values.shape[0]), [f"Layer {index + 1}" for index in range(values.shape[0])])
    ax.set_xlabel("Candidate action")
    ax.set_ylabel("Transformer layer")
    ax.set_title("Mean row-relative opener candidate attention by Transformer layer")
    fig.colorbar(image, ax=ax, label="row-relative attention (row maximum = 1.0)")
    save_figure(fig, path)

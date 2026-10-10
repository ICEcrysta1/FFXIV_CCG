"""完整因果上下文的逐层与逐 head 注意力矩阵。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

from common.policy.model.input_encoder import ROLE_SCENE
from training import TrainingCollator

from ..common import (
    HEATMAP_CELL_SIZE,
    AnalysisContext,
    create_grid_figure,
    hide_empty_tiles,
    save_figure,
)


def plot_standard_attention_outputs(
    context: AnalysisContext,
    *,
    steps: int = 28,
) -> tuple[Path, Path]:
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
    batch = context.encode_batch(collator([representative]))
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
        # 失效 scene 与 batch padding 都不是模型上下文；同步裁去 query/key，
        # 保留真实顺序与权重，避免灰色空位占据热图。
        valid_positions = np.flatnonzero(~padding_values)
        attention_arrays = tuple(
            attention[:, valid_positions, :][:, :, valid_positions]
            for attention in attention_arrays
        )
        blocked = blocked[np.ix_(valid_positions, valid_positions)]
        role_ids = role_ids[valid_positions]
    del batch, trace, padding
    if context.device.type == "cuda":
        torch.cuda.empty_cache()

    matrix_path = context.output_dir / "08_attention_matrix_by_layer.png"
    heads_path = context.output_dir / "09_attention_matrix_last_layer_heads.png"
    _plot_attention_matrix_grid(attention_arrays, blocked, role_ids, matrix_path)
    _plot_attention_head_grid(attention_arrays[-1], blocked, role_ids, heads_path)
    return matrix_path, heads_path


def _sample_token_count(sample: object) -> int:
    """按 dataset 的 raw 字段估算样本 token 数，供代表样本选择使用。"""
    if not isinstance(sample, Mapping):
        raise TypeError(
            "expected mapping sample for representative selection, "
            f"got {type(sample).__name__}"
        )
    total = 1  # 显式最新状态 token。
    for key in ("scene_abs_values",):
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
            total += 2 * len(history_values)
        except TypeError as exc:
            raise TypeError(
                "sample field 'history_skill_ids' must be sized for representative selection"
            ) from exc
    else:
        # 增量 history bank 只保留 bank 游标和 history_length，模型装配前没有显式历史张量。
        history_length = sample.get("history_length")
        if history_length is not None:
            try:
                total += 2 * int(history_length)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError(
                    "sample field 'history_length' must be an integer for representative selection"
                ) from exc
    return total


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


def _context_spans(role_ids: np.ndarray) -> list[tuple[int, int, str]]:
    """按场景、交错历史、当前状态分区，不按每次 state/skill 切换分区。"""
    if role_ids.size == 0:
        return []
    spans: list[tuple[int, int, str]] = []
    non_scene = np.flatnonzero(role_ids != ROLE_SCENE)
    scene_end = int(non_scene[0]) if non_scene.size else role_ids.size
    if scene_end:
        spans.append((0, scene_end, f"scene\n{scene_end} tokens"))
    if scene_end < role_ids.size:
        history_end = role_ids.size - 1
        if scene_end < history_end:
            history_count = history_end - scene_end
            spans.append((scene_end, history_end, f"history: state / skill\n{history_count} tokens"))
        spans.append((history_end, role_ids.size, "current\nstate"))
    return spans


def _draw_attention_structure(ax, role_ids: np.ndarray) -> None:
    """仅标出大区段边界和因果对角线，避免交错历史形成密集网格。"""
    spans = _context_spans(role_ids)
    ax.grid(False, which="both")
    centers = []
    labels = []
    for start, end, label in spans:
        centers.append((start + end - 1) / 2.0)
        labels.append(label)
        if end < role_ids.size:
            boundary = end - 0.5
            ax.axvline(boundary, color="white", linewidth=0.6, alpha=0.5)
            ax.axhline(boundary, color="white", linewidth=0.6, alpha=0.5)
    if centers:
        ax.set_xticks(centers, labels, fontsize=8)
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
        image = ax.imshow(display, cmap=cmap, norm=color_norm, aspect="equal", interpolation="nearest")
        _draw_attention_structure(ax, role_ids)
        ax.set_title(f"Layer {index + 1} · head mean")
        ax.set_xlabel("Key tokens (valid sequence order)")
        ax.set_ylabel("Query tokens (valid sequence order)")
    hide_empty_tiles(axes, len(matrices))
    assert image is not None
    fig.suptitle(
        "Full attention matrix by layer — valid tokens only\n"
        "Gray = blocked by mask; each query row normalized independently"
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
        image = ax.imshow(display, cmap=cmap, norm=color_norm, aspect="equal", interpolation="nearest")
        _draw_attention_structure(ax, role_ids)
        ax.set_title(f"Last layer · head {index + 1}")
        ax.set_xlabel("Key tokens (valid sequence order)")
        ax.set_ylabel("Query tokens (valid sequence order)")
    hide_empty_tiles(axes, head_count)
    assert image is not None
    fig.suptitle(
        "Full attention matrix by head — valid tokens only\n"
        "Gray = blocked by mask; each query row normalized independently"
    )
    fig.colorbar(
        image,
        ax=list(axes.flat),
        label="row-relative attention (row maximum = 1.0)",
        shrink=0.82,
    )
    save_figure(fig, path)

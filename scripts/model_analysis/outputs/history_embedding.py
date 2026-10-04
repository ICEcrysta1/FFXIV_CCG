"""独立历史技能与状态 embedding 的 PCA 视图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.torch_runtime import move_batch
from training import TrainingCollator

from ..common import (
    ATTENTION_CMAP_NAME,
    AnalysisContext,
    create_figure,
    pca_projection,
    save_figure,
    skill_name_by_vocab_id,
)


def plot_history_embeddings(
    context: AnalysisContext,
    *,
    batch_size: int = 16,
    max_points: int = 6000,
) -> tuple[Path, Path]:
    """分别投影真实历史技能和状态，两张图各自拟合 PCA。"""
    if batch_size <= 0:
        raise ValueError("history embedding batch size must be positive")
    if max_points < 2:
        raise ValueError("history embedding max_points must be at least 2")
    samples = [context.dataset[index] for index in range(len(context.dataset))]
    collator = TrainingCollator()
    embedding_rows: dict[str, list[np.ndarray]] = {"skill": [], "state": []}
    skill_id_rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = move_batch(collator(samples[start : start + batch_size]), context.device)
            with context.autocast():
                embeddings = context.model.input_encoder.embed_history(batch)
            valid = batch["history_mask"]
            if bool(valid.any()):
                for kind in embedding_rows:
                    embedding_rows[kind].append(embeddings[kind][valid].float().cpu().numpy())
                skill_id_rows.append(batch["history_skill_ids"][valid].detach().cpu().numpy())
            del batch, embeddings

    paths = tuple(
        context.output_dir / f"05_history_{kind}_embedding_pca.png"
        for kind in embedding_rows
    )
    if not skill_id_rows:
        for kind, path in zip(embedding_rows, paths):
            fig, ax = create_figure((15.0, 10.0))
            ax.text(0.5, 0.5, f"No valid history {kind} tokens in this context", ha="center", va="center")
            ax.set_axis_off()
            save_figure(fig, path)
        return paths

    skill_ids = np.concatenate(skill_id_rows).astype(np.int64, copy=False)
    indices = np.linspace(0, len(skill_ids) - 1, min(max_points, len(skill_ids)), dtype=int)
    skill_ids = skill_ids[indices]
    names = skill_name_by_vocab_id(context)
    for kind, path in zip(embedding_rows, paths):
        values = np.concatenate(embedding_rows[kind])[indices]
        _plot_history_embedding(values, skill_ids, names, kind, path)
    return paths


def _plot_history_embedding(values, skill_ids, names, kind: str, path: Path) -> None:
    """按同一历史条目的技能身份染色，不将技能与状态融合成一个点。"""
    if len(values) < 2:
        coordinates, explained = np.zeros((len(values), 2)), np.zeros(2)
    else:
        components = min(2, len(values), values.shape[1])
        coordinates, explained = pca_projection(values, components)
        coordinates = np.pad(coordinates, ((0, 0), (0, 2 - components)))
        explained = np.pad(explained, (0, 2 - components))
    unique_ids = sorted(int(value) for value in np.unique(skill_ids) if value > 0)
    color_map = plt.get_cmap(ATTENTION_CMAP_NAME, max(1, len(unique_ids)))
    fig, ax = create_figure((15.0, 10.0))
    for color_index, skill_id in enumerate(unique_ids):
        mask = skill_ids == skill_id
        label = names.get(skill_id, f"未知技能({skill_id})")
        ax.scatter(
            coordinates[mask, 0], coordinates[mask, 1], s=12, alpha=0.42,
            color=color_map(color_index), label=label, linewidths=0,
        )
        ax.annotate(
            label, coordinates[mask].mean(axis=0), xytext=(4, 4),
            textcoords="offset points", fontsize=7, color=color_map(color_index),
        )
    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax.set_title(f"History {kind.title()} Embedding PCA ({values.shape[1]}D; {len(values)} real tokens)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, borderaxespad=0.0)
    save_figure(fig, path)

"""真实候选技能-状态 pair embedding 的 PCA 投影。"""

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


def plot_pair_embedding(
    context: AnalysisContext,
    *,
    batch_size: int = 16,
    max_points: int = 6000,
) -> Path:
    """使用真实候选状态，绘制 pair fusion 输出的 PCA。"""
    if batch_size <= 0:
        raise ValueError("pair embedding batch size must be positive")
    if max_points < 2:
        raise ValueError("pair embedding max_points must be at least 2")
    if len(context.dataset) == 0:
        raise ValueError("dataset is empty, cannot plot pair embeddings")

    samples = [context.dataset[index] for index in range(len(context.dataset))]
    collator = TrainingCollator()
    embedding_rows: list[np.ndarray] = []
    skill_id_rows: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = move_batch(
                collator(samples[start : start + batch_size]),
                context.device,
            )
            with context.autocast():
                pair_embeddings = context.model.input_encoder.embed_pairs(batch)["candidate"]
            embedding_rows.append(pair_embeddings.float().cpu().numpy().reshape(-1, pair_embeddings.shape[-1]))
            skill_id_rows.append(batch["candidate_skill_ids"].detach().cpu().numpy().reshape(-1))
            del batch, pair_embeddings

    embeddings = np.concatenate(embedding_rows, axis=0)
    skill_ids = np.concatenate(skill_id_rows, axis=0).astype(np.int64, copy=False)
    if len(embeddings) > max_points:
        indices = np.linspace(0, len(embeddings) - 1, max_points, dtype=int)
        embeddings = embeddings[indices]
        skill_ids = skill_ids[indices]

    coordinates, explained = pca_projection(embeddings, 2)
    unique_skill_ids = sorted(int(value) for value in np.unique(skill_ids) if int(value) > 0)
    color_map = plt.get_cmap(ATTENTION_CMAP_NAME, max(1, len(unique_skill_ids)))
    skill_names = skill_name_by_vocab_id(context)
    labels = {skill_id: skill_names[skill_id] for skill_id in unique_skill_ids}

    fig, ax = create_figure((15.0, 10.0))
    for color_index, skill_id in enumerate(unique_skill_ids):
        mask = skill_ids == skill_id
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=12,
            alpha=0.42,
            color=color_map(color_index),
            label=labels[skill_id],
            linewidths=0,
        )
        centroid = coordinates[mask].mean(axis=0)
        ax.annotate(
            labels[skill_id],
            centroid,
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color=color_map(color_index),
        )

    pair_dim = int(embeddings.shape[1])
    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax.set_title(
        "Candidate Pair Embedding PCA "
        f"({pair_dim}D skill + state fusion; {len(embeddings)} real candidate pairs)"
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        ncol=1,
        fontsize=8,
        borderaxespad=0.0,
    )
    path = context.output_dir / "05_pair_embedding_pca.png"
    save_figure(fig, path)
    return path

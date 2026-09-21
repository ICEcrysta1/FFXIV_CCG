"""技能 embedding 的 PCA 投影。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ..common import (
    ATTENTION_CMAP_NAME,
    AnalysisContext,
    create_figure,
    pca_projection,
    save_figure,
    skill_name_by_vocab_id,
)


def plot_skill_embedding(context: AnalysisContext) -> Path:
    """输出真实技能 embedding 的 PCA，并标注可解析的 action key。"""
    path = context.output_dir / "05_skill_embedding_pca.png"
    weights = context.model.input_encoder.skill_embed.weight.detach().float().cpu().numpy()[1:]
    coordinates, explained = pca_projection(weights, 2)
    labels = _skill_labels(context)

    fig, ax = create_figure((12.0, 9.0))
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=35,
        color=plt.get_cmap(ATTENTION_CMAP_NAME)(0.72),
        alpha=0.8,
    )
    for index, (x_coord, y_coord) in enumerate(coordinates):
        ax.annotate(labels[index], (x_coord, y_coord), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax.set_title(f"Raw Skill ID Embedding PCA ({weights.shape[1]}D)")
    save_figure(fig, path)
    return path


def _skill_labels(context: AnalysisContext) -> list[str]:
    labels = skill_name_by_vocab_id(context)
    return [labels[vocab_id] for vocab_id in range(1, context.vocab.size())]

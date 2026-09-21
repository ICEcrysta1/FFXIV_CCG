"""Hidden 分布与按 token role 的范数统计图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..common import (
    ATTENTION_CMAP_NAME,
    ROLE_COLORS,
    ROLE_NAMES,
    SUPTITLE_FONTSIZE,
    AnalysisContext,
    create_figure,
    create_grid_figure,
    hide_empty_tiles,
    save_figure,
)


def plot_hidden_statistics(context: AnalysisContext) -> tuple[Path, Path]:
    """输出 hidden 值分布和 role 范数分布。"""
    layer_count = len(context.layer_vectors)
    distribution_path = context.output_dir / "01_layer_hidden_distribution.png"
    norm_path = context.output_dir / "04_layer_norm_by_role.png"

    fig, axes = create_grid_figure(layer_count)
    for index, vectors in enumerate(context.layer_vectors):
        ax = axes.flat[index]
        values = vectors.reshape(-1)
        attention_colors = plt.get_cmap(ATTENTION_CMAP_NAME)
        ax.hist(values, bins=80, color=attention_colors(0.62), alpha=0.85, density=True)
        ax.axvline(
            float(values.mean()),
            color=attention_colors(0.92),
            linestyle="--",
            linewidth=1.2,
            label="mean",
        )
        ax.axvline(
            float(np.median(values)),
            color=attention_colors(0.38),
            linestyle=":",
            linewidth=1.2,
            label="median",
        )
        ax.set_title(f"Layer {index + 1} | mean={values.mean():.3f}, std={values.std():.3f}")
        ax.set_xlabel("hidden value")
        ax.set_ylabel("density")
        ax.legend()
    hide_empty_tiles(axes, layer_count)
    fig.suptitle("Hidden Value Distribution by Transformer Layer", fontsize=SUPTITLE_FONTSIZE)
    save_figure(fig, distribution_path)

    fig, ax = create_figure((12.0, 6.0))
    positions = np.arange(layer_count)
    width = 0.13
    for role_id, role_name in ROLE_NAMES.items():
        medians = []
        lower = []
        upper = []
        for vectors, roles in zip(context.layer_vectors, context.layer_roles):
            norms = np.linalg.norm(vectors[roles == role_id], axis=1)
            if len(norms) == 0:
                medians.append(np.nan)
                lower.append(np.nan)
                upper.append(np.nan)
            else:
                medians.append(float(np.median(norms)))
                lower.append(float(np.percentile(norms, 25)))
                upper.append(float(np.percentile(norms, 75)))
        offset = (role_id - 2.5) * width
        ax.errorbar(
            positions + offset,
            medians,
            yerr=[np.array(medians) - np.array(lower), np.array(upper) - np.array(medians)],
            fmt="o-",
            capsize=3,
            linewidth=1.2,
            markersize=4,
            color=ROLE_COLORS[role_id],
            label=role_name,
        )
    ax.set_xticks(positions, [f"Layer {index + 1}" for index in positions])
    ax.set_ylabel("L2 norm, median with IQR")
    ax.set_title("Hidden Norm by Token Role")
    ax.legend(ncol=3)
    save_figure(fig, norm_path)
    return distribution_path, norm_path

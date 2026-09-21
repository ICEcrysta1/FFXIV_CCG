"""逐层 hidden 的 2D/3D PCA 科研图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from common.policy.model.input_encoder import ROLE_CANDIDATE

from ..common import (
    ANALYSIS_FEATURES,
    ATTENTION_CMAP_NAME,
    ROLE_COLORS,
    ROLE_NAMES,
    SUPTITLE_FONTSIZE,
    AnalysisContext,
    create_grid_figure,
    hide_empty_tiles,
    pca_projection,
    save_figure,
)
from ..token_metadata import ANALYSIS_MP_MAX


FEATURE_LABELS = {
    "fight_id": "fight_id",
    "step_index": "step index",
    "candidate_index": "candidate index",
    "skill_id": "skill_id",
    "legal": "legal / illegal",
    "invalid_reason": "invalid_reason",
    "elemental_state": "AF/UI",
    "mp_bucket": "MP",
    "label_rank": "label rank",
    "model_logit": "model logit",
}
NUMERIC_FEATURES = frozenset(
    {"step_index", "candidate_index", "skill_id", "mp_bucket", "label_rank", "model_logit"}
)
# 3D 子图需要额外高度容纳 z 轴标签与刻度，其余按统一网格基准。
PCA_3D_CELL_SIZE = (5.6, 5.6)
# 决策变量染色图的 suptitle 含逐层最强轴摘要，行数多时会撑满全宽，保留更小字号。
FEATURE_SUPTITLE_FONTSIZE = 13


def plot_layer_pca(context: AnalysisContext) -> list[Path]:
    """输出角色 PCA、3D PCA，以及按决策变量染色的多组 2D PCA。"""
    layer_count = len(context.layer_vectors)
    path_2d = context.output_dir / "02_layers_pca_2d.png"
    path_3d = context.output_dir / "03_layers_pca_3d.png"

    fig_2d, axes_2d = create_grid_figure(layer_count)
    fig_3d, axes_3d = create_grid_figure(
        layer_count,
        cell_size=PCA_3D_CELL_SIZE,
        projection="3d",
    )

    layer_projections, candidate_projections = _build_pca_cache(context)
    for layer_index, roles in enumerate(context.layer_roles):
        coordinates_3d, explained_3d = layer_projections[layer_index]
        coordinates_2d = coordinates_3d[:, :2]
        explained_2d = explained_3d[:2]
        ax_2d = axes_2d.flat[layer_index]
        ax_3d = axes_3d.flat[layer_index]
        _scatter_by_role(ax_2d, coordinates_2d, roles)
        _scatter_by_role(ax_3d, coordinates_3d, roles)
        ax_2d.set_title(
            f"Layer {layer_index + 1} | PC1={explained_2d[0]:.1%}, PC2={explained_2d[1]:.1%}"
        )
        ax_2d.set_xlabel("PC1")
        ax_2d.set_ylabel("PC2")
        ax_3d.set_title(
            f"Layer {layer_index + 1} | PC1={explained_3d[0]:.1%}, "
            f"PC2={explained_3d[1]:.1%}, PC3={explained_3d[2]:.1%}"
        )
        ax_3d.set_xlabel("PC1")
        ax_3d.set_ylabel("PC2")
        ax_3d.set_zlabel("PC3")
        ax_3d.view_init(elev=24, azim=-58)

    hide_empty_tiles(axes_2d, layer_count)
    hide_empty_tiles(axes_3d, layer_count)
    handles, labels = _legend_handles()
    fig_2d.legend(handles, labels, loc="outside lower center", ncol=3, frameon=True)
    fig_3d.legend(handles, labels, loc="outside lower center", ncol=3, frameon=True)
    fig_2d.suptitle(
        "Transformer Hidden Representation PCA — 2D",
        fontsize=SUPTITLE_FONTSIZE,
    )
    fig_3d.suptitle(
        "Transformer Hidden Representation PCA — 3D",
        fontsize=SUPTITLE_FONTSIZE,
    )
    save_figure(fig_2d, path_2d)
    save_figure(fig_3d, path_3d)
    feature_paths = [
        _plot_feature_pca_2d(context, feature, candidate_projections)
        for feature in ANALYSIS_FEATURES
    ]
    return [path_2d, path_3d, *feature_paths]


def _build_pca_cache(
    context: AnalysisContext,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray]],
]:
    """一次计算逐层和候选 token 的 PCA，供所有图复用。"""
    layer_projections = []
    candidate_projections = []
    for vectors, roles in zip(context.layer_vectors, context.layer_roles):
        # hidden 已经是 CPU numpy；留在 CPU 做低内存协方差分解，避免 CUDA SVD
        # 工作区与模型、attention trace 争抢显存。
        layer_projections.append(pca_projection(vectors, 3))
        candidate_projections.append(
            pca_projection(
                vectors[roles == ROLE_CANDIDATE],
                2,
            )
        )
    return layer_projections, candidate_projections


def _plot_feature_pca_2d(
    context: AnalysisContext,
    feature: str,
    candidate_projections: list[tuple[np.ndarray, np.ndarray]],
) -> Path:
    """只显示 candidate skill token，并按指定决策变量染色。"""
    layer_count = len(context.layer_vectors)
    path = context.output_dir / f"02_layers_pca_2d_{feature}.png"
    label = FEATURE_LABELS.get(feature, feature)
    numeric = feature in NUMERIC_FEATURES

    all_values = [
        metadata[feature][roles == ROLE_CANDIDATE]
        for metadata, roles in zip(context.layer_metadata, context.layer_roles)
    ]
    if numeric:
        numeric_parts = [values[np.isfinite(values)] for values in all_values if len(values)]
        numeric_values = np.concatenate(numeric_parts, axis=0) if numeric_parts else np.array([])
        if feature == "mp_bucket":
            vmin, vmax = 0.0, ANALYSIS_MP_MAX
        else:
            vmin = float(numeric_values.min()) if len(numeric_values) else 0.0
            vmax = float(numeric_values.max()) if len(numeric_values) else 1.0
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        normalizer = Normalize(vmin=vmin, vmax=vmax)
        color_map = plt.get_cmap(ATTENTION_CMAP_NAME)
        categories: list[str] = []
    else:
        categories = sorted(
            {
                str(value)
                for values in all_values
                for value in values
            }
        )
        category_index = {value: index for index, value in enumerate(categories)}
        normalizer = Normalize(vmin=0, vmax=max(1, len(categories) - 1))
        color_map = plt.get_cmap(ATTENTION_CMAP_NAME)

    fig, axes = create_grid_figure(layer_count)
    strongest_axes: list[str] = []
    strongest_scores: list[float] = []
    for layer_index, (roles, metadata) in enumerate(
        zip(context.layer_roles, context.layer_metadata)
    ):
        candidate_mask = roles == ROLE_CANDIDATE
        raw_values = metadata[feature][candidate_mask]
        coordinates, explained = candidate_projections[layer_index]
        axis_name, association = _strongest_pca_axis(coordinates, raw_values, numeric)
        strongest_axes.append(axis_name)
        strongest_scores.append(association)
        ax = axes.flat[layer_index]
        if numeric:
            colors = color_map(normalizer(raw_values.astype(float)))
        else:
            color_values = np.array(
                [category_index[str(value)] for value in raw_values],
                dtype=float,
            )
            colors = color_map(normalizer(color_values))
        ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=colors,
            s=12,
            alpha=0.58,
            linewidths=0,
        )
        ax.set_title(
            f"Layer {layer_index + 1} | strongest={axis_name} "
            f"score={association:.3f} | PC1={explained[0]:.1%}, PC2={explained[1]:.1%}"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    hide_empty_tiles(axes, layer_count)

    if not numeric and len(categories) <= 12:
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=category,
                markerfacecolor=color_map(normalizer(index)),
                markersize=6,
            )
            for index, category in enumerate(categories)
        ]
        fig.legend(
            handles,
            categories,
            loc="outside lower center",
            ncol=min(4, len(categories)),
        )

    axis_summary = ", ".join(
        f"{axis}:{score:.2f}" for axis, score in zip(strongest_axes, strongest_scores)
    )
    fig.suptitle(
        f"Candidate Skill PCA 2D — color={label}\n"
        f"strongest PCA axis by layer: {axis_summary}",
        fontsize=FEATURE_SUPTITLE_FONTSIZE,
    )
    if numeric:
        scalar_map = plt.cm.ScalarMappable(norm=normalizer, cmap=color_map)
        scalar_map.set_array([])
        fig.colorbar(scalar_map, ax=list(axes.flat), label=label, shrink=0.82)
    save_figure(fig, path)
    return path


def _strongest_pca_axis(
    coordinates: np.ndarray,
    values: np.ndarray,
    numeric: bool,
) -> tuple[str, float]:
    """找与染色变量关联最强的 PC1/PC2 轴。"""
    scores = []
    if numeric:
        numeric_values = values.astype(float)
        for axis_index in range(2):
            axis_values = coordinates[:, axis_index]
            if np.std(axis_values) <= 1e-12 or np.std(numeric_values) <= 1e-12:
                scores.append(0.0)
            else:
                scores.append(abs(float(np.corrcoef(axis_values, numeric_values)[0, 1])))
    else:
        categories = np.asarray([str(value) for value in values], dtype=object)
        for axis_index in range(2):
            axis_values = coordinates[:, axis_index]
            total_variance = float(np.var(axis_values))
            if total_variance <= 1e-12:
                scores.append(0.0)
                continue
            weighted_between = 0.0
            for category in np.unique(categories):
                group = axis_values[categories == category]
                weighted_between += len(group) * float((group.mean() - axis_values.mean()) ** 2)
            scores.append(weighted_between / (len(axis_values) * total_variance))
    strongest_index = int(np.argmax(scores))
    return f"PC{strongest_index + 1}", float(scores[strongest_index])


def _scatter_by_role(ax, coordinates: np.ndarray, roles: np.ndarray) -> None:
    for role_id, role_name in ROLE_NAMES.items():
        mask = roles == role_id
        if not np.any(mask):
            continue
        kwargs = {
            "c": ROLE_COLORS[role_id],
            "label": role_name,
            "s": 10,
            "alpha": 0.5,
            "depthshade": False,
        }
        if hasattr(ax, "zaxis"):
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], coordinates[mask, 2], **kwargs)
        else:
            kwargs.pop("depthshade", None)
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], **kwargs)


def _legend_handles():
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="w", label=name, markerfacecolor=ROLE_COLORS[role], markersize=7)
        for role, name in ROLE_NAMES.items()
    ]
    return handles, list(ROLE_NAMES.values())

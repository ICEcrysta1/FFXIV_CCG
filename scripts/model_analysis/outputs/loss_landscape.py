"""逐 Transformer 层的高精度 loss landscape。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from time import perf_counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common.policy.model import CandidateTransformerModel
from common.policy.model.repetition import build_repetition_penalty_mask
from common.policy.model.split_encoder import run_split_layer
from common.torch_runtime import move_batch
from training import TrainingCollator

from ..common import (
    SUPTITLE_FONTSIZE,
    ModelAnalysisContext,
    create_grid_figure,
    hide_empty_tiles,
    save_figure,
)


LOSS_LANDSCAPE_CMAP = "viridis"
LOSS_LANDSCAPE_DPI = 300
LOSS_LANDSCAPE_CELL_SIZE_2D = (5.6, 4.8)
LOSS_LANDSCAPE_CELL_SIZE_3D = (5.8, 5.4)


@dataclass(frozen=True)
class LayerDirections:
    """一层参数空间中的两条逐 filter 归一化正交方向。"""

    parameter_names: tuple[str, ...]
    parameters: tuple[nn.Parameter, ...]
    base_values: tuple[torch.Tensor, ...]
    direction_x: tuple[torch.Tensor, ...]
    direction_y: tuple[torch.Tensor, ...]
    x_norm: float
    y_norm: float
    cosine: float


def plot_loss_landscape(
    context: ModelAnalysisContext,
    *,
    resolution: int = 31,
    radius: float = 0.5,
    max_samples: int = 0,
    batch_size: int = 16,
    seed: int = 3407,
) -> list[Path]:
    """导出每个 Transformer 层的二维/三维损失地图和原始 float64 网格。

    每个子图仅扰动对应 encoder layer 的矩阵参数。两条方向在每个输出
    filter 内分别按 checkpoint 权重范数归一化并正交化；偏置和 LayerNorm
    这类一维参数保持不变，避免尺度很小的参数主导坐标轴。
    """
    _validate_options(
        resolution=resolution,
        radius=radius,
        max_samples=max_samples,
        batch_size=batch_size,
    )
    layers = tuple(context.model.encoder.layers)
    if not layers:
        raise ValueError("loss landscape requires at least one Transformer layer")

    sample_count = (
        len(context.dataset)
        if max_samples <= 0
        else min(max_samples, len(context.dataset))
    )
    if sample_count <= 0:
        raise ValueError("dataset is empty, cannot compute loss landscape")
    samples = [context.dataset[index] for index in range(sample_count)]
    batches = _prepare_batches(samples, batch_size=batch_size)
    coordinates = np.linspace(-radius, radius, resolution, dtype=np.float64)
    center_index = resolution // 2
    evaluated_points = len(layers) * (resolution * resolution - 1) + 1
    print(
        "[loss-landscape] "
        f"layers={len(layers)}, grid={resolution}x{resolution}, "
        f"samples={sample_count}, parameter points={evaluated_points}"
    )

    started = perf_counter()
    model = context.model
    was_training = model.training
    model.eval()
    try:
        baseline_loss = _mean_cross_entropy(context, batches)
        layer_losses: list[np.ndarray] = []
        direction_metadata: list[dict[str, object]] = []
        for layer_index, layer in enumerate(layers):
            print(f"[loss-landscape] evaluating layer {layer_index + 1}/{len(layers)}")
            directions = _build_layer_directions(layer, seed=seed + layer_index)
            losses = _evaluate_layer_grid(
                context,
                batches,
                directions,
                coordinates=coordinates,
                baseline_loss=baseline_loss,
                center_index=center_index,
            )
            layer_losses.append(losses)
            # 每完成一层保留数值快照，长任务中断时仍能检查已完成结果。
            context.output_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                context.output_dir / "11_loss_landscape_partial.npz",
                coordinates=coordinates,
                losses=np.stack(layer_losses),
                baseline_loss=np.asarray(baseline_loss, dtype=np.float64),
            )
            direction_metadata.append(
                {
                    "layer": layer_index + 1,
                    "parameter_names": list(directions.parameter_names),
                    "direction_x_l2_norm": directions.x_norm,
                    "direction_y_l2_norm": directions.y_norm,
                    "direction_cosine": directions.cosine,
                }
            )
    finally:
        model.train(was_training)

    loss_grid = np.stack(layer_losses, axis=0).astype(np.float64, copy=False)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    surface_path = context.output_dir / "11_loss_landscape_3d.png"
    contour_path = context.output_dir / "11_loss_landscape_contour.png"
    values_path = context.output_dir / "11_loss_landscape_values.npz"
    metadata_path = context.output_dir / "11_loss_landscape_metadata.json"

    evaluation_seconds = perf_counter() - started
    _plot_surface_grid(loss_grid, coordinates, baseline_loss, surface_path)
    _plot_contour_grid(loss_grid, coordinates, baseline_loss, contour_path)
    np.savez_compressed(
        values_path,
        coordinates=coordinates,
        losses=loss_grid,
        baseline_loss=np.asarray(baseline_loss, dtype=np.float64),
    )
    metadata = {
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": perf_counter() - started,
        "prefix_cache": _supports_prefix_cache(model),
        "batch_schedule": "one device transfer per batch per layer; GPU grid accumulation",
        "checkpoint": str(getattr(context, "checkpoint_path", "")),
        "source": str(getattr(context, "source_path", "")),
        "method": "per-layer filter-normalized orthogonal parameter directions",
        "loss": "mean cross entropy",
        "precision": context.precision,
        "loss_accumulation_dtype": "float64",
        "model_forward_dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "color_scale": "normalized independently within each layer",
        "resolution": resolution,
        "radius": radius,
        "seed": seed,
        "sample_count": sample_count,
        "batch_size": batch_size,
        "baseline_loss": baseline_loss,
        "layers": direction_metadata,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return [surface_path, contour_path, values_path, metadata_path]


def _validate_options(
    *,
    resolution: int,
    radius: float,
    max_samples: int,
    batch_size: int,
) -> None:
    if resolution < 3 or resolution % 2 == 0:
        raise ValueError("loss landscape resolution must be an odd integer >= 3")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("loss landscape radius must be finite and > 0")
    if batch_size <= 0:
        raise ValueError("loss landscape batch size must be positive")


def _prepare_batches(
    samples: list[dict[str, object]],
    *,
    batch_size: int,
) -> tuple[dict[str, object], ...]:
    collator = TrainingCollator()
    return tuple(
        collator(samples[start : start + batch_size])
        for start in range(0, len(samples), batch_size)
    )


def _build_layer_directions(layer: nn.Module, *, seed: int) -> LayerDirections:
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in layer.named_parameters()
        if parameter.requires_grad and parameter.ndim > 1
    )
    if not named_parameters:
        raise ValueError("Transformer layer has no matrix parameters for loss landscape")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    names: list[str] = []
    parameters: list[nn.Parameter] = []
    base_values: list[torch.Tensor] = []
    direction_x: list[torch.Tensor] = []
    direction_y: list[torch.Tensor] = []
    for name, parameter in named_parameters:
        base = parameter.detach().clone()
        x_cpu, y_cpu = _filter_normalized_orthogonal_pair(base, generator=generator)
        names.append(name)
        parameters.append(parameter)
        base_values.append(base)
        direction_x.append(x_cpu.to(device=parameter.device, dtype=parameter.dtype))
        direction_y.append(y_cpu.to(device=parameter.device, dtype=parameter.dtype))

    x_norm, y_norm, cosine = _direction_statistics(direction_x, direction_y)
    return LayerDirections(
        parameter_names=tuple(names),
        parameters=tuple(parameters),
        base_values=tuple(base_values),
        direction_x=tuple(direction_x),
        direction_y=tuple(direction_y),
        x_norm=x_norm,
        y_norm=y_norm,
        cosine=cosine,
    )


def _filter_normalized_orthogonal_pair(
    weight: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在每个第一维 filter 内生成等权重范数且互相正交的方向。"""
    weight_cpu = weight.detach().to(device="cpu", dtype=torch.float64)
    flat_weight = weight_cpu.reshape(weight_cpu.shape[0], -1)
    raw_x = torch.randn(flat_weight.shape, generator=generator, dtype=torch.float64)
    raw_y = torch.randn(flat_weight.shape, generator=generator, dtype=torch.float64)
    eps = torch.finfo(torch.float64).eps

    weight_norm = torch.linalg.vector_norm(flat_weight, dim=1, keepdim=True)
    x_norm = torch.linalg.vector_norm(raw_x, dim=1, keepdim=True).clamp_min(eps)
    normalized_x = raw_x * (weight_norm / x_norm)

    x_norm_squared = normalized_x.square().sum(dim=1, keepdim=True)
    projection = (raw_y * normalized_x).sum(
        dim=1,
        keepdim=True,
    ) / x_norm_squared.clamp_min(eps)
    orthogonal_y = raw_y - projection * normalized_x
    y_norm = torch.linalg.vector_norm(orthogonal_y, dim=1, keepdim=True).clamp_min(eps)
    normalized_y = orthogonal_y * (weight_norm / y_norm)

    zero_filter = weight_norm <= eps
    normalized_x = torch.where(zero_filter, torch.zeros_like(normalized_x), normalized_x)
    normalized_y = torch.where(zero_filter, torch.zeros_like(normalized_y), normalized_y)
    return normalized_x.reshape(weight.shape), normalized_y.reshape(weight.shape)


def _direction_statistics(
    direction_x: list[torch.Tensor],
    direction_y: list[torch.Tensor],
) -> tuple[float, float, float]:
    x_squared = sum(
        float(value.detach().double().square().sum().cpu())
        for value in direction_x
    )
    y_squared = sum(
        float(value.detach().double().square().sum().cpu())
        for value in direction_y
    )
    dot = sum(
        float((x.detach().double() * y.detach().double()).sum().cpu())
        for x, y in zip(direction_x, direction_y)
    )
    x_norm = float(np.sqrt(x_squared))
    y_norm = float(np.sqrt(y_squared))
    denominator = max(x_norm * y_norm, np.finfo(np.float64).eps)
    return x_norm, y_norm, dot / denominator


def _supports_prefix_cache(model: nn.Module) -> bool:
    """仅对普通 split encoder 复用前层；Full AttnRes 保留完整前向语义。"""
    return (
        isinstance(model, CandidateTransformerModel)
        and getattr(model.encoder, "attention_residual", None) is None
    )


class _LayerForward:
    """分析专用的单 batch 前层缓存，不修改模型或持有全数据集 GPU 激活。"""

    def __init__(self, context, batch, layer_index: int):
        self.context = context
        self.batch = batch
        self.layer_index = layer_index
        self.encoded = None
        if _supports_prefix_cache(context.model):
            # 重复动作惩罚只依赖固定输入，每批只构建并传输一次。
            self.batch = dict(batch)
            self.batch["repetition_penalty_mask"] = build_repetition_penalty_mask(
                batch, context.model.repetition, device=context.device,
            )
            self.batch["repetition_penalty_config"] = context.model.repetition
            with context.autocast():
                encoded = context.model.input_encoder(batch)
                self.encoded = encoded
                prefix_length = int(encoded["prefix_length"])
                candidate_count = int(encoded["candidate_count"])
                tokens = encoded["tokens"]
                hidden = (
                    tokens[:, :prefix_length],
                    tokens[:, prefix_length : prefix_length + candidate_count],
                    tokens[:, prefix_length + candidate_count :],
                )
                for layer in context.model.encoder.layers[:layer_index]:
                    hidden = self._step(layer, hidden)
                self.hidden = hidden

    def _step(self, layer, hidden):
        encoded = self.encoded
        return run_split_layer(
            layer, *hidden,
            prefix_valid=encoded["prefix_valid"],
            candidate_valid=encoded["candidate_valid"],
            cls_valid=encoded["cls_valid"],
            position_ids=encoded["position_ids"],
            rotary_position_encoding=self.context.model.encoder.rotary_position_encoding,
        )[:3]

    def logits(self):
        model = self.context.model
        with self.context.autocast():
            if self.encoded is None:
                # 不计算训练用 top-k 和重复的低精度交叉熵。
                return model({k: v for k, v in self.batch.items() if k != "label_index"})["logits"]
            hidden = self.hidden
            for layer in model.encoder.layers[self.layer_index:]:
                hidden = self._step(layer, hidden)
            if model.encoder.norm is not None:
                hidden = tuple(model.encoder.norm(value) for value in hidden)
            return model.score_hidden(self.encoded, torch.cat(hidden, dim=1), self.batch)


@torch.inference_mode()
def _evaluate_layer_grid(
    context: ModelAnalysisContext,
    batches: tuple[dict[str, object], ...],
    directions: LayerDirections,
    *,
    coordinates: np.ndarray,
    baseline_loss: float,
    center_index: int,
) -> np.ndarray:
    # batch 放在外层：输入只搬运一次，缓存仅占一个 batch 的显存。
    size = len(coordinates)
    totals = torch.zeros((size, size), dtype=torch.float64, device=context.device)
    layer_index = next(
        index for index, layer in enumerate(context.model.encoder.layers)
        if any(parameter is directions.parameters[0] for parameter in layer.parameters())
    )
    sample_count = 0
    started = last_report = perf_counter()
    try:
        for batch_index, batch_cpu in enumerate(batches):
            batch = move_batch(batch_cpu, context.device)
            # 上一 batch 的最后一个点仍有扰动，先还原再构建前层缓存。
            _restore_parameters(directions)
            forward = _LayerForward(context, batch, layer_index)
            labels = batch["label_index"]
            sample_count += int(labels.shape[0])
            for y_index, y_value in enumerate(coordinates):
                for x_index, x_value in enumerate(coordinates):
                    if x_index == center_index and y_index == center_index:
                        continue
                    _set_perturbed_parameters(
                        directions, x_value=float(x_value), y_value=float(y_value),
                    )
                    logits = forward.logits()
                    totals[y_index, x_index] += F.cross_entropy(
                        logits.double(), labels, reduction="sum",
                    )
                if perf_counter() - last_report >= 30:
                    print(
                        f"[loss-landscape] layer={layer_index + 1}, "
                        f"batch={batch_index + 1}/{len(batches)}, row={y_index + 1}/{size}",
                        flush=True,
                    )
                    last_report = perf_counter()
            del forward, batch, labels, logits
    finally:
        _restore_parameters(directions)
    if sample_count <= 0:
        raise ValueError("loss landscape received no samples")
    losses = (totals / sample_count).cpu().numpy()
    losses[center_index, center_index] = baseline_loss
    if not np.isfinite(losses).all():
        raise FloatingPointError("loss landscape contains non-finite values")
    print(f"[loss-landscape] layer={layer_index + 1} completed in {perf_counter() - started:.2f}s", flush=True)
    return losses


@torch.no_grad()
def _set_perturbed_parameters(
    directions: LayerDirections,
    *,
    x_value: float,
    y_value: float,
) -> None:
    for parameter, base, direction_x, direction_y in zip(
        directions.parameters,
        directions.base_values,
        directions.direction_x,
        directions.direction_y,
    ):
        parameter.copy_(base + x_value * direction_x + y_value * direction_y)


@torch.no_grad()
def _restore_parameters(directions: LayerDirections) -> None:
    for parameter, base in zip(directions.parameters, directions.base_values):
        parameter.copy_(base)


@torch.inference_mode()
def _mean_cross_entropy(
    context: ModelAnalysisContext,
    batches: tuple[dict[str, object], ...],
) -> float:
    loss_sum = torch.zeros((), dtype=torch.float64, device=context.device)
    sample_count = 0
    for batch_cpu in batches:
        batch = move_batch(batch_cpu, context.device)
        with context.autocast():
            output = context.model(batch)
        logits = output["logits"]
        labels = batch["label_index"]
        batch_loss = F.cross_entropy(logits.double(), labels, reduction="sum")
        loss_sum += batch_loss.detach()
        sample_count += int(labels.shape[0])
        del batch, output, logits, labels, batch_loss
    if sample_count <= 0:
        raise ValueError("loss landscape received no samples")
    return float(loss_sum.item() / sample_count)


def _loss_normalizer(losses: np.ndarray) -> Normalize:
    minimum = float(losses.min())
    maximum = float(losses.max())
    if minimum == maximum:
        padding = max(abs(minimum) * 1e-6, 1e-9)
        minimum -= padding
        maximum += padding
    return Normalize(vmin=minimum, vmax=maximum)


def _plot_surface_grid(
    losses: np.ndarray,
    coordinates: np.ndarray,
    baseline_loss: float,
    path: Path,
) -> None:
    layer_count = losses.shape[0]
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    color_map = plt.get_cmap(LOSS_LANDSCAPE_CMAP)
    fig, axes = create_grid_figure(
        layer_count,
        cell_size=LOSS_LANDSCAPE_CELL_SIZE_3D,
        projection="3d",
    )
    for layer_index, layer_loss in enumerate(losses):
        ax = axes.flat[layer_index]
        # 中心点位于曲面上，使用显式层级避免深度排序把标记遮住。
        ax.computed_zorder = False
        normalizer = _loss_normalizer(layer_loss)
        ax.plot_surface(
            x_grid,
            y_grid,
            layer_loss,
            cmap=color_map,
            norm=normalizer,
            rstride=1,
            cstride=1,
            linewidth=0.12,
            edgecolor=(1.0, 1.0, 1.0, 0.24),
            antialiased=True,
            shade=False,
        )
        ax.scatter(
            [0.0],
            [0.0],
            [baseline_loss],
            color="#ff7f0e",
            edgecolors="black",
            linewidths=0.4,
            s=38,
            depthshade=False,
            zorder=5,
        )
        ax.set_title(
            f"Layer {layer_index + 1} | center={baseline_loss:.6f} | "
            f"min={layer_loss.min():.6f}"
        )
        ax.set_xlabel("direction x")
        ax.set_ylabel("direction y")
        ax.set_zlabel("cross entropy")
        ax.view_init(elev=28, azim=-58)
    hide_empty_tiles(axes, layer_count)
    scalar_map = plt.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap=color_map)
    scalar_map.set_array([])
    fig.colorbar(
        scalar_map,
        ax=list(axes.flat[:layer_count]),
        label="relative loss within each layer",
        orientation="horizontal",
        shrink=0.55,
        fraction=0.025,
        pad=0.03,
    )
    fig.suptitle(
        "Per-layer Loss Landscape — Filter-normalized Orthogonal Directions\n"
        "color normalized independently per layer; z axis shows absolute cross entropy",
        fontsize=SUPTITLE_FONTSIZE,
    )
    save_figure(fig, path, dpi=LOSS_LANDSCAPE_DPI)


def _plot_contour_grid(
    losses: np.ndarray,
    coordinates: np.ndarray,
    baseline_loss: float,
    path: Path,
) -> None:
    layer_count = losses.shape[0]
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    color_map = plt.get_cmap(LOSS_LANDSCAPE_CMAP)
    fig, axes = create_grid_figure(layer_count, cell_size=LOSS_LANDSCAPE_CELL_SIZE_2D)
    for layer_index, layer_loss in enumerate(losses):
        ax = axes.flat[layer_index]
        normalizer = _loss_normalizer(layer_loss)
        levels = np.linspace(normalizer.vmin, normalizer.vmax, 48)
        ax.contourf(
            x_grid,
            y_grid,
            layer_loss,
            levels=levels,
            cmap=color_map,
            norm=normalizer,
        )
        ax.contour(
            x_grid,
            y_grid,
            layer_loss,
            levels=levels[::4],
            colors="white",
            linewidths=0.3,
            alpha=0.6,
        )
        ax.scatter(
            [0.0],
            [0.0],
            color="#ff7f0e",
            edgecolors="black",
            linewidths=0.4,
            s=22,
            zorder=3,
        )
        ax.set_title(
            f"Layer {layer_index + 1} | center={baseline_loss:.6f} | "
            f"range={np.ptp(layer_loss):.6f}"
        )
        ax.set_xlabel("direction x")
        ax.set_ylabel("direction y")
        ax.set_aspect("equal", adjustable="box")
    hide_empty_tiles(axes, layer_count)
    scalar_map = plt.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap=color_map)
    scalar_map.set_array([])
    fig.colorbar(
        scalar_map,
        ax=list(axes.flat[:layer_count]),
        label="relative loss within each layer",
        shrink=0.82,
    )
    fig.suptitle(
        "Per-layer Loss Landscape — Contours\n"
        "color normalized independently per layer; titles show absolute cross entropy",
        fontsize=SUPTITLE_FONTSIZE,
    )
    save_figure(fig, path, dpi=LOSS_LANDSCAPE_DPI)

"""模型分析中心：加载数据、提取每层 hidden，并提供科研绘图工具。"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import matplotlib

# 后端必须在导入 pyplot 之前固定，保证无显示环境下也能出图。
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from common.config import PROJECT_ROOT, load_project_config
from common.cache_compilation import compile_raw_training_cache
from common.project_config import resolve_project_job_tag
from common.torch_runtime import autocast_context, model_dtype, move_batch
from common.torch_serialization import safe_torch_load
from common.policy.data import Normalizer, SkillVocab
from training import TrainingCollator, TrainingDataset
from common.policy.data.policy_actions import load_policy_actions
from common.policy.data import DataSpec
from common.policy.model import (
    CandidateTransformerModel,
    repetition_config_from_checkpoint,
)
from .token_metadata import ANALYSIS_FEATURES, build_token_metadata


logger = logging.getLogger(__name__)

ROLE_NAMES = {
    0: "scene",
    1: "history_pair",
    2: "candidate_pair",
    3: "cls",
}

# 模型分析统一使用 attention 图的黑色到白金色阶；有明确正负语义的图仍保留专用发散色阶。
ATTENTION_CMAP_NAME = "magma"
ROLE_COLORS = {
    0: "#321067",
    1: "#8c2981",
    2: "#e34f66",
    3: "#febd82",
}


def skill_name_by_vocab_id(context: "AnalysisContext") -> dict[int, str]:
    """按正式职业配置把词表 ID 映射为中文技能名。"""
    if len(context.dataset) == 0:
        raise ValueError("dataset is empty, cannot resolve skill labels")

    configured_job_tag = resolve_project_job_tag(
        project_root=PROJECT_ROOT,
    )
    project_config = load_project_config(job_tag=configured_job_tag)
    if project_config.job.key != context.data_spec.job_tag:
        raise ValueError(
            f"loaded config job {project_config.job.key!r} != analysis job "
            f"{context.data_spec.job_tag!r}"
        )

    definitions = (*project_config.system.skills, *project_config.job.skills)
    names_by_raw_id = {definition.game_id: definition.name for definition in definitions}
    # policy 动作不进入 SkillBook，但仍是词表中的正式 token（例如 raw_id=0 的空输出）。
    names_by_raw_id.update({action.raw_id: action.name for action in load_policy_actions()})
    labels: dict[int, str] = {}
    for vocab_id in range(1, context.vocab.size()):
        raw_skill_id = context.vocab.reverse_lookup(vocab_id)
        labels[vocab_id] = names_by_raw_id.get(
            raw_skill_id,
            f"未知技能({raw_skill_id})",
        )
    return labels


@dataclass
class ModelAnalysisContext:
    """所有模型分析输出共享的运行时契约。"""

    checkpoint_path: Path
    source_path: Path
    output_dir: Path
    model: CandidateTransformerModel
    dataset: TrainingDataset
    data_spec: DataSpec
    vocab: SkillVocab
    device: torch.device
    precision: str

    def autocast(self):
        """统一模型前向精度；所有输出都以模型 YAML 的 precision 为权威。"""
        return autocast_context(self.device, self.precision)


@dataclass
class AnalysisContext(ModelAnalysisContext):
    """需要逐层 hidden/attention 的完整分析上下文。"""

    layer_vectors: list[np.ndarray]
    layer_roles: list[np.ndarray]
    layer_metadata: list[dict[str, np.ndarray]]


def load_analysis_context(
    *,
    checkpoint_path: Path,
    source_path: Path | None,
    raw_root: Path,
    cache_dir: Path,
    max_history: int,
    cache_shard_size: int,
    cache_max_shards: int,
    output_dir: Path,
    max_samples: int,
    max_tokens: int,
    batch_size: int,
    device_name: str,
    precision: str,
    candidate_order_file: Path | None = None,
) -> AnalysisContext:
    """加载 checkpoint/compiled cache 并提取每个 Transformer layer 的有效 token hidden。"""
    runtime = _load_model_analysis_context(
        checkpoint_path=checkpoint_path,
        source_path=source_path,
        raw_root=raw_root,
        cache_dir=cache_dir,
        max_history=max_history,
        cache_shard_size=cache_shard_size,
        cache_max_shards=cache_max_shards,
        candidate_order_file=candidate_order_file,
        output_dir=output_dir,
        device_name=device_name,
        precision=precision,
    )
    model = runtime.model
    dataset = runtime.dataset
    data_spec = runtime.data_spec
    device = runtime.device
    if max_samples <= 0:
        max_samples = len(dataset)
    samples = [dataset[index] for index in range(min(max_samples, len(dataset)))]
    collator = TrainingCollator()

    layer_rows: list[list[np.ndarray]] = [[] for _ in model.encoder.layers]
    role_rows: list[list[np.ndarray]] = [[] for _ in model.encoder.layers]
    metadata_rows: list[dict[str, list[np.ndarray]]] = [
        {feature: [] for feature in ANALYSIS_FEATURES}
        for _ in model.encoder.layers
    ]
    collected_tokens = [0 for _ in model.encoder.layers]
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            sample_batch = samples[start : start + batch_size]
            batch = collator(sample_batch)
            batch_device = move_batch(batch, device)
            with runtime.autocast():
                trace = model.trace(batch_device)
            encoded = trace.encoded
            valid = ~encoded["padding_mask"]
            layer_hidden_rows = [
                layer_hidden.detach().float().cpu().numpy()
                for layer_hidden in trace.layer_hidden
            ]
            valid_rows = valid.detach().cpu().numpy()
            role_rows_batch = encoded["role_ids"].detach().cpu().numpy()

            with runtime.autocast():
                logits = model.score_hidden(encoded, trace.hidden, batch_device)
            token_metadata = build_token_metadata(
                sample_batch,
                batch=batch_device,
                encoded=encoded,
                schema=dataset.schema,
                job_tag=data_spec.job_tag,
                logits=logits.detach().float().cpu().numpy(),
            )
            for layer_index, hidden_rows in enumerate(layer_hidden_rows):
                if collected_tokens[layer_index] >= max_tokens:
                    continue
                flat_vectors = hidden_rows[valid_rows]
                flat_roles = role_rows_batch[valid_rows]
                flat_metadata = {
                    feature: values[valid_rows]
                    for feature, values in token_metadata.items()
                }
                remaining = max_tokens - collected_tokens[layer_index]
                if remaining <= 0:
                    continue
                if len(flat_vectors) > remaining:
                    indices = np.linspace(0, len(flat_vectors) - 1, remaining, dtype=int)
                    flat_vectors = flat_vectors[indices]
                    flat_roles = flat_roles[indices]
                    flat_metadata = {
                        feature: values[indices]
                        for feature, values in flat_metadata.items()
                    }
                layer_rows[layer_index].append(flat_vectors)
                role_rows[layer_index].append(flat_roles)
                for feature, values in flat_metadata.items():
                    metadata_rows[layer_index][feature].append(values)
                collected_tokens[layer_index] += len(flat_vectors)
            # trace 会持有所有层 attention/hidden。显式结束本批生命周期，避免下一批
            # 前向计算时旧 trace 仍占用显存，形成接近双倍的瞬时峰值。
            del (
                batch_device,
                trace,
                encoded,
                valid,
                layer_hidden_rows,
                valid_rows,
                role_rows_batch,
                logits,
                token_metadata,
            )
            if device.type == "cuda":
                # 科研 trace 的序列长度会随 batch 波动；及时归还 allocator 中
                # 不再使用的分段，避免碎片化 reserved 显存累计到整张卡。
                torch.cuda.empty_cache()

    layer_vectors = [
        _concat_arrays(rows, empty_shape=(0, 0), dtype=np.float32)
        for rows in layer_rows
    ]
    layer_roles = [
        _concat_arrays(rows, empty_shape=(0,), dtype=np.int64)
        for rows in role_rows
    ]
    layer_metadata = [
        {
            feature: _concat_arrays(values, empty_shape=(0,), dtype=object)
            for feature, values in rows.items()
        }
        for rows in metadata_rows
    ]
    return AnalysisContext(
        checkpoint_path=runtime.checkpoint_path,
        source_path=runtime.source_path,
        output_dir=runtime.output_dir,
        model=model,
        dataset=dataset,
        data_spec=data_spec,
        vocab=runtime.vocab,
        device=device,
        precision=runtime.precision,
        layer_vectors=layer_vectors,
        layer_roles=layer_roles,
        layer_metadata=layer_metadata,
    )


def load_loss_landscape_context(
    *,
    checkpoint_path: Path,
    source_path: Path | None,
    raw_root: Path,
    cache_dir: Path,
    max_history: int,
    cache_shard_size: int,
    cache_max_shards: int,
    output_dir: Path,
    device_name: str,
    precision: str,
    candidate_order_file: Path | None = None,
) -> ModelAnalysisContext:
    """只加载损失地图所需模型与数据，不提取 hidden、attention 或 PCA 输入。"""
    return _load_model_analysis_context(
        checkpoint_path=checkpoint_path,
        source_path=source_path,
        raw_root=raw_root,
        cache_dir=cache_dir,
        max_history=max_history,
        cache_shard_size=cache_shard_size,
        cache_max_shards=cache_max_shards,
        candidate_order_file=candidate_order_file,
        output_dir=output_dir,
        device_name=device_name,
        precision=precision,
    )


def _load_model_analysis_context(
    *,
    checkpoint_path: Path,
    source_path: Path | None,
    raw_root: Path,
    cache_dir: Path,
    max_history: int,
    cache_shard_size: int,
    cache_max_shards: int,
    output_dir: Path,
    device_name: str,
    precision: str,
    candidate_order_file: Path | None,
) -> ModelAnalysisContext:
    """统一加载分析模型和 dataset；模型 dtype 与 autocast 共用同一 precision。"""
    checkpoint_path = Path(checkpoint_path)
    device = _resolve_device(device_name)
    if precision != "float32" and device.type != "cuda":
        raise RuntimeError(f"model analysis precision {precision} requires CUDA")
    checkpoint = safe_torch_load(checkpoint_path)
    data_spec = DataSpec.from_dict(checkpoint["data_spec"])
    job_tag = resolve_project_job_tag(project_root=PROJECT_ROOT)
    if job_tag != data_spec.job_tag:
        raise ValueError(
            f"configured job_tag {job_tag!r} does not match checkpoint job_tag {data_spec.job_tag!r}"
        )
    vocab = SkillVocab.build_from_job_tag(job_tag)
    if source_path is None:
        source_path = _find_default_raw(raw_root)
    source_path = Path(source_path)
    model_config = CandidateTransformerModel.checkpoint_model_config(checkpoint)
    repetition_config = repetition_config_from_checkpoint(checkpoint)
    model = CandidateTransformerModel(
        data_spec,
        model_config,
        vocab_size=vocab.size(),
        repetition=repetition_config,
    )
    model.to(device=device, dtype=model_dtype(precision))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    del checkpoint
    dataset = _load_analysis_dataset(
        source_path=source_path,
        cache_dir=cache_dir,
        max_history=max_history,
        cache_shard_size=cache_shard_size,
        cache_max_shards=cache_max_shards,
        candidate_order_file=candidate_order_file,
        job_tag=job_tag,
        skill_vocab=vocab,
    )
    if len(dataset) == 0:
        raise ValueError("dataset is empty, cannot load model analysis context")
    dataset_spec = DataSpec.from_dataset(dataset)
    data_spec.assert_compatible_with(dataset_spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    return ModelAnalysisContext(
        checkpoint_path=checkpoint_path,
        source_path=source_path,
        output_dir=output_dir,
        model=model,
        dataset=dataset,
        data_spec=data_spec,
        vocab=vocab,
        device=device,
        precision=precision,
    )




def pca_projection(
    values: np.ndarray,
    components: int,
    *,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 PCA；CPU 路径使用特征协方差分解，避免创建完整左奇异矩阵。"""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] < components:
        raise ValueError(f"PCA requires at least {components} rows, got {values.shape}")

    resolved_device = torch.device(device) if device is not None else None
    if resolved_device is not None and resolved_device.type == "cuda":
        tensor = torch.as_tensor(values, dtype=torch.float32, device=resolved_device)
        centered = tensor - tensor.mean(dim=0, keepdim=True)
        _, singular_values, right_vectors = torch.linalg.svd(centered, full_matrices=False)
        coordinates = centered @ right_vectors[:components].transpose(0, 1)
        variance = singular_values.square()
        total_variance = max(float(variance.sum().item()), np.finfo(np.float32).eps)
        explained = variance[:components] / total_variance
        return (
            coordinates.cpu().numpy(),
            explained.cpu().numpy(),
        )

    centered = np.array(values, dtype=np.float64, copy=True)
    centered -= centered.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    principal_axes = eigenvectors[:, order[:components]]
    coordinates = centered @ principal_axes
    explained = eigenvalues[:components] / max(
        float(eigenvalues.sum()),
        np.finfo(float).eps,
    )
    return coordinates, explained


def configure_matplotlib() -> None:
    """统一科研图表的字体、网格和负号显示。"""
    matplotlib.use("Agg")
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


# ---------------------------------------------------------------------------
# 统一图表布局：所有输出模块共用同一列数规则、尺寸推导与收尾方式。
# ---------------------------------------------------------------------------

# 子图基准尺寸 (宽, 高)，单位英寸；figure 尺寸由网格行列数乘以该基准得到。
GRID_CELL_SIZE = (5.6, 4.6)
# 热图子图基准尺寸：内容为方阵，宽度收紧接近方形避免 imshow 被横向拉伸。
HEATMAP_CELL_SIZE = (4.8, 4.6)
SUPTITLE_FONTSIZE = 16
SAVE_PAD_INCHES = 0.2


def grid_shape(tile_count: int) -> tuple[int, int]:
    """按近正方形规则返回 (rows, columns)。

    先最小化 |行数 - 列数|，同一差异下取更少列，因此网格偏竖向而非横向铺开：
    12 -> (4, 3)、16 -> (4, 4)、19 -> (5, 4)、3 -> (2, 2)、8 -> (3, 3)。
    """
    if tile_count <= 0:
        raise ValueError(f"grid tile count must be positive, got {tile_count}")

    best_rows, best_columns, best_score = tile_count, 1, None
    for columns in range(1, tile_count + 1):
        rows = -(-tile_count // columns)
        score = (abs(rows - columns), columns)
        if best_score is None or score < best_score:
            best_rows, best_columns, best_score = rows, columns, score
    return best_rows, best_columns


def create_grid_figure(
    tile_count: int,
    *,
    cell_size: tuple[float, float] = GRID_CELL_SIZE,
    **subplot_kwargs: object,
) -> tuple[Figure, np.ndarray]:
    """创建统一网格 figure；返回的 axes 恒为二维数组，形状与 grid_shape 一致。"""
    rows, columns = grid_shape(tile_count)
    return plt.subplots(
        rows,
        columns,
        figsize=(columns * cell_size[0], rows * cell_size[1]),
        squeeze=False,
        constrained_layout=True,
        subplot_kw=subplot_kwargs or None,
    )


def create_figure(figsize: tuple[float, float], **subplot_kwargs: object) -> tuple[Figure, Axes]:
    """创建单轴 figure，与网格图共用同一布局引擎和保存方式。"""
    return plt.subplots(
        figsize=figsize,
        constrained_layout=True,
        subplot_kw=subplot_kwargs or None,
    )


def hide_empty_tiles(axes: np.ndarray, tile_count: int) -> None:
    """隐藏网格末端的空位；保留占位以维持网格对齐。"""
    for ax in axes.flat[tile_count:]:
        ax.set_visible(False)


def save_figure(fig: Figure, path: Path, *, dpi: int | None = None) -> None:
    """统一收尾：按同一留白保存并关闭 figure。"""
    fig.savefig(path, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES, dpi=dpi)
    plt.close(fig)


def _find_default_raw(raw_root: Path) -> Path:
    root = Path(raw_root)
    candidates = sorted(root.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"no raw JSON dataset found: {root}")
    return candidates[0]


def _load_analysis_dataset(
    *,
    source_path: Path,
    cache_dir: Path,
    max_history: int,
    cache_shard_size: int,
    cache_max_shards: int,
    job_tag: str,
    skill_vocab: SkillVocab,
    candidate_order_file: Path | None = None,
) -> TrainingDataset:
    """读取分析用 cache；缺失或过期时调用转换 CLI 后重试。"""
    normalizer = Normalizer()
    normalizer.configure_job_resources(job_tag)
    dataset_kwargs = {
        "normalizer": normalizer,
        "job_tag": job_tag,
        "skill_vocab": skill_vocab,
        "max_history": max_history,
        "int_dtype": torch.int32,
        "float_dtype": torch.float32,
        "cache_dir": cache_dir,
        "compiled_cache_shard_size": cache_shard_size,
        "compiled_cache_max_shards": cache_max_shards,
        "candidate_order_file": candidate_order_file,
    }
    try:
        return TrainingDataset([source_path], **dataset_kwargs)
    except FileNotFoundError:
        logger.info("模型分析 cache 缺失或过期，调用转换脚本: %s", source_path)
        compile_raw_training_cache(
            source_path=source_path,
            cache_dir=cache_dir,
            cache_shard_size=cache_shard_size,
            job_tag=job_tag,
        )
        return TrainingDataset([source_path], **dataset_kwargs)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name == "cuda":
        return torch.device("cpu")
    return torch.device(device_name)


def _concat_arrays(
    rows: list[np.ndarray],
    *,
    empty_shape: tuple[int, ...],
    dtype: object,
) -> np.ndarray:
    """拼接分析数组；无数据时返回指定形状和 dtype 的空数组。"""
    if not rows:
        return np.zeros(empty_shape, dtype=dtype)
    return np.concatenate(rows, axis=0)

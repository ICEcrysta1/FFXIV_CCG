"""训练 checkpoint 的保存、续训校验和运行状态恢复。"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from common.policy.data import DataSpec, ModelInputContract
from common.policy.model import CandidateTransformerModel
from common.torch_serialization import safe_torch_load

from ..config import RunConfig

logger = logging.getLogger(__name__)


def _top1_val_ppg_average(top1_accuracy: float, normalized_val_ppg: float) -> float:
    """把 top1 与独立验证集 PPG（已归一）合成主评分。"""
    return (float(top1_accuracy) + float(normalized_val_ppg)) / 2.0


def _best_metric_key(
    metrics: dict[str, float],
    *,
    ppg_enabled: bool,
) -> tuple[float, float, float, float]:
    """按主评分、top1、top3、value loss 构造可直接比较的排序键。"""
    score = (
        metrics["top1_val_ppg_average"]
        if ppg_enabled
        else metrics["top1_accuracy"]
    )
    return (
        float(score),
        float(metrics["top1_accuracy"]),
        float(metrics["top3_accuracy"]),
        -float(metrics["value_preference_loss"]),
    )


def _epoch_checkpoint_name(epoch: int, metrics: dict[str, float]) -> str:
    """生成包含本轮 PPG 的 checkpoint 文件名。"""
    ppg = metrics.get("val_ppg")
    if ppg is None:
        return f"epoch_{epoch:03d}.pt"
    return f"epoch_{epoch:03d}_val_ppg_{float(ppg):.2f}.pt"


def _load_resume_checkpoint(path: Path) -> dict[str, object]:
    """读取并校验续训 checkpoint。"""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    payload = safe_torch_load(path)
    if not isinstance(payload, Mapping):
        raise ValueError("resume checkpoint must be a mapping")
    return dict(payload)


def _normalized_checkpoint_epoch(checkpoint: Mapping[str, object]) -> int:
    """校验并返回载荷中的 epoch；续训校验与列目录共用这一条规则。"""
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool):
        raise ValueError("resume checkpoint epoch must be an integer")
    try:
        normalized_epoch = int(epoch)
    except (TypeError, ValueError) as exc:
        raise ValueError("resume checkpoint epoch must be an integer") from exc
    if normalized_epoch < 1:
        raise ValueError("resume checkpoint epoch must be >= 1")
    return normalized_epoch


@dataclass(frozen=True)
class CheckpointCandidate:
    """恢复训练候选项：文件、真实 epoch 与保存时的数据上限。"""

    path: Path
    epoch: int
    max_files: int | None


def _read_checkpoint_metadata(path: Path) -> Mapping[str, object]:
    """只读取 checkpoint 元数据，不加载权重 storage。"""
    path = Path(path)
    # meta 设备只避免把 tensor 放到 CPU；mmap 才能避免为列目录读取完整 storage。
    payload = safe_torch_load(path, mmap=True, map_location="meta")
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint must be a mapping: {path}")
    return payload


def _checkpoint_max_files(checkpoint: Mapping[str, object]) -> int | None:
    """读取 checkpoint 保存的数据上限；旧格式缺省时按不限量兼容。"""
    checkpoint_run_config = checkpoint.get("run_config")
    if checkpoint_run_config is None:
        return None
    if not isinstance(checkpoint_run_config, Mapping):
        raise ValueError("resume checkpoint run_config must be a mapping")
    max_files = checkpoint_run_config.get("max_files")
    if max_files is not None and (
        isinstance(max_files, bool) or not isinstance(max_files, int) or max_files < 1
    ):
        raise ValueError(
            "resume checkpoint run_config.max_files must be a positive integer or null"
        )
    return max_files


def read_checkpoint_epoch(path: Path) -> int:
    """只读取 checkpoint 载荷里的 epoch，不加载权重数据。"""
    payload = _read_checkpoint_metadata(path)
    return _normalized_checkpoint_epoch(payload)


def collect_checkpoint_candidates(
    output_dir: Path,
    *,
    max_epochs: int,
) -> tuple[tuple[CheckpointCandidate, ...], tuple[CheckpointCandidate, ...]]:
    """按 checkpoint 载荷中的真实 epoch，把目录内的 `.pt` 分成可续训与被拒绝两组。"""
    if max_epochs < 1:
        raise ValueError("max_epochs must be >= 1")
    resumable: list[CheckpointCandidate] = []
    rejected: list[CheckpointCandidate] = []
    for path in sorted(Path(output_dir).glob("*.pt")):
        payload = _read_checkpoint_metadata(path)
        candidate = CheckpointCandidate(
            path=path,
            epoch=_normalized_checkpoint_epoch(payload),
            max_files=_checkpoint_max_files(payload),
        )
        # training_loop 拒绝 resume_epoch >= max_epochs，这里保持同一条边界。
        target = rejected if candidate.epoch >= max_epochs else resumable
        target.append(candidate)
    return tuple(resumable), tuple(rejected)


def _validate_resume_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    data_spec: DataSpec,
    dataset,
    config: RunConfig,
    input_contract: ModelInputContract,
    force_resume_data_mismatch: bool = False,
) -> int:
    """确保续训 checkpoint 与当前数据、模型和归一化契约完全一致。"""
    checkpoint_data_spec = checkpoint.get("data_spec")
    if not isinstance(checkpoint_data_spec, Mapping):
        raise ValueError("resume checkpoint missing data_spec")
    DataSpec.from_dict(dict(checkpoint_data_spec)).assert_compatible_with(data_spec)

    checkpoint_contract = ModelInputContract.from_checkpoint(checkpoint)
    checkpoint_contract.assert_matches_data_spec(data_spec)
    checkpoint_contract.schema.assert_compatible_with(dataset.schema)
    if checkpoint_contract.normalizer_contract != input_contract.normalizer_contract:
        raise ValueError("resume checkpoint normalization contract mismatch")

    checkpoint_model_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_model_config, Mapping):
        raise ValueError("resume checkpoint missing model_config")
    normalized_model_config = asdict(
        CandidateTransformerModel.checkpoint_model_config(dict(checkpoint))
    )
    if normalized_model_config != asdict(config.model):
        raise ValueError("resume checkpoint model config mismatch")

    if config.model_variant is not None:
        checkpoint_model_variant = checkpoint.get("model_variant")
        if not isinstance(checkpoint_model_variant, str) or not checkpoint_model_variant.strip():
            raise ValueError("resume checkpoint missing model_variant")
        if checkpoint_model_variant.strip() != config.model_variant:
            raise ValueError(
                "resume checkpoint model variant mismatch: "
                f"{checkpoint_model_variant!r} != {config.model_variant!r}"
            )

    # max_files 加入 checkpoint 前的载荷等价于不限量；当前配置若仍为
    # null 可以兼容恢复，但切换到新的有限上限必须拒绝。
    checkpoint_max_files = _checkpoint_max_files(checkpoint)
    if checkpoint_max_files != config.max_files:
        mismatch_message = (
            "resume checkpoint max_files mismatch: "
            f"checkpoint={checkpoint_max_files!r} != current={config.max_files!r}"
        )
        if not force_resume_data_mismatch:
            raise ValueError(mismatch_message)
        logger.warning(
            "%s; force_resume_data_mismatch=True，继续使用当前 YAML/CLI 数据上限。",
            mismatch_message,
        )

    checkpoint_precision = checkpoint.get("training_precision")
    if checkpoint_precision is not None and str(checkpoint_precision) != config.precision:
        raise ValueError("resume checkpoint training precision mismatch")
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        raise ValueError("resume checkpoint missing model_state_dict")
    if not isinstance(checkpoint.get("optimizer_state_dict"), Mapping):
        raise ValueError("resume checkpoint missing optimizer_state_dict")

    return _normalized_checkpoint_epoch(checkpoint)


def _restore_scheduler_state(scheduler, checkpoint: Mapping[str, object], *, completed_steps: int) -> None:
    """恢复 scheduler；旧 checkpoint 没有 scheduler 时按已完成 batch 定位。"""
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if isinstance(scheduler_state, Mapping):
        scheduler.load_state_dict(dict(scheduler_state))
        return

    scheduler.last_epoch = int(completed_steps)
    scheduler._step_count = max(1, int(completed_steps) + 1)
    scheduler._last_lr = [group["lr"] for group in scheduler.optimizer.param_groups]


def _checkpoint_metrics(checkpoint: Mapping[str, object]) -> dict[str, float]:
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, Mapping):
        return {}
    return {str(key): float(value) for key, value in metrics.items()}


def _metric_state_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    ppg_enabled: bool,
) -> tuple[tuple[float, float, float, float] | None, dict[str, float]]:
    metrics = checkpoint.get("best_val_metrics")
    if not isinstance(metrics, Mapping):
        metrics = checkpoint.get("metrics")
    if not isinstance(metrics, Mapping):
        return None, {}
    normalized_metrics = {str(key): float(value) for key, value in metrics.items()}
    try:
        key = _best_metric_key(normalized_metrics, ppg_enabled=ppg_enabled)
    except KeyError:
        return None, {}
    return key, normalized_metrics


def _restore_best_state(
    checkpoint: Mapping[str, object],
    resume_path: Path,
    *,
    ppg_enabled: bool,
) -> tuple[tuple[float, float, float, float] | None, dict[str, float]]:
    """恢复 best 指标；兼容尚未保存 best 元数据的 epoch checkpoint。"""
    best_key = checkpoint.get("best_key")
    best_metrics = checkpoint.get("best_val_metrics")
    if isinstance(best_key, (list, tuple)) and len(best_key) == 4 and isinstance(best_metrics, Mapping):
        return (
            tuple(float(value) for value in best_key),
            {str(key): float(value) for key, value in best_metrics.items()},
        )

    best_path = Path(resume_path).resolve().parent / "best.pt"
    if best_path.is_file() and best_path != Path(resume_path).resolve():
        best_checkpoint = _load_resume_checkpoint(best_path)
        restored = _metric_state_from_checkpoint(best_checkpoint, ppg_enabled=ppg_enabled)
        if restored[0] is not None:
            return restored

    return _metric_state_from_checkpoint(checkpoint, ppg_enabled=ppg_enabled)


def _capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(checkpoint: Mapping[str, object]) -> None:
    state = checkpoint.get("rng_state")
    if not isinstance(state, Mapping):
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if (
        cuda_state is not None
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 0
    ):
        if not isinstance(cuda_state, (list, tuple)):
            raise ValueError("checkpoint CUDA RNG state must be a list")
        device_count = torch.cuda.device_count()
        if len(cuda_state) != device_count:
            raise ValueError(
                "checkpoint CUDA RNG state device count mismatch: "
                f"checkpoint={len(cuda_state)}, current={device_count}"
            )
        torch.cuda.set_rng_state_all(cuda_state)


def _save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    config,
    data_spec,
    metrics,
    *,
    input_contract: ModelInputContract,
    scheduler=None,
    best_key: tuple[float, float, float, float] | None = None,
    best_val_metrics: Mapping[str, float] | None = None,
) -> None:
    model_variant = getattr(config, "model_variant", None)
    if not isinstance(model_variant, str) or not model_variant.strip():
        raise ValueError("checkpoint model_variant must be configured before saving")
    model_variant = model_variant.strip()
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(config.model),
        "data_spec": asdict(data_spec),
        "job_tag": data_spec.job_tag,
        "model_variant": model_variant,
        "input_contract": input_contract.to_dict(),
        "training_precision": config.precision,
        "run_config": asdict(config),
        "metrics": metrics,
        "rng_state": _capture_rng_state(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if best_key is not None:
        payload["best_key"] = tuple(float(value) for value in best_key)
    if best_val_metrics is not None:
        payload["best_val_metrics"] = dict(best_val_metrics)
    torch.save(payload, path)

"""训练初始化、epoch 循环、验证和调度器推进。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import logging
import math
from pathlib import Path
import random

import torch

from common.torch_runtime import autocast_context, model_dtype, move_batch
from common.policy.config import PROJECT_ROOT, resolve_policy_cache_dir
from common.project_config import resolve_registered_job_tags
from common.policy.data import ModelInputContract, SkillVocab, DataSpec

from .checkpoint import (
    _best_metric_key,
    _checkpoint_metrics,
    _epoch_checkpoint_name,
    _load_resume_checkpoint,
    _restore_best_state,
    _restore_rng_state,
    _restore_scheduler_state,
    _save_checkpoint,
    _top1_val_ppg_average,
    _validate_resume_checkpoint,
)
from ..config import RunConfig, ValuePreferenceConfig
from .dataloaders import build_dataloaders
from common.training.metrics import MetricAccumulator
from common.policy.model import CandidateTransformerModel
from common.policy.model.repetition import RepetitionConfig, prepare_repetition_penalty
from ..runtime.runtime_debug import RuntimeDebugRecorder
from .value_preference import compute_value_preference_loss


logger = logging.getLogger(__name__)
_EPOCH_METRICS = (
    "loss", "cross_entropy_loss", "value_preference_loss", "top1_accuracy", "top3_accuracy",
)


def _debug_stage(recorder: RuntimeDebugRecorder | None, name: str):
    """在关闭调试时返回零开销的空上下文。"""
    if recorder is None:
        return nullcontext()
    return recorder.stage(name)


def run_training(
    config: RunConfig,
    *,
    raw_paths: list[Path],
    output_dir: Path | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    device_name: str = "cuda",
    validation_metrics_callback=None,
    resume_path: Path | None = None,
    _build_dataloaders=None,
    _train_epoch=None,
    _validate=None,
    _save_checkpoint=None,
    _resolve_cache_dir=None,
    _registered_job_tags=None,
    _skill_vocab=None,
    _model_class=None,
) -> dict[str, object]:
    """执行完整 BC 训练并保存逐轮、best/final checkpoint。"""
    build_dataloaders_fn = _build_dataloaders or build_dataloaders
    train_epoch_fn = _train_epoch or train_epoch
    validate_fn = _validate or validate
    save_checkpoint_fn = _save_checkpoint or _save_checkpoint_default
    resolve_cache_dir_fn = _resolve_cache_dir or resolve_policy_cache_dir
    registered_job_tags_fn = _registered_job_tags or (lambda: resolve_registered_job_tags(PROJECT_ROOT))
    skill_vocab_cls = _skill_vocab or SkillVocab
    model_cls = _model_class or CandidateTransformerModel

    if output_dir is not None:
        config = replace(config, output_dir=Path(output_dir))
    if max_epochs is not None:
        config = replace(config, max_epochs=max_epochs)
    if batch_size is not None:
        config = replace(config, batch_size=batch_size)
    if learning_rate is not None:
        config = replace(config, learning_rate=learning_rate)

    resume_checkpoint = _load_resume_checkpoint(resume_path) if resume_path is not None else None

    _seed_everything(config.seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by TRAINING_DEVICE=cuda, but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    if not raw_paths:
        raise FileNotFoundError("no prepared raw JSON files supplied for training")

    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders_fn(
        raw_paths,
        config,
        int_dtype=torch.int32,
        float_dtype=torch.float32,
        cache_dir=(resolve_cache_dir_fn(config.job_tag) if config.job_tag else None),
    )
    data_spec = DataSpec.from_dataset(train_dataset)
    if config.job_tag is not None and config.job_tag != data_spec.job_tag:
        raise ValueError(f"config job_tag {config.job_tag!r} != cache job_tag {data_spec.job_tag!r}")
    if data_spec.job_tag not in registered_job_tags_fn():
        raise ValueError(
            f"cache job_tag {data_spec.job_tag!r} has no registered combat state-machine route"
        )
    normalizer = train_dataset.normalizer
    if normalizer is None:
        raise ValueError("training dataset must expose its normalizer")
    input_contract = ModelInputContract.from_training(
        data_spec=data_spec,
        schema=train_dataset.schema,
        normalizer=normalizer,
    )
    resume_epoch = 0
    if resume_checkpoint is not None:
        resume_epoch = _validate_resume_checkpoint(
            resume_checkpoint,
            data_spec=data_spec,
            dataset=train_dataset,
            config=config,
            input_contract=input_contract,
        )
        if resume_epoch >= config.max_epochs:
            raise ValueError(
                f"resume checkpoint already reached epoch {resume_epoch}; "
                f"configured max_epochs is only {config.max_epochs}"
            )
    if config.ppg.enabled and validation_metrics_callback is None:
        raise ValueError(
            "PPG validation is enabled but validation_metrics_callback was not supplied"
        )
    vocab = skill_vocab_cls.build_from_job_tag(data_spec.job_tag)
    model = model_cls(
        data_spec,
        config.model,
        vocab_size=vocab.size(),
        repetition=config.repetition,
    ).to(
        device=device,
        dtype=model_dtype(config.precision),
    )
    if config.activation_checkpoint_ffn:
        model.enable_activation_checkpoint_ffn()
    if config.activation_checkpoint_attention:
        model.enable_activation_checkpoint_attention()
    runtime_debug = RuntimeDebugRecorder(
        device=device,
        config=config.runtime_debug,
        output_path=config.output_dir / config.runtime_debug.output_filename,
    )
    debug_recorder = runtime_debug if runtime_debug.enabled else None
    set_runtime_debug = getattr(model, "set_runtime_debug", None)
    if callable(set_runtime_debug):
        set_runtime_debug(debug_recorder)
    logger.info(
        "模型: job=%s model_variant=%s candidates=%d state=%d scene=%d skill_features=%d layers=%d d_model=%d activation=%s ff_dim=%d precision=%s ffn_checkpoint=%s attention_checkpoint=%s device=%s",
        data_spec.job_tag,
        config.model_variant,
        data_spec.num_candidates,
        data_spec.state_dim,
        data_spec.scene_dim,
        data_spec.skill_feature_dim,
        config.model.n_layers,
        config.model.d_model,
        config.model.transformer_activation,
        config.model.ff_dim,
        config.precision,
        config.activation_checkpoint_ffn,
        config.activation_checkpoint_attention,
        device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = max(1, len(train_loader) * config.max_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_lambda(step, config.warmup_steps, total_steps),
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_key: tuple[float, float, float, float] | None = None
    best_val_metrics: dict[str, float] = {}
    last_val_metrics: dict[str, float] = {}
    start_epoch = 1
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        _restore_scheduler_state(
            scheduler,
            resume_checkpoint,
            completed_steps=resume_epoch * len(train_loader),
        )
        start_epoch = resume_epoch + 1
        best_key, best_val_metrics = _restore_best_state(
            resume_checkpoint,
            Path(resume_path),
            ppg_enabled=config.ppg.enabled,
        )
        last_val_metrics = _checkpoint_metrics(resume_checkpoint)
        _restore_rng_state(resume_checkpoint)
        logger.info(
            "从 checkpoint=%s 的第 %d 轮继续训练，下一轮为 %d",
            resume_path,
            resume_epoch,
            start_epoch,
        )

    for epoch in range(start_epoch, config.max_epochs + 1):
        train_metrics = train_epoch_fn(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            config.precision,
            value_preference=config.value_preference,
            runtime_debug=debug_recorder,
            epoch=epoch,
        )
        val_metrics = validate_fn(
            model,
            val_loader,
            device,
            config.precision,
            value_preference=config.value_preference,
        )
        if config.ppg.enabled:
            extra_metrics = validation_metrics_callback(
                model=model,
                dataset=val_dataset,
                data_spec=data_spec,
                vocab=vocab,
                config=config,
                device=device,
            )
            if not isinstance(extra_metrics, dict):
                raise TypeError("validation_metrics_callback must return a dict")
            val_metrics.update(
                {str(key): float(value) for key, value in extra_metrics.items()}
            )
            required_ppg_metrics = (
                "val_ppg",
                "val_ppg_normalized",
                "none_ppg",
                "none_ppg_normalized",
            )
            if any(key not in val_metrics for key in required_ppg_metrics):
                raise ValueError(
                    "validation_metrics_callback must return independent val_ppg and none_ppg metrics"
                )
            val_metrics["top1_val_ppg_average"] = _top1_val_ppg_average(
                val_metrics["top1_accuracy"],
                val_metrics["val_ppg_normalized"],
            )
        last_val_metrics = val_metrics
        logger.info(
            "Epoch %3d | train loss=%.4f ce=%.4f value_aux=%.4f top1=%.4f top3=%.4f | "
            "val loss=%.4f ce=%.4f value_aux=%.4f top1=%.4f top3=%.4f val_ppg=%.2f none_ppg=%.2f",
            epoch,
            train_metrics["loss"],
            train_metrics["cross_entropy_loss"],
            train_metrics["value_preference_loss"],
            train_metrics["top1_accuracy"],
            train_metrics["top3_accuracy"],
            val_metrics["loss"],
            val_metrics["cross_entropy_loss"],
            val_metrics["value_preference_loss"],
            val_metrics["top1_accuracy"],
            val_metrics["top3_accuracy"],
            val_metrics.get("val_ppg", float("nan")),
            val_metrics.get("none_ppg", float("nan")),
        )

        current_key = _best_metric_key(val_metrics, ppg_enabled=config.ppg.enabled)
        is_best = best_key is None or current_key > best_key
        if is_best:
            best_key = current_key
            best_val_metrics = dict(val_metrics)

        epoch_checkpoint_name = _epoch_checkpoint_name(epoch, val_metrics)
        save_checkpoint_fn(
            config.output_dir / epoch_checkpoint_name,
            model,
            optimizer,
            epoch,
            config,
            data_spec,
            val_metrics,
            input_contract=input_contract,
            scheduler=scheduler,
            best_key=best_key,
            best_val_metrics=best_val_metrics,
        )
        if is_best:
            save_checkpoint_fn(
                config.output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                config,
                data_spec,
                val_metrics,
                input_contract=input_contract,
                scheduler=scheduler,
                best_key=best_key,
                best_val_metrics=best_val_metrics,
            )

    save_checkpoint_fn(
        config.output_dir / "final.pt",
        model,
        optimizer,
        config.max_epochs,
        config,
        data_spec,
        last_val_metrics,
        input_contract=input_contract,
        scheduler=scheduler,
        best_key=best_key,
        best_val_metrics=best_val_metrics,
    )
    return {
        "data_spec": data_spec,
        "best_val_score": None if best_key is None else best_key[0],
        "best_val_top1_accuracy": best_val_metrics.get("top1_accuracy", 0.0),
        "best_val_ppg": best_val_metrics.get("val_ppg", 0.0),
        "last_val_metrics": last_val_metrics,
        "output_dir": config.output_dir,
    }


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    precision: str = "float32",
    *,
    value_preference: ValuePreferenceConfig | None = None,
    runtime_debug: RuntimeDebugRecorder | None = None,
    epoch: int = 0,
) -> dict[str, float]:
    model.train()
    totals = MetricAccumulator(_EPOCH_METRICS, device=device)
    repetition = getattr(model, "repetition", RepetitionConfig())
    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        if runtime_debug is not None:
            runtime_debug.begin_step(epoch=epoch, step=step, batch=batch)
        try:
            with _debug_stage(runtime_debug, "batch_to_device"):
                batch = prepare_repetition_penalty(batch, repetition)
                batch = move_batch(batch, device, non_blocking=device.type == "cuda")
            with _debug_stage(runtime_debug, "forward"):
                with autocast_context(device, precision):
                    output = model(batch)
            with _debug_stage(runtime_debug, "training_loss"):
                total_loss, value_loss = _resolve_training_loss(
                    output,
                    batch,
                    value_preference=value_preference,
                )
            with _debug_stage(runtime_debug, "backward"):
                total_loss.backward()
            with _debug_stage(runtime_debug, "clip_grad_norm"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            with _debug_stage(runtime_debug, "optimizer_step"):
                optimizer.step()
            with _debug_stage(runtime_debug, "scheduler_step"):
                scheduler.step()
        finally:
            if runtime_debug is not None:
                runtime_debug.end_step()
        count = batch["label_index"].shape[0]
        totals.update(
            {
                "loss": total_loss,
                "cross_entropy_loss": output["loss"],
                "value_preference_loss": value_loss,
                "top1_accuracy": output["top1_accuracy"],
                "top3_accuracy": output["top3_accuracy"],
            },
            weight=count,
        )
    return totals.mean()


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    precision: str = "float32",
    *,
    value_preference: ValuePreferenceConfig | None = None,
) -> dict[str, float]:
    model.eval()
    totals = MetricAccumulator(_EPOCH_METRICS, device=device)
    repetition = getattr(model, "repetition", RepetitionConfig())
    for batch in loader:
        batch = prepare_repetition_penalty(batch, repetition)
        batch = move_batch(batch, device, non_blocking=device.type == "cuda")
        with autocast_context(device, precision):
            output = model(batch)
        total_loss, value_loss = _resolve_training_loss(
            output,
            batch,
            value_preference=value_preference,
        )
        count = batch["label_index"].shape[0]
        totals.update(
            {
                "loss": total_loss,
                "cross_entropy_loss": output["loss"],
                "value_preference_loss": value_loss,
                "top1_accuracy": output["top1_accuracy"],
                "top3_accuracy": output["top3_accuracy"],
            },
            weight=count,
        )
    return totals.mean()


def _resolve_training_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    value_preference: ValuePreferenceConfig | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """组合基础交叉熵和可选的价值辅助损失。"""
    config = value_preference or ValuePreferenceConfig()
    if not config.enabled:
        return output["loss"], output["loss"].new_zeros(())
    value_loss = compute_value_preference_loss(
        output["logits"],
        batch,
        config,
    )
    return output["loss"] + config.loss_weight * value_loss, value_loss


def _save_checkpoint_default(*args, **kwargs):
    return _save_checkpoint(*args, **kwargs)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lr_lambda(current_step: int, warmup_steps: int, total_steps: int) -> float:
    if current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

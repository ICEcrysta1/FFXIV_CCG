"""BC 训练编排：初始化、逐轮调用、监控和 checkpoint 生命周期。"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from common.configuration.environment import resolve_registered_job_tags
from common.policy.config import PROJECT_ROOT, resolve_policy_cache_dir
from common.policy.data import DataSpec, ModelInputContract
from common.policy.data.encoding.context_encoding import ContextEncoder
from common.policy.model import CausalPolicyModel
from common.policy.model.execution.causal_encoder import run_causal_encoder
from common.pytorch.precision import load_precision_config
from common.pytorch.runtime import autocast_context, model_dtype
from common.training.optimizer import build_optimizer
from common.training.tensorboard import (
    close_tensorboard_writer,
    create_tensorboard_writer,
    write_scalar_metrics,
)
from common.training.tensorboard_monitor import TensorBoardMonitor

from ..config import RunConfig
from ..runtime import RuntimeDebugRecorder
from .checkpoint import (
    _best_metric_key,
    _checkpoint_metrics,
    _epoch_checkpoint_name,
    _has_resume_data_mismatch,
    _load_resume_checkpoint,
    _restore_best_state,
    _restore_rng_state,
    _restore_scheduler_state,
    _save_checkpoint,
    _top1_val_ppg_average,
    _validate_resume_checkpoint,
)
from .dataloaders import build_dataloaders
from .epochs import train_epoch, validate
from .progress import format_metrics

logger = logging.getLogger(__name__)


def run_training(
    config: RunConfig,
    *,
    raw_paths: list[Path],
    validation_paths: list[Path],
    output_dir: Path | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    device_name: str = "cuda",
    validation_metrics_callback=None,
    resume_path: Path | None = None,
    force_resume_data_mismatch: bool = False,
) -> dict[str, object]:
    """执行完整 BC 训练并保存逐轮、best/final checkpoint。"""
    tensorboard_output_dir = config.output_dir
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
    model_variant = config.model_variant
    if not isinstance(model_variant, str) or not model_variant.strip():
        raise ValueError("training model_variant must be configured before initialization")
    if model_variant != model_variant.strip():
        config = replace(config, model_variant=model_variant.strip())
    if not validation_paths:
        raise FileNotFoundError("no prepared VAL compiled caches supplied for validation")

    precision = load_precision_config()
    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(
        raw_paths,
        config,
        validation_paths=validation_paths,
        int_dtype=precision.resolve_int_dtype(),
        float_dtype=precision.resolve_float_dtype(),
        cache_dir=(resolve_policy_cache_dir(config.job_tag) if config.job_tag else None),
    )
    data_spec = DataSpec.from_dataset(train_dataset)
    if config.job_tag is not None and config.job_tag != data_spec.job_tag:
        raise ValueError(f"config job_tag {config.job_tag!r} != cache job_tag {data_spec.job_tag!r}")
    if data_spec.job_tag not in resolve_registered_job_tags(PROJECT_ROOT):
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
        skill_vocab=train_dataset.skill_vocab,
    )
    resume_epoch = 0
    resume_data_mismatch = False
    if resume_checkpoint is not None:
        resume_data_mismatch = _has_resume_data_mismatch(resume_checkpoint, config)
        resume_epoch = _validate_resume_checkpoint(
            resume_checkpoint,
            data_spec=data_spec,
            dataset=train_dataset,
            config=config,
            input_contract=input_contract,
            force_resume_data_mismatch=force_resume_data_mismatch,
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
    context_encoder = ContextEncoder(normalizer, input_contract.schema, layout=input_contract.schema.state_layout(data_spec.action_keys)).to(device=device)
    vocab = train_dataset.skill_vocab
    model = CausalPolicyModel(
        data_spec,
        config.model,
        vocab_size=vocab.size(),
        repetition=config.repetition,
    ).to(device=device)
    # 输入投影与 HC 保留 FP32；主干与输出头沿用既有参数和梯度精度。
    parameter_dtype = torch.float32 if config.fp32_gradients else model_dtype(config.precision)
    model.set_compute_dtype(parameter_dtype)
    if config.activation_checkpoint_ffn:
        model.enable_activation_checkpoint_ffn()
    if config.activation_checkpoint_attention:
        model.enable_activation_checkpoint_attention(
            block=config.activation_checkpoint_attention_block
        )
    runtime_debug = RuntimeDebugRecorder(
        device=device,
        config=config.runtime_debug,
        output_path=config.output_dir / config.runtime_debug.output_filename,
        context_capacity=config.model.context_capacity,
    )
    debug_recorder = runtime_debug if runtime_debug.enabled else None
    set_runtime_debug = getattr(model, "set_runtime_debug", None)
    if callable(set_runtime_debug):
        set_runtime_debug(debug_recorder)
    logger.info(
        "模型: job=%s model_variant=%s actions=%d state=%d scene=%d skill_features=%d layers=%d d_model=%d activation=%s ff_dim=%d precision=%s"
        " input_encoder_parameter_dtype=%s encoder_parameter_dtype=%s output_head_parameter_dtype=%s"
        " ffn_checkpoint=%s attention_checkpoint=%s device=%s",
        data_spec.job_tag,
        config.model_variant,
        data_spec.num_actions,
        data_spec.state_dim,
        data_spec.scene_dim,
        data_spec.skill_feature_dim,
        config.model.n_layers,
        config.model.d_model,
        config.model.transformer_activation,
        config.model.ff_dim,
        config.precision,
        next(model.input_encoder.parameters()).dtype,
        next(model.encoder.parameters()).dtype,
        next(model.output_head.parameters()).dtype,
        config.activation_checkpoint_ffn,
        config.activation_checkpoint_attention,
        device,
    )

    optimizer = build_optimizer(
        model,
        config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    logger.info(
        "BC 优化器: %s learning_rate=%g weight_decay=%g",
        config.optimizer.name,
        config.learning_rate,
        config.weight_decay,
    )
    total_steps = max(1, len(train_loader) * config.max_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_lambda(step, config.warmup_steps, total_steps),
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_writer = create_tensorboard_writer(
        config.tensorboard,
        tensorboard_output_dir,
        run_name="bc",
        model_variant=config.model_variant,
    )
    tensorboard_monitor = None
    global_step = 0
    last_train_metrics: dict[str, float] = {}
    last_val_metrics: dict[str, float] = {}
    try:
        if tensorboard_writer is not None:
            logger.info("TensorBoard events: %s", tensorboard_writer.log_dir)
        best_key: tuple[float, float, float, float] | None = None
        best_val_metrics: dict[str, float] = {}
        start_epoch = 1
        if resume_checkpoint is not None:
            model.load_state_dict(resume_checkpoint["model_state_dict"])
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            for group in optimizer.param_groups:
                group["weight_decay"] = config.weight_decay
            _restore_scheduler_state(
                scheduler,
                resume_checkpoint,
                completed_steps=resume_epoch * len(train_loader),
            )
            start_epoch = resume_epoch + 1
            if resume_data_mismatch:
                logger.info(
                    "当前以强制数据上限不匹配模式续训，清空旧 checkpoint 的 best 基线，"
                    "改用当前训练/验证数据重新评估。"
                )
            else:
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

        if device.type == "cuda":
            _warmup_compiled_training(model, config)

        # 调度器保存真实累计更新数；强制调整数据量续训时不能按新 loader 长度重算历史步号。
        global_step = max(0, int(scheduler.last_epoch))
        tensorboard_monitor = (
            TensorBoardMonitor(
                tensorboard_writer,
                config.tensorboard,
                model=model,
                vocab=vocab,
                initial_step=global_step,
                metadata={
                    "training": asdict(config),
                    "resume_path": str(resume_path) if resume_path is not None else None,
                    "resume_epoch": resume_epoch,
                },
            )
            if tensorboard_writer is not None
            else None
        )
        for epoch in range(start_epoch, config.max_epochs + 1):
            train_epoch_kwargs = {
                "action_quality": config.action_quality_loss,
                "value_preference": config.value_preference,
                "runtime_debug": debug_recorder,
                "epoch": epoch,
                "max_epochs": config.max_epochs,
                "context_encoder": context_encoder,
            }
            if tensorboard_writer is not None:
                train_epoch_kwargs.update(
                    tensorboard_writer=tensorboard_writer,
                    tensorboard_monitor=tensorboard_monitor,
                    tensorboard_log_every_steps=config.tensorboard.log_every_steps,
                    global_step_offset=global_step,
                )
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                config.precision,
                **train_epoch_kwargs,
            )
            global_step += len(train_loader)
            last_train_metrics = train_metrics
            validation_kwargs = {
                "value_preference": config.value_preference,
                "epoch": epoch,
                "max_epochs": config.max_epochs,
                "context_encoder": context_encoder,
            }
            if tensorboard_monitor is not None:
                validation_kwargs.update(
                    tensorboard_monitor=tensorboard_monitor,
                    global_step=global_step,
                )
            val_metrics = validate(
                model,
                val_loader,
                device,
                config.precision,
                **validation_kwargs,
            )
            last_val_metrics = val_metrics
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
            if tensorboard_writer is not None:
                epoch_step = global_step
                write_scalar_metrics(
                    tensorboard_writer,
                    train_metrics,
                    prefix="train/epoch",
                    global_step=epoch_step,
                )
                write_scalar_metrics(
                    tensorboard_writer,
                    val_metrics,
                    prefix="validation",
                    global_step=epoch_step,
                )
                write_scalar_metrics(
                    tensorboard_writer, {"index": epoch}, prefix="epoch", global_step=epoch_step,
                )
                tensorboard_monitor.log_epoch(global_step=epoch_step)
            if config.ppg.enabled:
                logger.info(
                    "\n%s",
                    format_metrics(val_metrics, epoch=f"{epoch}/{config.max_epochs}"),
                )

            current_key = _best_metric_key(val_metrics, ppg_enabled=config.ppg.enabled)
            is_best = best_key is None or current_key > best_key
            if is_best:
                best_key = current_key
                best_val_metrics = dict(val_metrics)

            checkpoint_names = [_epoch_checkpoint_name(epoch, val_metrics)]
            if is_best:
                checkpoint_names.append("best.pt")
            for checkpoint_name in checkpoint_names:
                _save_checkpoint(
                    config.output_dir / checkpoint_name,
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

        _save_checkpoint(
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
    finally:
        try:
            if tensorboard_monitor is not None:
                tensorboard_monitor.finish(
                    {
                        **{f"train/epoch/{key}": value for key, value in last_train_metrics.items()},
                        **{f"validation/{key}": value for key, value in last_val_metrics.items()},
                    },
                    global_step=max(global_step, int(scheduler.last_epoch)),
                )
        finally:
            close_tensorboard_writer(tensorboard_writer)


def _warmup_compiled_training(model: CausalPolicyModel, config: RunConfig) -> None:
    """正式计时前预热真实 encoder 的前后向，不读取数据或改写参数梯度。"""
    if config.model.full_attention_residuals and config.model.transformer_activation != "swiglu":
        return
    device = next(model.parameters()).device
    started = time.perf_counter()
    # 1 会单独特化；多 token 先用与固定特征宽度不同的长度，避免绑定无关维度。
    batch_sizes = (2, 1) if config.batch_size > 1 else (1,)
    lengths = tuple(dict.fromkeys((min(7, config.model.context_capacity), 1)))
    logger.info("CUDA 编译预热开始：首次运行可能需要数分钟，完成后开始训练计时")
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        for batch_size in batch_sizes:
            for length in lengths:
                logger.info("编译预热：batch=%d tokens=%d，执行前向和反向", batch_size, length)
                tokens = torch.randn(
                    batch_size, length, config.model.d_model,
                    device=device, dtype=torch.float32, requires_grad=True,
                )
                with autocast_context(device, config.precision):
                    hidden, _, _ = run_causal_encoder(model.encoder, {
                        "tokens": tokens,
                        "valid": torch.ones(batch_size, length, device=device, dtype=torch.bool),
                    })
                # 只请求输入梯度，触发所有局部 AOT 反向但不写入 parameter.grad。
                torch.autograd.grad(hidden.sum(), tokens)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    logger.info("CUDA 编译预热完成，耗时 %.1f 秒", time.perf_counter() - started)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _lr_lambda(current_step: int, warmup_steps: int, total_steps: int) -> float:
    if current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

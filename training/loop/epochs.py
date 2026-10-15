"""单轮训练与验证：批次编码、损失、参数更新及指标采集。"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from common.policy.data.context_encoding import ContextEncoder
from common.policy.model.repetition import RepetitionConfig, prepare_repetition_penalty
from common.pytorch.runtime import autocast_context, move_batch
from common.training.metrics import MetricAccumulator
from common.training.tensorboard import write_scalar_metrics
from common.training.tensorboard_monitor import TensorBoardMonitor

from ..config import ActionQualityLossConfig, ValuePreferenceConfig
from ..runtime import RuntimeDebugRecorder
from .loss import LossBreakdown, compose_training_loss, configured_auxiliary_losses

_EPOCH_METRICS = ("loss", "cross_entropy_loss", "top1_accuracy", "top3_accuracy")


def _batch_metrics(output, losses: LossBreakdown) -> dict[str, torch.Tensor]:
    """训练和验证共用分项指标，汇总时仍按各批样本数加权。"""
    return {
        "loss": losses.total,
        "cross_entropy_loss": losses.primary,
        **losses.auxiliary,
        "top1_accuracy": output["top1_accuracy"],
        "top3_accuracy": output["top3_accuracy"],
    }


def _debug_stage(recorder: RuntimeDebugRecorder | None, name: str):
    """在关闭调试时返回零开销的空上下文。"""
    if recorder is None:
        return nullcontext()
    return recorder.stage(name)


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    precision: str = "float32",
    *,
    action_quality: ActionQualityLossConfig | None = None,
    value_preference: ValuePreferenceConfig | None = None,
    runtime_debug: RuntimeDebugRecorder | None = None,
    epoch: int = 0,
    tensorboard_writer=None,
    tensorboard_monitor: TensorBoardMonitor | None = None,
    tensorboard_log_every_steps: int = 50,
    global_step_offset: int = 0,
    context_encoder: ContextEncoder | None = None,
) -> dict[str, float]:
    model.train()
    auxiliary_losses = configured_auxiliary_losses(value_preference=value_preference)
    totals = MetricAccumulator(
        (*_EPOCH_METRICS, *(term.metric_name for term in auxiliary_losses)),
        device=device,
    )
    if tensorboard_writer is not None and tensorboard_log_every_steps < 1:
        raise ValueError("tensorboard_log_every_steps must be >= 1")
    interval_totals = (
        MetricAccumulator((*totals.names, "learning_rate", "gradient_norm"), device=device)
        if tensorboard_writer is not None
        else None
    )
    repetition = getattr(model, "repetition", RepetitionConfig())
    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        if runtime_debug is not None:
            runtime_debug.begin_step(epoch=epoch, step=step, batch=batch)
        try:
            with _debug_stage(runtime_debug, "batch_to_device"):
                batch = prepare_repetition_penalty(batch, repetition)
                batch = move_batch(batch, device, non_blocking=device.type == "cuda")
            if context_encoder is not None:
                with _debug_stage(runtime_debug, "context_encoding"):
                    batch = context_encoder.encode(batch)
            if tensorboard_monitor is not None:
                with autocast_context(device, precision):
                    tensorboard_monitor.log_graph(batch)
            with _debug_stage(runtime_debug, "forward"):
                with autocast_context(device, precision):
                    output = model(batch)
            with _debug_stage(runtime_debug, "training_loss"):
                losses = compose_training_loss(
                    output,
                    batch,
                    auxiliary_losses,
                    action_quality=action_quality,
                )
            with _debug_stage(runtime_debug, "backward"):
                losses.total.backward()
            with _debug_stage(runtime_debug, "clip_grad_norm"):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            global_step = global_step_offset + step
            if tensorboard_monitor is not None:
                tensorboard_monitor.log_gradients(global_step, grad_norm=grad_norm)
            # 记录这次更新真正使用的学习率，调度器推进后的值属于下一次更新。
            learning_rate = optimizer.param_groups[0]["lr"]
            with _debug_stage(runtime_debug, "optimizer_step"):
                optimizer.step()
            with _debug_stage(runtime_debug, "scheduler_step"):
                scheduler.step()
        finally:
            if runtime_debug is not None:
                runtime_debug.end_step()
        count = batch["label_index"].shape[0]
        step_metrics = _batch_metrics(output, losses)
        totals.update(step_metrics, weight=count)
        if interval_totals is not None:
            interval_totals.update(
                {
                    **step_metrics,
                    "learning_rate": torch.tensor(learning_rate, dtype=torch.float64, device=device),
                    "gradient_norm": torch.as_tensor(grad_norm, device=device),
                },
                weight=count,
            )
        if tensorboard_monitor is not None:
            tensorboard_monitor.log_step(
                output["logits"], global_step=global_step, labels=batch["label_index"],
            )
        if tensorboard_writer is not None and (
            step % tensorboard_log_every_steps == 0 or step == len(loader)
        ):
            write_scalar_metrics(
                tensorboard_writer,
                interval_totals.mean(),
                prefix="train/step",
                global_step=global_step_offset + step,
            )
            # 每个点汇总本日志区间的所有样本，末尾不足一个区间也会写入。
            interval_totals = MetricAccumulator(interval_totals.names, device=device)
    return totals.mean()


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    precision: str = "float32",
    *,
    value_preference: ValuePreferenceConfig | None = None,
    context_encoder: ContextEncoder | None = None,
    tensorboard_monitor: TensorBoardMonitor | None = None,
    global_step: int = 0,
) -> dict[str, float]:
    model.eval()
    auxiliary_losses = configured_auxiliary_losses(value_preference=value_preference)
    totals = MetricAccumulator(
        (*_EPOCH_METRICS, *(term.metric_name for term in auxiliary_losses)),
        device=device,
    )
    repetition = getattr(model, "repetition", RepetitionConfig())
    if tensorboard_monitor is not None:
        tensorboard_monitor.begin_validation()
    for batch in loader:
        batch = prepare_repetition_penalty(batch, repetition)
        batch = move_batch(batch, device, non_blocking=device.type == "cuda")
        if context_encoder is not None:
            batch = context_encoder.encode(batch)
        with autocast_context(device, precision):
            output = model(batch)
        if tensorboard_monitor is not None:
            tensorboard_monitor.update_validation(output["logits"], batch["label_index"])
        losses = compose_training_loss(
            output,
            batch,
            auxiliary_losses,
        )
        count = batch["label_index"].shape[0]
        totals.update(
            _batch_metrics(output, losses),
            weight=count,
        )
    if tensorboard_monitor is not None:
        tensorboard_monitor.end_validation(global_step=global_step)
    return totals.mean()

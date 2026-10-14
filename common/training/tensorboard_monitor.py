"""BC 与 GRPO 共用的模型诊断；只观察已有训练数据，不参与优化。"""

from __future__ import annotations

import json
import logging
import math
import warnings
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass

import torch
from torch import nn

from .tensorboard_labels import tensorboard_tag

logger = logging.getLogger(__name__)


def _serializable(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def _hparams(value, prefix="") -> dict[str, object]:
    """HParams 只接受扁平标量；配置序列保存为可读 JSON。"""
    value = _serializable(value)
    if isinstance(value, Mapping):
        result = {}
        for name, item in value.items():
            result.update(_hparams(item, f"{prefix}/{name}".strip("/")))
        return result
    if value is None:
        value = "null"
    elif isinstance(value, float) and not math.isfinite(value):
        value = str(value)
    elif not isinstance(value, (str, bool, int, float)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return {prefix: value} if prefix else {}


def _metadata_cell(value) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


class _GraphModel(nn.Module):
    """用张量位置参数和单张量输出适配 TensorBoard 的 JIT tracer。"""

    def __init__(self, model, keys, metadata):
        super().__init__()
        self.model = model
        self.keys = keys
        self.metadata = metadata

    def forward(self, *values):
        return self.model({**self.metadata, **dict(zip(self.keys, values))})["logits"]


class TensorBoardMonitor:
    """有界、确定性的诊断采样，禁用时不额外前向或读取设备标量。"""

    def __init__(self, writer, config, *, model, vocab=None, metadata=None, initial_step=0):
        self.writer = writer
        self.config = config
        self.model = model
        self.vocab = vocab
        self.enabled = writer is not None and config.enabled
        self._graph_attempted = False
        self._finished = False
        self._epoch = 0
        self._last_embedding_step = None
        self.action_keys = tuple(getattr(getattr(model, "data_spec", None), "action_keys", ()))
        self._run_config = {
            "model": _serializable(getattr(model, "config", {})),
            "data_spec": _serializable(getattr(model, "data_spec", {})),
            "run": dict(metadata or {}),
        }
        self.begin_validation()
        if not self.enabled:
            return
        if self._supports("add_text"):
            self.writer.add_text(
                tensorboard_tag("run/configuration"),
                "```json\n" + json.dumps(self._run_config, ensure_ascii=False, indent=2, default=str) + "\n```",
                initial_step,
            )
            self.writer.add_text(tensorboard_tag("run/model"), "```text\n" + str(model) + "\n```", initial_step)
            self.writer.add_text(
                tensorboard_tag("run/metric_semantics"),
                "步号为累计优化器更新次数；训练曲线为记录区间的样本加权均值。"
                "learning_rate 是区间内更新实际使用的值。验证混淆矩阵覆盖全部有效验证样本，"
                "拒绝的非有限输出或无效标签由 diagnostic_rejected_samples 单独计数。"
                "行为克隆 PR 曲线使用有界确定性样本。Embedding 分别展示输入技能表和独立输出头，排除 padding 行。"
                "Graph 是首个真实编码样本的 eval 前向静态图，形状与分支只代表该样本。",
                initial_step,
            )
        if self._supports("add_custom_scalars"):
            grpo = self._run_config["run"].get("stage") == "grpo"
            panels = ({
                "损失": ["grpo/update/loss", "grpo/update/policy_loss", "grpo/update/kl"],
                "回放PPG": ["grpo/iteration/greedy_ppg_before", "grpo/iteration/greedy_ppg_after"],
                "学习率": ["grpo/update/learning_rate", "grpo/update/next_learning_rate"],
            } if grpo else {
                "损失": ["train/step/loss", "train/epoch/loss", "validation/loss"],
                "准确率": ["train/step/top1_accuracy", "validation/top1_accuracy", "validation/top3_accuracy"],
                "回放PPG": ["validation/val_ppg", "validation/none_ppg"],
                "学习率": ["train/step/learning_rate"],
            })
            self.writer.add_custom_scalars({
                "GRPO训练" if grpo else "行为克隆训练": {
                    title: ["Multiline", [tensorboard_tag(tag) for tag in tags]]
                    for title, tags in panels.items()
                },
            })
        self._log_embeddings(global_step=initial_step)

    def _supports(self, name):
        return self.enabled and callable(getattr(self.writer, name, None))

    def should_log_histograms(self, global_step: int) -> bool:
        return self.enabled and (global_step == 1 or global_step % self.config.histogram_every_steps == 0)

    def log_graph(self, batch) -> None:
        """仅追踪一次真实输入；完整恢复模式、随机数、KV 与运行时诊断状态。"""
        if self._graph_attempted or not self.config.graph or not self._supports("add_graph"):
            return
        self._graph_attempted = True
        inputs = {
            name: value.detach()[:1] if value.ndim else value.detach()
            for name, value in batch.items()
            if isinstance(value, torch.Tensor) and name != "label_index"
        }
        if not inputs:
            return
        metadata = {
            name: value[:1] if isinstance(value, (list, tuple)) else value
            for name, value in batch.items() if not isinstance(value, torch.Tensor)
        }
        modules = list(self.model.modules())
        training_modes = [module.training for module in modules]
        saved = [
            (module, name, getattr(module, name))
            for module in modules
            for name in ("_kv_cache_enabled", "_kv_cache", "_runtime_debug")
            if hasattr(module, name)
        ]
        devices = sorted({tensor.device.index for tensor in inputs.values() if tensor.is_cuda})
        parameter = next(self.model.parameters(), None)
        # FP32 参数训练可在外层使用 BF16 autocast，图追踪必须沿用计算精度。
        autocast_dtype = (
            torch.get_autocast_dtype(parameter.device.type)
            if parameter is not None and torch.is_autocast_enabled(parameter.device.type)
            else parameter.dtype if parameter is not None else None
        )
        autocast = (
            torch.autocast(device_type=parameter.device.type, dtype=autocast_dtype, cache_enabled=False)
            if parameter is not None and autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad(), autocast:
                self.model.eval()
                for module, name, _value in saved:
                    setattr(module, name, False if name == "_kv_cache_enabled" else None)
                with warnings.catch_warnings():
                    # 静态图固定真实样本的形状；shape 分支警告不代表训练路径错误。
                    warnings.simplefilter("ignore", torch.jit.TracerWarning)
                    self.writer.add_graph(_GraphModel(self.model, tuple(inputs), metadata), tuple(inputs.values()))
        except Exception as exc:
            # 可视化失败不应终止一次训练，但必须向日志和面板报告。
            logger.warning("TensorBoard 模型图写入失败：%s", exc)
            if self._supports("add_text"):
                self.writer.add_text(tensorboard_tag("diagnostics/graph_error"), str(exc), 0)
        finally:
            for module, name, value in saved:
                setattr(module, name, value)
            for module, training in zip(modules, training_modes):
                module.training = training

    def _histogram(self, tag, values, global_step):
        if not self._supports("add_histogram"):
            return
        values = torch.as_tensor(values).detach()
        if values.is_sparse:
            values = values.coalesce().values()
        values = values.reshape(-1)
        stride = max(1, math.ceil(values.numel() / self.config.max_histogram_values))
        values = values[::stride].float().cpu()
        finite = torch.isfinite(values)
        invalid = int((~finite).sum())
        if invalid and self._supports("add_scalar"):
            self.writer.add_scalar(tensorboard_tag(f"diagnostics/nonfinite_samples/{tag}"), invalid, global_step)
        values = values[finite]
        if values.numel():
            self.writer.add_histogram(tensorboard_tag(tag), values, global_step, bins=64)

    def log_gradients(self, global_step: int, grad_norm=None) -> None:
        if not self.should_log_histograms(global_step):
            return
        if grad_norm is not None and self._supports("add_scalar"):
            norm = float(torch.as_tensor(grad_norm).detach().cpu())
            if math.isfinite(norm):
                self.writer.add_scalar(tensorboard_tag("train/grad_norm"), norm, global_step)
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                self._histogram(f"gradients/{name}", parameter.grad, global_step)

    def log_step(self, logits, *, global_step: int, prefix="train", labels=None) -> None:
        if not self.should_log_histograms(global_step):
            return
        logits = logits.detach().float()
        self.log_distributions({
            "logits": logits,
            "probabilities": logits.softmax(dim=-1),
            "predicted_action": logits.argmax(dim=-1),
            **({"label_action": labels} if labels is not None else {}),
        }, global_step=global_step, prefix=prefix)
        for name, parameter in self.model.named_parameters():
            self._histogram(f"parameters/{name}", parameter, global_step)

    def log_distributions(self, metrics: Mapping, *, global_step: int, prefix="rollout") -> None:
        if not self.enabled:
            return
        for name, values in metrics.items():
            if values is not None:
                self._histogram(f"{prefix}/{name}", values, global_step)

    def _log_embeddings(self, *, global_step):
        if not self._supports("add_embedding") or self._last_embedding_step == global_step:
            return
        self._last_embedding_step = global_step
        data_spec = getattr(self.model, "data_spec", None)
        encoder = getattr(self.model, "input_encoder", None)
        embedding = getattr(encoder, "skill_embed", None)
        action_rows = tuple(getattr(data_spec, "action_to_vocab_id", ()))
        is_gcd = tuple(getattr(data_spec, "action_is_gcd", ()))
        actions = {row: (key, "gcd" if gcd else "ogcd") for row, key, gcd in zip(action_rows, self.action_keys, is_gcd)}
        header = ["技能标识", "原始技能ID", "词表ID", "技能类型"]
        if embedding is not None and embedding.weight.shape[0] > 1:
            rows = range(1, embedding.weight.shape[0])
            metadata = []
            for row in rows:
                raw_id = self.vocab.reverse_lookup(row) if self.vocab is not None else ""
                key, kind = actions.get(row, (f"raw_skill_{raw_id}" if raw_id != "" else f"vocab_{row}", "history_only"))
                metadata.append([_metadata_cell(cell) for cell in (key, raw_id, row, kind)])
            self.writer.add_embedding(
                embedding.weight.detach()[1:].float().cpu(), metadata=metadata,
                metadata_header=header, tag=tensorboard_tag("embeddings/input_skills"), global_step=global_step,
            )
        output_head = getattr(self.model, "output_head", None)
        if output_head is not None and len(self.action_keys) == output_head.weight.shape[0]:
            metadata = [
                [_metadata_cell(cell) for cell in (key, self.vocab.reverse_lookup(row) if self.vocab else "", row, "gcd" if gcd else "ogcd")]
                for key, row, gcd in zip(self.action_keys, action_rows, is_gcd)
            ]
            self.writer.add_embedding(
                output_head.weight.detach().float().cpu(), metadata=metadata,
                metadata_header=header, tag=tensorboard_tag("embeddings/output_actions"), global_step=global_step,
            )

    def log_epoch(self, *, global_step: int) -> None:
        if not self.enabled:
            return
        self._epoch += 1
        for name, parameter in self.model.named_parameters():
            self._histogram(f"parameters/{name}", parameter, global_step)
        if self._epoch % self.config.embedding_every_epochs == 0:
            self._log_embeddings(global_step=global_step)

    def begin_validation(self) -> None:
        count = len(self.action_keys)
        self._confusion = torch.zeros((count, count), dtype=torch.int64)
        self._validation_seen = 0
        self._validation_rejected = 0
        self._pr_labels = torch.empty(0, dtype=torch.int64)
        self._pr_probs = torch.empty((0, count))
        self._pr_priorities = torch.empty(0, dtype=torch.int64)

    def update_validation(self, logits, labels) -> None:
        if not self.enabled or not self.action_keys:
            return
        count = len(self.action_keys)
        logits = logits.detach().float().cpu()
        labels = labels.detach().long().cpu().reshape(-1)
        if logits.shape != (labels.numel(), count):
            raise ValueError("TensorBoard validation logits must match labels and action_keys")
        valid = (labels >= 0) & (labels < count) & torch.isfinite(logits).all(dim=-1)
        self._validation_rejected += int((~valid).sum())
        logits, labels = logits[valid], labels[valid]
        predictions = logits.argmax(dim=-1)
        self._confusion += torch.bincount(labels * count + predictions, minlength=count * count).reshape(count, count)
        start = self._validation_seen
        self._validation_seen += labels.numel()
        if not self.config.pr_curves:
            return
        # 索引的可逆整数混合构成确定性 bottom-k reservoir，不占用训练 RNG，
        # 并避免仅保留前 N 个样本导致的副本/动作顺序偏差。
        priorities = torch.arange(start, self._validation_seen, dtype=torch.int64)
        for _ in range(2):
            priorities = ((priorities ^ (priorities >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
        priorities ^= priorities >> 16
        priorities = torch.cat((self._pr_priorities, priorities))
        labels = torch.cat((self._pr_labels, labels))
        probs = torch.cat((self._pr_probs, logits.softmax(dim=-1)))
        keep = priorities.topk(min(priorities.numel(), self.config.pr_curve_max_samples), largest=False).indices
        self._pr_priorities, self._pr_labels, self._pr_probs = priorities[keep], labels[keep], probs[keep]

    def end_validation(self, *, global_step: int) -> None:
        if not self.enabled:
            return
        if self._supports("add_scalar"):
            self.writer.add_scalar(tensorboard_tag("validation/diagnostic_samples"), self._validation_seen, global_step)
            self.writer.add_scalar(tensorboard_tag("validation/diagnostic_rejected_samples"), self._validation_rejected, global_step)
            self.writer.add_scalar(tensorboard_tag("validation/pr_sample_count"), self._pr_labels.numel(), global_step)
        if not self._validation_seen:
            return
        support = self._confusion.sum(dim=1)
        recall = self._confusion.diag().float() / support.clamp_min(1)
        precision = self._confusion.diag().float() / self._confusion.sum(dim=0).clamp_min(1)
        if self._supports("add_scalar"):
            for index, key in enumerate(self.action_keys):
                for name, values in (("support", support), ("recall", recall), ("precision", precision)):
                    self.writer.add_scalar(tensorboard_tag(f"validation/actions/{key}/{name}"), float(values[index]), global_step)
        if self._supports("add_text"):
            rows = ["| 真实动作 / 预测动作 | " + " | ".join(self.action_keys) + " |", "|---|" + "---|" * len(self.action_keys)]
            rows.extend("| " + key + " | " + " | ".join(map(str, row.tolist())) + " |" for key, row in zip(self.action_keys, self._confusion))
            self.writer.add_text(tensorboard_tag("validation/confusion_counts"), "\n".join(rows), global_step)
        if self._supports("add_figure"):
            from matplotlib import font_manager
            from matplotlib.figure import Figure

            # 仅为本图选择中文字体，不修改训练进程的全局绘图设置。
            available_fonts = {font.name for font in font_manager.fontManager.ttflist}
            family = next((name for name in (
                "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei",
            ) if name in available_fonts), "sans-serif")
            # 使用字体族列表，避免 sans-serif 被当作 fontconfig 表达式解析。
            font = font_manager.FontProperties(family=[family])
            figure = Figure(figsize=(max(6, len(self.action_keys) * 0.32), max(5, len(self.action_keys) * 0.30)))
            axis = figure.subplots()
            image = axis.imshow(self._confusion.float() / support.clamp_min(1).unsqueeze(1), vmin=0, vmax=1, cmap="Blues")
            axis.set_xticks(range(len(self.action_keys)), self.action_keys, rotation=90, fontsize=7)
            axis.set_yticks(range(len(self.action_keys)), self.action_keys, fontsize=7)
            axis.set_xlabel("预测动作", fontproperties=font)
            axis.set_ylabel("真实动作", fontproperties=font)
            axis.set_title("验证动作混淆矩阵（按真实动作归一化）", fontproperties=font)
            figure.colorbar(image, ax=axis)
            figure.tight_layout()
            self.writer.add_figure(tensorboard_tag("validation/confusion_matrix"), figure, global_step, close=True)
        if self.config.pr_curves and self._supports("add_pr_curve"):
            for index, key in enumerate(self.action_keys):
                targets = self._pr_labels == index
                if targets.any() and not targets.all():
                    self.writer.add_pr_curve(tensorboard_tag(f"validation/actions/{key}"), targets, self._pr_probs[:, index], global_step)

    def finish(self, metrics: Mapping, *, global_step: int) -> None:
        """只结束一次 HParams session，沿用当前 run，避免额外生成伪 run。"""
        if self._finished or not self._supports("add_hparams"):
            return
        self._finished = True
        finite_metrics = {}
        for name, value in metrics.items():
            try:
                scalar = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(scalar):
                finite_metrics[tensorboard_tag(name)] = scalar
        if finite_metrics:
            self.writer.add_hparams(_hparams(self._run_config), finite_metrics, run_name=".", global_step=global_step)

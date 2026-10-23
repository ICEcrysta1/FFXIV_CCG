"""训练优化器装配：线性矩阵、归一化及词表参数分别更新。"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy

import torch
from torch import nn

from common.policy.model.layers.residual_mix import HyperConnections

from .master_weights import LOW_PRECISION_DTYPES, FP32MasterOptimizer
from .optimizer_config import OptimizerConfig
from .sinkhorn import Sinkhorn

logger = logging.getLogger(__name__)


def _parameter_groups(
    model: nn.Module, weight_decay: float = 0.0, *, use_muon: bool = True,
) -> tuple[dict, ...]:
    """按实际模块职责分组，共享词表权重优先免衰减且只更新一次。"""
    matrix_names: set[str] = set()
    vocabulary_names: set[str] = set()
    no_decay_names: set[str] = set()
    output_head = getattr(model, "output_head", None)
    for prefix, module in model.named_modules(remove_duplicate=False):
        for local_name, parameter in module.named_parameters(recurse=False):
            name = f"{prefix}.{local_name}" if prefix else local_name
            if isinstance(module, nn.Embedding) or module is output_head:
                if parameter.ndim == 2:
                    vocabulary_names.add(name)
                no_decay_names.add(name)
            elif isinstance(module, HyperConnections):
                if local_name == "fn":
                    matrix_names.add(name)
                else:
                    no_decay_names.add(name)
            elif local_name.endswith("bias") or local_name.startswith("bias_"):
                no_decay_names.add(name)
            elif (isinstance(module, nn.Linear) and local_name == "weight") or (
                isinstance(module, nn.MultiheadAttention) and local_name in {
                    "in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight",
                }
            ):
                matrix_names.add(name)
    aliases: dict[int, set[str]] = defaultdict(set)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases[id(parameter)].add(name)
    groups = tuple(
        {"params": [], "param_names": [], "optimizer_name": name, "weight_decay": decay}
        for name, decay in (("muon", weight_decay), ("adamw", weight_decay),
                            ("adamw", 0.0), ("sinkhorn", 0.0))
    )
    for name, parameter in model.named_parameters():
        names = aliases[id(parameter)]
        if use_muon and names & vocabulary_names:
            group = groups[3]
        elif use_muon and parameter.ndim == 2 and names <= matrix_names:
            group = groups[0]
        else:
            group = groups[2 if names & no_decay_names else 1]
        group["params"].append(parameter)
        group["param_names"].append(name)
    if use_muon and not groups[0]["params"]:
        raise ValueError("Muon requires linear weight matrices")
    return groups if use_muon else groups[1:3]


class _HybridOptimizer(torch.optim.Optimizer):
    """组合各类更新规则，并向调度器暴露实际参数组。"""

    _STATE_FORMAT = "muon_adamw_sinkhorn_v1"

    def __init__(self, muon: torch.optim.Optimizer, adamw: torch.optim.AdamW, sinkhorn: Sinkhorn):
        self._optimizers = (muon, adamw, sinkhorn)
        super().__init__([group for optimizer in self._optimizers for group in optimizer.param_groups], defaults={})
        self._parameter_contract = [
            {
                "optimizer_name": group["optimizer_name"],
                "parameters": [
                    {"name": name, "shape": list(parameter.shape)}
                    for name, parameter in zip(
                        group["param_names"], group["params"], strict=True,
                    )
                ],
            }
            for group in self.param_groups
        ]
        self._bind_native_state()

    def _bind_native_state(self) -> None:
        """原生 load 会替换分组字典，恢复后必须重新绑定调度器所见的引用。"""
        self.param_groups = [
            group for optimizer in self._optimizers for group in optimizer.param_groups
        ]
        self.state = defaultdict(dict)
        for optimizer in self._optimizers:
            self.state.update(optimizer.state)

    @torch.no_grad()
    def step(self, closure=None):
        """closure 只执行一次，再分别更新互不重叠的参数组。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self._optimizers:
            optimizer.step()
        self._bind_native_state()
        return loss

    def state_dict(self) -> dict:
        """分别保存原生状态，额外记录参数名称与形状，防止错组恢复。"""
        return {
            "format": self._STATE_FORMAT,
            "parameter_contract": deepcopy(self._parameter_contract),
            **{name: optimizer.state_dict() for name, optimizer in
               zip(("muon", "adamw", "sinkhorn"), self._optimizers, strict=True)},
        }

    def load_state_dict(self, state_dict: Mapping) -> None:
        """拒绝 AdamW 旧状态及结构、顺序不一致的混合优化器状态。"""
        if state_dict.get("format") != self._STATE_FORMAT:
            raise ValueError("Muon optimizer state format mismatch")
        if state_dict.get("parameter_contract") != self._parameter_contract:
            raise ValueError("Muon optimizer parameter grouping mismatch")
        # 先检查所有分组再执行恢复，避免后续组不匹配时前面的状态已被改写。
        for name, optimizer in zip(("muon", "adamw", "sinkhorn"), self._optimizers, strict=True):
            native_state = state_dict.get(name)
            if not isinstance(native_state, Mapping) or not isinstance(
                native_state.get("state"), Mapping,
            ):
                raise TypeError(f"Muon optimizer missing {name} state")
            saved_groups = native_state.get("param_groups")
            if not isinstance(saved_groups, list) or len(saved_groups) != len(optimizer.param_groups):
                raise ValueError(f"Muon optimizer {name} grouping mismatch")
            for saved_group, expected in zip(saved_groups, optimizer.param_groups, strict=True):
                if (
                    not isinstance(saved_group, Mapping)
                    or saved_group.get("optimizer_name") != name
                    or saved_group.get("param_names") != expected["param_names"]
                    or not isinstance(saved_group.get("params"), list)
                    or len(saved_group["params"]) != len(expected["params"])
                ):
                    raise ValueError(f"Muon optimizer {name} grouping mismatch")
        for name, optimizer in zip(("muon", "adamw", "sinkhorn"), self._optimizers, strict=True):
            optimizer.load_state_dict(dict(state_dict[name]))
        self._bind_native_state()


def build_optimizer(
    model: nn.Module,
    optimizer_config: OptimizerConfig,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """统一算法分组；低精度模型通过 FP32 主权重累计更新。"""
    if optimizer_config.name not in {"adamw", "muon"}:
        raise ValueError(f"unsupported optimizer: {optimizer_config.name}")
    use_muon = optimizer_config.name == "muon"
    muon_class = getattr(torch.optim, "Muon", None)
    if use_muon and muon_class is None:
        raise RuntimeError("Muon requires a PyTorch version with torch.optim.Muon")
    groups = list(_parameter_groups(model, weight_decay, use_muon=use_muon))
    primary = optimizer_config.muon if use_muon else optimizer_config.adamw
    reference_lr = learning_rate if primary.learning_rate is None else primary.learning_rate
    for group in groups:
        settings = getattr(optimizer_config, group["optimizer_name"])
        # 阶段学习率覆盖按同一比例作用于各组，保留各算法配置的相对学习率。
        group_lr = reference_lr if settings.learning_rate is None else settings.learning_rate
        group["lr"] = learning_rate * (group_lr / reference_lr)
    if use_muon and optimizer_config.adamw.weight_decay is not None:
        groups[1]["weight_decay"] = optimizer_config.adamw.weight_decay

    def build_native(groups: list[dict]) -> torch.optim.Optimizer:
        adamw_config = optimizer_config.adamw
        adamw = torch.optim.AdamW(
            groups[1:3] if use_muon else groups, lr=learning_rate, weight_decay=weight_decay,
            betas=adamw_config.betas, eps=adamw_config.eps, amsgrad=adamw_config.amsgrad,
        )
        if not use_muon:
            return adamw
        muon_config = optimizer_config.muon
        muon = muon_class(
            [groups[0]], lr=learning_rate, weight_decay=weight_decay,
            momentum=muon_config.momentum, nesterov=muon_config.nesterov,
            ns_steps=muon_config.ns_steps, ns_coefficients=muon_config.ns_coefficients,
            adjust_lr_fn=muon_config.adjust_lr_fn, eps=muon_config.eps,
        )
        sinkhorn_config = optimizer_config.sinkhorn
        sinkhorn = Sinkhorn(
            [groups[3]], lr=learning_rate, momentum=sinkhorn_config.momentum,
            nesterov=sinkhorn_config.nesterov, steps=sinkhorn_config.iters,
            threshold=sinkhorn_config.threshold, eps=sinkhorn_config.eps,
            lr_scale=sinkhorn_config.lr_scale,
        )
        return _HybridOptimizer(muon, adamw, sinkhorn)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    for group in groups:
        count = sum(parameter.numel() for parameter in group["params"])
        logger.info(
            "优化器分组: optimizer=%s learning_rate=%g weight_decay=%g tensors=%d parameters=%d proportion=%.4f%%",
            group["optimizer_name"], group["lr"], group["weight_decay"], len(group["params"]), count,
            100.0 * count / total_parameters,
        )
    logger.info("AdamW 参数: %s", optimizer_config.adamw)
    if use_muon:
        logger.info("Muon 参数: %s；Sinkhorn 参数: %s", optimizer_config.muon, optimizer_config.sinkhorn)
    if any(parameter.dtype in LOW_PRECISION_DTYPES for parameter in model.parameters()):
        return FP32MasterOptimizer(groups, build_native)
    return build_native(groups)

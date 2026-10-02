"""训练优化器装配：主干矩阵使用 Muon，其余参数保留 AdamW。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
import logging

import torch
from torch import nn

from .master_weights import FP32MasterOptimizer, LOW_PRECISION_DTYPES
from .optimizer_config import OptimizerConfig


logger = logging.getLogger(__name__)


def _muon_parameter_names(model: nn.Module) -> set[str]:
    """只选择 Transformer 层内已知 attention/FFN 的线性权重。"""
    names: set[str] = set()
    layers = getattr(getattr(model, "encoder", None), "layers", None)
    if not isinstance(layers, nn.Module):
        return names
    for layer_name, layer in layers.named_children():
        prefix = f"encoder.layers.{layer_name}"
        attention = getattr(layer, "self_attn", None)
        if isinstance(attention, nn.Module):
            for module_name, module in attention.named_modules():
                if isinstance(module, nn.Linear):
                    path = f".{module_name}" if module_name else ""
                    names.add(f"{prefix}.self_attn{path}.weight")
            if isinstance(attention, nn.MultiheadAttention):
                # 原生 MHA 的 Q/K/V 并非 Linear 子模块，需显式处理其投影权重。
                for projection in (
                    "in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight",
                ):
                    if getattr(attention, projection, None) is not None:
                        names.add(f"{prefix}.self_attn.{projection}")
        for projection in ("linear1", "linear2", "gate_proj"):
            if isinstance(getattr(layer, projection, None), nn.Linear):
                names.add(f"{prefix}.{projection}.weight")
    return names


def _parameter_groups(model: nn.Module) -> tuple[dict, dict]:
    """按参数身份去重，并让与 embedding 等共享的权重保守留在 AdamW。"""
    allowed_names = _muon_parameter_names(model)
    aliases: dict[int, set[str]] = defaultdict(set)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases[id(parameter)].add(name)
    groups = tuple(
        {"params": [], "param_names": [], "optimizer_name": name}
        for name in ("muon", "adamw")
    )
    for name, parameter in model.named_parameters():
        use_muon = parameter.ndim == 2 and aliases[id(parameter)] <= allowed_names
        group = groups[0 if use_muon else 1]
        group["params"].append(parameter)
        group["param_names"].append(name)
    if not groups[0]["params"]:
        raise ValueError("Muon requires attention/FFN matrices under encoder.layers")
    return groups


class _MuonAdamW(torch.optim.Optimizer):
    """组合原生优化器，并向现有调度器暴露两组共享的参数字典。"""

    _STATE_FORMAT = "muon_adamw_v1"

    def __init__(self, muon: torch.optim.Optimizer, adamw: torch.optim.AdamW):
        self._optimizers = (muon, adamw)
        super().__init__([*muon.param_groups, *adamw.param_groups], defaults={})
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
        """closure 只执行一次，再分别更新互不重叠的两组参数。"""
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
            "muon": self._optimizers[0].state_dict(),
            "adamw": self._optimizers[1].state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping) -> None:
        """拒绝 AdamW 旧状态及结构、顺序不一致的混合优化器状态。"""
        if state_dict.get("format") != self._STATE_FORMAT:
            raise ValueError("Muon optimizer state format mismatch")
        if state_dict.get("parameter_contract") != self._parameter_contract:
            raise ValueError("Muon optimizer parameter grouping mismatch")
        # 先检查两套状态再执行恢复，避免第二组不匹配时第一组已经被改写。
        for name, expected in zip(("muon", "adamw"), self.param_groups, strict=True):
            native_state = state_dict.get(name)
            if not isinstance(native_state, Mapping) or not isinstance(
                native_state.get("state"), Mapping,
            ):
                raise ValueError(f"Muon optimizer missing {name} state")
            saved_groups = native_state.get("param_groups")
            if not isinstance(saved_groups, list) or len(saved_groups) != 1:
                raise ValueError(f"Muon optimizer {name} grouping mismatch")
            saved_group = saved_groups[0]
            if (
                not isinstance(saved_group, Mapping)
                or saved_group.get("optimizer_name") != name
                or saved_group.get("param_names") != expected["param_names"]
                or not isinstance(saved_group.get("params"), list)
                or len(saved_group["params"]) != len(expected["params"])
            ):
                raise ValueError(f"Muon optimizer {name} grouping mismatch")
        for name, optimizer in zip(("muon", "adamw"), self._optimizers, strict=True):
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
    if optimizer_config.name == "adamw":
        named_parameters = list(model.named_parameters())
        if any(parameter.dtype in LOW_PRECISION_DTYPES for _, parameter in named_parameters):
            return FP32MasterOptimizer(
                [{
                    "params": [parameter for _, parameter in named_parameters],
                    "param_names": [name for name, _ in named_parameters],
                    "optimizer_name": "adamw",
                }],
                lambda groups: torch.optim.AdamW(
                    groups, lr=learning_rate, weight_decay=weight_decay,
                ),
            )
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
    if optimizer_config.name != "muon":
        raise ValueError(f"unsupported optimizer: {optimizer_config.name}")
    muon_class = getattr(torch.optim, "Muon", None)
    if muon_class is None:
        raise RuntimeError("Muon requires a PyTorch version with torch.optim.Muon")
    muon_group, adamw_group = _parameter_groups(model)
    def build_native(groups: list[dict]) -> torch.optim.Optimizer:
        muon = muon_class(
            [groups[0]], lr=learning_rate, weight_decay=weight_decay,
            momentum=optimizer_config.momentum,
            nesterov=optimizer_config.nesterov,
            ns_steps=optimizer_config.ns_steps,
            adjust_lr_fn=optimizer_config.adjust_lr_fn,
        )
        adamw = torch.optim.AdamW(
            [groups[1]], lr=learning_rate, weight_decay=weight_decay,
        )
        return _MuonAdamW(muon, adamw)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    for group in (muon_group, adamw_group):
        count = sum(parameter.numel() for parameter in group["params"])
        logger.info(
            "优化器分组: optimizer=%s tensors=%d parameters=%d proportion=%.4f%%",
            group["optimizer_name"], len(group["params"]), count,
            100.0 * count / total_parameters,
        )
    logger.info(
        "Muon 参数: momentum=%s nesterov=%s ns_steps=%s adjust_lr_fn=%s",
        optimizer_config.momentum, optimizer_config.nesterov,
        optimizer_config.ns_steps, optimizer_config.adjust_lr_fn,
    )
    groups = [muon_group, adamw_group]
    if any(parameter.dtype in LOW_PRECISION_DTYPES for parameter in model.parameters()):
        return FP32MasterOptimizer(groups, build_native)
    return build_native(groups)

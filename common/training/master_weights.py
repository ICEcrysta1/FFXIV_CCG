"""低精度模型的 FP32 主权重：跨更新保留尚不足以改变 BF16 数值的小增量。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import logging

import torch


logger = logging.getLogger(__name__)
LOW_PRECISION_DTYPES = (torch.bfloat16, torch.float16)


class FP32MasterOptimizer(torch.optim.Optimizer):
    """模型负责低精度前后向，原生优化器只更新 FP32 主权重及其动量。"""

    _STATE_FORMAT = "fp32_master_v1"

    def __init__(
        self,
        parameter_groups: list[dict],
        optimizer_factory: Callable[[list[dict]], torch.optim.Optimizer],
    ):
        self._master_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._parameter_contract = []
        master_groups = []
        for group in parameter_groups:
            master_group = {**group, "params": []}
            contract = {"optimizer_name": group["optimizer_name"], "parameters": []}
            for name, parameter in zip(group["param_names"], group["params"], strict=True):
                contract["parameters"].append({
                    "name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                })
                if parameter.dtype in LOW_PRECISION_DTYPES:
                    master = parameter.detach().to(dtype=torch.float32, copy=True)
                    self._master_pairs.append((parameter, master))
                else:
                    # 混合精度模型中原本为 FP32 的参数不额外复制。
                    master = parameter
                master_group["params"].append(master)
            self._parameter_contract.append(contract)
            master_groups.append(master_group)
        self._optimizer = optimizer_factory(master_groups)
        super().__init__(self._optimizer.param_groups, self._optimizer.defaults)
        self._bind_native_state()
        logger.info(
            "低精度优化器启用 FP32 主权重与状态: tensors=%d parameters=%d；模型前向精度不变",
            len(self._master_pairs), sum(master.numel() for _, master in self._master_pairs),
        )

    def _bind_native_state(self) -> None:
        """调度器始终修改原生优化器当前使用的分组字典。"""
        self.param_groups = self._optimizer.param_groups
        self.state = self._optimizer.state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        try:
            for parameter, master in self._master_pairs:
                # 梯度裁剪已在模型梯度上完成；缺失梯度仍跳过更新和权重衰减。
                master.grad = None if parameter.grad is None else parameter.grad.detach().float()
            self._optimizer.step()
            for parameter, master in self._master_pairs:
                parameter.copy_(master)
            self._bind_native_state()
        finally:
            # FP32 梯度只服务本次更新，避免在下一次前后向期间额外常驻。
            for _, master in self._master_pairs:
                master.grad = None
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optimizer.zero_grad(set_to_none=set_to_none)
        for parameter, _ in self._master_pairs:
            if parameter.grad is None:
                continue
            if set_to_none:
                parameter.grad = None
            else:
                if parameter.grad.grad_fn is not None:
                    parameter.grad.detach_()
                else:
                    parameter.grad.requires_grad_(False)
                parameter.grad.zero_()

    def state_dict(self) -> dict:
        """主权重必须随动量保存，否则续训会丢失还没写入低精度模型的小更新。"""
        return {
            "format": self._STATE_FORMAT,
            "parameter_contract": deepcopy(self._parameter_contract),
            "master_weights": [master.detach() for _, master in self._master_pairs],
            "optimizer": self._optimizer.state_dict(),
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict: Mapping) -> None:
        if state_dict.get("format") == self._STATE_FORMAT:
            if state_dict.get("parameter_contract") != self._parameter_contract:
                raise ValueError("FP32 master optimizer parameter contract mismatch")
            weights = state_dict.get("master_weights")
            if not isinstance(weights, list) or len(weights) != len(self._master_pairs):
                raise ValueError("FP32 master optimizer weights mismatch")
            for weight, (_, master) in zip(weights, self._master_pairs, strict=True):
                if (
                    not isinstance(weight, torch.Tensor)
                    or weight.shape != master.shape
                    or weight.dtype != torch.float32
                ):
                    raise ValueError("FP32 master optimizer weight shape or dtype mismatch")
            native_state = state_dict.get("optimizer")
            if not isinstance(native_state, Mapping):
                raise ValueError("FP32 master optimizer missing native state")
            self._optimizer.load_state_dict(dict(native_state))
            for weight, (parameter, master) in zip(weights, self._master_pairs, strict=True):
                master.copy_(weight)
                parameter.copy_(master)
        else:
            if "master_weights" in state_dict or str(state_dict.get("format", "")).startswith("fp32_master"):
                raise ValueError("FP32 master optimizer state format mismatch")
            # 旧 checkpoint 没有亚 BF16 精度信息；由调用方先恢复模型，再据此初始化主权重。
            # 原生 load 会把旧低精度动量转换成主权重的 FP32 dtype。
            self._optimizer.load_state_dict(dict(state_dict))
            for parameter, master in self._master_pairs:
                master.copy_(parameter)
            logger.warning(
                "旧 optimizer checkpoint 没有 FP32 主权重，已从恢复后的模型权重初始化；"
                "此前被舍入的小更新无法恢复，后续更新将使用 FP32 累积。"
            )
        self._bind_native_state()
        self.zero_grad(set_to_none=True)

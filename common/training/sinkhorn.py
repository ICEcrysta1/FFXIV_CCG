"""按词表行和隐藏特征列平衡更新幅度的动量优化器。"""

from __future__ import annotations

import math

import torch


class Sinkhorn(torch.optim.Optimizer):
    """embedding 和输出矩阵使用行列 L2 平衡，不施加权重衰减。"""

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        steps: int = 11,
        threshold: float = 1e-3,
        eps: float = 1e-20,
        lr_scale: float = 0.2,
    ) -> None:
        for name, value in (
            ("lr", lr), ("momentum", momentum), ("threshold", threshold),
            ("eps", eps), ("lr_scale", lr_scale),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Sinkhorn {name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Sinkhorn {name} must be finite and >= 0")
        if momentum >= 1 or eps == 0 or lr_scale == 0:
            raise ValueError("Sinkhorn requires momentum < 1, eps > 0 and lr_scale > 0")
        if not isinstance(nesterov, bool):
            raise TypeError("Sinkhorn nesterov must be boolean")
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("Sinkhorn steps must be a positive odd integer")
        if steps < 1 or steps % 2 != 1:
            raise ValueError("Sinkhorn steps must be a positive odd integer")
        super().__init__(params, {
            "lr": lr, "momentum": momentum, "nesterov": nesterov,
            "steps": steps, "threshold": threshold, "eps": eps,
            "lr_scale": lr_scale, "weight_decay": 0.0,
        })
        for group in self.param_groups:
            if group["weight_decay"] != 0.0:
                raise ValueError("Sinkhorn does not support weight decay")
            if any(parameter.ndim != 2 or parameter.is_complex() for parameter in group["params"]):
                raise ValueError("Sinkhorn requires real matrix parameters")

    @torch.no_grad()
    def step(self, closure=None):
        """累计 FP32 动量，屏蔽近零行后交替归一化，最后恢复单位行 RMS。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            momentum = group["momentum"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("Sinkhorn does not support sparse gradients")
                gradient = parameter.grad.float()
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(gradient, alpha=1.0 - momentum)
                update = (
                    buffer.mul(momentum).add_(gradient, alpha=1.0 - momentum)
                    if group["nesterov"] else buffer.clone()
                )
                row_norm = torch.linalg.vector_norm(update, dim=1, keepdim=True)
                update.masked_fill_(row_norm <= group["threshold"] * row_norm.mean(), 0.0)
                for step in range(group["steps"]):
                    norm = torch.linalg.vector_norm(update, dim=1 if step % 2 == 0 else 0, keepdim=True)
                    update.div_(norm.add_(group["eps"]))
                scale = group["lr"] * group["lr_scale"] * math.sqrt(parameter.shape[1])
                parameter.add_(update, alpha=-scale)
        return loss

"""全模型共用的激活算法：统一由 ``model.transformer_activation`` 决定。"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from ..config import TRANSFORMER_ACTIVATIONS


# SwiGLU 是唯一带门控的取值，其余取值都是点到点激活。
GATED_ACTIVATION = "swiglu"
_POINTWISE_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "gelu": F.gelu,
    "relu": F.relu,
    # SwiGLU 的门控非线性就是 SiLU，非门控位置沿用同一个非线性。
    "swiglu": F.silu,
}


def _assert_supported(activation: str) -> None:
    if activation not in TRANSFORMER_ACTIVATIONS:
        raise ValueError("activation must be gelu, relu or swiglu")


def resolve_pointwise_activation(
    activation: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """返回配置取值对应的点到点激活函数。"""
    _assert_supported(activation)
    return _POINTWISE_ACTIVATIONS[activation]


def uses_gate(activation: str) -> bool:
    """判断该取值是否需要门控投影：SwiGLU 需要，GELU/ReLU 不需要。"""
    _assert_supported(activation)
    return activation == GATED_ACTIVATION


def gated_hidden_dim(
    activation: str,
    *,
    in_features: int,
    out_features: int,
    hidden_dim: int,
) -> int:
    """折算隐层宽度：门控三投影的矩阵参数量 ≈ 原来的两投影。"""
    if not uses_gate(activation):
        return hidden_dim
    two_projection = in_features * hidden_dim + hidden_dim * out_features
    return max(1, two_projection // (2 * in_features + out_features))


def activation_hidden(
    pointwise: Callable[[torch.Tensor], torch.Tensor],
    up: torch.Tensor,
    *,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """把上行投影合成为隐层激活：门控取值多乘一道非线性门控。

    ``pointwise`` 来自 ``resolve_pointwise_activation``；SwiGLU 传 ``gate``
    （即 ``down(SiLU(gate(x)) * up(x))`` 的前半段），其余取值只传 ``up``。
    """
    if gate is None:
        return pointwise(up)
    return pointwise(gate) * up

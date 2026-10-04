"""因果 Transformer 策略模型及其运行组件。"""

from .attention import build_causal_attention_mask
from .position_encoding import RotaryPositionEncoding
from .repetition import (
    RepetitionConfig,
    parse_repetition_config,
    repetition_config_from_checkpoint,
)


def __getattr__(name: str):
    """按需加载完整模型，避免配置与模型包互相初始化。"""
    if name == "CausalPolicyModel":
        from .model import CausalPolicyModel

        return CausalPolicyModel
    raise AttributeError(name)


__all__ = [
    "CausalPolicyModel",
    "RepetitionConfig",
    "RotaryPositionEncoding",
    "build_causal_attention_mask",
    "parse_repetition_config",
    "repetition_config_from_checkpoint",
]

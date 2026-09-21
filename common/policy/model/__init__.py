"""候选 Transformer 策略模型及其运行组件。"""

from .attention import build_split_attention_mask
from .position_encoding import RotaryPositionEncoding
from .repetition import (
    RepetitionConfig,
    parse_repetition_config,
    repetition_config_from_checkpoint,
)


def __getattr__(name: str):
    """按需加载完整模型，避免配置与模型包互相初始化。"""
    if name == "CandidateTransformerModel":
        from .model import CandidateTransformerModel

        return CandidateTransformerModel
    raise AttributeError(name)


__all__ = [
    "CandidateTransformerModel",
    "RepetitionConfig",
    "RotaryPositionEncoding",
    "build_split_attention_mask",
    "parse_repetition_config",
    "repetition_config_from_checkpoint",
]

"""预训练与 GRPO 共同使用的训练辅助组件。"""

from .metrics import MetricAccumulator
from .tensorboard import TensorBoardConfig

__all__ = ["MetricAccumulator", "TensorBoardConfig"]

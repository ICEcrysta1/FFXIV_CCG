"""科研分析 PNG 输出模块。"""

from .hidden_statistics import plot_hidden_statistics
from .attention import plot_opener_attention, plot_standard_attention_outputs
from .loss_landscape import plot_loss_landscape
from .pca_layers import plot_layer_pca
from .pair_embedding import plot_pair_embedding
from .skill_embedding import plot_skill_embedding

__all__ = [
    "plot_hidden_statistics",
    "plot_layer_pca",
    "plot_loss_landscape",
    "plot_pair_embedding",
    "plot_opener_attention",
    "plot_standard_attention_outputs",
    "plot_skill_embedding",
]

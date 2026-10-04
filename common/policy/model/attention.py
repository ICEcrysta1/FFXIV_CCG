"""因果注意力的可视化 mask 公共入口。"""

from .attention_masks import build_causal_attention_mask

__all__ = ["build_causal_attention_mask"]

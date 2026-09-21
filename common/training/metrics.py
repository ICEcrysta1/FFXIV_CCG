"""训练指标的设备端加权累积与集中读取。"""

from collections.abc import Mapping, Sequence

import torch


class MetricAccumulator:
    """只保留脱离计算图的定长累计值，汇报时一次搬回 CPU。"""

    def __init__(self, names: Sequence[str], *, device):
        self.names = tuple(names)
        # 与原来的 Python float 累加精度一致，避免长 epoch 的低精度累计误差。
        self.totals = torch.zeros(len(self.names), dtype=torch.float64, device=device)
        self.weight = 0

    def update(self, metrics: Mapping[str, torch.Tensor], *, weight: int = 1) -> None:
        values = torch.stack([metrics[name].detach() for name in self.names]).to(
            dtype=torch.float64,
        )
        self.totals.add_(values, alpha=weight)
        self.weight += weight

    def mean(self) -> dict[str, float]:
        totals = self.totals.cpu().tolist()
        return {name: value / max(1, self.weight) for name, value in zip(self.names, totals)}


__all__ = ["MetricAccumulator"]

"""加载 FFLogs 批量下载的数据集配额配置。"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from common.yaml_config import load_yaml_mapping

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "fflogs_scraper" / "batch.yaml"


def load_validation_ratio(path: Path = CONFIG_PATH) -> Decimal:
    """读取验证目标占训练目标的比例，并拒绝无效配置。"""
    config = load_yaml_mapping(path, description="FFLogs batch config")
    value = config.get("validation_ratio")
    if type(value) not in (int, float):
        raise ValueError("FFLogs batch validation_ratio 必须是大于 0 且不超过 1 的数值")
    ratio = Decimal(str(value))
    if not ratio.is_finite() or not 0 < ratio <= 1:
        raise ValueError("FFLogs batch validation_ratio 必须是大于 0 且不超过 1 的数值")
    return ratio

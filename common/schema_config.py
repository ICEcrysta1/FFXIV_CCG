"""共享 schema 配置加载（config/schema.yaml）。

C#（FightEngine）与 Python（common/）都以本文件为单一事实来源，
承载共享常量、scene context schema、canonical 输出 schema 与
状态向量字段模板；动态部分（职业量谱/Buff/DoT 列表）仍由运行时推导。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .yaml_config import load_yaml_mapping

SCHEMA_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "schema.yaml"


@lru_cache(maxsize=1)
def load_schema_config() -> dict[str, object]:
    """加载共享 schema 定义（进程内一次性缓存）。"""
    return load_yaml_mapping(SCHEMA_CONFIG_PATH)

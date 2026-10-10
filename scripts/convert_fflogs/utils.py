"""FFLogs 转换使用的职业静态配置和技能表。"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，以解析 common 配置/模型导入
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.config import load_project_config  # noqa: E402
from common.skills import SkillBook  # noqa: E402


def load_job_project_config(job_tag: str):
    """加载职业项目配置（静态数据，不依赖状态机后端）。"""
    return load_project_config(job_tag=job_tag)


def build_skill_book(project_config):
    """从项目配置构建技能索引（静态数据，系统技能 + 职业技能合并）。"""
    return SkillBook.from_project_config(project_config)

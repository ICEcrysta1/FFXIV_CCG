"""项目根目录环境变量加载。"""

import logging
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """只加载项目根目录 `.env`（环境变量优先）。"""
    project_root = Path(__file__).resolve().parents[3]
    env_path = project_root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
        logger.debug("已加载项目根目录 .env: %s", env_path)

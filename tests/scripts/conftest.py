"""scripts 测试公共 fixture：进程内 Rust 状态机后端。"""

from __future__ import annotations

import pytest

from scripts.common.inprocess_backend import InProcessEngine, _load_native_module
from scripts.convert_fflogs.utils import build_skill_book, load_job_project_config


def _require_inprocess_backend() -> None:
    try:
        _load_native_module()
    except RuntimeError as exc:
        if isinstance(exc.__cause__, ImportError):
            pytest.skip("Rust 状态机原生扩展未构建；请先执行 ./setup.sh")
        raise


@pytest.fixture
def cs_backend():
    """黑魔状态机后端（每个测试独立会话，结束后释放）。"""
    _require_inprocess_backend()
    with InProcessEngine("black_mage") as engine:
        with engine.create_backend(max_history=None) as backend:
            yield backend


@pytest.fixture
def cs_skill_book():
    """黑魔技能索引（静态配置数据）。"""
    return build_skill_book(load_job_project_config("black_mage"))

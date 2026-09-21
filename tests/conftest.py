"""测试公共引导。

这个文件只做路径注入，保证 `pytest` 能稳定找到项目根目录下的新包。
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """为每个 pytest 进程分配独立临时目录，避免 Windows 共享目录权限冲突。"""
    if config.option.basetemp is not None:
        return
    temp_parent = PROJECT_ROOT / ".tmp" / "pytest_runs"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_base = temp_parent / f"pytest-{os.getpid()}-{uuid.uuid4().hex}"
    config.option.basetemp = str(temp_base)
    atexit.register(shutil.rmtree, temp_base, ignore_errors=True)


@pytest.fixture(autouse=True)
def project_job_tag(monkeypatch):
    """为测试显式提供项目统一职业标签。"""
    monkeypatch.setenv("FFXIV_JOB_TAG", "black_mage")

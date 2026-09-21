"""scripts 测试公共 fixture：C# 状态机后端（SidecarHost 长驻进程）。

转换与自回归回放测试共用同一后端客户端；SidecarHost 未构建时跳过
依赖后端的测试（提示先构建）。
"""

from __future__ import annotations

import pytest

from scripts.common.cs_backend import SidecarBackend, _SIDECAR_EXE
from scripts.convert_fflogs.utils import build_skill_book, load_job_project_config


def _require_sidecar_host() -> None:
    if not _SIDECAR_EXE.exists():
        pytest.skip("SidecarHost 未构建，请先运行 dotnet build Combat.Sim/SidecarHost/SidecarHost.csproj")


@pytest.fixture
def cs_backend():
    """黑魔 C# 状态机后端（每个测试独立进程，结束后关闭）。"""
    _require_sidecar_host()
    backend = SidecarBackend("black_mage")
    yield backend
    backend.close()


@pytest.fixture
def cs_skill_book():
    """黑魔技能索引（静态配置数据）。"""
    return build_skill_book(load_job_project_config("black_mage"))

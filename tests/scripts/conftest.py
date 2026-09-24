"""scripts 测试公共 fixture：进程内 C# 状态机后端。

转换与自回归回放测试共用同一后端客户端；托管运行文件未构建时跳过
依赖后端的测试（提示先构建 SidecarHost 项目以生成依赖文件）。
"""

from __future__ import annotations

import pytest

from scripts.common.inprocess_backend import (
    InProcessBackend,
    _FIGHT_ENGINE_DLL,
    _RUNTIME_CONFIG,
    _YAML_DOTNET_DLL,
)
from scripts.convert_fflogs.utils import build_skill_book, load_job_project_config


def _require_inprocess_backend() -> None:
    missing = [
        path
        for path in (_RUNTIME_CONFIG, _FIGHT_ENGINE_DLL, _YAML_DOTNET_DLL)
        if not path.is_file()
    ]
    if missing:
        pytest.skip(
            "C# 状态机运行文件未构建，请先运行 "
            "dotnet build Combat.Sim/SidecarHost/SidecarHost.csproj"
        )
    try:
        import pythonnet  # noqa: F401
    except ImportError:
        pytest.skip("缺少 pythonnet，请先安装 requirements.txt")


@pytest.fixture
def cs_backend():
    """黑魔 C# 状态机后端（每个测试独立会话，结束后释放）。"""
    _require_inprocess_backend()
    backend = InProcessBackend("black_mage")
    yield backend
    backend.close()


@pytest.fixture
def cs_skill_book():
    """黑魔技能索引（静态配置数据）。"""
    return build_skill_book(load_job_project_config("black_mage"))

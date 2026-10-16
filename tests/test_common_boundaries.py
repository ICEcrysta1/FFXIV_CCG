"""公共组件的依赖方向与基础包导入边界回归。"""

import ast
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", [
    "common.configuration.environment",
    "common.configuration.yaml",
    "common.context.scene",
    "common.game.config",
    "common.game.skills",
    "common.pytorch.precision",
    "common.pytorch.serialization",
    "common.training",
    "common.training.optimizer_config",
    "common.training.tensorboard",
])
def test_foundation_imports_do_not_load_training_or_torch(module):
    """非计算入口可独立使用，不隐式初始化脚本、训练或 PyTorch。"""
    probe = """
import importlib
import sys

importlib.import_module(sys.argv[1])
unexpected = sorted(
    name for name in sys.modules
    if name.split('.')[0] in {'scripts', 'training', 'grpo', 'torch'}
)
assert not unexpected, unexpected
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, module],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_common_does_not_import_application_modules():
    """共享组件不能反向加载业务入口，包括函数内的延迟导入。"""
    root = Path(__file__).resolve().parents[1]
    forbidden = {"scripts", "training", "grpo"}
    violations = []
    for path in sorted((root / "common").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                if module.split(".")[0] in forbidden:
                    violations.append(f"{path.relative_to(root)}:{node.lineno}: {module}")
    assert not violations, violations

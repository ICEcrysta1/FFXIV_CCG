"""公共基础包的导入边界回归。"""

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

"""PyTorch 文件读取的安全边界测试。"""

from __future__ import annotations

import pickle
from pathlib import Path, PureWindowsPath, WindowsPath

import pytest
import torch

from common.pytorch.serialization import safe_torch_load, stringify_paths


_EXECUTED: list[bool] = []


def _unsafe_constructor() -> None:
    _EXECUTED.append(True)


class _MaliciousPayload:
    def __reduce__(self):
        return _unsafe_constructor, ()


def test_safe_torch_load_rejects_executable_pickle(tmp_path: Path):
    checkpoint_path = tmp_path / "malicious.pt"
    torch.save(_MaliciousPayload(), checkpoint_path)

    with pytest.raises(pickle.UnpicklingError):
        safe_torch_load(checkpoint_path)

    assert _EXECUTED == []


def test_safe_torch_load_allows_legacy_path_metadata(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"run_config": {"output_dir": tmp_path}}, checkpoint_path)

    payload = safe_torch_load(checkpoint_path)

    assert payload["run_config"]["output_dir"] == tmp_path


class _LegacyWindowsPath:
    def __reduce__(self):
        # 生成真正引用 pathlib.WindowsPath 的旧格式 pickle，无需在 Linux 构造它。
        return WindowsPath, ("C:\\training", "artifacts", "best.pt")


def test_safe_torch_load_reads_windows_checkpoint_paths_on_linux(tmp_path: Path):
    checkpoint_path = tmp_path / "windows.pt"
    torch.save({"run_config": {"output_dir": _LegacyWindowsPath()}, "weight": torch.ones(2)}, checkpoint_path)

    loaded = safe_torch_load(checkpoint_path)

    assert loaded["run_config"]["output_dir"] == PureWindowsPath("C:/training/artifacts/best.pt")
    torch.testing.assert_close(loaded["weight"], torch.ones(2))
    assert Path is __import__("pathlib").Path


def test_stringify_paths_preserves_tensors_and_container_types(tmp_path: Path):
    weight = torch.ones(2)
    original = {"config": {"root": tmp_path}, "paths": [PureWindowsPath("C:/data")], "values": (tmp_path, weight)}

    saved = stringify_paths(original)
    checkpoint_path = tmp_path / "portable.pt"
    torch.save(saved, checkpoint_path)
    loaded = torch.load(checkpoint_path, weights_only=True)

    assert saved["values"][1] is weight
    assert original["config"]["root"] == tmp_path
    assert loaded["config"]["root"] == str(tmp_path)
    assert loaded["paths"] == ["C:\\data"]
    assert isinstance(loaded["values"], tuple)
    torch.testing.assert_close(loaded["values"][1], weight)

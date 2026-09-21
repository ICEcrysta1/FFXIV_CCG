"""PyTorch 文件读取的安全边界测试。"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

from common.torch_serialization import safe_torch_load


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

"""autoregressive_replay 测试公共 fixture。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.autoregressive_replay import config as replay_config_module


@pytest.fixture(autouse=True)
def _isolate_replay_dotenv(monkeypatch):
    """单元测试不读取开发机根目录 `.env`，需要的变量由用例显式提供。"""
    monkeypatch.delenv("AUTOREGRESSIVE_REPLAY_BACKEND", raising=False)
    monkeypatch.delenv("AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE", raising=False)
    monkeypatch.setattr(replay_config_module, "load_root_dotenv", lambda _root: None)


@pytest.fixture
def prepare_replay_files(monkeypatch, tmp_path):
    """构造最小 checkpoint/scene，并注入 replay 配置环境变量。"""
    checkpoint = tmp_path / "model.pt"
    scene = tmp_path / "scene.json"
    torch.save(
        {
            "job_tag": "black_mage",
            "data_spec": {"job_tag": "black_mage"},
        },
        checkpoint,
    )
    scene.write_text("{}", encoding="utf-8", newline="\n")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_BACKEND", "pytorch")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_JSON", str(scene))
    monkeypatch.setenv("FFXIV_JOB_TAG", "black_mage")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_DEVICE", "cpu")
    return checkpoint, scene

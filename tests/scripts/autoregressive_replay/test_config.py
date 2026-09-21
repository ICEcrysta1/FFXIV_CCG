"""自回归回放配置加载测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.autoregressive_replay import config as replay_config_module
from scripts.autoregressive_replay.config import load_replay_config


def test_replay_config_loads_dotenv_before_resolving_backend(monkeypatch, tmp_path):
    package = tmp_path / "deployment"
    package.mkdir()
    (package / "manifest.json").write_text(
        '{"contract":{"job_tag":"black_mage","capacity":{"history_capacity":384}},"model":{"model_variant":"artzip"}}',
        encoding="utf-8",
    )
    scene = tmp_path / "scene.json"
    scene.write_text("{}", encoding="utf-8")

    for name in (
        "AUTOREGRESSIVE_REPLAY_BACKEND",
        "AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE",
        "AUTOREGRESSIVE_REPLAY_SCENE_JSON",
        "AUTOREGRESSIVE_REPLAY_SCENE_MODE",
    ):
        monkeypatch.delenv(name, raising=False)

    def load_test_dotenv(_root):
        monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_BACKEND", "onnxruntime")
        monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE", str(package))
        monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_JSON", str(scene))
        monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_MODE", "empty")
        monkeypatch.setenv("FFXIV_JOB_TAG", "black_mage")

    monkeypatch.setattr(replay_config_module, "load_root_dotenv", load_test_dotenv)

    config = load_replay_config()

    assert config.backend == "onnxruntime"
    assert config.scene_mode == "empty"
    assert config.onnx_package_path == package.resolve()
    assert config.checkpoint_path is None


def test_replay_config_reads_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_CHECKPOINT", str(tmp_path / "model.pt"))
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_OUTPUT", str(tmp_path / "rollout.md"))
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_JSON", str(tmp_path / "scene.json"))
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_MODE", "empty")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_SAMPLE", "3")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_MAX_STEPS", "42")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_MAX_GCDS", "36")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_TOP_K", "6")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_TOP_P", "0.9")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_TEMPERATURE", "0.8")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_DEVICE", "cpu")
    monkeypatch.setenv("FFXIV_JOB_TAG", "black_mage")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_INITIAL_ACTION", "fire_iii")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_INITIAL_TIME", "-3.5")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_BASE_GCD", "2.17")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_USE_KV_CACHE", "false")

    checkpoint = tmp_path / "model.pt"
    scene = tmp_path / "scene.json"
    torch.save(
        {
            "job_tag": "black_mage",
            "model_variant": "artzip",
            "data_spec": {"job_tag": "black_mage"},
        },
        checkpoint,
    )
    scene.write_text("{}", encoding="utf-8", newline="\n")
    config = load_replay_config()

    assert config.checkpoint_path == checkpoint.resolve()
    assert config.output_path == (tmp_path / "rollout.md").resolve()
    assert config.scene_json_path == scene.resolve()
    assert config.scene_mode == "empty"
    assert config.scene_sample_index == 3
    assert config.max_steps == 42
    assert config.max_gcds == 36
    assert config.top_k == 6
    assert config.top_p == 0.9
    assert config.temperature == 0.8
    assert config.max_history == 384
    assert config.device == "cpu"
    assert config.job_tag == "black_mage"
    assert config.initial_action == "fire_iii"
    assert config.initial_time_seconds == -3.5
    assert config.base_gcd == 2.17
    assert config.use_kv_cache is False


def test_replay_config_enables_kv_cache_by_default(monkeypatch, tmp_path, prepare_replay_files):
    monkeypatch.delenv("AUTOREGRESSIVE_REPLAY_USE_KV_CACHE", raising=False)

    config = load_replay_config()

    assert config.use_kv_cache is True


def test_replay_config_rejects_checkpoint_model_variant_mismatch(
    monkeypatch,
    prepare_replay_files,
):
    checkpoint, _scene = prepare_replay_files
    payload = torch.load(checkpoint, weights_only=False)
    payload["model_variant"] = "other_variant"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="checkpoint model_variant"):
        load_replay_config()


def test_replay_config_allows_explicit_float32_cpu_analysis(monkeypatch, tmp_path, prepare_replay_files):
    config = load_replay_config(
        device="cpu",
        policy_precision="float32",
    )

    assert config.device == "cpu"
    assert config.policy_precision == "float32"


def test_replay_config_routes_onnx_job_from_manifest(monkeypatch, tmp_path):
    package = tmp_path / "deployment"
    package.mkdir()
    (package / "manifest.json").write_text(
        '{"contract":{"job_tag":"black_mage","capacity":{"history_capacity":384}},"model":{"model_variant":"artzip"}}',
        encoding="utf-8",
    )
    scene = tmp_path / "scene.json"
    scene.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_BACKEND", "onnxruntime")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE", str(package))
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_JSON", str(scene))
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_MODE", "empty")
    monkeypatch.setenv("FFXIV_JOB_TAG", "black_mage")
    monkeypatch.setenv("FFXIV_JOB_TAG", "black_mage")
    monkeypatch.delenv("AUTOREGRESSIVE_REPLAY_ORT_PROVIDER", raising=False)

    config = load_replay_config()

    assert config.backend == "onnxruntime"
    assert config.onnx_package_path == package.resolve()
    assert config.checkpoint_path is None
    assert config.job_tag == "black_mage"
    assert config.max_history == 384
    assert config.cache_shard_size == 768
    assert config.cache_max_shards == 16
    assert config.device == "cpu"
    assert config.use_kv_cache is False
    assert config.ort_provider == "CUDAExecutionProvider"


def test_replay_config_rejects_onnx_job_mismatch(monkeypatch, tmp_path):
    package = tmp_path / "deployment"
    package.mkdir()
    (package / "manifest.json").write_text(
        '{"contract":{"job_tag":"machinist","capacity":{"history_capacity":384}},"model":{"model_variant":"artzip"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_BACKEND", "onnxruntime")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE", str(package))
    monkeypatch.setenv("FFXIV_JOB_TAG", "black_mage")

    with pytest.raises(ValueError, match="ONNX job_tag"):
        load_replay_config()


def test_replay_top_p_parser_defaults_to_no_filter(monkeypatch):
    monkeypatch.delenv("AUTOREGRESSIVE_REPLAY_TOP_P", raising=False)

    assert replay_config_module._probability_float(
        None,
        replay_config_module.AUTOREGRESSIVE_REPLAY_TOP_P_ENV,
        1.0,
    ) == 1.0


def test_replay_config_rejects_invalid_modes_and_limits(monkeypatch, tmp_path, prepare_replay_files):
    checkpoint, _scene = prepare_replay_files

    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_MODE", "invalid")
    with pytest.raises(ValueError, match="must be cache or empty"):
        load_replay_config()

    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_MODE", "cache")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_DEVICE", "metal")
    with pytest.raises(ValueError, match="must be cuda or cpu"):
        load_replay_config()

    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_DEVICE", "cpu")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_MAX_STEPS", "0")
    with pytest.raises(ValueError, match="MAX_STEPS must be >= 1"):
        load_replay_config()

    monkeypatch.delenv("AUTOREGRESSIVE_REPLAY_MAX_STEPS")
    with pytest.raises(ValueError, match="max_history must be >= 0"):
        load_replay_config(max_history=-1)

    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_TEMPERATURE", "-0.1")
    with pytest.raises(ValueError, match="TEMPERATURE must be >= 0"):
        load_replay_config()

    monkeypatch.delenv("AUTOREGRESSIVE_REPLAY_TEMPERATURE")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_TOP_P", "0")
    with pytest.raises(ValueError, match="TOP_P must be > 0 and <= 1"):
        load_replay_config()

    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_TOP_P", "1.1")
    with pytest.raises(ValueError, match="TOP_P must be > 0 and <= 1"):
        load_replay_config()

    assert replay_config_module._resolve_checkpoint(Path("model.yaml"), checkpoint) == checkpoint
    assert replay_config_module._optional_text("  ") is None
    assert replay_config_module._optional_text(3) == "3"
    assert replay_config_module._optional_float(None) is None
    assert replay_config_module._optional_positive_float(None) is None
    with pytest.raises(ValueError, match="INITIAL_TIME"):
        replay_config_module._optional_float("1")
    with pytest.raises(ValueError, match="BASE_GCD"):
        replay_config_module._optional_positive_float("0")

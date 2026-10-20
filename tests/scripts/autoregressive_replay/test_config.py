"""自回归回放配置加载测试。"""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from common.policy.data import ActionSpace, DataSpec
from common.policy.config import load_model_config
from scripts.autoregressive_replay import config as replay_config_module
from scripts.autoregressive_replay.config import load_replay_config
from scripts.common.json_io import atomic_write_json


def test_replay_scene_defaults_to_prepared_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_JSON", " ")
    monkeypatch.setenv("MODEL_ANALYSIS_SCENE_JSON", "analysis.json.br")
    scene = tmp_path / "90-100" / "scene.json.br"
    atomic_write_json(scene, {})
    calls = []

    def select(root, **kwargs):
        calls.append((root, kwargs))
        return scene

    monkeypatch.setattr(replay_config_module, "find_prepared_scene_source", select)
    normalizer, vocab = object(), object()
    spec = DataSpec("black_mage", 2, 5, 1, 1, 1, 1, ("a", "b"), ("kind",), (1, 2), (True, False))
    monkeypatch.setattr(
        replay_config_module.ModelInputContract, "from_dict",
        lambda _payload: SimpleNamespace(data_spec=asdict(spec), create_normalizer=lambda: normalizer,
                                        create_skill_vocab=lambda: vocab),
    )
    kwargs = dict(raw_root=tmp_path, cache_dir=tmp_path / ".cache",
                  job_tag="black_mage", cache_shard_size=768, input_contract_payload={})
    assert replay_config_module._resolve_scene_json(None, **kwargs) == scene
    assert calls == [(tmp_path, dict(cache_dir=tmp_path / ".cache",
                                   job_tag="black_mage", cache_shard_size=768,
                                   normalizer=normalizer, expected_action_space=ActionSpace.from_data_spec(spec),
                                   expected_skill_vocab=vocab))]

    explicit = tmp_path / "explicit.json.br"
    atomic_write_json(explicit, {})
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_JSON", str(scene))
    assert replay_config_module._resolve_scene_json(explicit, **kwargs) == explicit
    assert len(calls) == 1


def test_replay_config_loads_dotenv_before_resolving_backend(monkeypatch, tmp_path):
    package = tmp_path / "deployment"
    package.mkdir()
    (package / "manifest.json").write_text(
        '{"contract":{"job_tag":"black_mage","capacity":{"context_capacity":1024}},"model":{"model_variant":"artzip"}}',
        encoding="utf-8",
    )
    scene = tmp_path / "scene.json.br"
    atomic_write_json(scene, {})

    for name in (
        "AUTOREGRESSIVE_REPLAY_BACKEND",
        "AUTOREGRESSIVE_REPLAY_SCENE_JSON",
        "AUTOREGRESSIVE_REPLAY_SCENE_MODE",
    ):
        monkeypatch.delenv(name, raising=False)

    def load_test_dotenv(_root):
        monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_BACKEND", "onnxruntime")
        monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_JSON", str(scene))
        monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_MODE", "empty")
        monkeypatch.setenv("FFXIV_JOB_TAG", "black_mage")

    monkeypatch.setattr(replay_config_module, "load_root_dotenv", load_test_dotenv)

    config = load_replay_config(onnx_package=package)

    assert config.backend == "onnxruntime"
    assert config.scene_mode == "empty"
    assert config.onnx_package_path == package.resolve()
    assert config.checkpoint_path is None


def test_replay_config_reads_env_overrides(monkeypatch, tmp_path, prepare_replay_files):
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_OUTPUT", str(tmp_path / "rollout.md"))
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_MODE", "empty")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_SAMPLE", "3")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_MAX_STEPS", "42")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_MAX_GCDS", "36")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_TOP_K", "6")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_TOP_P", "0.9")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_TEMPERATURE", "0.8")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_INITIAL_ACTION", "fire_iii")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_INITIAL_TIME", "-3.5")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_BASE_GCD", "2.17")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_USE_KV_CACHE", "false")

    checkpoint, scene = prepare_replay_files
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
    assert config.max_history is None
    assert config.device == "cpu"
    assert config.job_tag == "black_mage"
    assert config.initial_action == "fire_iii"
    assert config.initial_time_seconds == -3.5
    assert config.base_gcd == 2.17
    assert config.use_kv_cache is False


def test_replay_capacity_follows_checkpoint_without_default_history_limit(prepare_replay_files):
    checkpoint, _scene = prepare_replay_files
    payload = torch.load(checkpoint, weights_only=False)
    assert load_model_config(
        replay_config_module.resolve_policy_model_config_path(),
    ).context_capacity == 1024
    payload["model_config"]["context_capacity"] = 2048
    torch.save(payload, checkpoint)

    config = load_replay_config()
    assert config.model_context_capacity == 2048
    assert asdict(config)["model_context_capacity"] == 2048
    assert config.max_history is None
    for limit in (None, 0, 4096):
        assert replace(config, max_history=limit).max_history == limit
    assert load_replay_config(max_history=900).max_history == 900
    assert load_replay_config(max_history=4096).max_history == 4096


def test_replay_does_not_parse_unused_local_model_config(monkeypatch, prepare_replay_files):
    from common.policy import config as policy_config_module

    path = replay_config_module.resolve_policy_model_config_path()
    local = replay_config_module.load_policy_config(path)
    local["model"] = {"context_capacity": 1, "n_heads": 0}
    monkeypatch.setattr(policy_config_module, "load_policy_config", lambda _path: local)
    monkeypatch.setattr(replay_config_module, "load_policy_config", lambda _path: local)

    config = load_replay_config()
    assert config.model_context_capacity == 1024
    assert config.max_history is None


@pytest.mark.parametrize(("value", "message"), (
    (-1, "max_history must be >= 0"),
    (True, "max_history must be an integer"),
    (1.5, "max_history must be an integer"),
    ("8", "max_history must be an integer"),
))
def test_replay_config_validates_history_limit_at_construction(prepare_replay_files, value, message):
    config = load_replay_config()
    with pytest.raises(ValueError, match=message):
        replace(config, max_history=value)
    with pytest.raises(ValueError, match=message):
        load_replay_config(max_history=value)


def test_replay_config_rejects_checkpoint_missing_model_config(prepare_replay_files):
    checkpoint, _scene = prepare_replay_files
    payload = torch.load(checkpoint, weights_only=False)
    del payload["model_config"]
    torch.save(payload, checkpoint)

    with pytest.raises(TypeError, match="checkpoint missing model_config"):
        load_replay_config()


def test_replay_config_enables_kv_cache_by_default(monkeypatch, prepare_replay_files):
    monkeypatch.delenv("AUTOREGRESSIVE_REPLAY_USE_KV_CACHE", raising=False)

    config = load_replay_config()

    assert config.use_kv_cache is True


def test_replay_config_rejects_checkpoint_model_variant_mismatch(
    prepare_replay_files,
):
    checkpoint, _scene = prepare_replay_files
    payload = torch.load(checkpoint, weights_only=False)
    payload["model_variant"] = "other_variant"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="checkpoint model_variant"):
        load_replay_config()


def test_replay_config_allows_explicit_float32_cpu_analysis(prepare_replay_files):
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
        '{"contract":{"job_tag":"black_mage","capacity":{"context_capacity":2048}},"model":{"model_variant":"artzip"}}',
        encoding="utf-8",
    )
    scene = tmp_path / "scene.json.br"
    atomic_write_json(scene, {})
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_BACKEND", "onnxruntime")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_JSON", str(scene))
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_MODE", "empty")
    monkeypatch.setenv("FFXIV_JOB_TAG", "black_mage")
    monkeypatch.delenv("AUTOREGRESSIVE_REPLAY_ORT_PROVIDER", raising=False)

    config = load_replay_config(onnx_package=package)

    assert config.backend == "onnxruntime"
    assert config.onnx_package_path == package.resolve()
    assert config.checkpoint_path is None
    assert config.job_tag == "black_mage"
    assert config.model_context_capacity == 2048
    assert config.max_history is None
    assert load_replay_config(onnx_package=package, max_history=4096).max_history == 4096
    assert config.cache_shard_size == 768
    assert config.cache_max_shards == 16
    assert config.device == "cpu"
    assert config.use_kv_cache is False
    assert config.ort_provider == "CUDAExecutionProvider"


def test_replay_config_derives_onnx_package_from_explicit_checkpoint(monkeypatch, tmp_path):
    """显式选择 checkpoint 时 ORT 回放必须按该 checkpoint 推导部署包。"""
    from scripts.onnx_export.config import config as export_config_module

    monkeypatch.setattr(replay_config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(export_config_module, "PROJECT_ROOT", tmp_path)
    checkpoint_dir = tmp_path / "artifacts" / "checkpoints" / "black_mage" / "artzip_hotstart"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "epoch_003.pt"
    checkpoint.write_bytes(b"checkpoint")
    package = tmp_path / "artifacts" / "exports" / "black_mage" / "artzip_hotstart"
    package.mkdir(parents=True)
    (package / "manifest.json").write_text(
        '{"contract":{"job_tag":"black_mage","capacity":{"context_capacity":1024}},'
        '"model":{"model_variant":"artzip"}}',
        encoding="utf-8",
    )
    default_package = tmp_path / "artifacts" / "exports" / "black_mage" / "artzip_bc"
    scene = tmp_path / "scene.json.br"
    atomic_write_json(scene, {})
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_BACKEND", "onnxruntime")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_JSON", str(scene))
    monkeypatch.setenv("FFXIV_JOB_TAG", "black_mage")

    config = load_replay_config(checkpoint=checkpoint)

    assert config.onnx_package_path == package.resolve()
    assert config.onnx_package_path != default_package.resolve()


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (
            '{"contract":{"job_tag":"machinist","capacity":{"context_capacity":1024}},"model":{"model_variant":"artzip"}}',
            "ONNX job_tag",
        ),
        (
            '{"contract":{"job_tag":"black_mage","capacity":{"context_capacity":1024}},"model":{"model_variant":"other_variant"}}',
            "ONNX model_variant",
        ),
        (
            '{"contract":{"job_tag":"black_mage","capacity":{"context_capacity":1024}},"model":{}}',
            "missing model_variant; re-export the package",
        ),
        (
            '{"contract":{"capacity":{"context_capacity":1024}},"model":{"model_variant":"artzip"}}',
            "missing replay routing field 'job_tag'",
        ),
        (
            '{"model":{"model_variant":"artzip"}}',
            "missing replay routing field 'contract'",
        ),
        (
            '{"contract":{"job_tag":"black_mage","capacity":{"context_capacity":1024}}}',
            "missing replay routing field 'model'",
        ),
        (
            '{"contract":{"job_tag":"black_mage"},"model":{"model_variant":"artzip"}}',
            "missing replay routing field 'capacity'",
        ),
        (
            '{"contract":{"job_tag":"black_mage","capacity":{}},"model":{"model_variant":"artzip"}}',
            "missing replay routing field 'context_capacity'",
        ),
    ],
)
def test_replay_config_rejects_invalid_onnx_routing(
    monkeypatch,
    tmp_path,
    manifest,
    message,
):
    package = tmp_path / "deployment"
    package.mkdir()
    (package / "manifest.json").write_text(manifest, encoding="utf-8")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_BACKEND", "onnxruntime")
    monkeypatch.setenv("FFXIV_JOB_TAG", "black_mage")

    with pytest.raises(ValueError, match=message):
        load_replay_config(onnx_package=package)


def test_replay_top_p_parser_defaults_to_no_filter(monkeypatch):
    monkeypatch.delenv("AUTOREGRESSIVE_REPLAY_TOP_P", raising=False)

    assert replay_config_module._probability_float(
        None,
        replay_config_module.AUTOREGRESSIVE_REPLAY_TOP_P_ENV,
        1.0,
    ) == 1.0


def test_replay_config_rejects_invalid_modes_and_limits(monkeypatch, prepare_replay_files):
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

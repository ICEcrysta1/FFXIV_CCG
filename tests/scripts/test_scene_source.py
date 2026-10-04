"""参考场景只读选择与缓存有效性回归测试。"""

from pathlib import Path
import json

import pytest
import torch

from common.config import load_precision_config
from common.policy.data import ModelInputContract, Normalizer
from common.policy.data.schema import TrainingSchema
from common.policy.data.action_space import ActionSpace
from common.policy.data.compiled_cache import (
    CACHE_FORMAT,
    build_cache_signature,
    cache_path_for_source,
)
from common.torch_serialization import safe_torch_load
from scripts.common.json_io import atomic_write_json
from scripts.common.scene_source import find_prepared_scene_source


def _scene_data_spec():
    actions = ActionSpace.from_job_tag("black_mage")
    return {
        "job_tag": "black_mage", "num_actions": len(actions.action_keys), "state_dim": 1,
        "scene_dim": 0, "skill_feature_dim": 1, "num_scene_types": 0,
        "action_keys": list(actions.action_keys), "action_to_vocab_id": list(actions.action_to_vocab_id),
        "action_is_gcd": list(actions.action_is_gcd), "skill_feature_names": ["potency"],
    }


def _scene_schema():
    return TrainingSchema(
        serialization_format="test", sample_schema_version=1,
        context_schema_version=1, scene_context_mode="absolute", scene_windows=(),
        state_group_feature_keys={"player_state": ("before.time_seconds",)},
        skill_history_fields=("potency",),
    )


def _scene_sample():
    actions = _scene_data_spec()
    return {
        "action_keys": actions["action_keys"],
        "current_state_vectors": torch.zeros(1),
        "current_state_null_mask": torch.zeros(1, dtype=torch.bool),
        "action_values": torch.ones(actions["num_actions"]),
        "action_legal_mask": torch.ones(actions["num_actions"], dtype=torch.bool),
        "label_index": 0, "label_action_key": actions["action_keys"][0],
    }


@pytest.fixture
def scene_cache(tmp_path):
    """构造可由正式缓存 reader 校验的最小 manifest 和分片。"""
    root = tmp_path / "annotated"
    cache_dir = tmp_path / ".cache"
    normalizer = Normalizer()
    normalizer.ensure_job_resources("black_mage")
    precision = load_precision_config()

    def create(relative, *, invalid=None, legacy=False, cache_normalizer=None):
        source = root / relative
        atomic_write_json(source, {})
        if invalid == "missing":
            return source
        manifest = cache_path_for_source(cache_dir, source)
        if legacy:
            manifest = cache_dir / manifest.name
        manifest.parent.mkdir(parents=True, exist_ok=True)
        shard = manifest.with_suffix(".shard.pt")
        torch.save({"cache_format": CACHE_FORMAT, "samples": [_scene_sample()]}, shard)
        signature = build_cache_signature(
            source, normalizer=normalizer if cache_normalizer is None else cache_normalizer,
            int_dtype=precision.resolve_int_dtype(),
            float_dtype=precision.resolve_float_dtype(), shard_size=768,
        )
        bank = {key: torch.zeros(1) for key in (
            "skill_ids", "skill_features", "state_vectors", "state_null_mask",
            "skill_potencies", "cumulative_dot_potencies",
        )}
        bank["action_keys"] = [""]
        actions = _scene_data_spec()
        payload = {
            "cache_format": CACHE_FORMAT, "cache_signature": signature,
            "schema": _scene_schema(), "job_tag": "black_mage", "num_samples": 1,
            "num_actions": actions["num_actions"], "skill_feature_names": ["potency"],
            "action_keys": actions["action_keys"], "action_to_vocab_id": actions["action_to_vocab_id"],
            "action_is_gcd": actions["action_is_gcd"], "shard_size": 768,
            "history_bank": bank, "shard_files": [shard.name],
        }
        if invalid == "stale":
            signature["source_size"] += 1
        elif invalid == "format":
            payload["cache_format"] = "old"
        elif invalid == "job":
            payload["job_tag"] = "machinist"
        elif invalid == "empty":
            payload["num_samples"] = 0
        elif invalid == "shard":
            payload["shard_files"] = ["missing.pt"]
        torch.save(payload, manifest)
        if invalid == "corrupt":
            manifest.write_bytes(b"broken")
        return source

    return root, cache_dir, create


def _select(root: Path, cache_dir: Path) -> Path:
    return find_prepared_scene_source(
        root, cache_dir=cache_dir, job_tag="black_mage", cache_shard_size=768,
    )


def test_scene_source_prefers_high_bucket_across_encounters(scene_cache):
    root, cache_dir, create = scene_cache
    create("AAA/00-10/a.json.br")
    create("AAA/80-90/a.json.br")
    create("VAL/AAA/a.json.br")
    create("ZZZ/90-100/b.json.br")
    expected = create("ZZZ/90-100/a.json.br")
    assert _select(root, cache_dir) == expected
    assert _select(root, cache_dir) == expected


@pytest.mark.parametrize("invalid", ["missing", "stale", "format", "job", "empty", "shard", "corrupt"])
def test_scene_source_skips_invalid_cache_without_writing(scene_cache, invalid):
    root, cache_dir, create = scene_cache
    create("AAA/90-100/a.json.br", invalid=invalid)
    expected = create("ZZZ/80-90/a.json.br")
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in cache_dir.rglob("*.pt")}
    assert _select(root, cache_dir) == expected
    after = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in cache_dir.rglob("*.pt")}
    assert before == after


def test_scene_source_accepts_legacy_flat_cache(scene_cache):
    root, cache_dir, create = scene_cache
    expected = create("FRU/90-100/a.json.br", legacy=True)
    assert _select(root, cache_dir) == expected


def test_scene_source_requires_existing_valid_cache(scene_cache):
    root, cache_dir, create = scene_cache
    create("FRU/90-100/a.json.br", invalid="missing")
    with pytest.raises(FileNotFoundError, match="先运行训练文件转换"):
        _select(root, cache_dir)
    assert not cache_dir.exists()


@pytest.mark.parametrize("later_shard", [False, True])
@pytest.mark.parametrize("damage", [
    "truncated", "missing_format", "missing_samples", "invalid_samples",
    "short_samples", "invalid_sample",
])
def test_scene_source_skips_damaged_shard(scene_cache, later_shard, damage, caplog):
    root, cache_dir, create = scene_cache
    broken = create("AAA/90-100/a.json.br")
    expected = create("ZZZ/80-90/a.json.br")
    manifest = cache_path_for_source(cache_dir, broken)
    payload = safe_torch_load(manifest, safe_globals=(TrainingSchema,))
    shard = manifest.parent / payload["shard_files"][0]
    if later_shard:
        # 第一片完整且可读，损坏只发生在后续分片。
        torch.save({"cache_format": CACHE_FORMAT, "samples": [_scene_sample()] * 768}, shard)
        shard = manifest.with_suffix(".second.pt")
        payload["num_samples"] = 769
        payload["shard_files"].append(shard.name)
        torch.save(payload, manifest)

    shard_payload = {"cache_format": CACHE_FORMAT, "samples": [_scene_sample()]}
    if damage == "missing_format":
        del shard_payload["cache_format"]
    elif damage == "missing_samples":
        del shard_payload["samples"]
    elif damage == "invalid_samples":
        shard_payload["samples"] = {}
    elif damage == "short_samples":
        shard_payload["samples"] = []
    elif damage == "invalid_sample":
        shard_payload["samples"] = [None]
    torch.save(shard_payload, shard)
    if damage == "truncated":
        shard.write_bytes(shard.read_bytes()[:64])

    before = {path: path.read_bytes() for path in cache_dir.rglob("*.pt")}
    assert _select(root, cache_dir) == expected
    assert "跳过分片不可读取的参考场景" in caplog.text
    assert before == {path: path.read_bytes() for path in cache_dir.rglob("*.pt")}


def test_scene_source_reports_no_usable_cache_when_all_shards_are_broken(scene_cache):
    root, cache_dir, create = scene_cache
    source = create("FRU/90-100/a.json.br")
    manifest = cache_path_for_source(cache_dir, source)
    shard = manifest.with_suffix(".shard.pt")
    shard.write_bytes(b"broken")
    with pytest.raises(FileNotFoundError, match="先运行训练文件转换"):
        _select(root, cache_dir)


def test_scene_source_does_not_hide_memory_error(scene_cache, monkeypatch):
    from common.policy.data.compiled_cache import CompiledCacheReader

    root, cache_dir, create = scene_cache
    create("FRU/90-100/a.json.br")

    def fail(*_args):
        raise MemoryError("test allocation failure")

    monkeypatch.setattr(CompiledCacheReader, "samples", fail)
    with pytest.raises(MemoryError, match="test allocation failure"):
        _select(root, cache_dir)


@pytest.mark.parametrize("backend", ["pytorch", "onnxruntime"])
@pytest.mark.parametrize("current_cache_exists", [False, True])
def test_default_replay_uses_saved_contract_without_recompiling(
    scene_cache, monkeypatch, tmp_path, backend, current_cache_exists,
):
    from scripts.autoregressive_replay import config as config_module
    from scripts.autoregressive_replay import replay as replay_module

    root, cache_dir, create = scene_cache
    normalizer = Normalizer()
    normalizer.ensure_job_resources("black_mage")
    saved_normalizer = normalizer.normalization_contract
    saved_normalizer["config"]["mp_max"] *= 2
    saved_normalizer["resource_limits"] = {
        key: value * 2 for key, value in saved_normalizer["resource_limits"].items()
    }
    contract = ModelInputContract(
        job_tag="black_mage", data_spec=_scene_data_spec(),
        schema=_scene_schema(),
        normalizer_contract=saved_normalizer,
    )
    if current_cache_exists:
        create("AAA/90-100/current.json.br")
    expected = create(
        "ZZZ/80-90/saved.json.br", cache_normalizer=contract.create_normalizer(),
    )
    checkpoint = tmp_path / "model.pt"
    torch.save({"data_spec": _scene_data_spec(), "model_variant": "artzip",
                "input_contract": contract.to_dict()}, checkpoint)
    package = tmp_path / "deployment"
    package.mkdir()
    (package / "manifest.json").write_text(json.dumps({
        "model": {"model_variant": "artzip"},
        "contract": {"job_tag": "black_mage", "capacity": {"history_capacity": 384},
                     "model_input_contract": contract.to_dict()},
    }), encoding="utf-8", newline="\n")

    monkeypatch.setattr(config_module, "load_root_dotenv", lambda _root: None)
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_JSON", "")
    monkeypatch.setenv("FFXIV_JOB_TAG", "black_mage")
    monkeypatch.setenv("FFXIV_MODEL_VARIANT", "artzip")
    original_load = config_module.load_policy_config

    def load_config(path):
        payload = original_load(path)
        return {**payload, "raw_data_dir": str(root),
                "training": {**payload["training"], "compiled_cache_shard_size": 768}}

    monkeypatch.setattr(config_module, "load_policy_config", load_config)
    monkeypatch.setattr(config_module, "resolve_policy_cache_dir", lambda _job: cache_dir)
    monkeypatch.setattr(
        replay_module, "precompile_raw_training_caches",
        lambda *_args, **_kwargs: pytest.fail("默认场景必须直接复用模型兼容缓存"),
    )
    config = config_module.load_replay_config(
        checkpoint=checkpoint, backend=backend, onnx_package=package,
        scene_mode="cache", device="cpu", use_kv_cache=False,
    )
    assert config.scene_json_path == expected
    reader = replay_module._load_replay_cache(
        config, "black_mage", contract.create_normalizer(), engine=object(),
    )
    sample = reader.sample(0)
    assert sample["action_keys"] == _scene_data_spec()["action_keys"]
    assert sample["label_action_key"] == sample["action_keys"][sample["label_index"]]
    assert torch.equal(sample["current_state_vectors"], torch.zeros(1))


@pytest.mark.parametrize("payload", [None, {"version": 1}])
def test_default_replay_rejects_missing_or_old_contract(scene_cache, monkeypatch, payload):
    from scripts.autoregressive_replay.config import _resolve_scene_json

    root, cache_dir, create = scene_cache
    create("FRU/90-100/current.json.br")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_JSON", "")
    with pytest.raises(ValueError, match="input contract"):
        _resolve_scene_json(
            None, raw_root=root, cache_dir=cache_dir, job_tag="black_mage",
            cache_shard_size=768, input_contract_payload=payload,
        )

"""参考场景只读选择与缓存有效性回归测试。"""

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json
import io

import pytest
import torch

from common.pytorch.precision import load_precision_config
from common.context.output import CANONICAL_CONTEXT_SCHEMA_VERSION
from common.policy.data import DataSpec, ModelInputContract, Normalizer, SkillVocab
from common.policy.data.schema import TRAINING_SAMPLE_SCHEMA_VERSION, TRAINING_SOURCE_FORMAT, TrainingSchema, StateFeatureGroup
from common.policy.data.contracts.action_space import ActionSpace
from common.policy.data.cache.compiled_cache import (
    CACHE_FORMAT,
    build_cache_signature,
    cache_path_for_source,
)
from common.pytorch.serialization import safe_torch_load
from scripts.common.json_io import atomic_write_json
from scripts.common.scene_source import find_prepared_scene_source


def _scene_data_spec():
    actions = ActionSpace.from_job_tag("black_mage")
    return {
        "job_tag": "black_mage", "num_actions": len(actions.action_keys), "state_dim": 1 + 6 * len(actions.action_keys), "base_state_dim": 1,
        "scene_dim": 0, "skill_feature_dim": 1, "num_scene_types": 0,
        "action_keys": list(actions.action_keys), "action_to_vocab_id": list(actions.action_to_vocab_id),
        "action_is_gcd": list(actions.action_is_gcd), "skill_feature_names": ["potency"],
    }


def _changed_actions(space, drift):
    if drift == "enabled":
        return ActionSpace(space.action_keys[1:], space.action_to_vocab_id[1:], space.action_is_gcd[1:])
    return replace(space, action_is_gcd=(not space.action_is_gcd[0], *space.action_is_gcd[1:]))


def _scene_schema():
    return TrainingSchema(
        serialization_format=TRAINING_SOURCE_FORMAT, sample_schema_version=TRAINING_SAMPLE_SCHEMA_VERSION,
        context_schema_version=CANONICAL_CONTEXT_SCHEMA_VERSION, scene_context_mode="absolute", scene_windows=(),
        state_groups=(
            StateFeatureGroup("player_state", "player_state_feature_keys", ("request_state.time_seconds",), "anchored_delta"),
            StateFeatureGroup("skill_state", "skill_state_feature_keys",
                tuple(f"{prefix}.{key}.{field}" for prefix in ("previous_action_after", "request_state")
                      for key in ActionSpace.from_job_tag("black_mage").action_keys
                      for field in ("available", "current_charges", "next_cooldown_seconds")), "absolute_numeric"),
        ),
        state_snapshots=("previous_action_after", "request_state"),
        skill_history_fields=("potency",),
    )


def _scene_sample():
    actions = _scene_data_spec()
    return {
        "action_keys": actions["action_keys"],
        "current_state_abs_values": torch.zeros(1),
        "current_state_delta_values": torch.zeros(1),
        "current_state_null_mask": torch.zeros(1, dtype=torch.bool),
        "current_state_delta_reset_mask": torch.ones(1, dtype=torch.bool),
        "current_state_skill_state": torch.ones(6 * actions["num_actions"], dtype=torch.float32),
        "history_end": 1, "history_length": 0,
        "scene_vectors": torch.zeros(0, 0), "scene_types": torch.zeros(0, dtype=torch.int32),
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
        # 有效基线遵守当前 schema 和矩阵形状，使损坏测试能进入各自要覆盖的读取边界。
        bank = {
            "skill_ids": torch.zeros(1, dtype=precision.resolve_int_dtype()),
            "skill_features": torch.zeros(1, 1, dtype=precision.resolve_float_dtype()),
            "state_abs_values": torch.zeros(1, 1, dtype=torch.float32),
            "state_delta_values": torch.zeros(1, 1, dtype=torch.float32),
            "state_null_mask": torch.zeros(1, 1, dtype=torch.bool),
            "state_delta_reset_mask": torch.zeros(1, 1, dtype=torch.bool),
            "state_skill_state": torch.zeros(1, 6 * _scene_data_spec()["num_actions"], dtype=torch.float32),
            "skill_potencies": torch.zeros(1, dtype=precision.resolve_float_dtype()),
            "cumulative_dot_potencies": torch.zeros(1, dtype=precision.resolve_float_dtype()),
            "action_keys": [""],
        }
        actions = _scene_data_spec()
        payload = {
            "cache_format": CACHE_FORMAT, "cache_signature": signature,
            "source_path": str(source.resolve()),
            "schema": _scene_schema(), "job_tag": "black_mage", "num_samples": 1,
            "num_actions": actions["num_actions"], "skill_feature_names": ["potency"],
            "action_keys": actions["action_keys"], "action_to_vocab_id": actions["action_to_vocab_id"],
            "action_is_gcd": actions["action_is_gcd"], "shard_size": 768,
            "history_bank": bank, "shard_files": [shard.name],
            "history_markers": (),
            "vocab_signature": tuple(SkillVocab.build_from_job_tag("black_mage")),
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
        expected_action_space=ActionSpace.from_job_tag("black_mage"),
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


@pytest.mark.parametrize("config_problem", ["missing", "missing_movement", "invalid_yaml"])
def test_readonly_cache_consumers_do_not_load_conversion_yaml(scene_cache, monkeypatch, config_problem):
    """转换 YAML 缺失或损坏时，已有缓存仍可供训练、VAL 和回放读取。"""
    import builtins

    from common.policy.data.cache.prepared_sources import select_prepared_training_sources, select_prepared_validation_sources
    from scripts.convert_fflogs.cache.cache_load import load_raw_compiled_cache
    from scripts.convert_fflogs.config import CONVERT_DEFAULT_CONFIG_PATH
    from training import TrainingDataset

    root, cache_dir, create = scene_cache
    source = create("FRU/90-100/a.json.br")
    validation_source = create("VAL/FRU/a.json.br")
    before = {path: path.read_bytes() for path in cache_dir.rglob("*.pt")}
    original_open = builtins.open

    def open_with_broken_conversion_config(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file).resolve() == CONVERT_DEFAULT_CONFIG_PATH:
            if config_problem == "missing":
                raise FileNotFoundError(file)
            return io.StringIO("convert_fflogs: {}" if config_problem == "missing_movement" else "[")
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_with_broken_conversion_config)
    precision = load_precision_config()
    normalizer = Normalizer()
    normalizer.ensure_job_resources("black_mage")
    actions = ActionSpace.from_job_tag("black_mage")
    args = dict(int_dtype=precision.resolve_int_dtype(), float_dtype=precision.resolve_float_dtype(),
                cache_dir=cache_dir)
    assert _select(root, cache_dir) == source
    assert select_prepared_training_sources(root, max_files=1, job_tag="black_mage", shard_size=768, **args) == [source]
    assert select_prepared_validation_sources(max_files=1, data_dir=root, job_tag="black_mage",
                                             shard_size=768, **args) == [validation_source]
    dataset = TrainingDataset([source], expected_action_space=actions, normalizer=normalizer,
                              context_capacity=1024,
                              job_tag="black_mage", compiled_cache_shard_size=768, **args)
    restored = object.__new__(TrainingDataset)
    restored.__setstate__(dataset.__getstate__())
    assert len(dataset) == len(restored) == 1
    reader = load_raw_compiled_cache(source, expected_action_space=actions, normalizer=normalizer,
                                    shard_size=768, **args)
    assert reader is not None and reader.num_samples == 1
    assert before == {path: path.read_bytes() for path in cache_dir.rglob("*.pt")}


@pytest.mark.parametrize("invalid", ["missing", "stale", "format", "job", "empty", "shard", "corrupt"])
def test_scene_source_skips_invalid_cache_without_writing(scene_cache, invalid):
    root, cache_dir, create = scene_cache
    create("AAA/90-100/a.json.br", invalid=invalid)
    expected = create("ZZZ/80-90/a.json.br")
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in cache_dir.rglob("*.pt")}
    assert _select(root, cache_dir) == expected
    after = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in cache_dir.rglob("*.pt")}
    assert before == after


def test_scene_source_rejects_legacy_flat_cache(scene_cache):
    root, cache_dir, create = scene_cache
    create("FRU/90-100/a.json.br", legacy=True)
    with pytest.raises(FileNotFoundError, match="先运行训练文件转换"):
        _select(root, cache_dir)


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
    payload = safe_torch_load(manifest, safe_globals=(TrainingSchema, StateFeatureGroup))
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
    from common.policy.data.cache.compiled_cache import CompiledCacheReader

    root, cache_dir, create = scene_cache
    create("FRU/90-100/a.json.br")

    def fail(*_args):
        raise MemoryError("test allocation failure")

    monkeypatch.setattr(CompiledCacheReader, "samples", fail)
    with pytest.raises(MemoryError, match="test allocation failure"):
        _select(root, cache_dir)


@pytest.mark.parametrize("backend", ["pytorch", "onnxruntime"])
@pytest.mark.parametrize("current_cache_exists", [False, True])
@pytest.mark.parametrize("action_drift", [None, "enabled", "kind"])
def test_default_replay_uses_saved_contract_without_recompiling(
    scene_cache, monkeypatch, tmp_path, backend, current_cache_exists, action_drift,
):
    from scripts.autoregressive_replay import config as config_module
    from scripts.autoregressive_replay import replay as replay_module
    from dataclasses import asdict
    from common.policy.config import ModelConfig

    root, cache_dir, create = scene_cache
    normalizer = Normalizer()
    normalizer.ensure_job_resources("black_mage")
    saved_normalizer = normalizer.normalization_contract
    saved_normalizer["config"]["mp_max"] *= 2
    saved_normalizer["resource_limits"] = {
        key: value * 2 for key, value in saved_normalizer["resource_limits"].items()
    }
    saved_spec = _scene_data_spec()
    saved_actions = ActionSpace.from_data_spec(DataSpec.from_dict(saved_spec))
    contract = ModelInputContract(
        job_tag="black_mage", data_spec=saved_spec,
        schema=_scene_schema(),
        normalizer_contract=saved_normalizer,
        skill_vocab_entries=tuple(SkillVocab.build_from_job_tag("black_mage")),
    )
    if current_cache_exists:
        create("AAA/90-100/current.json.br")
    expected = create(
        "ZZZ/80-90/saved.json.br", cache_normalizer=contract.create_normalizer(),
    )
    checkpoint = tmp_path / "model.pt"
    torch.save({"data_spec": saved_spec, "model_variant": "artzip", "model_config": asdict(ModelConfig()),
                "input_contract": contract.to_dict()}, checkpoint)
    package = tmp_path / "deployment"
    package.mkdir()
    (package / "manifest.json").write_text(json.dumps({
        "model": {"model_variant": "artzip"},
        "contract": {"job_tag": "black_mage", "capacity": {"context_capacity": 1024},
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
    if action_drift is not None:
        changed = _changed_actions(saved_actions, action_drift)
        monkeypatch.setattr(ActionSpace, "from_job_tag", lambda _job: changed)
    config = config_module.load_replay_config(
        checkpoint=checkpoint, backend=backend, onnx_package=package,
        scene_mode="cache", device="cpu", use_kv_cache=False,
    )
    assert config.scene_json_path == expected
    reader = replay_module._load_replay_cache(
        config, "black_mage", contract.create_normalizer(), engine=object(),
        expected_action_space=saved_actions,
        expected_skill_vocab=contract.create_skill_vocab(),
    )
    sample = reader.sample(0)
    assert sample["action_keys"] == saved_spec["action_keys"]
    assert sample["label_action_key"] == sample["action_keys"][sample["label_index"]]
    assert torch.equal(sample["current_state_abs_values"], torch.zeros(1))


@pytest.mark.parametrize("drift", ["enabled", "kind"])
def test_model_cache_prepare_reuses_saved_actions_and_training_rejects_drift(scene_cache, monkeypatch, drift):
    from scripts.autoregressive_replay.replay import ReplayCacheStore
    from scripts.convert_fflogs.cache import cache_compile
    from common.policy.data.cache.prepared_sources import select_prepared_training_sources
    from training.loop.dataloaders import _build_dataset
    from training import TrainingDataset

    root, cache_dir, create = scene_cache
    source = create("FRU/90-100/saved.json.br")
    saved_actions = ActionSpace.from_job_tag("black_mage")
    normalizer = Normalizer()
    normalizer.ensure_job_resources("black_mage")
    changed = _changed_actions(saved_actions, drift)
    engine = SimpleNamespace(job_tag="black_mage", capacity=2)
    config = SimpleNamespace(
        scene_json_path=source, cache_dir=cache_dir, job_tag="black_mage",
        cache_shard_size=768, cache_max_shards=2,
        compiled_cache_shard_size=768, compiled_cache_max_shards=2,
        model=SimpleNamespace(context_capacity=17, history_reset_keep=2),
    )
    before = {path: path.read_bytes() for path in cache_dir.rglob("*.pt")}
    monkeypatch.setattr(ActionSpace, "from_job_tag", lambda _job: changed)
    with pytest.raises(FileNotFoundError, match="cache not found or stale"):
        _build_dataset([source], config, normalizer, torch.int32, torch.float32, cache_dir)
    with pytest.raises(FileNotFoundError, match="训练缓存缺失或已过期"):
        select_prepared_training_sources(
            root, job_tag="black_mage", max_files=1, cache_dir=cache_dir,
            int_dtype=torch.int32, float_dtype=torch.float32, shard_size=768,
        )
    # 正式转换入口未指定模型契约时，仍须按当前 YAML 判定旧 cache 需要重编译。
    compiled = []
    monkeypatch.setattr(
        cache_compile, "_compile_raw_source_worker",
        lambda task, **_kw: compiled.append(task[0]) or (str(task[0]), 1, 1),
    )
    assert cache_compile.precompile_raw_training_caches(
        [source], job_tag="black_mage", normalizer=normalizer, cache_dir=cache_dir,
        int_dtype=torch.int32, float_dtype=torch.float32, shard_size=768, engine=engine,
    ) == [source]
    assert compiled == [source]

    # checkpoint caller 提供保存的契约后，不应重新读取当前动作配置或启动转换器。
    monkeypatch.setattr(ActionSpace, "from_job_tag", lambda _job: pytest.fail("已有模型兼容缓存不应重建动作空间"))
    monkeypatch.setattr(cache_compile, "_compile_raw_source_worker", lambda *_a, **_kw: pytest.fail("不应补编译"))
    store = ReplayCacheStore(max_shards=2)
    kwargs = dict(job_tag="black_mage", normalizer=normalizer, expected_action_space=saved_actions,
                  expected_skill_vocab=SkillVocab.build_from_job_tag("black_mage"), engine=engine)
    store.prepare([config], workers=2, **kwargs)
    reader = store.load(config, **kwargs)
    assert reader.action_keys == saved_actions.action_keys
    assert reader.action_to_vocab_id == saved_actions.action_to_vocab_id
    assert reader.action_is_gcd == saved_actions.action_is_gcd
    dataset = TrainingDataset(
        [source], normalizer=normalizer, expected_action_space=saved_actions,
        context_capacity=1024,
        skill_vocab=SkillVocab.build_from_job_tag("black_mage"), job_tag="black_mage",
        int_dtype=torch.int32, float_dtype=torch.float32, cache_dir=cache_dir,
        compiled_cache_shard_size=768,
    )
    assert ActionSpace.from_data_spec(DataSpec.from_dataset(dataset)) == saved_actions
    assert before == {path: path.read_bytes() for path in cache_dir.rglob("*.pt")}


@pytest.mark.parametrize("existing_cache", [False, True])
@pytest.mark.parametrize("drift", ["enabled", "kind"])
def test_model_cache_recompile_rejects_current_actions_before_writing(scene_cache, monkeypatch, existing_cache, drift):
    from scripts.autoregressive_replay.replay import ReplayCacheStore
    from scripts.convert_fflogs.cache import cache_compile

    _, cache_dir, create = scene_cache
    source = create("FRU/90-100/saved.json.br", invalid=None if existing_cache else "missing")
    saved_actions = ActionSpace.from_job_tag("black_mage")
    changed = _changed_actions(saved_actions, drift)
    if existing_cache:
        # 请求旧模型契约，但已有 cache 是当前 YAML 的另一套动作契约。
        manifest = cache_path_for_source(cache_dir, source)
        payload = safe_torch_load(manifest, safe_globals=(TrainingSchema, StateFeatureGroup))
        payload.update(action_keys=changed.action_keys, action_to_vocab_id=changed.action_to_vocab_id,
                       action_is_gcd=changed.action_is_gcd, num_actions=len(changed.action_keys))
        torch.save(payload, manifest)
    before = {path: path.read_bytes() for path in cache_dir.rglob("*.pt")}
    monkeypatch.setattr(ActionSpace, "from_job_tag", lambda _job: changed)
    monkeypatch.setattr(cache_compile, "InProcessEngine", lambda *_a, **_kw: pytest.fail("应在启动引擎前拒绝"))
    monkeypatch.setattr(cache_compile, "_compile_raw_source_worker", lambda *_a, **_kw: pytest.fail("应在写 cache 前拒绝"))
    config = SimpleNamespace(scene_json_path=source, cache_dir=cache_dir, cache_shard_size=768, cache_max_shards=2)
    normalizer = Normalizer()
    engine = SimpleNamespace(job_tag="black_mage", capacity=2)
    kwargs = dict(job_tag="black_mage", normalizer=normalizer, expected_action_space=saved_actions,
                  expected_skill_vocab=SkillVocab.build_from_job_tag("black_mage"), engine=engine)
    store = ReplayCacheStore(max_shards=2)
    with pytest.raises(ValueError, match="current YAML differs.*model action contract"):
        store.prepare([config], workers=2, **kwargs)
    with pytest.raises(ValueError, match="current YAML differs.*model action contract"):
        store.load(config, **kwargs)
    assert before == {path: path.read_bytes() for path in cache_dir.rglob("*.pt")}
    if not existing_cache:
        assert not cache_dir.exists()


@pytest.mark.parametrize("payload,error_type", [(None, TypeError), ({"version": 1}, ValueError)])
def test_default_replay_rejects_missing_or_old_contract(scene_cache, monkeypatch, payload, error_type):
    from scripts.autoregressive_replay.config import _resolve_scene_json

    root, cache_dir, create = scene_cache
    create("FRU/90-100/current.json.br")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_JSON", "")
    with pytest.raises(error_type, match="input contract"):
        _resolve_scene_json(
            None, raw_root=root, cache_dir=cache_dir, job_tag="black_mage",
            cache_shard_size=768, input_contract_payload=payload,
        )


def _cache_with_inactive_vocab(scene_cache):
    _, cache_dir, create = scene_cache
    source = create("FRU/90-100/full_vocab.json.br")
    base = SkillVocab.build_from_job_tag("black_mage")
    vocab = SkillVocab.from_entries([*base, (900001, base.size()), (900002, base.size() + 1)])
    path = cache_path_for_source(cache_dir, source)
    payload = safe_torch_load(path, safe_globals=(TrainingSchema, StateFeatureGroup))
    payload["vocab_signature"] = tuple(vocab)
    torch.save(payload, path)
    return source, cache_dir, vocab


def test_model_cache_reuses_complete_vocab_without_current_yaml(scene_cache, monkeypatch):
    from scripts.autoregressive_replay.replay import ReplayCacheStore

    source, cache_dir, saved_vocab = _cache_with_inactive_vocab(scene_cache)
    saved_actions = ActionSpace.from_job_tag("black_mage")
    normalizer = Normalizer()
    normalizer.ensure_job_resources("black_mage")
    config = SimpleNamespace(scene_json_path=source, cache_dir=cache_dir, cache_shard_size=768, cache_max_shards=2)
    engine = SimpleNamespace(job_tag="black_mage", capacity=2)
    before = {path: path.read_bytes() for path in cache_dir.rglob("*.pt")}
    monkeypatch.setattr(SkillVocab, "build_from_job_tag", lambda *_a, **_kw: pytest.fail("兼容缓存不读取当前词表 YAML"))
    monkeypatch.setattr(ActionSpace, "from_job_tag", lambda *_a, **_kw: pytest.fail("兼容缓存不重建动作空间"))
    store = ReplayCacheStore(max_shards=2)
    kwargs = dict(job_tag="black_mage", normalizer=normalizer, expected_action_space=saved_actions,
                  expected_skill_vocab=saved_vocab, engine=engine)
    store.prepare([config], workers=2, **kwargs)
    reader = store.load(config, **kwargs)
    saved_vocab.assert_matches(reader.vocab_signature, context="test")
    assert before == {path: path.read_bytes() for path in cache_dir.rglob("*.pt")}


def test_model_cache_rejects_inactive_row_drift_before_recompiling(scene_cache, monkeypatch):
    from common.policy.data.cache.compiled_cache import CompiledShardCache, load_compiled_cache
    from scripts.convert_fflogs.cache import cache_compile

    source, cache_dir, saved_vocab = _cache_with_inactive_vocab(scene_cache)
    path = cache_path_for_source(cache_dir, source)
    payload = safe_torch_load(path, safe_globals=(TrainingSchema, StateFeatureGroup))
    entries = list(saved_vocab)
    entries[-2], entries[-1] = (entries[-2][0], entries[-1][1]), (entries[-1][0], entries[-2][1])
    changed = SkillVocab.from_entries(entries)
    payload["vocab_signature"] = tuple(changed)
    torch.save(payload, path)
    actions = ActionSpace.from_job_tag("black_mage")
    # 输出动作、词表行数和 bank 形状都相同，完整 raw-id 映射仍然必须一致。
    assert load_compiled_cache(path, source, signature=payload["cache_signature"], expected_action_space=actions,
                               expected_skill_vocab=saved_vocab, shard_cache=CompiledShardCache(1)) is None
    before = {p: p.read_bytes() for p in cache_dir.rglob("*.pt")}
    monkeypatch.setattr(SkillVocab, "build_from_job_tag", lambda *_a, **_kw: changed)
    monkeypatch.setattr(cache_compile, "InProcessEngine", lambda *_a, **_kw: pytest.fail("词表不匹配时不启动引擎"))
    with pytest.raises(ValueError, match="cache compilation skill vocab mismatch.*raw_skill_id=900001"):
        cache_compile.precompile_raw_training_caches(
            [source], job_tag="black_mage", normalizer=Normalizer(), expected_action_space=actions,
            expected_skill_vocab=saved_vocab, cache_dir=cache_dir, shard_size=768,
            int_dtype=torch.int32, float_dtype=torch.float32,
        )
    assert before == {p: p.read_bytes() for p in cache_dir.rglob("*.pt")}


@pytest.mark.parametrize("saved", [None, {"movement": {"speed": 0.5}}])
def test_compiled_cache_compares_caller_config_map_without_reading_yaml(scene_cache, saved):
    from common.policy.data.cache.compiled_cache import CompiledShardCache, load_compiled_cache

    _, cache_dir, create = scene_cache
    source = create("FRU/90-100/config.json.br")
    artifact = cache_path_for_source(cache_dir, source)
    payload = safe_torch_load(artifact, safe_globals=(TrainingSchema, StateFeatureGroup))
    if saved is not None:
        payload["config_dependencies"] = saved
    else:
        payload.pop("config_dependencies", None)
    torch.save(payload, artifact)
    args = dict(signature=payload["cache_signature"], expected_action_space=ActionSpace.from_job_tag("black_mage"),
                shard_cache=CompiledShardCache(1))
    assert load_compiled_cache(artifact, source, **args) is not None
    current = {"movement": {"speed": 0.5}}
    matched = load_compiled_cache(artifact, source, expected_config=current, **args)
    assert (matched is not None) == (saved is not None)
    assert load_compiled_cache(artifact, source, expected_config={"movement": {"speed": 0.6}}, **args) is None


def test_conversion_quota_rejects_stale_config_before_counting_cached_sources(scene_cache, monkeypatch):
    from common.policy.data.cache.source_selection import RawTrainingPathGroup
    from scripts.convert_fflogs.cache import cache_compile

    root, cache_dir, create = scene_cache
    stale = create("FRU/90-100/stale.json.br")
    current = create("FRU/90-100/current.json.br")
    expected = {"movement": {"speed": 0.5}}
    for source, speed in ((stale, 0.6), (current, 0.5)):
        artifact = cache_path_for_source(cache_dir, source)
        payload = safe_torch_load(artifact, safe_globals=(TrainingSchema, StateFeatureGroup))
        payload["config_dependencies"] = {"movement": {"speed": speed}}
        torch.save(payload, artifact)
    groups = (RawTrainingPathGroup("FRU", 1, (stale, current)),)
    monkeypatch.setattr(cache_compile, "select_training_raw_path_groups", lambda *_a: groups)
    monkeypatch.setattr(cache_compile, "build_conversion_config_dependencies", lambda *_a, **_kw: expected)
    monkeypatch.setattr(cache_compile, "precompile_raw_training_caches", lambda *_a, **_kw: pytest.fail("同组有效缓存已满足配额"))
    assert cache_compile.prepare_training_caches(
        root, max_files=1, job_tag="black_mage", int_dtype=torch.int32, float_dtype=torch.float32,
        cache_dir=cache_dir, shard_size=768,
    ) == [current]


def test_conversion_rebuilds_legacy_and_changed_maps_but_reuses_comment_changes(scene_cache, tmp_path, monkeypatch):
    from common.configuration.yaml import load_yaml_mapping
    from scripts.convert_fflogs.cache import cache_compile

    _, cache_dir, create = scene_cache
    source = create("FRU/90-100/rebuild.json.br")
    artifact = cache_path_for_source(cache_dir, source)
    config_path = tmp_path / "conversion.yaml"
    config_path.write_text("movement: {speed: 0.5, gap: 1}\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(cache_compile, "build_conversion_config_dependencies",
                        lambda *_a, **_kw: load_yaml_mapping(config_path))
    compiled = []

    def compile_worker(task, *, engine):
        assert engine.job_tag == "black_mage"
        payload = safe_torch_load(artifact, safe_globals=(TrainingSchema, StateFeatureGroup))
        payload["config_dependencies"] = task[-1]
        torch.save(payload, artifact)
        compiled.append(task[0])
        return str(task[0]), 1, 1

    monkeypatch.setattr(cache_compile, "_compile_raw_source_worker", compile_worker)
    args = dict(job_tag="black_mage", normalizer=Normalizer(), int_dtype=torch.int32,
                float_dtype=torch.float32, cache_dir=cache_dir, shard_size=768,
                engine=SimpleNamespace(job_tag="black_mage", capacity=1))
    assert cache_compile.precompile_raw_training_caches([source], **args) == [source]
    assert compiled == [source]
    config_path.write_text("# 新注释\nmovement: {gap: 1, speed: 0.5}\n", encoding="utf-8", newline="\n")
    assert cache_compile.precompile_raw_training_caches([source], **args) == [source]
    assert compiled == [source]
    config_path.write_text("movement: {speed: 0.6, gap: 1}\n", encoding="utf-8", newline="\n")
    assert cache_compile.precompile_raw_training_caches([source], **args) == [source]
    assert compiled == [source, source]
    payload = safe_torch_load(artifact, safe_globals=(TrainingSchema, StateFeatureGroup))
    assert payload["config_dependencies"] == {"movement": {"speed": 0.6, "gap": 1}}


def test_conversion_inventory_only_reloads_changed_and_new_caches(scene_cache, monkeypatch):
    """预检复用真实 PT 的校验结果，变更 raw/manifest/分片后恢复正式校验。"""
    from scripts.convert_fflogs.cache import cache_compile, cache_load

    _, cache_dir, create = scene_cache
    sources = [create(f"FRU/90-100/{name}.json.br") for name in ("a", "b", "c", "d")]
    actions = ActionSpace.from_job_tag("black_mage")
    vocab = SkillVocab.build_from_job_tag("black_mage")
    precision = load_precision_config()
    args = dict(job_tag="black_mage", normalizer=Normalizer(), cache_dir=cache_dir, shard_size=768,
                int_dtype=precision.resolve_int_dtype(), float_dtype=precision.resolve_float_dtype(),
                expected_action_space=actions, expected_skill_vocab=vocab,
                engine=SimpleNamespace(job_tag="black_mage", capacity=2))
    assert cache_compile.precompile_raw_training_caches(sources, **args) == sources
    original_load = cache_load._load_cache
    reads, compiled = [], []

    def load(source, **kwargs):
        reads.append(source)
        return original_load(source, **kwargs)

    monkeypatch.setattr(cache_load, "_load_cache", load)
    assert cache_compile.precompile_raw_training_caches(sources, **args) == sources
    assert reads == []

    atomic_write_json(sources[1], {"changed": True})
    manifest = cache_path_for_source(cache_dir, sources[2])
    manifest.write_bytes(b"broken")
    shard = cache_path_for_source(cache_dir, sources[3]).with_suffix(".shard.pt")
    shard.unlink()
    new = create("FRU/90-100/z-new.json.br", invalid="missing")
    expected = {sources[1], sources[2], sources[3], new}

    def compile_worker(task, *, engine):
        compiled.append(Path(task[0]))
        return str(task[0]), 1, 1

    monkeypatch.setattr(cache_compile, "_compile_raw_source_worker", compile_worker)
    assert cache_compile.precompile_raw_training_caches([*sources, new], **args) == [*sources, new]
    assert set(compiled) == expected
    assert set(reads) == expected

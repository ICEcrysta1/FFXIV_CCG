"""训练公共层 PT、compiled cache 与 batch 测试。"""

from __future__ import annotations

import pytest

from common.config import load_precision_config
from scripts.convert_fflogs.source.source_reader import TrainingSourceReader
from scripts.convert_fflogs.training.sample_builder import TrainingSampleBuilder
from scripts.convert_fflogs.utils import build_skill_book, load_job_project_config
from common.policy.config import ModelConfig
from common.policy.data import ActionSpace, CompiledCacheReader, DataSpec, Normalizer
from common.policy.data.prepared_sources import select_prepared_validation_sources
from training import ShardBatchSampler, TrainingCollator, WeightedShardBatchSampler
from training.config import RunConfig
from common.policy.model.input_encoder import CausalInputEncoder
from training.loop import build_dataloaders
from tests.training._common_fixtures import (
    enabled_black_mage_config,
    make_dataset,
    make_demo_pt,
    make_illegal_action_pt,
    write_test_compiled_cache,
)
from common.policy.data.compiled_cache import (
    CACHE_FORMAT,
    CompiledShardCache,
    load_compiled_cache,
)


def test_precision_config_separates_value_and_operator_index_dtypes(tmp_path):
    torch = pytest.importorskip("torch")
    config_path = tmp_path / "precision.yaml"
    config_path.write_text(
        "precision:\n  int_dtype: int32\n  index_dtype: int64\n  float_dtype: float32\n",
        encoding="utf-8", newline="\n",
    )
    precision = load_precision_config(config_path)
    assert precision.resolve_int_dtype() == torch.int32
    assert precision.resolve_index_dtype() == torch.int64
    assert precision.resolve_float_dtype() == torch.float32

    config_path.write_text(
        "precision:\n  int_dtype: int32\n  index_dtype: int32\n  float_dtype: float32\n",
        encoding="utf-8", newline="\n",
    )
    with pytest.raises(ValueError, match="index_dtype"):
        load_precision_config(config_path)


def test_training_dataset_accepts_single_and_multi_sample_pt_with_same_contract(tmp_path):
    pytest.importorskip("torch")
    single_pt = make_demo_pt(tmp_path, ["fire_iii"], fight_id="single_demo")
    multi_pt = make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="multi_demo")

    dataset = make_dataset([single_pt, multi_pt])

    assert len(dataset) == 3
    assert dataset.skill_feature_names


def test_training_dataset_returns_grouped_sample_and_uses_config_vocab(tmp_path):
    pytest.importorskip("torch")
    pt_path = make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="dataset_demo")

    dataset = make_dataset([pt_path])
    sample = dataset[1]
    fire_iii_vocab_id = dataset.skill_vocab.lookup(
        build_skill_book(load_job_project_config("black_mage")).get("fire_iii").game_id
    )

    assert sample["history_length"] == 1
    history_start = sample["history_end"] - sample["history_length"]
    assert sample["history_bank_skill_ids"][history_start].item() == fire_iii_vocab_id
    assert sample["history_bank_skill_features"].shape[1] == len(dataset.skill_feature_names)
    assert "kind" in dataset.skill_feature_names
    kind_index = dataset.skill_feature_names.index("kind")
    assert sample["history_bank_skill_features"][history_start, kind_index].item() == pytest.approx(1.0)
    assert sample["history_bank_skill_potencies"].shape == sample["history_bank_cumulative_dot_potencies"].shape
    assert sample["history_bank_state_vectors"].shape == sample["history_bank_state_null_mask"].shape
    assert sample["action_values"].shape == (len(sample["action_keys"]),)
    assert sample["current_state_vectors"].shape == sample["current_state_null_mask"].shape == (dataset.state_dim,)
    assert sample["scene_vectors"].shape[0] == sample["scene_types"].shape[0]
    assert sample["label_action_key"] == "fire_iv"


def test_training_dataset_compiles_and_reuses_disk_cache(tmp_path):
    pytest.importorskip("torch")
    pt_path = make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="compiled_cache_demo")
    cache_dir = tmp_path / "cache"

    first = make_dataset([pt_path], cache_dir=cache_dir)
    first_sample = first[1]
    cache_files = list(cache_dir.glob("*.compiled.pt"))

    assert len(cache_files) == 1
    assert isinstance(first._readers[0], CompiledCacheReader)

    second = make_dataset([pt_path], cache_dir=cache_dir)
    second_sample = second[1]

    assert isinstance(second._readers[0], CompiledCacheReader)
    assert second_sample["history_end"] == first_sample["history_end"]
    assert second_sample["history_length"] == first_sample["history_length"]
    assert second_sample["history_bank_skill_ids"].equal(first_sample["history_bank_skill_ids"])
    assert second_sample["current_state_vectors"].equal(first_sample["current_state_vectors"])
    assert len(first._shard_cache) == 1

    restored = object.__new__(type(first))
    restored.__setstate__(first.__getstate__())
    assert isinstance(restored._readers[0], CompiledCacheReader)
    assert restored[1]["current_state_vectors"].equal(first_sample["current_state_vectors"])


def test_compiled_history_and_current_state_use_compact_state_contract(tmp_path):
    """真实 C# 转换生成的当前请求与完整历史 bank 必须同时采用新输入维度。"""
    pytest.importorskip("torch")
    source_path = make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="seconds_contract")
    dataset = make_dataset([source_path])
    sample = dataset[1]
    removed_fields = {
        "gcd_index", "fight_remaining_seconds", "gcd_remaining_gcds",
        "weave_window_gcds", "ogcd_window_gcds", "downtime_remaining_gcds",
        "gcd_remaining_seconds", "weave_window_seconds", "ogcd_window_seconds",
        "next_untargetable_in_gcds", "remaining_gcds",
        "ogcds_weaved", "max_ogcd_per_window",
    }

    assert "gcd_index" not in dataset.skill_feature_names
    assert len(dataset.skill_feature_names) == 19
    assert "job_resources_consumed.polyglot" in dataset.skill_feature_names
    assert dataset.schema.state_vector_dim() == 86
    player_keys = dataset.schema.state_group_feature_keys["player_state"]
    assert len(player_keys) == 18
    assert "before.current_gcd_seconds" in player_keys
    assert "after.downtime_remaining_seconds" in player_keys
    assert len(dataset.schema.state_group_feature_keys["buff_state"]) == 40
    assert len(dataset.schema.state_group_feature_keys["target_buff_state"]) == 14
    resource_keys = dataset.schema.state_group_feature_keys["resource_state"]
    assert len(resource_keys) == 14
    assert all(key.startswith("before.") for key in resource_keys[:7])
    assert all(key.startswith("after.") for key in resource_keys[7:])
    assert not any(
        key.rsplit(".", 1)[-1] in removed_fields or key.endswith("_gcds")
        or key.startswith("consumed.") or ".manaward." in key or ".surecast." in key
        for keys in dataset.schema.state_group_feature_keys.values()
        for key in keys
    )
    assert sample["history_bank_skill_features"].shape[-1] == 19
    for prefix in ("history_bank", "current"):
        assert sample[f"{prefix}_state_vectors"].shape[-1] == 86
        assert sample[f"{prefix}_state_null_mask"].shape == sample[f"{prefix}_state_vectors"].shape


@pytest.mark.parametrize("old_contract", ["cache_format", "conversion_version"])
def test_previous_state_layout_cache_is_rejected(tmp_path, old_contract):
    """旧输入字段缓存不可复用，即使 raw 文件身份和其余编译参数一致。"""
    torch = pytest.importorskip("torch")
    from common.policy.data.compiled_cache import cache_path_for_source
    from common.policy.data.schema import SceneWindowSchema, TrainingSchema
    from common.torch_serialization import safe_torch_load

    source_path = make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="old_progress_cache")
    cache_path = cache_path_for_source(tmp_path / ".cache", source_path)
    payload = safe_torch_load(cache_path, safe_globals=(SceneWindowSchema, TrainingSchema))
    signature = dict(payload["cache_signature"])
    if old_contract == "cache_format":
        payload["cache_format"] = "raw_json_compiled_samples_v17_seconds_only_state"
    else:
        payload["cache_signature"]["conversion_version"] = "raw_json_to_compiled_v18_seconds_only_state"
    torch.save(payload, cache_path)

    assert load_compiled_cache(
        cache_path, source_path, signature=signature, shard_cache=CompiledShardCache(),
    ) is None


def test_training_dataset_history_window_reuses_full_cache(tmp_path):
    pytest.importorskip("torch")
    pt_path = make_demo_pt(
        tmp_path,
        ["fire_iii", "fire_iv", "fire_iv", "fire_iii"],
        fight_id="history_window_cache_demo",
    )
    cache_dir = tmp_path / "cache"

    short_window = make_dataset([pt_path], cache_dir=cache_dir, max_history=1)
    full_window = make_dataset([pt_path], cache_dir=cache_dir, max_history=2)
    sample_index = next(
        index
        for index in range(len(full_window))
        if int(full_window[index]["history_length"]) >= 2
    )

    assert short_window[sample_index]["history_length"] == 1
    assert full_window[sample_index]["history_length"] == 2
    assert (
        short_window._readers[0].history_bank["skill_ids"].shape
        == full_window._readers[0].history_bank["skill_ids"].shape
    )


def test_compiled_cache_manifest_uses_mmap_for_history_bank(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pt_path = make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="manifest_mmap_demo")
    cache_dir = tmp_path / "cache"
    make_dataset([pt_path], cache_dir=cache_dir)

    import importlib

    compiled_cache_module = importlib.import_module("common.policy.data.compiled_cache")
    original_load = compiled_cache_module.safe_torch_load
    mmap_values = []

    def tracking_load(path, **kwargs):
        mmap_values.append(bool(kwargs.get("mmap", False)))
        return original_load(path, **kwargs)

    monkeypatch.setattr(compiled_cache_module, "safe_torch_load", tracking_load)
    make_dataset([pt_path], cache_dir=cache_dir)

    assert mmap_values
    assert mmap_values[0] is True


@pytest.mark.parametrize(
    "corruption",
    [
        "non_tensor_field",
        "tensor_action_keys",
        "shape_mismatch",
        "bank_size_mismatch",
        "missing_history_bank",
        "overflow_num_samples",
        "overflow_shard_size",
    ],
)
def test_corrupt_history_bank_manifest_falls_back_to_recompile(
    tmp_path, monkeypatch, corruption
):
    """损坏的 history bank manifest 必须返回 None，交给上层重编译。"""
    torch = pytest.importorskip("torch")
    import importlib

    compiled_cache_module = importlib.import_module("common.policy.data.compiled_cache")
    cache_path = tmp_path / "corrupt.compiled.pt"
    cache_path.write_bytes(b"corrupt manifest")
    payload_num_samples = 2
    space = ActionSpace.from_job_tag("black_mage")
    history_bank = {
        "skill_ids": torch.zeros((2,), dtype=torch.int32),
        "skill_features": torch.zeros((2, 1)),
        "state_vectors": torch.zeros((2, 1)),
        "state_null_mask": torch.zeros((2, 1), dtype=torch.bool),
        "action_keys": ("", "fire_iii"),
        "skill_potencies": torch.zeros((2,)),
        "cumulative_dot_potencies": torch.zeros((2,)),
    }
    if corruption == "non_tensor_field":
        history_bank["skill_features"] = []
    elif corruption == "tensor_action_keys":
        history_bank["action_keys"] = torch.zeros((2, 2), dtype=torch.int32)
    elif corruption == "shape_mismatch":
        history_bank["skill_ids"] = torch.zeros((1,), dtype=torch.int32)
    elif corruption == "bank_size_mismatch":
        payload_num_samples = 1
    payload = {
        "cache_format": CACHE_FORMAT,
        "cache_signature": {},
        "schema": None,
        "job_tag": "black_mage",
        "num_samples": payload_num_samples,
        "num_actions": len(space.action_keys),
        "skill_feature_names": (),
        "action_keys": space.action_keys,
        "action_to_vocab_id": space.action_to_vocab_id,
        "action_is_gcd": space.action_is_gcd,
        "shard_size": float("inf") if corruption == "overflow_shard_size" else 1,
        "shard_files": [],
        "history_bank": history_bank,
    }
    if corruption == "missing_history_bank":
        payload.pop("history_bank")
    elif corruption == "overflow_num_samples":
        payload["num_samples"] = float("inf")
    monkeypatch.setattr(
        compiled_cache_module,
        "safe_torch_load",
        lambda *args, **kwargs: payload,
    )

    assert load_compiled_cache(
        cache_path,
        tmp_path / "source.json",
        signature={},
        shard_cache=CompiledShardCache(),
    ) is None


def test_empty_compiled_cache_accepts_sentinel_history_bank(tmp_path, monkeypatch):
    """空源缓存允许只保存 history bank 的 sentinel 行。"""
    torch = pytest.importorskip("torch")
    import importlib

    compiled_cache_module = importlib.import_module("common.policy.data.compiled_cache")
    cache_path = tmp_path / "empty.compiled.pt"
    cache_path.write_bytes(b"empty manifest")
    space = ActionSpace.from_job_tag("black_mage")
    payload = {
        "cache_format": CACHE_FORMAT,
        "cache_signature": {},
        "schema": None,
        "job_tag": "black_mage",
        "num_samples": 0,
        "num_actions": len(space.action_keys),
        "skill_feature_names": (),
        "action_keys": space.action_keys,
        "action_to_vocab_id": space.action_to_vocab_id,
        "action_is_gcd": space.action_is_gcd,
        "shard_size": 1,
        "shard_files": [],
        "history_bank": {
            "skill_ids": torch.zeros((1,), dtype=torch.int32),
            "skill_features": torch.zeros((1, 1)),
            "state_vectors": torch.zeros((1, 1)),
            "state_null_mask": torch.zeros((1, 1), dtype=torch.bool),
            "action_keys": ("",),
            "skill_potencies": torch.zeros((1,)),
            "cumulative_dot_potencies": torch.zeros((1,)),
        },
    }
    monkeypatch.setattr(
        compiled_cache_module,
        "safe_torch_load",
        lambda *args, **kwargs: payload,
    )

    reader = load_compiled_cache(
        cache_path,
        tmp_path / "source.json",
        signature={},
        shard_cache=CompiledShardCache(),
    )

    assert reader is not None
    assert reader.num_samples == 0
    assert reader.history_bank["action_keys"] == ("",)


def test_compiled_cache_reader_does_not_swallow_memory_error(tmp_path, monkeypatch):
    """reader 自身的内存错误不能被误判为缓存损坏。"""
    pytest.importorskip("torch")
    import importlib

    compiled_cache_module = importlib.import_module("common.policy.data.compiled_cache")
    cache_path = tmp_path / "manifest.pt"
    cache_path.write_bytes(b"manifest")
    space = ActionSpace.from_job_tag("black_mage")
    payload = {
        "cache_format": CACHE_FORMAT,
        "cache_signature": {},
        "job_tag": "black_mage",
        "action_keys": space.action_keys,
        "action_to_vocab_id": space.action_to_vocab_id,
        "action_is_gcd": space.action_is_gcd,
        "shard_files": [],
    }
    monkeypatch.setattr(
        compiled_cache_module,
        "safe_torch_load",
        lambda *args, **kwargs: payload,
    )

    def raise_memory_error(*args, **kwargs):
        raise MemoryError("simulated reader allocation failure")

    monkeypatch.setattr(compiled_cache_module, "CompiledCacheReader", raise_memory_error)

    with pytest.raises(MemoryError, match="simulated reader allocation failure"):
        load_compiled_cache(
            cache_path,
            tmp_path / "source.json",
            signature={},
            shard_cache=CompiledShardCache(),
        )


def test_compact_collator_deduplicates_copied_bank_by_explicit_id(tmp_path):
    pytest.importorskip("torch")
    pt_path = make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="bank_id_demo")
    dataset = make_dataset([pt_path])
    original = dataset[1]
    copied = dict(original)
    for key in (
        "history_bank_skill_ids",
        "history_bank_skill_features",
        "history_bank_state_vectors",
        "history_bank_state_null_mask",
    ):
        copied[key] = copied[key].clone()

    batch = TrainingCollator()([original, copied])

    assert batch["history_bank_skill_ids"].shape[0] == original["history_bank_skill_ids"].shape[0]
    assert batch["history_ends"].tolist() == [original["history_end"], original["history_end"]]


def test_training_dataloader_uses_shard_batch_sampler_after_precompile(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    first_pt = make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="loader_first")
    second_pt = make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="loader_second")
    config_path = enabled_black_mage_config(tmp_path)
    config = RunConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        job_tag="black_mage",
        batch_size=1,
        compiled_cache_shard_size=4,
        config_path=config_path,
    )
    precision = load_precision_config()
    cache_dir = tmp_path / "cache"
    for source_path in (first_pt, second_pt):
        from tests.training._common_fixtures import _TEST_TRAINING_PAYLOADS

        write_test_compiled_cache(
            source_path,
            _TEST_TRAINING_PAYLOADS[source_path.resolve()],
            cache_dir,
            {
                "normalizer": Normalizer(),
                "int_dtype": precision.resolve_int_dtype(),
                "float_dtype": precision.resolve_float_dtype(),
                "compiled_cache_shard_size": config.compiled_cache_shard_size,
            },
        )
    train_loader, val_loader, _, _ = build_dataloaders(
        [first_pt],
        config,
        validation_paths=[second_pt],
        int_dtype=precision.resolve_int_dtype(),
        float_dtype=precision.resolve_float_dtype(),
        cache_dir=cache_dir,
    )

    assert isinstance(train_loader.batch_sampler, WeightedShardBatchSampler)
    assert isinstance(val_loader.batch_sampler, ShardBatchSampler)
    assert all(reader.shard_size == config.compiled_cache_shard_size for reader in train_loader.dataset._readers)
    assert all(reader.shard_size == config.compiled_cache_shard_size for reader in val_loader.dataset._readers)
    assert next(iter(train_loader))["label_index"].shape == (1,)


def test_validation_selector_reads_real_compiled_pt_from_val_layout(tmp_path):
    """真实 manifest、签名和分片只会从指定阶段的 VAL 副本被选中。"""
    source = make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="val_fru")
    from tests.training._common_fixtures import _TEST_TRAINING_PAYLOADS

    validation = tmp_path / "annotated" / "VAL" / "FRU" / "val_fru.json.br"
    validation.parent.mkdir(parents=True)
    validation.write_text("{}", encoding="utf-8", newline="\n")
    cache_dir = tmp_path / ".cache"
    precision = load_precision_config()
    write_test_compiled_cache(
        validation, _TEST_TRAINING_PAYLOADS[source.resolve()], cache_dir,
        {"normalizer": Normalizer(),
         "int_dtype": precision.resolve_int_dtype(),
         "float_dtype": precision.resolve_float_dtype()},
    )
    assert select_prepared_validation_sources(
        cache_dir, data_dir=tmp_path / "annotated", max_files=1,
        job_tag="black_mage", int_dtype=precision.resolve_int_dtype(),
        float_dtype=precision.resolve_float_dtype(),
    ) == [validation]


def test_training_dataset_cache_loads_shards_lazily_with_global_bound(tmp_path):
    pytest.importorskip("torch")
    pt_path = make_demo_pt(
        tmp_path,
        ["fire_iii", "fire_iv", "fire_iv"],
        fight_id="lazy_compiled_cache_demo",
    )
    cache_dir = tmp_path / "cache"

    dataset = make_dataset(
        [pt_path],
        cache_dir=cache_dir,
        compiled_cache_shard_size=1,
        compiled_cache_max_shards=1,
    )
    assert len(list(cache_dir.glob("*.shard-*.pt"))) == len(dataset)
    assert len(dataset._shard_cache) == 0

    dataset[0]
    assert len(dataset._shard_cache) == 1
    dataset[1]
    assert len(dataset._shard_cache) == 1
    dataset[0]
    assert len(dataset._shard_cache) == 1


def test_training_dataset_batch_fetch_preserves_requested_order(tmp_path):
    pytest.importorskip("torch")
    pt_path = make_demo_pt(
        tmp_path,
        ["fire_iii", "fire_iv", "fire_iv"],
        fight_id="batch_fetch_demo",
    )
    dataset = make_dataset(
        [pt_path],
        cache_dir=tmp_path / "cache",
        compiled_cache_shard_size=2,
    )

    requested = [2, 0, 1]
    batch = dataset.__getitems__(requested)

    assert [sample["metadata"]["step"] for sample in batch] == [
        dataset[index]["metadata"]["step"] for index in requested
    ]


def test_training_dataset_treats_raw_skill_id_zero_as_real_ogcd_wait(tmp_path):
    pytest.importorskip("torch")
    pt_path = make_demo_pt(tmp_path, ["fire_iii"], fight_id="ogcd_wait_demo")

    dataset = make_dataset([pt_path])
    sample = dataset[0]
    ogcd_wait_vocab_id = dataset.skill_vocab.lookup(0)

    assert ogcd_wait_vocab_id > 0
    assert "ogcd_wait" in sample["action_keys"]
    assert dataset.action_to_vocab_id[sample["action_keys"].index("ogcd_wait")] == ogcd_wait_vocab_id


def test_training_dataset_keeps_current_request_state_for_illegal_output_action(tmp_path):
    torch = pytest.importorskip("torch")
    pt_path = make_illegal_action_pt(tmp_path)

    dataset = make_dataset([pt_path], normalizer=Normalizer())
    sample = dataset[0]
    fire_iii_index = sample["action_keys"].index("fire_iii")
    null_mask = sample["current_state_null_mask"]
    values = sample["current_state_vectors"]

    assert bool(sample["action_legal_mask"][fire_iii_index].item()) is False
    assert bool(null_mask.any().item()) is False
    assert torch.isfinite(values).all()


def test_training_collator_pads_history_and_scene_lengths(tmp_path):
    pytest.importorskip("torch")
    short_pt = make_demo_pt(tmp_path, ["fire_iii"], fight_id="short_demo")
    long_pt = make_demo_pt(
        tmp_path,
        ["fire_iii", "fire_iv", "fire_iv"],
        fight_id="long_demo",
    )

    short_dataset = make_dataset([short_pt])
    long_dataset = make_dataset([long_pt])
    collator = TrainingCollator()
    batch = collator([short_dataset[0], long_dataset[2]])

    assert "history_skill_ids" not in batch
    assert batch["history_lengths"].tolist() == [0, 2]
    assert batch["history_mask"].shape == (2, 2)
    assert batch["history_mask"][0].sum().item() == 0
    assert batch["history_mask"][1].sum().item() == 2
    assert batch["scene_vectors"].ndim == 3
    assert batch["scene_mask"].shape[:2] == batch["scene_vectors"].shape[:2]
    assert batch["current_state_vectors"].shape == batch["current_state_null_mask"].shape == (2, short_dataset.state_dim)


@pytest.mark.parametrize("levels, status, percentile, error", [
    ([0], "attributed_label", 50.0, "levels"),
    ([4], "attributed_label", 50.0, "levels"),
    ([3], "unannotated", 50.0, "available annotation"),
    ([3], "attributed_label", None, "percentile"),
    ([3], "attributed_label", -1.0, "percentile"),
    ([3], "attributed_label", 101.0, "percentile"),
    ([3], "attributed_label", float("nan"), "percentile"),
    ([3], "attributed_label", float("inf"), "percentile"),
    ([3], "attributed_label", -float("inf"), "percentile"),
])
def test_training_collator_rejects_invalid_quality_supervision_on_cpu(
    tmp_path, levels, status, percentile, error,
):
    torch = pytest.importorskip("torch")
    pt_path = make_demo_pt(tmp_path, ["fire_iii"], fight_id="invalid_quality")
    sample = make_dataset([pt_path])[0]
    sample = {
        **sample,
        "quality_label_levels": torch.tensor(levels, dtype=torch.int32),
        "metadata": {
            **sample["metadata"], "quality_label_status": status, "percentile": percentile,
        },
    }
    with pytest.raises(ValueError, match=error):
        TrainingCollator(require_quality_percentile=True)([sample])


@pytest.mark.parametrize("collator_options", [{}, {"require_quality_percentile": False}])
def test_unweighted_dataloader_accepts_tagged_sample_without_percentile(
    tmp_path, collator_options,
):
    torch = pytest.importorskip("torch")
    from training.loop.loss import compose_training_loss

    pt_path = make_demo_pt(tmp_path, ["fire_iii"], fight_id="legacy_tagged_quality")
    sample = make_dataset([pt_path])[0]
    sample = {
        **sample,
        "quality_label_levels": torch.tensor([3], dtype=torch.int32),
        "metadata": {
            **sample["metadata"], "quality_label_status": "attributed_label", "percentile": None,
        },
    }
    loader = torch.utils.data.DataLoader(
        [sample], batch_size=1, collate_fn=TrainingCollator(**collator_options),
    )
    batch = next(iter(loader))
    assert batch["quality_label_levels"].tolist() == [[3]]
    assert batch["quality_annotation_available"].tolist() == [True]
    assert batch["source_quality"].tolist() == [-1.0]
    logits = torch.zeros_like(batch["action_legal_mask"], dtype=torch.float32)
    loss = compose_training_loss({"logits": logits}, batch)
    expected = torch.nn.functional.cross_entropy(logits, batch["label_index"])
    torch.testing.assert_close(loss.primary, expected)


@pytest.mark.parametrize("levels, status, error", [
    ([0], "attributed_label", "levels"),
    ([4], "attributed_label", "levels"),
    ([3], "unannotated", "available annotation"),
])
def test_unweighted_collator_still_rejects_invalid_quality_labels(tmp_path, levels, status, error):
    torch = pytest.importorskip("torch")
    pt_path = make_demo_pt(tmp_path, ["fire_iii"], fight_id="invalid_unweighted_quality")
    sample = make_dataset([pt_path])[0]
    sample = {
        **sample,
        "quality_label_levels": torch.tensor(levels, dtype=torch.int32),
        "metadata": {
            **sample["metadata"], "quality_label_status": status, "percentile": None,
        },
    }
    with pytest.raises(ValueError, match=error):
        TrainingCollator(require_quality_percentile=False)([sample])


def test_training_collator_allows_unknown_percentile_for_untagged_samples(tmp_path):
    torch = pytest.importorskip("torch")
    pt_path = make_demo_pt(tmp_path, ["fire_iii"], fight_id="unknown_quality")
    sample = make_dataset([pt_path])[0]
    samples = [
        {
            **sample,
            "quality_label_levels": torch.empty(0, dtype=torch.int32),
            "metadata": {**sample["metadata"], "percentile": percentile},
        }
        for percentile in (None, float("nan"))
    ]
    batch = TrainingCollator(require_quality_percentile=True)(samples)
    assert batch["quality_label_levels"].shape == (2, 0)
    assert batch["source_quality"][0].item() == -1.0
    assert torch.isnan(batch["source_quality"][1])


def test_compact_history_materialization_matches_legacy_dense_builder(tmp_path):
    torch = pytest.importorskip("torch")
    pt_path = make_demo_pt(
        tmp_path,
        ["fire_iii", "fire_iv", "fire_iv", "fire_iii"],
        fight_id="compact_history_equivalence",
    )
    dataset = make_dataset([pt_path])
    from tests.training._common_fixtures import _TEST_TRAINING_PAYLOADS

    reader = TrainingSourceReader(_TEST_TRAINING_PAYLOADS[pt_path.resolve()])
    normalizer = Normalizer()
    normalizer.configure_job_resources(reader.job_tag)
    normalizer.register_schema(reader.schema)
    precision = load_precision_config()
    dense_builder = TrainingSampleBuilder(
        torch=torch,
        normalizer=normalizer,
        skill_vocab=dataset.skill_vocab,
        skill_feature_names=reader.skill_feature_names,
        int_dtype=precision.resolve_int_dtype(),
        float_dtype=precision.resolve_float_dtype(),
        num_actions=reader.num_actions,
    )
    indices = [0, 1, 3]
    dense_batch = TrainingCollator()(
        [dense_builder.build(reader, index) for index in indices]
    )
    compact_batch = TrainingCollator()([dataset[index] for index in indices])

    encoder = CausalInputEncoder(
        DataSpec.from_dataset(dataset),
        ModelConfig(d_model=16, pair_embedding_dim=8, n_layers=1, n_heads=2, ff_dim=32),
        vocab_size=dataset.skill_vocab.size(),
    )
    encoder._materialize_compact_history(compact_batch)
    for key in (
        "history_mask",
        "history_skill_ids",
        "history_skill_features",
        "history_state_vectors",
        "history_state_null_mask",
    ):
        assert torch.equal(compact_batch[key], dense_batch[key]), key
    assert compact_batch["history_action_keys"] == dense_batch["history_action_keys"]

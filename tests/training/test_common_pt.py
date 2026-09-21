"""训练公共层 PT、compiled cache 与 batch 测试。"""

from __future__ import annotations

import pytest

from common.config import load_precision_config
from scripts.convert_fflogs.sample_builder import TrainingSampleBuilder
from scripts.convert_fflogs.source_reader import TrainingSourceReader
from scripts.convert_fflogs.utils import build_skill_book, load_job_project_config
from common.policy.config import ModelConfig
from common.policy.data import CompiledCacheReader, DataSpec, Normalizer
from training import ShardBatchSampler, TrainingCollator, WeightedShardBatchSampler
from training.config import RunConfig
from common.policy.model.input_encoder import CandidateInputEncoder
from training.loop import build_dataloaders
from tests.training._common_fixtures import (
    enabled_black_mage_config,
    make_dataset,
    make_demo_pt,
    make_illegal_candidate_pt,
    write_test_compiled_cache,
)
from common.policy.data.compiled_cache import (
    CACHE_FORMAT,
    CompiledShardCache,
    load_compiled_cache,
)


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
    assert sample["candidate_skill_ids"].shape[0] == len(sample["candidate_action_keys"])
    assert sample["candidate_skill_features"].shape[1] == len(dataset.skill_feature_names)
    assert sample["candidate_values"].shape[0] == len(sample["candidate_action_keys"])
    assert sample["candidate_state_vectors"].shape == sample["candidate_state_null_mask"].shape
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
    assert second_sample["candidate_state_vectors"].equal(first_sample["candidate_state_vectors"])
    assert len(first._shard_cache) == 1

    restored = object.__new__(type(first))
    restored.__setstate__(first.__getstate__())
    assert isinstance(restored._readers[0], CompiledCacheReader)
    assert restored[1]["candidate_state_vectors"].equal(first_sample["candidate_state_vectors"])


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
        "num_candidates": 1,
        "skill_feature_names": (),
        "candidate_action_keys": (),
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
    payload = {
        "cache_format": CACHE_FORMAT,
        "cache_signature": {},
        "schema": None,
        "job_tag": "black_mage",
        "num_samples": 0,
        "num_candidates": 1,
        "skill_feature_names": (),
        "candidate_action_keys": (),
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
    payload = {
        "cache_format": CACHE_FORMAT,
        "cache_signature": {},
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
        compiled_cache_workers=1,
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
        [first_pt, second_pt],
        config,
        int_dtype=precision.resolve_int_dtype(),
        float_dtype=precision.resolve_float_dtype(),
        cache_dir=cache_dir,
    )

    assert isinstance(train_loader.batch_sampler, WeightedShardBatchSampler)
    assert isinstance(val_loader.batch_sampler, ShardBatchSampler)
    assert all(reader.shard_size == config.compiled_cache_shard_size for reader in train_loader.dataset._readers)
    assert all(reader.shard_size == config.compiled_cache_shard_size for reader in val_loader.dataset._readers)
    assert next(iter(train_loader))["label_index"].shape == (1,)


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
    assert "ogcd_wait" in sample["candidate_action_keys"]
    assert sample["candidate_skill_ids"][sample["candidate_action_keys"].index("ogcd_wait")].item() == ogcd_wait_vocab_id


def test_training_dataset_uses_null_mask_for_illegal_candidate_state(tmp_path):
    torch = pytest.importorskip("torch")
    pt_path = make_illegal_candidate_pt(tmp_path)

    dataset = make_dataset([pt_path], normalizer=Normalizer())
    sample = dataset[0]
    fire_iii_index = sample["candidate_action_keys"].index("fire_iii")
    null_mask = sample["candidate_state_null_mask"][fire_iii_index]
    values = sample["candidate_state_vectors"][fire_iii_index]

    assert bool(sample["candidate_legal_mask"][fire_iii_index].item()) is False
    assert bool(null_mask.any().item()) is True
    assert torch.count_nonzero(values[null_mask]).item() > 0
    assert torch.all(values[null_mask] == -1.0)


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
    assert batch["candidate_state_vectors"].shape == batch["candidate_state_null_mask"].shape


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
        num_candidates=reader.num_candidates,
    )
    indices = [0, 1, 3]
    dense_batch = TrainingCollator()(
        [dense_builder.build(reader, index) for index in indices]
    )
    compact_batch = TrainingCollator()([dataset[index] for index in indices])

    encoder = CandidateInputEncoder(
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

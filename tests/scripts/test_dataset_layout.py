"""数据阶段目录映射与下载、转换共用布局的回归测试。"""

import pytest

from common.dataset_layout import (
    PERCENTILE_BUCKETS,
    find_dataset_json_files,
    map_dataset_output_path,
    percentile_bucket,
    percentile_directory,
)
from common.policy.data.compiled_cache import cache_path_for_source
from common.policy.data.source_selection import select_training_raw_path_groups
from scripts.convert_fflogs.cli import _resolve_input_files


def test_download_and_annotation_preserve_every_bucket(tmp_path):
    raw_root = tmp_path / "raw"
    annotated_root = tmp_path / "annotated"
    for percentile in range(0, 101, 10):
        bucket = percentile_bucket(percentile)
        source = percentile_directory(raw_root / "FRU", bucket) / f"fight_{percentile}.json.br"
        output = map_dataset_output_path(source, source_root=raw_root, output_root=annotated_root)
        assert output == annotated_root / "FRU" / bucket / source.name
        assert output.with_suffix(".pt").parent == output.parent
    assert PERCENTILE_BUCKETS[0] == "90-100"
    assert PERCENTILE_BUCKETS[-1] == "00-10"
    assert len(PERCENTILE_BUCKETS) == 10
    assert not annotated_root.exists()


@pytest.mark.parametrize("bucket", ["0-10", "40-60", "../FRU", "/90-100", ""])
def test_invalid_bucket_cannot_change_output_directory(tmp_path, bucket):
    with pytest.raises(ValueError, match="invalid percentile bucket"):
        percentile_directory(tmp_path / "FRU", bucket)


@pytest.mark.parametrize("source", ["../outside.json.br", "."])
def test_mapping_requires_source_inside_input_root(tmp_path, source):
    with pytest.raises(ValueError):
        map_dataset_output_path(
            tmp_path / "raw" / source,
            source_root=tmp_path / "raw",
            output_root=tmp_path / "annotated",
        )


@pytest.mark.parametrize("relative", ["old.json.br", "FRU/old.json.br", "FRU/00-10/fight.json.br"])
def test_mapping_preserves_existing_layout_and_filename(tmp_path, relative):
    output = map_dataset_output_path(
        tmp_path / "raw" / relative,
        source_root=tmp_path / "raw",
        output_root=tmp_path / "annotated",
    )
    assert output == tmp_path / "annotated" / relative


def test_conversion_groups_by_encounter_and_bucket(tmp_path):
    for relative in ["FRU/00-10/low.json.br", "FRU/90-100/high.json.br", "M5s/old.json.br"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8", newline="\n")
    (tmp_path / "FRU/readme.txt").write_text("", encoding="utf-8")
    discovered = find_dataset_json_files(tmp_path)
    assert len(discovered) == 3
    assert _resolve_input_files([str(tmp_path)]) == sorted(discovered, key=lambda path: str(path).casefold())
    groups = select_training_raw_path_groups(tmp_path)
    assert [(group.directory_name, group.target_count) for group in groups] == [
        ("FRU/90-100", 1), ("FRU/00-10", 1), ("M5s", 1),
    ]
    assert {path for group in groups for path in group.candidates} == set(discovered)


def test_dataset_discovery_ignores_plain_json(tmp_path):
    (tmp_path / "old.json").write_text("{}", encoding="utf-8")
    assert find_dataset_json_files(tmp_path) == []


@pytest.mark.parametrize("stage", ["raw", "annotated"])
def test_training_selection_excludes_val_from_stage_root_but_accepts_explicit_val_root(tmp_path, stage):
    training = tmp_path / stage / "FRU/90-100/train.json.br"
    validation = tmp_path / stage / "VAL/FRU/val.json.br"
    for path in (training, validation):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8", newline="\n")

    stage_groups = select_training_raw_path_groups(tmp_path / stage)
    assert [path for group in stage_groups for path in group.primary_paths] == [training]
    val_groups = select_training_raw_path_groups(tmp_path / stage / "VAL")
    assert [path for group in val_groups for path in group.primary_paths] == [validation]


def test_max_files_balances_buckets_and_keeps_fallback_within_bucket(tmp_path):
    for bucket, count in (("90-100", 4), ("00-10", 4), ("40-50", 1)):
        directory = tmp_path / "FRU" / bucket
        directory.mkdir(parents=True)
        for index in range(count):
            (directory / f"fight-{index}.json.br").write_text("{}", encoding="utf-8")

    groups = select_training_raw_path_groups(tmp_path, max_files=5)
    assert [(group.directory_name, group.target_count) for group in groups] == [
        ("FRU/90-100", 2), ("FRU/40-50", 1), ("FRU/00-10", 2),
    ]
    assert all({path.parent.name for path in group.candidates} == {group.directory_name.split("/")[-1]} for group in groups)


@pytest.mark.parametrize("input_root", ["annotated", "annotated/FRU"])
def test_encounter_root_and_stage_root_balance_the_same_buckets(tmp_path, input_root):
    stage_root = tmp_path / "annotated"
    for bucket, count in (("90-100", 100), ("00-10", 10)):
        directory = stage_root / "FRU" / bucket
        directory.mkdir(parents=True)
        for index in range(count):
            (directory / f"fight-{index:03d}.json.br").write_text("{}", encoding="utf-8")

    groups = select_training_raw_path_groups(tmp_path / input_root, max_files=10)
    assert [(group.directory_name, group.target_count) for group in groups] == [
        ("FRU/90-100", 5), ("FRU/00-10", 5),
    ]
    assert all(
        {path.parent.name for path in group.candidates} == {group.directory_name.split("/")[-1]}
        for group in groups
    )


def test_bucket_root_preserves_encounter_and_bucket(tmp_path):
    bucket_root = tmp_path / "annotated" / "FRU" / "90-100"
    bucket_root.mkdir(parents=True)
    for index in range(4):
        (bucket_root / f"fight-{index}.json.br").write_text("{}", encoding="utf-8")

    groups = select_training_raw_path_groups(bucket_root, max_files=2)
    assert [(group.directory_name, group.target_count) for group in groups] == [
        ("FRU/90-100", 2),
    ]


def test_fru_small_quotas_span_buckets_and_larger_quotas_stay_balanced(tmp_path):
    for bucket in PERCENTILE_BUCKETS:
        directory = tmp_path / "FRU" / bucket
        directory.mkdir(parents=True)
        for index in range(20):
            (directory / f"fight-{index:02d}.json.br").write_text("{}", encoding="utf-8")

    groups = select_training_raw_path_groups(tmp_path, max_files=100)
    assert len(groups) == 10
    assert [group.directory_name for group in groups] == [f"FRU/{bucket}" for bucket in PERCENTILE_BUCKETS]
    assert [group.target_count for group in groups] == [10] * 10
    assert all(len(group.candidates) == 20 for group in groups)

    assert [group.target_count for group in select_training_raw_path_groups(tmp_path, max_files=1)] == [
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    ]
    assert [group.target_count for group in select_training_raw_path_groups(tmp_path, max_files=2)] == [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ]
    assert [group.target_count for group in select_training_raw_path_groups(tmp_path, max_files=5)] == [
        1, 0, 1, 0, 1, 0, 0, 1, 0, 1,
    ]
    assert [group.target_count for group in select_training_raw_path_groups(tmp_path, max_files=128)] == [
        13, 13, 13, 13, 13, 13, 13, 13, 12, 12,
    ]


def test_missing_dataset_directory_remains_empty(tmp_path):
    assert find_dataset_json_files(tmp_path / "missing") == []
    assert select_training_raw_path_groups(tmp_path / "missing") == ()


@pytest.mark.parametrize("stage", ["raw", "annotated", ".cache"])
def test_stage_root_is_inferred_without_each_script_deciding_layout(tmp_path, stage):
    source = tmp_path / stage / "FRU/00-10/fight.json.br"
    output = map_dataset_output_path(source, output_root=tmp_path / ".cache")
    assert output == tmp_path / ".cache/FRU/00-10/fight.json.br"


def test_standalone_input_remains_flat_and_custom_source_root_is_supported(tmp_path):
    source = tmp_path / "custom/FRU/00-10/fight.json.br"
    assert map_dataset_output_path(source, output_root=tmp_path / ".cache") == tmp_path / ".cache/fight.json.br"
    assert map_dataset_output_path(
        source, source_root=tmp_path / "custom", output_root=tmp_path / ".cache",
    ) == tmp_path / ".cache/FRU/00-10/fight.json.br"


def test_compiled_cache_uses_shared_layout_and_keeps_source_identity(tmp_path):
    paths = [
        tmp_path / "raw/FRU/00-10/fight.json.br",
        tmp_path / "raw/FRU/90-100/fight.json.br",
        tmp_path / "annotated/FRU/00-10/fight.json.br",
    ]
    outputs = [cache_path_for_source(tmp_path / ".cache", source) for source in paths]
    assert outputs[0].parent == outputs[2].parent == tmp_path / ".cache/FRU/00-10"
    assert outputs[1].parent == tmp_path / ".cache/FRU/90-100"
    assert len({output.name for output in outputs}) == 3
    assert all(output.name.endswith(".compiled.pt") for output in outputs)

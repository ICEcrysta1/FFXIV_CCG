"""数据阶段目录映射与下载、转换共用布局的回归测试。"""

import pytest

from scripts.common.dataset_layout import (
    PERCENTILE_BUCKETS,
    find_dataset_json_files,
    map_dataset_output_path,
    percentile_bucket,
    percentile_directory,
)
from scripts.convert_fflogs.cache.cache_paths import select_training_raw_path_groups
from scripts.convert_fflogs.cli import _resolve_input_files


def test_download_and_annotation_preserve_every_bucket(tmp_path):
    raw_root = tmp_path / "raw"
    annotated_root = tmp_path / "annotated"
    for percentile in range(0, 101, 10):
        bucket = percentile_bucket(percentile)
        source = percentile_directory(raw_root / "FRU", bucket) / f"fight_{percentile}.json"
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


@pytest.mark.parametrize("source", ["../outside.json", "."])
def test_mapping_requires_source_inside_input_root(tmp_path, source):
    with pytest.raises(ValueError):
        map_dataset_output_path(
            tmp_path / "raw" / source,
            source_root=tmp_path / "raw",
            output_root=tmp_path / "annotated",
        )


@pytest.mark.parametrize("relative", ["old.json", "FRU/old.json", "FRU/00-10/fight.json"])
def test_mapping_preserves_existing_layout_and_filename(tmp_path, relative):
    output = map_dataset_output_path(
        tmp_path / "raw" / relative,
        source_root=tmp_path / "raw",
        output_root=tmp_path / "annotated",
    )
    assert output == tmp_path / "annotated" / relative


def test_conversion_discovers_bucket_files_but_still_groups_by_encounter(tmp_path):
    for relative in ["FRU/00-10/low.json", "FRU/90-100/high.json", "M5s/old.json"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8", newline="\n")
    (tmp_path / "FRU/readme.txt").write_text("", encoding="utf-8")
    discovered = find_dataset_json_files(tmp_path)
    assert len(discovered) == 3
    assert _resolve_input_files([str(tmp_path)]) == sorted(discovered, key=lambda path: str(path).casefold())
    groups = select_training_raw_path_groups(tmp_path)
    assert [(group.directory_name, group.target_count) for group in groups] == [("FRU", 2), ("M5s", 1)]
    assert {path for group in groups for path in group.candidates} == set(discovered)


def test_missing_dataset_directory_remains_empty(tmp_path):
    assert find_dataset_json_files(tmp_path / "missing") == []
    assert select_training_raw_path_groups(tmp_path / "missing") == ()

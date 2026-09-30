"""旧数据一次性压缩迁移，不向正式读取链路引入普通 JSON 兼容。"""

import json

import pytest

from scripts.common.json_io import read_json
from scripts.dataset_compression import main, migrate_file


def _write_plain(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")


def test_migrate_raw_and_matching_annotation_without_hash(tmp_path):
    raw = tmp_path / "raw/FRU/00-10/fight.json"
    annotated = tmp_path / "annotated/FRU/00-10/fight.json"
    fight = {"report_code": "REPORT", "fight_id": 5, "source_id": 2, "events": [{"type": "cast"}]}
    analysis = {"schema_version": 2, "bridge_version": 3,
                "source": {"report_code": "REPORT", "fight_id": 5, "sha256": "old"}}
    _write_plain(raw, fight)
    _write_plain(annotated, {**fight, "analysis": analysis})

    assert main([str(tmp_path)]) == 0
    compressed_raw = raw.with_name(raw.name + ".br")
    compressed_annotation = annotated.with_name(annotated.name + ".br")
    assert read_json(compressed_raw) == fight
    migrated = read_json(compressed_annotation)
    assert migrated["analysis"]["bridge_version"] == 4
    assert "sha256" not in migrated["analysis"]["source"]
    assert raw.is_file() and annotated.is_file()
    assert migrate_file(raw) == (compressed_raw, False)
    assert main([str(tmp_path), "--remove-original"]) == 0
    assert not raw.exists() and not annotated.exists()
    assert read_json(compressed_annotation) == migrated


def test_migration_refuses_annotation_with_mismatched_raw(tmp_path):
    raw = tmp_path / "raw/FRU/00-10/fight.json"
    annotated = tmp_path / "annotated/FRU/00-10/fight.json"
    _write_plain(raw, {"report_code": "A", "fight_id": 1})
    _write_plain(annotated, {
        "report_code": "A", "fight_id": 2,
        "analysis": {"schema_version": 2, "bridge_version": 3,
                     "source": {"report_code": "A", "fight_id": 2}},
    })
    with pytest.raises(ValueError, match="不一致"):
        migrate_file(annotated)
    assert not annotated.with_name(annotated.name + ".br").exists()


def test_remove_original_only_after_verified_compressed_copy(tmp_path):
    raw = tmp_path / "raw/FRU/00-10/fight.json"
    _write_plain(raw, {"fight_id": 1})
    target, created = migrate_file(raw, remove_original=True)
    assert created and read_json(target) == {"fight_id": 1}
    assert not raw.exists()

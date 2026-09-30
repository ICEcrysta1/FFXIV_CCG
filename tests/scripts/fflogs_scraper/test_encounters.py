"""下载目标代号与 FFLogs 两级 ID 映射测试。"""

from pathlib import Path

import pytest

from scripts.fflogs_scraper.config.encounters import (
    load_download_encounters,
    resolve_download_encounter,
)


def test_all_configured_encounters_have_unique_ids_and_expected_zones():
    entries = load_download_encounters()
    assert len(entries) == 15
    assert resolve_download_encounter("fru").encounter_id == 1079
    assert resolve_download_encounter("65").alias == "FRU"
    assert [resolve_download_encounter(f"M{index}s").zone_id for index in range(1, 12)] == (
        [62] * 4 + [68] * 4 + [73] * 3
    )
    assert resolve_download_encounter("M12sI").encounter_id == 104
    assert resolve_download_encounter("M12sII").encounter_id == 105
    assert resolve_download_encounter("DancingMad").encounter_id == 1085
    assert resolve_download_encounter("DancingMad").zone_id == 76
    assert resolve_download_encounter("76").alias == "DancingMad"


def test_multi_encounter_zone_requires_alias():
    with pytest.raises(ValueError, match="M1s.*M4s"):
        resolve_download_encounter("62")


def test_invalid_or_duplicate_config_is_rejected(tmp_path: Path):
    path = tmp_path / "encounters.yaml"
    path.write_text(
        "encounters:\n  FRU:\n    zone_id: 65\n    encounter_id: 1079\n"
        "    group: ultimate\n  Other:\n    zone_id: 68\n"
        "    encounter_id: 1079\n    group: raid\n",
        encoding="utf-8", newline="\n",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_download_encounters(path)

"""FFLogs 下载器 io 职责回归测试。"""

import json

import pytest

from scripts.common.json_io import atomic_write_json
from scripts.fflogs_scraper import parse_fflogs_url
from scripts.fflogs_scraper.io.filenames import (
    _build_batch_output_filename,
    _build_output_filename,
)


@pytest.mark.parametrize("report_code", ["../secret", r"..\secret", "report/name", ""])
def test_output_filename_rejects_path_like_report_codes(report_code: str):
    with pytest.raises(ValueError, match="invalid report code"):
        _build_output_filename(report_code, 1, None)


def test_parse_fflogs_url_keeps_alphanumeric_report_code():
    parsed = parse_fflogs_url(
        "https://www.fflogs.com/reports/ABC123?fight=4&source=9"
    )

    assert parsed["report_code"] == "ABC123"
    assert parsed["fight_id"] == 4
    assert parsed["source_id"] == 9


@pytest.mark.parametrize("report_code", ["../secret", r"..\secret", "report/name"])
def test_batch_output_filename_rejects_path_like_report_codes(report_code: str):
    with pytest.raises(ValueError, match="invalid report code"):
        _build_batch_output_filename(report_code, 1, "Player")


def test_batch_output_filename_sanitizes_windows_path_separator():
    filename = _build_batch_output_filename("ABC123", 1, r"Player\Name / Test")

    assert filename == "fflogs_ABC123_f1_Player-Name_-_Test.json"


def test_parse_fflogs_url_ignores_non_numeric_source_id():
    parsed = parse_fflogs_url(
        "https://www.fflogs.com/reports/ABC123?fight=4&source=bad"
    )

    assert parsed == {"report_code": "ABC123", "fight_id": 4}


def test_write_json_preserves_existing_file_on_failure_and_uses_utf8_lf(tmp_path):
    path = tmp_path / "report.json"
    atomic_write_json(path, {"title": "中文报告"})
    data = path.read_bytes()
    assert b"\r\n" not in data
    assert json.loads(data)["title"] == "中文报告"
    with pytest.raises(TypeError):
        atomic_write_json(path, {"invalid": object()})
    assert path.read_bytes() == data
    assert list(tmp_path.glob("*.tmp")) == []

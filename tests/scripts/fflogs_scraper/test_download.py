"""FFLogs 下载器 download 职责回归测试。"""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts.fflogs_scraper import FFLogsV2Client
from scripts.fflogs_scraper.contracts.rankings import HistoricalReport
from scripts.fflogs_scraper.download import batch, single
from scripts.fflogs_scraper.download.sampling import _allocate_percentile_quotas


def _stub_report_download(monkeypatch, client, meta):
    monkeypatch.setattr(client, "resolve_source_id", lambda *args: 1)
    monkeypatch.setattr(client, "get_report_fights", lambda code: replace(meta, code=code))
    monkeypatch.setattr(client, "get_damage_table", lambda *args: {"entries": []})
    monkeypatch.setattr(client, "get_fight_events", lambda *args, **kwargs: [])


def _record(code, percentile, name="Player"):
    return HistoricalReport(code, 33, name, 123, percentile, 1, 7.5, True)


@pytest.mark.parametrize("mode", ["events-only", "default", "damage-only"])
def test_batch_download_saves_full_analysis_context(monkeypatch, tmp_path, analysis_meta, mode):
    client = FFLogsV2Client("id", "secret")
    monkeypatch.setattr(client, "resolve_source_id", lambda *args: 1)
    monkeypatch.setattr(client, "get_report_fights", lambda code: analysis_meta)
    monkeypatch.setattr(client, "get_damage_table", lambda *args: {"entries": []})
    requests = []
    def get_events(code, fight, source_id=None, *, require_complete=False):
        requests.append((source_id, require_complete))
        return []
    monkeypatch.setattr(client, "get_fight_events", get_events)
    batch._batch_download(client, [("ABC123", 33, "Player", 123)], str(tmp_path), mode)
    payload = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert payload["source_id"] == 1
    assert payload["lang"] == "cn"
    assert payload["events_complete"] is (mode != "damage-only")
    assert requests == ([] if mode == "damage-only" else [(None, True)])


@pytest.mark.parametrize("events_only", [True, False])
def test_single_download_resolves_last_fight_and_fetches_full_events(monkeypatch, tmp_path, analysis_meta, events_only):
    client = FFLogsV2Client("id", "secret")
    monkeypatch.setattr(client, "get_report_fights", lambda code: analysis_meta)
    requests = []
    def get_events(code, fight, source_id=None, *, require_complete=False):
        requests.append((fight.id, source_id, require_complete))
        return []
    monkeypatch.setattr(client, "get_fight_events", get_events)
    monkeypatch.setattr(client, "get_damage_table", lambda *args: {"entries": []})
    output = tmp_path / "single.json"
    args = SimpleNamespace(url="https://www.fflogs.com/reports/ABC123?fight=last&source=1",
                           report=None, fight=None, source=None, events_only=events_only,
                           damage_only=False, output=str(output), output_dir=None)
    single._cmd_single(client, args)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["fight_id"] == 33
    assert payload["source_id"] == 1
    assert payload["events_complete"] is True
    assert requests == [(33, None, True)]
    assert ("damage_table" in payload) is not events_only


def test_incomplete_download_does_not_create_output(monkeypatch, tmp_path, analysis_meta):
    client = FFLogsV2Client("id", "secret")
    monkeypatch.setattr(client, "get_report_fights", lambda code: analysis_meta)
    def fail(*args, **kwargs):
        raise RuntimeError("不完整事件")
    monkeypatch.setattr(client, "get_fight_events", fail)
    output = tmp_path / "report.json"
    args = SimpleNamespace(url=None, report="ABC123", fight=33, source=1,
                           events_only=True, damage_only=False, output=str(output), output_dir=None)
    with pytest.raises(RuntimeError, match="不完整事件"):
        single._cmd_single(client, args)
    assert not output.exists()


def test_stratified_download_saves_bucket_metadata_and_replaces_failed_candidate(monkeypatch, tmp_path, analysis_meta):
    client = FFLogsV2Client("id", "secret")
    _stub_report_download(monkeypatch, client, analysis_meta)
    monkeypatch.setattr(client, "resolve_source_id", lambda code, *args: None if code == "FAIL" else 1)
    records = [_record("FAIL", 95), _record("TOP", 100), _record("TOP", 100)]
    records += [_record(f"R{lower}", lower) for lower in range(80, -1, -10)]
    result = batch._stratified_batch_download(client, records, _allocate_percentile_quotas(10), str(tmp_path), "events-only")
    assert result["success"] == 10
    assert result["failed"] == 1
    assert set(result["counts"].values()) == {1}
    assert len(list(tmp_path.rglob("*.json"))) == 10
    for path in tmp_path.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["ranking"]["percentile_bucket"] == path.parent.name
        assert payload["ranking"]["metric"] == "rdps"
        assert payload["ranking"]["timeframe"] == "historical"
        assert payload["events_complete"] is True
    assert not list(tmp_path.rglob("*FAIL*"))


def test_complete_quotas_stop_before_requesting_another_candidate(monkeypatch, tmp_path):
    client = FFLogsV2Client("id", "secret")
    requested = []
    def candidates():
        yield _record("FIRST", 95)
        pytest.fail("配额完成后不应继续请求历史候选")
    def download(*args, **kwargs):
        requested.append(args[1])
        return "success"
    monkeypatch.setattr(batch, "_download_report", download)
    result = batch._stratified_batch_download(client, candidates(), {"90-100": 1}, str(tmp_path))
    assert result["counts"] == {"90-100": 1}
    assert requested == ["FIRST"]


def test_anonymous_and_duplicate_records_do_not_fill_quota(monkeypatch, tmp_path, analysis_meta):
    client = FFLogsV2Client("id", "secret")
    _stub_report_download(monkeypatch, client, analysis_meta)
    records = [_record("PRIVATE", 95, "Anonymous"), _record("a:DfrP27RKwgqBkQGA", 95, "Masked"),
               _record("PUBLIC", 95), _record("PUBLIC", 95)]
    result = batch._stratified_batch_download(client, records, {"90-100": 2}, str(tmp_path))
    assert result["counts"] == {"90-100": 1}
    assert result["anonymous"] == 2
    assert len(list(tmp_path.rglob("*.json"))) == 1


@pytest.mark.parametrize("mode", ["events-only", "default", "damage-only"])
def test_valid_existing_files_count_towards_target_without_redownloading(monkeypatch, tmp_path, analysis_meta, mode):
    client = FFLogsV2Client("id", "secret")
    _stub_report_download(monkeypatch, client, analysis_meta)
    record = _record("ABC123", 95)
    first = batch._stratified_batch_download(client, [record], {"90-100": 1}, str(tmp_path), mode)
    assert first["success"] == 1
    def unexpected(*args):
        pytest.fail("已有有效文件不应再查询报告")
    monkeypatch.setattr(client, "resolve_source_id", unexpected)
    second = batch._stratified_batch_download(client, [record], {"90-100": 1}, str(tmp_path), mode)
    assert second["existing"] == 1
    assert second["success"] == 0
    assert second["counts"] == {"90-100": 1}


def test_incomplete_existing_file_does_not_count_and_survives_failed_replacement(monkeypatch, tmp_path, analysis_meta):
    client = FFLogsV2Client("id", "secret")
    _stub_report_download(monkeypatch, client, analysis_meta)
    directory = tmp_path / "90-100"
    directory.mkdir()
    output = directory / "fflogs_ABC123_f33_Player.json"
    original = '{"events_complete": false}\n'
    output.write_text(original, encoding="utf-8")
    def fail(*args, **kwargs):
        raise RuntimeError("不完整事件")
    monkeypatch.setattr(client, "get_fight_events", fail)
    result = batch._stratified_batch_download(client, [_record("ABC123", 95)], {"90-100": 1}, str(tmp_path))
    assert result["counts"] == {"90-100": 0}
    assert result["failed"] == 1
    assert output.read_text(encoding="utf-8") == original


def test_missing_bucket_does_not_borrow_surplus_from_another(monkeypatch, tmp_path, capsys):
    client = FFLogsV2Client("id", "secret")
    calls = []
    def download(*args, **kwargs):
        calls.append(args[1])
        return "success"
    monkeypatch.setattr(batch, "_download_report", download)
    result = batch._stratified_batch_download(
        client, [_record("TOP1", 95), _record("TOP2", 95)],
        {"90-100": 1, "00-10": 1}, str(tmp_path),
    )
    assert result["counts"] == {"90-100": 1, "00-10": 0}
    assert calls == ["TOP1"]
    assert "00-10: 0/1" in capsys.readouterr().out


@pytest.mark.parametrize("partition", [None, 25])
def test_cmd_batch_uses_api_default_or_explicit_partition(monkeypatch, tmp_path, partition):
    client = FFLogsV2Client("id", "secret")
    monkeypatch.setattr(client, "get_encounter_details", lambda encounter: {
        "name": "Futures Rewritten", "zone": {"partitions": [{"id": 1}, {"id": 25}]},
    })
    calls = []
    def discover(*args, **kwargs):
        calls.append(kwargs)
        return []
    monkeypatch.setattr(batch, "_iter_historical_reports", discover)
    args = SimpleNamespace(zone=None, encounter=1079, count=200, partition=partition,
                           spec_name="BlackMage", metric="rdps", bracket=0, max_pages=10,
                           output=str(tmp_path / "FRU"), mode="events-only")
    batch._cmd_batch(client, args)
    assert calls[0]["partition"] == partition
    assert "partitions" not in calls[0]
    assert "history_partition" not in calls[0]
    assert (tmp_path / "FRU/00-10").is_dir()
    assert (tmp_path / "FRU/90-100").is_dir()

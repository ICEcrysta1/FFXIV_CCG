"""FFLogs 下载器 download 职责回归测试。"""

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.common.json_io import read_json
from scripts.fflogs_scraper import FFLogsV2Client
from scripts.fflogs_scraper.contracts.rankings import HistoricalReport
from scripts.fflogs_scraper.download import batch, single
from scripts.fflogs_scraper.download.sampling import (
    _allocate_percentile_quotas,
    _iter_historical_reports,
)


def _stub_report_download(monkeypatch, client, meta):
    monkeypatch.setattr(
        client, "get_report_fights",
        lambda code, *, player_fight_id=None: replace(
            meta, code=code, players=[{"id": 1, "name": "Player"}],
        ),
    )
    monkeypatch.setattr(client, "get_damage_table", lambda *args: {"entries": []})
    monkeypatch.setattr(client, "get_fight_events", lambda *args, **kwargs: [])


def _record(code, percentile, name="Player"):
    return HistoricalReport(code, 33, name, 123, percentile, 1, 7.5, True)


@pytest.mark.parametrize("mode", ["events-only", "default", "damage-only"])
def test_stratified_download_saves_full_analysis_context(monkeypatch, tmp_path, analysis_meta, mode):
    client = FFLogsV2Client("id", "secret")
    monkeypatch.setattr(
        client, "get_report_fights",
        lambda code, *, player_fight_id=None: replace(
            analysis_meta, players=[{"id": 1, "name": "Player"}],
        ),
    )
    monkeypatch.setattr(client, "get_damage_table", lambda *args: {"entries": []})
    requests = []
    def get_events(code, fight, source_id=None, *, require_complete=False):
        requests.append((source_id, require_complete))
        return []
    monkeypatch.setattr(client, "get_fight_events", get_events)
    result = batch._stratified_batch_download(
        client, [_record("ABC123", 95)], {"90-100": 1}, str(tmp_path), mode,
    )
    assert result["success"] == 1
    assert result["counts"] == {"90-100": 1}
    payload = read_json(next(tmp_path.rglob("*.json.br")))
    assert payload["source_id"] == 1
    assert payload["lang"] == "cn"
    assert payload["events_complete"] is (mode != "damage-only")
    assert requests == ([] if mode == "damage-only" else [(None, True)])


def test_batch_download_uses_combined_metadata_and_player_query(monkeypatch, tmp_path, analysis_meta):
    client = FFLogsV2Client("id", "secret")
    calls = []

    def get_metadata(code, *, player_fight_id=None):
        calls.append((code, player_fight_id))
        return replace(analysis_meta, players=[{"id": 1, "name": "Player"}])

    monkeypatch.setattr(client, "get_report_fights", get_metadata)
    monkeypatch.setattr(
        client, "query",
        lambda *args: pytest.fail("不应再次单独查询玩家"),
    )
    monkeypatch.setattr(client, "get_fight_events", lambda *args, **kwargs: [])
    batch._download_report(client, "ABC123", 33, "Player", 123, str(tmp_path), "events-only")

    assert calls == [("ABC123", 33)]
    payload = read_json(next(tmp_path.glob("*.json.br")))
    assert payload["source_id"] == 1
    assert "playerDetails" not in payload


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
    output = tmp_path / "single.json.br"
    args = SimpleNamespace(url="https://www.fflogs.com/reports/ABC123?fight=last&source=1",
                           report=None, fight=None, source=None, events_only=events_only,
                           damage_only=False, output=str(output), output_dir=None)
    single._cmd_single(client, args)
    payload = read_json(output)
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
    output = tmp_path / "report.json.br"
    args = SimpleNamespace(url=None, report="ABC123", fight=33, source=1,
                           events_only=True, damage_only=False, output=str(output), output_dir=None)
    with pytest.raises(RuntimeError, match="不完整事件"):
        single._cmd_single(client, args)
    assert not output.exists()


def test_single_download_rejects_plain_json_before_api_request(monkeypatch, tmp_path):
    client = FFLogsV2Client("id", "secret")
    monkeypatch.setattr(client, "get_report_fights", lambda *_args: pytest.fail("must reject before API request"))
    args = SimpleNamespace(
        url=None, report="ABC123", fight=33, source=1,
        output=str(tmp_path / "report.json"),
    )
    with pytest.raises(ValueError, match=".json.br"):
        single._cmd_single(client, args)


def test_stratified_download_saves_bucket_metadata_and_replaces_failed_candidate(monkeypatch, tmp_path, analysis_meta):
    client = FFLogsV2Client("id", "secret")
    _stub_report_download(monkeypatch, client, analysis_meta)
    monkeypatch.setattr(
        client, "get_report_fights",
        lambda code, *, player_fight_id=None: replace(
            analysis_meta, code=code,
            players=[] if code == "FAIL" else [{"id": 1, "name": "Player"}],
        ),
    )
    records = [_record("FAIL", 95), _record("TOP", 100), _record("TOP", 100)]
    records += [_record(f"R{lower}", lower) for lower in range(80, -1, -10)]
    result = batch._stratified_batch_download(client, records, _allocate_percentile_quotas(10), str(tmp_path), "events-only")
    assert result["success"] == 10
    assert result["failed"] == 1
    assert set(result["counts"].values()) == {1}
    assert len(list(tmp_path.rglob("*.json.br"))) == 10
    for path in tmp_path.rglob("*.json.br"):
        payload = read_json(path)
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


def test_batch_uses_distinct_characters_before_repeating_one_bucket(monkeypatch, tmp_path):
    def rank(code):
        return {"report": {"code": code, "fightID": 1}, "spec": "BlackMage",
                "historicalPercent": 95, "amount": 100}

    client = SimpleNamespace(
        _cancelled=False,
        server_region=None,
        get_encounter_rankings=lambda *args, **kwargs: {
            "rankings": [
                {"name": "First", "lodestoneID": 1},
                {"name": "Second", "lodestoneID": 2},
            ],
            "hasMorePages": False,
        },
        get_character_history=lambda lodestone_id, *args, **kwargs: {
            "id": lodestone_id,
            "name": "First" if lodestone_id == 1 else "Second",
            "encounterRankings": {"ranks": (
                [rank("FIRST1"), rank("FIRST2"), rank("FIRST3")]
                if lodestone_id == 1 else [rank("SECOND1")]
            )},
        },
    )
    selected = []

    def download(_client, code, _fight_id, name, *args, **kwargs):
        selected.append((code, name))
        return "success"

    monkeypatch.setattr(batch, "_download_report", download)
    reports = _iter_historical_reports(
        client, 1079, spec_name="BlackMage", metric="rdps",
    )
    result = batch._stratified_batch_download(
        client, reports, {"90-100": 2}, tmp_path,
    )
    assert result["counts"] == {"90-100": 2}
    assert selected == [("FIRST1", "First"), ("SECOND1", "Second")]


def test_anonymous_and_duplicate_records_do_not_fill_quota(monkeypatch, tmp_path, analysis_meta):
    client = FFLogsV2Client("id", "secret")
    _stub_report_download(monkeypatch, client, analysis_meta)
    records = [_record("PRIVATE", 95, "Anonymous"), _record("a:DfrP27RKwgqBkQGA", 95, "Masked"),
               _record("PUBLIC", 95), _record("PUBLIC", 95)]
    result = batch._stratified_batch_download(client, records, {"90-100": 2}, str(tmp_path))
    assert result["counts"] == {"90-100": 1}
    assert result["anonymous"] == 2
    assert len(list(tmp_path.rglob("*.json.br"))) == 1


@pytest.mark.parametrize("mode", ["events-only", "default", "damage-only"])
def test_valid_existing_files_count_towards_target_without_redownloading(monkeypatch, tmp_path, analysis_meta, mode):
    client = FFLogsV2Client("id", "secret")
    _stub_report_download(monkeypatch, client, analysis_meta)
    record = _record("ABC123", 95)
    first = batch._stratified_batch_download(client, [record], {"90-100": 1}, str(tmp_path), mode)
    assert first["success"] == 1
    def unexpected(*args):
        pytest.fail("已有有效文件不应再查询报告")
    monkeypatch.setattr(client, "get_report_fights", unexpected)
    second = batch._stratified_batch_download(client, [record], {"90-100": 1}, str(tmp_path), mode)
    assert second["existing"] == 1
    assert second["success"] == 0
    assert second["counts"] == {"90-100": 1}


def test_incomplete_existing_file_does_not_count_and_survives_failed_replacement(monkeypatch, tmp_path, analysis_meta):
    client = FFLogsV2Client("id", "secret")
    _stub_report_download(monkeypatch, client, analysis_meta)
    directory = tmp_path / "90-100"
    directory.mkdir()
    output = directory / "fflogs_ABC123_f33_Player.json.br"
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
    client = FFLogsV2Client("id", "secret", server_region="CN")
    validation_client = FFLogsV2Client("id", "secret", server_region="NA")
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
                            output=str(tmp_path / "raw/FRU"), mode="events-only")
    batch._cmd_batch(client, validation_client, args)
    assert calls[0]["partition"] == partition
    assert all("server_region" not in call for call in calls)
    assert "partitions" not in calls[0]
    assert "history_partition" not in calls[0]
    assert (tmp_path / "raw/FRU/00-10").is_dir()
    assert (tmp_path / "raw/FRU/90-100").is_dir()
    assert (tmp_path / "raw/VAL/FRU").is_dir()
    assert not (tmp_path / "raw/VAL/FRU/90-100").exists()


def test_cmd_batch_rejects_output_outside_raw_before_api_call(monkeypatch, tmp_path):
    client = FFLogsV2Client("id", "secret", server_region="CN")
    monkeypatch.setattr(client, "get_encounter_details", lambda *args: pytest.fail("无效路径不应请求 API"))
    with pytest.raises(ValueError, match="raw/<副本>"):
        batch._cmd_batch(client, FFLogsV2Client("id", "secret", server_region="NA"), SimpleNamespace(
            zone=None, encounter=1079, count=200, output=str(tmp_path / "FRU"),
        ))


def test_cmd_batch_rejects_mismatched_data_sources():
    with pytest.raises(ValueError, match="CN 训练与 NA 验证"):
        batch._cmd_batch(
            FFLogsV2Client("id", "secret", server_region="NA"),
            FFLogsV2Client("id", "secret", server_region="CN"),
            SimpleNamespace(),
        )


def test_cmd_batch_uses_configured_validation_ratio(monkeypatch, tmp_path):
    client = FFLogsV2Client("id", "secret", server_region="CN")
    validation_client = FFLogsV2Client("id", "secret", server_region="NA")
    monkeypatch.setattr(client, "get_encounter_details", lambda encounter: {"name": "FRU"})
    calls = []
    monkeypatch.setattr(batch, "_iter_historical_reports", lambda *args, **kwargs: [])

    def download(_client, _reports, quotas, output, _mode, _metric, **kwargs):
        calls.append((_client.server_region, quotas, Path(output), kwargs))

    monkeypatch.setattr(batch, "_stratified_batch_download", download)
    for ratio, count, expected in (("0.10", 200, 20), ("0.10", 203, 21),
                                   ("0.07", 100, 7), ("0.20", 203, 41)):
        monkeypatch.setattr(batch, "load_validation_ratio", lambda ratio=ratio: Decimal(ratio))
        calls.clear()
        batch._cmd_batch(client, validation_client, SimpleNamespace(
            zone=None, encounter=1079, count=count, partition=None,
            spec_name="BlackMage", metric="rdps", bracket=0, max_pages=10,
            output=str(tmp_path / "raw/FRU"), mode="events-only",
        ))
        assert calls[0][0] == "CN"
        assert sum(calls[0][1].values()) == count
        assert calls[0][2] == tmp_path / "raw/FRU"
        assert calls[1] == (
            "NA", {"90-100": expected}, tmp_path / "raw/VAL/FRU",
            {"bucket_directories": False},
        )


@pytest.mark.parametrize("alias,encounter_id", [
    ("FRU", 1079), ("65", 1079), ("M1s", 93), ("M12sI", 104), ("M12sII", 105),
])
def test_named_batch_resolves_model_job_and_output_directory(monkeypatch, tmp_path, alias, encounter_id):
    client = FFLogsV2Client("id", "secret", server_region="CN")
    validation_client = FFLogsV2Client("id", "secret", server_region="NA")
    monkeypatch.setattr(batch, "resolve_policy_model_config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setattr(batch, "resolve_policy_model_job_tag", lambda _path: "black_mage")
    monkeypatch.setattr(batch, "load_policy_config", lambda _path: {
        "raw_data_dir": str(tmp_path / "annotated"),
    })
    monkeypatch.setattr(client, "get_encounter_details", lambda _id: {"name": alias})
    monkeypatch.setattr(batch, "_iter_historical_reports", lambda *args, **kwargs: [])
    calls = []
    monkeypatch.setattr(batch, "_stratified_batch_download", lambda *args, **kwargs: calls.append(args))
    args = SimpleNamespace(
        target=alias, zone=None, encounter=None, output=None, spec_name=None,
        count=200, partition=None, metric="rdps", bracket=0, max_pages=10, mode="default",
    )
    batch._cmd_batch(client, validation_client, args)
    assert args.encounter == encounter_id
    assert args.spec_name == "BlackMage"
    assert Path(args.output) == tmp_path / "raw" / ("FRU" if alias == "65" else alias)
    assert Path(calls[0][3]) == Path(args.output)
    assert Path(calls[1][3]) == Path(args.output).parent / "VAL" / Path(args.output).name


def test_existing_valid_inventory_fills_quota_without_querying_reports(monkeypatch, tmp_path, analysis_meta):
    client = FFLogsV2Client("id", "secret", server_region="CN")
    _stub_report_download(monkeypatch, client, analysis_meta)
    record = _record("EXISTING", 95)
    first = batch._stratified_batch_download(
        client, [record], {"90-100": 1}, tmp_path, "events-only",
    )
    assert first["success"] == 1

    class UnexpectedReports:
        def __iter__(self):
            pytest.fail("本地配额已满时不应请求历史候选")

    monkeypatch.setattr(client, "get_report_fights", lambda *args, **kwargs: pytest.fail("不应重新下载"))
    second = batch._stratified_batch_download(
        client, UnexpectedReports(), {"90-100": 1}, tmp_path, "events-only",
    )
    assert second["counts"] == {"90-100": 1}
    assert second["existing"] == 1 and second["success"] == 0


def test_percentile_drift_does_not_download_same_report_into_another_bucket(
    monkeypatch, tmp_path, analysis_meta,
):
    client = FFLogsV2Client("id", "secret", server_region="CN")
    _stub_report_download(monkeypatch, client, analysis_meta)
    first = batch._stratified_batch_download(
        client, [_record("SAME", 35)], {"30-40": 1}, tmp_path,
        "events-only",
    )
    assert first["success"] == 1
    second = batch._stratified_batch_download(
        client, [_record("SAME", 45), _record("OTHER", 45)],
        {"30-40": 1, "40-50": 1}, tmp_path,
        "events-only",
    )
    assert second["counts"] == {"30-40": 1, "40-50": 1}
    assert second["existing"] == 1 and second["success"] == 1
    assert [read_json(path)["report_code"] for path in (tmp_path / "40-50").glob("*.json.br")] == ["OTHER"]


def test_validation_download_is_flat_and_rejects_wrong_report_region(monkeypatch, tmp_path, analysis_meta):
    client = FFLogsV2Client("id", "secret", server_region="NA")
    requested = []

    def metadata(code, *, player_fight_id=None):
        requested.append(code)
        return replace(
            analysis_meta, code=code, region="CN" if code == "WRONG" else "NA",
            players=[{"id": 1, "name": "Player"}],
        )

    monkeypatch.setattr(client, "get_report_fights", metadata)
    monkeypatch.setattr(client, "get_fight_events", lambda *args, **kwargs: [])
    output = tmp_path / "raw/VAL/FRU"
    reports = [_record("LOW", 85), _record("WRONG", 95), _record("VALID", 97)]
    result = batch._stratified_batch_download(
        client, reports, {"90-100": 1}, output, "events-only",
        bucket_directories=False,
    )

    assert result["counts"] == {"90-100": 1}
    assert result["success"] == 1 and result["failed"] == 1
    assert requested == ["WRONG", "VALID"]
    files = list(output.glob("*.json.br"))
    assert len(files) == 1
    assert not (output / "90-100").exists()
    payload = read_json(files[0])
    assert payload["ranking"]["server_region"] == "NA"
    assert payload["ranking"]["percentile_bucket"] == "90-100"
    assert payload["report_region"] == "NA"

    monkeypatch.setattr(client, "get_report_fights", lambda *args, **kwargs: pytest.fail("已有验证文件应复用"))
    repeated = batch._stratified_batch_download(
        client, [_record("VALID", 97)], {"90-100": 1}, output, "events-only",
        bucket_directories=False,
    )
    assert repeated["existing"] == 1 and repeated["success"] == 0

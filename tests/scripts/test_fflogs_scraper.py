"""FFLogs 下载脚本的输入边界测试。"""

from __future__ import annotations

import logging

import pytest

from scripts import fflogs_scraper
from scripts.fflogs_scraper import (
    FFLogsV2Client,
    _build_batch_output_filename,
    _build_output_filename,
    parse_fflogs_url,
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


def test_get_report_events_stops_at_event_limit(monkeypatch, caplog):
    monkeypatch.setattr(fflogs_scraper, "MAX_EVENTS", 3)
    client = FFLogsV2Client("client-id", "client-secret")
    responses = iter([
        {
            "reportData": {
                "report": {
                    "events": {
                        "data": [{"id": 1}, {"id": 2}],
                        "nextPageTimestamp": 2,
                    }
                }
            }
        },
        {
            "reportData": {
                "report": {
                    "events": {
                        "data": [{"id": 3}, {"id": 4}],
                        "nextPageTimestamp": 4,
                    }
                }
            }
        },
    ])
    query_count = 0

    def fake_query(_gql):
        nonlocal query_count
        query_count += 1
        return next(responses)

    client.query = fake_query

    with caplog.at_level(logging.INFO):
        events = client.get_report_events("ABC123", end_time=10)

    assert events == [{"id": 1}, {"id": 2}, {"id": 3}]
    assert query_count == 2
    assert "事件累计达到上限 3，提前停止" in caplog.text
    assert "获取到 1 条事件 (累计 3)" in caplog.text
    assert "获取到 2 条事件 (累计 3)" not in caplog.text


def test_get_report_events_normalizes_fight_ids(monkeypatch):
    client = FFLogsV2Client("client-id", "client-secret")
    queries = []

    def fake_query(gql):
        queries.append(gql)
        return {"reportData": {"report": {"events": {"data": []}}}}

    monkeypatch.setattr(client, "query", fake_query)

    assert client.get_report_events("ABC123", fight_ids=[1, 2], end_time=10) == []
    assert "fightIDs: [1,2]" in queries[0]


@pytest.mark.parametrize("fight_ids", [["1) injected"], [0], [-1], "1,2"])
def test_get_report_events_rejects_unsafe_fight_ids(fight_ids):
    client = FFLogsV2Client("client-id", "client-secret")

    with pytest.raises(ValueError, match="invalid fight_ids"):
        client.get_report_events("ABC123", fight_ids=fight_ids)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spec_name", 'BlackMage"'),
        ("class_name", r"Caster\Injected"),
    ],
)
def test_get_encounter_rankings_rejects_unsafe_filter(field, value):
    client = FFLogsV2Client("client-id", "client-secret")

    with pytest.raises(ValueError, match=rf"invalid {field}"):
        client.get_encounter_rankings(1079, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metric", "dps) injected"),
        ("bracket", "0) injected"),
        ("page", "1) injected"),
        ("encounter_id", "1079) injected"),
    ],
)
def test_get_encounter_rankings_rejects_unsafe_graphql_scalars(field, value):
    client = FFLogsV2Client("client-id", "client-secret")
    kwargs = {"encounter_id": 1079, field: value}

    with pytest.raises(ValueError, match=rf"invalid {field}"):
        client.get_encounter_rankings(**kwargs)


def test_get_encounter_rankings_normalizes_integer_strings(monkeypatch):
    client = FFLogsV2Client("client-id", "client-secret")
    queries = []

    def fake_query(gql):
        queries.append(gql)
        return {"worldData": {"encounter": {"characterRankings": {}}}}

    monkeypatch.setattr(client, "query", fake_query)

    client.get_encounter_rankings(
        "1079",
        bracket="6",
        page="2",
        metric="rdps",
    )

    assert "encounter(id: 1079)" in queries[0]
    assert "metric: rdps" in queries[0]
    assert "bracket: 6" in queries[0]
    assert "page: 2" in queries[0]


def test_get_encounter_rankings_keeps_valid_filters(monkeypatch):
    client = FFLogsV2Client("client-id", "client-secret")
    queries = []

    def fake_query(gql):
        queries.append(gql)
        return {
            "worldData": {
                "encounter": {
                    "characterRankings": {"rankings": []},
                }
            }
        }

    monkeypatch.setattr(client, "query", fake_query)

    assert client.get_encounter_rankings(
        1079,
        spec_name="BlackMage",
        class_name="Caster",
    ) == {"rankings": []}
    assert 'specName: "BlackMage"' in queries[0]
    assert 'className: "Caster"' in queries[0]

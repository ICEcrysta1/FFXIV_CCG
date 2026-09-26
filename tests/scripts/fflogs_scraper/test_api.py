"""FFLogs 下载器 api 职责回归测试。"""

import logging

import pytest

from scripts.fflogs_scraper import FFLogsV2Client
from scripts.fflogs_scraper.api import client as api_client


def test_get_report_events_stops_at_event_limit(monkeypatch, caplog):
    monkeypatch.setattr(api_client, "MAX_EVENTS", 3)
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


@pytest.mark.parametrize("reason", ["events_limit", "pages_limit", "missing_response", "stalled"])
def test_complete_event_download_rejects_incomplete_results(monkeypatch, reason):
    client = FFLogsV2Client("id", "secret")
    wrapper = {"data": [{"timestamp": 1}, {"timestamp": 2}], "nextPageTimestamp": 3}
    if reason == "events_limit":
        monkeypatch.setattr(api_client, "MAX_EVENTS", 1)
    elif reason == "missing_response":
        wrapper = {}
    elif reason == "stalled":
        wrapper["nextPageTimestamp"] = 0
    monkeypatch.setattr(client, "query", lambda gql: {"reportData": {"report": {"events": wrapper}}})
    with pytest.raises(RuntimeError):
        client.get_report_events("ABC123", end_time=10, max_pages=1, require_complete=True)


def test_complete_event_download_uses_nested_abilities_and_keeps_page_order(monkeypatch):
    client = FFLogsV2Client("id", "secret")
    queries = []
    responses = iter([
        {"data": [{"timestamp": 1}, {"timestamp": 1}], "nextPageTimestamp": 2},
        {"data": [{"timestamp": 2}], "nextPageTimestamp": None},
    ])
    def query(gql):
        queries.append(gql)
        return {"reportData": {"report": {"events": next(responses)}}}
    monkeypatch.setattr(client, "query", query)
    events = client.get_report_events("ABC123", end_time=10, require_complete=True)
    assert [event["timestamp"] for event in events] == [1, 1, 2]
    assert "startTime: 2" in queries[1]
    assert "useAbilityIDs: false useActorIDs: true translate: true" in queries[0]

"""FFLogs 下载器 api 职责回归测试。"""

import logging

import pytest

from scripts.fflogs_scraper import FFLogsV2Client
from scripts.fflogs_scraper.api import client as api_client
from scripts.fflogs_scraper.contracts.models import FightInfo, ReportMeta


def test_report_metadata_fetches_fight_players_in_one_query(monkeypatch):
    client = FFLogsV2Client("id", "secret")
    queries = []

    def fake_query(gql):
        queries.append(gql)
        return {"reportData": {"report": {"playerDetails": {
            "data": {"playerDetails": {"dps": [{"id": 7, "name": "Player"}]}}
        }}}}

    monkeypatch.setattr(client, "query", fake_query)
    monkeypatch.setattr(
        api_client, "_adapt_report_metadata", lambda code, report: ReportMeta(code),
    )
    meta = client.get_report_fights("ABC123", player_fight_id=33)

    assert len(queries) == 1
    assert "playerDetails(fightIDs: [33])" in queries[0]
    assert "region { compactName }" in queries[0]
    assert meta.players == [{"id": 7, "name": "Player"}]


def test_report_metadata_without_player_fight_keeps_single_download_shape(monkeypatch):
    client = FFLogsV2Client("id", "secret")
    queries = []
    monkeypatch.setattr(client, "query", lambda gql: queries.append(gql) or {
        "reportData": {"report": {"playerDetails": {"data": {"dps": []}}}},
    })
    monkeypatch.setattr(
        api_client, "_adapt_report_metadata", lambda code, report: ReportMeta(code),
    )

    assert client.get_report_fights("ABC123").players == []
    assert "playerDetails" not in queries[0]


def test_report_metadata_rejects_invalid_player_fight_id():
    client = FFLogsV2Client("id", "secret")
    with pytest.raises(ValueError, match="player_fight_id"):
        client.get_report_fights("ABC123", player_fight_id="33) injected")


def test_http_session_is_reused_and_closed(monkeypatch):
    sessions = []

    class FakeResponse:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    class FakeSession:
        def __init__(self):
            self.calls = []
            self.closed = False
            sessions.append(self)

        def post(self, url, **kwargs):
            self.calls.append(url)
            if url == FFLogsV2Client.TOKEN_URL:
                return FakeResponse({"access_token": "token", "expires_in": 3600})
            return FakeResponse({"data": {"ok": True}})

        def close(self):
            self.closed = True

    monkeypatch.setattr(api_client.cf_requests, "Session", FakeSession)
    client = FFLogsV2Client("id", "secret")
    assert client.query("query { ok }") == {"ok": True}
    assert client.query("query { ok }") == {"ok": True}
    assert len(sessions) == 1
    assert sessions[0].calls == [FFLogsV2Client.TOKEN_URL, FFLogsV2Client.GQL_URL,
                                 FFLogsV2Client.GQL_URL]
    client.close()
    assert sessions[0].closed


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


def test_ranking_query_uses_bound_region_independently_of_partition(monkeypatch):
    cn_client = FFLogsV2Client("client-id", "client-secret", server_region="CN")
    na_client = FFLogsV2Client("client-id", "client-secret", server_region="NA")
    queries = []
    for client in (cn_client, na_client):
        monkeypatch.setattr(client, "query", lambda gql: queries.append(gql) or {
            "worldData": {"encounter": {"characterRankings": {"rankings": []}}},
        })

    cn_client.get_encounter_rankings(1079, partition=25)
    na_client.get_encounter_rankings(1079)

    assert 'serverRegion: "CN"' in queries[0] and "partition: 25" in queries[0]
    assert 'serverRegion: "NA"' in queries[1] and "partition:" not in queries[1]


def test_region_context_routes_every_query_to_one_endpoint(monkeypatch):
    class FakeResponse:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": self.data}

    calls = []
    class FakeSession:
        def post(self, url, **kwargs):
            calls.append(url)
            gql = kwargs["json"]["query"]
            if "worldData" in gql:
                data = {"worldData": {"encounter": {"characterRankings": {"rankings": []}}}}
            elif "rankedCharacters" in gql:
                data = {"reportData": {"report": {"rankedCharacters": []}}}
            elif "characterData" in gql:
                data = {"characterData": {"character": {"encounterRankings": {"ranks": []}}}}
            elif "events(" in gql:
                data = {"reportData": {"report": {
                    "events": {"data": [], "nextPageTimestamp": None},
                }}}
            elif "table(" in gql:
                data = {"reportData": {"report": {"table": {"entries": []}}}}
            else:
                data = {"reportData": {"report": {"title": "Report"}}}
            return FakeResponse(data)

        def close(self):
            pass

    monkeypatch.setattr(api_client, "_adapt_report_metadata", lambda code, _report: ReportMeta(code))
    for region, endpoint in (("CN", FFLogsV2Client.CN_GQL_URL),
                             ("NA", FFLogsV2Client.GQL_URL)):
        client = FFLogsV2Client("id", "secret", server_region=region)
        client._token = "test-token"
        monkeypatch.setattr(client, "_http_session", FakeSession)
        client.get_encounter_rankings(1079)
        client.resolve_ranking_character_id("ABC123", "Player", 81)
        client.get_character_history(1, 1079, spec_name="BlackMage")
        client.get_report_fights("ABC123")
        fight = FightInfo(id=1, name="FRU", start_time=0, end_time=10)
        client.get_report_events("ABC123", fight_ids=[fight.id], end_time=10)
        client.get_damage_table("ABC123", fight, 1)
        assert calls[-6:] == [endpoint] * 6


@pytest.mark.parametrize("region", ["US", "cn", 'CN"'])
def test_region_context_rejects_unsupported_values(region):
    with pytest.raises(ValueError, match="数据源地区"):
        FFLogsV2Client("id", "secret", server_region=region)


def test_ranking_query_uses_partition_and_surfaces_json_errors(monkeypatch):
    client = FFLogsV2Client("id", "secret")
    queries = []
    def query(gql):
        queries.append(gql)
        return {"worldData": {"encounter": {"characterRankings": {"error": "Invalid partition specified."}}}}
    monkeypatch.setattr(client, "query", query)
    with pytest.raises(RuntimeError, match="Invalid partition"):
        client.get_encounter_rankings(1079, partition=25)
    assert "partition: 25" in queries[0]


@pytest.mark.parametrize("partition", [None, 25])
def test_character_history_uses_api_partition_and_historical_percentile(monkeypatch, partition):
    client = FFLogsV2Client("id", "secret")
    queries = []
    character = {"name": "Player", "hidden": False, "encounterRankings": {"ranks": []}}
    def query(gql):
        queries.append(gql)
        return {"characterData": {"character": character}}
    monkeypatch.setattr(client, "query", query)
    assert client.get_character_history(123, 1079, spec_name="BlackMage", partition=partition) == character
    assert "lodestoneID: 123" in queries[0]
    if partition is None:
        assert "partition:" not in queries[0]
    else:
        assert f"partition: {partition}" in queries[0]
    assert "timeframe: Historical" in queries[0]
    assert 'specName: "BlackMage"' in queries[0]


@pytest.mark.parametrize("field,value", [
    ("partition", 0), ("partition", -1), ("partition", -2), ("lodestone_id", True),
    ("spec_name", 'BlackMage"'), ("metric", "injected"),
])
def test_character_history_rejects_invalid_parameters(field, value):
    client = FFLogsV2Client("id", "secret")
    kwargs = {"lodestone_id": 1, "encounter_id": 1079, "spec_name": "BlackMage", field: value}
    with pytest.raises(ValueError):
        client.get_character_history(**kwargs)


def test_unlinked_character_resolves_by_name_and_server_before_history_query(monkeypatch):
    client = FFLogsV2Client("id", "secret")
    queries = []
    def query(gql):
        queries.append(gql)
        if "rankedCharacters" in gql:
            return {"reportData": {"report": {"rankedCharacters": [
                {"id": 1, "name": "Player", "hidden": False, "server": {"id": 81}},
                {"id": 2, "name": "Player", "hidden": False, "server": {"id": 82}},
                {"id": 3, "name": "Hidden", "hidden": True, "server": {"id": 81}},
            ]}}}
        return {"characterData": {"character": {"id": 1, "name": "Player", "encounterRankings": {"ranks": []}}}}
    monkeypatch.setattr(client, "query", query)
    assert client.resolve_ranking_character_id("ABC123", "Player", 81) == 1
    assert client.resolve_ranking_character_id("ABC123", "Hidden", 81) is None
    assert client.get_character_history(None, 1079, character_id=1, spec_name="BlackMage")["id"] == 1
    assert "character(id: 1)" in queries[-1]


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

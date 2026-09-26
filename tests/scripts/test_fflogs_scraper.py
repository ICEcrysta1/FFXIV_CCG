"""FFLogs 下载脚本的输入边界测试。"""

from __future__ import annotations

import logging
import json
from types import SimpleNamespace

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


@pytest.fixture
def analysis_meta():
    """覆盖 V2 玩家、敌人实例和宠物归属。"""
    return fflogs_scraper._adapt_report_metadata("ABC123", {
        "title": "中文报告", "startTime": 1778332651064, "endTime": 1778332661064,
        "owner": {"name": "Uploader"}, "zone": {"id": 66, "name": "Ranking zone"},
        "masterData": {"lang": "cn", "actors": [
            {"id": 1, "gameID": 1000001, "name": "黑魔", "type": "Player", "subType": "BlackMage"},
            {"id": 2, "gameID": 17839, "name": "Boss", "type": "NPC", "subType": "Boss"},
            {"id": 3, "gameID": 13961, "name": "宠物", "type": "Pet", "petOwner": 1},
        ]},
        "fights": [{
            "id": 33, "name": "Futures Rewritten", "encounterID": 1079,
            "startTime": 5000, "endTime": 10000, "combatTime": 5000,
            "difficulty": 100, "kill": True, "size": 8,
            "bossPercentage": 12.5, "fightPercentage": 25.0,
            "gameZone": {"id": 1238, "name": "A Future Rewritten"},
            "friendlyPlayers": [1], "enemyPlayers": [], "friendlyNPCs": [],
            "enemyNPCs": [{"id": 2, "instanceCount": 2, "groupCount": 1}],
            "friendlyPets": [{"id": 3, "instanceCount": 3}], "enemyPets": [],
        }],
    })


def test_analysis_payload_preserves_report_and_training_contract(analysis_meta):
    payload = fflogs_scraper._build_download_payload(analysis_meta, analysis_meta.fights[0], 1)
    assert payload["start"] == 1778332651064
    assert payload["lang"] == "cn"
    assert payload["code"] == payload["report_code"] == "ABC123"
    assert payload["loading"] is False
    fight = payload["fights"][0]
    assert (fight["boss"], fight["zoneID"], fight["zoneName"]) == (1079, 1238, "A Future Rewritten")
    assert fight["fightPercentage"] == 2500
    assert fight["combatTime"] == 5000
    assert payload["friendlies"][0]["type"] == "BlackMage"
    assert payload["enemies"][0]["fights"] == [{"id": 33, "instances": 2, "groups": 1}]
    assert payload["friendlyPets"][0]["petOwner"] == 1
    event = {"timestamp": 5000, "type": "cast", "fight": 33, "sourceID": 1,
             "targetID": 2, "targetInstance": 2, "packetID": 7,
             "ability": {"guid": 3577, "name": "Fire IV", "type": 1024, "abilityIcon": "fire.png"}}
    fflogs_scraper._attach_analysis_events(payload, [event])
    saved = payload["events"][0]
    assert saved["abilityGameID"] == 3577
    assert saved["targetInstance"] == 2
    assert saved["timestamp"] == 5000
    assert saved["sourceIsFriendly"] is True
    assert saved["targetIsFriendly"] is False
    assert payload["events_complete"] is True
    assert "abilityGameID" not in event
    from scripts.convert_fflogs.extraction.extraction import _get_ability_id
    assert _get_ability_id(saved) == 3577


@pytest.mark.parametrize("reason", ["events_limit", "pages_limit", "missing_response", "stalled"])
def test_complete_event_download_rejects_incomplete_results(monkeypatch, reason):
    client = FFLogsV2Client("id", "secret")
    wrapper = {"data": [{"timestamp": 1}, {"timestamp": 2}], "nextPageTimestamp": 3}
    if reason == "events_limit":
        monkeypatch.setattr(fflogs_scraper, "MAX_EVENTS", 1)
    elif reason == "missing_response":
        wrapper = {}
    elif reason == "stalled":
        wrapper["nextPageTimestamp"] = 0
    monkeypatch.setattr(client, "query", lambda gql: {"reportData": {"report": {"events": wrapper}}})
    with pytest.raises(RuntimeError):
        client.get_report_events("ABC123", end_time=10, max_pages=1, require_complete=True)


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
    fflogs_scraper._batch_download(client, [("ABC123", 33, "Player", 123)], str(tmp_path), mode)
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
    fflogs_scraper._cmd_single(client, args)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["fight_id"] == 33
    assert payload["source_id"] == 1
    assert payload["events_complete"] is True
    assert requests == [(33, None, True)]
    assert ("damage_table" in payload) is not events_only


def test_write_json_preserves_existing_file_on_failure_and_uses_utf8_lf(tmp_path):
    path = tmp_path / "report.json"
    fflogs_scraper._write_download_json(str(path), {"title": "中文报告"})
    data = path.read_bytes()
    assert b"\r\n" not in data
    assert json.loads(data)["title"] == "中文报告"
    with pytest.raises(TypeError):
        fflogs_scraper._write_download_json(str(path), {"invalid": object()})
    assert path.read_bytes() == data
    assert list(tmp_path.glob("*.tmp")) == []


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


def test_metadata_rejects_missing_date_language_and_actor(analysis_meta):
    # 数据缺失必须失败，不能用下载日期或默认地区填补。
    raw = {"startTime": None, "masterData": {"lang": "en"}}
    with pytest.raises(ValueError, match="startTime"):
        fflogs_scraper._adapt_report_metadata("ABC123", raw)
    raw["startTime"] = 1778332651064
    raw["endTime"] = 1778332661064
    raw["masterData"]["lang"] = None
    with pytest.raises(ValueError, match="语言"):
        fflogs_scraper._adapt_report_metadata("ABC123", raw)
    raw["masterData"]["lang"] = "en"
    raw["title"] = "Report"
    raw["zone"] = {"id": 1, "name": "Zone"}
    raw["fights"] = [{"id": 1, "name": "Fight", "encounterID": 1079,
                      "startTime": 0, "endTime": 1, "friendlyPlayers": [99]}]
    with pytest.raises(ValueError, match="masterData"):
        fflogs_scraper._adapt_report_metadata("ABC123", raw)


def test_analysis_events_reject_flat_ability(analysis_meta):
    payload = fflogs_scraper._build_download_payload(analysis_meta, analysis_meta.fights[0], 1)
    with pytest.raises(ValueError, match="嵌套 ability"):
        fflogs_scraper._attach_analysis_events(payload, [{"type": "cast", "abilityGameID": 3577}])
    assert payload["events_complete"] is False


def test_payload_filters_actor_memberships_without_mutating_report(analysis_meta):
    other_fight = {**analysis_meta.report["fights"][0], "id": 34}
    analysis_meta.report["fights"].append(other_fight)
    analysis_meta.report["friendlies"][0]["fights"].append({"id": 34})
    payload = fflogs_scraper._build_download_payload(analysis_meta, analysis_meta.fights[0], 1)
    assert [f["id"] for f in payload["fights"]] == [33]
    assert payload["friendlies"][0]["fights"] == [{"id": 33}]
    assert len(analysis_meta.report["friendlies"][0]["fights"]) == 2
    with pytest.raises(ValueError, match="source=99"):
        fflogs_scraper._build_download_payload(analysis_meta, analysis_meta.fights[0], 99)
    assert fflogs_scraper._find_fight(analysis_meta, 1) is None


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
        fflogs_scraper._cmd_single(client, args)
    assert not output.exists()


def test_events_keep_incoming_buffs_and_targetability(analysis_meta):
    payload = fflogs_scraper._build_download_payload(analysis_meta, analysis_meta.fights[0], 1)
    events = [
        {"timestamp": 5000, "type": "applybuff", "fight": 33, "sourceID": 3, "targetID": 1,
         "ability": {"guid": 1000048, "name": "Well Fed", "type": 32, "abilityIcon": "food.png"}},
        {"timestamp": 5100, "type": "targetabilityupdate", "fight": 33,
         "sourceID": 2, "targetID": 2, "targetable": 0},
    ]
    fflogs_scraper._attach_analysis_events(payload, events)
    buff, targetability = payload["events"]
    assert buff["abilityGameID"] == 1000048
    assert buff["sourceIsFriendly"] is True
    assert buff["targetIsFriendly"] is True
    assert targetability["targetable"] == 0

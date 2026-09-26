"""FFLogs 下载器 contracts 职责回归测试。"""

import pytest

from scripts.fflogs_scraper.contracts import events as event_contract
from scripts.fflogs_scraper.contracts import report


def test_analysis_payload_preserves_report_and_training_contract(analysis_meta):
    payload = report._build_download_payload(analysis_meta, analysis_meta.fights[0], 1)
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
    event_contract._attach_analysis_events(payload, [event])
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


def test_metadata_rejects_missing_date_language_and_actor(analysis_meta):
    # 数据缺失必须失败，不能用下载日期或默认地区填补。
    raw = {"startTime": None, "masterData": {"lang": "en"}}
    with pytest.raises(ValueError, match="startTime"):
        report._adapt_report_metadata("ABC123", raw)
    raw["startTime"] = 1778332651064
    raw["endTime"] = 1778332661064
    raw["masterData"]["lang"] = None
    with pytest.raises(ValueError, match="语言"):
        report._adapt_report_metadata("ABC123", raw)
    raw["masterData"]["lang"] = "en"
    raw["title"] = "Report"
    raw["zone"] = {"id": 1, "name": "Zone"}
    raw["fights"] = [{"id": 1, "name": "Fight", "encounterID": 1079,
                      "startTime": 0, "endTime": 1, "friendlyPlayers": [99]}]
    with pytest.raises(ValueError, match="masterData"):
        report._adapt_report_metadata("ABC123", raw)


def test_analysis_events_reject_flat_ability(analysis_meta):
    payload = report._build_download_payload(analysis_meta, analysis_meta.fights[0], 1)
    with pytest.raises(ValueError, match="嵌套 ability"):
        event_contract._attach_analysis_events(payload, [{"type": "cast", "abilityGameID": 3577}])
    assert payload["events_complete"] is False


def test_payload_filters_actor_memberships_without_mutating_report(analysis_meta):
    other_fight = {**analysis_meta.report["fights"][0], "id": 34}
    analysis_meta.report["fights"].append(other_fight)
    analysis_meta.report["friendlies"][0]["fights"].append({"id": 34})
    payload = report._build_download_payload(analysis_meta, analysis_meta.fights[0], 1)
    assert [f["id"] for f in payload["fights"]] == [33]
    assert payload["friendlies"][0]["fights"] == [{"id": 33}]
    assert len(analysis_meta.report["friendlies"][0]["fights"]) == 2
    with pytest.raises(ValueError, match="source=99"):
        report._build_download_payload(analysis_meta, analysis_meta.fights[0], 99)
    assert report._find_fight(analysis_meta, 1) is None


def test_events_keep_incoming_buffs_and_targetability(analysis_meta):
    payload = report._build_download_payload(analysis_meta, analysis_meta.fights[0], 1)
    events = [
        {"timestamp": 5000, "type": "applybuff", "fight": 33, "sourceID": 3, "targetID": 1,
         "ability": {"guid": 1000048, "name": "Well Fed", "type": 32, "abilityIcon": "food.png"}},
        {"timestamp": 5100, "type": "targetabilityupdate", "fight": 33,
         "sourceID": 2, "targetID": 2, "targetable": 0},
    ]
    event_contract._attach_analysis_events(payload, events)
    buff, targetability = payload["events"]
    assert buff["abilityGameID"] == 1000048
    assert buff["sourceIsFriendly"] is True
    assert buff["targetIsFriendly"] is True
    assert targetability["targetable"] == 0

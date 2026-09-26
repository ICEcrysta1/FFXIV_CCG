"""FFLogs 下载器 download 职责回归测试。"""

import json
from types import SimpleNamespace

import pytest

from scripts.fflogs_scraper import FFLogsV2Client
from scripts.fflogs_scraper.download import batch, single


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

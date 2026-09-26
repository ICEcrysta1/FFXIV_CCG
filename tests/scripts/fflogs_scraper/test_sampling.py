"""历史记录发现、匿名排除与百分位配额测试。"""

from types import SimpleNamespace

import pytest

from scripts.fflogs_scraper.contracts.rankings import (
    _historical_report_from_rank,
    _percentile_bucket,
)
from scripts.fflogs_scraper.download.sampling import (
    _allocate_percentile_quotas,
    _iter_historical_reports,
)


def test_200_reports_are_evenly_allocated_to_ten_buckets():
    quotas = _allocate_percentile_quotas(200)
    assert list(quotas) == [f"{lower:02d}-{lower + 10}" for lower in range(90, -1, -10)]
    assert list(quotas.values()) == [20] * 10


@pytest.mark.parametrize("total", [1, 9, 10, 203])
def test_remainders_preserve_total_and_differ_by_at_most_one(total):
    quotas = _allocate_percentile_quotas(total)
    assert sum(quotas.values()) == total
    assert max(quotas.values()) - min(quotas.values()) <= 1
    assert list(quotas.values()) == sorted(quotas.values(), reverse=True)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_total_is_rejected(value):
    with pytest.raises(ValueError):
        _allocate_percentile_quotas(value)


@pytest.mark.parametrize("value,bucket", [
    (0, "00-10"), (9.999, "00-10"), (10, "10-20"), (89.999, "80-90"),
    (90, "90-100"), (100, "90-100"),
])
def test_bucket_boundaries(value, bucket):
    assert _percentile_bucket(value) == bucket


@pytest.mark.parametrize("value", [None, True, -0.1, 100.1, float("nan"), float("inf")])
def test_unknown_or_invalid_percentile_is_not_guessed(value):
    with pytest.raises((TypeError, ValueError)):
        _percentile_bucket(value)


def _rank(code, percentile, **extra):
    return {
        "report": {"code": code, "fightID": 33}, "spec": "BlackMage",
        "historicalPercent": percentile, "amount": 123, "lockedIn": True, **extra,
    }


def test_parse_requires_historical_percentile_and_preserves_zero():
    rank = _rank("ABC123", 0)
    record = _historical_report_from_rank(rank, player_name="Player", lodestone_id=1, spec_name="BlackMage")
    assert record.percentile == 0
    assert record.bucket == "00-10"
    assert record.ranking_metadata("rdps")["timeframe"] == "historical"
    del rank["historicalPercent"]
    rank["rankPercent"] = 90
    assert _historical_report_from_rank(rank, player_name="Player", lodestone_id=1, spec_name="BlackMage") is None


@pytest.mark.parametrize("partition", [None, 25])
def test_history_uses_non_best_records_and_excludes_anonymous_hidden_and_duplicates(partition):
    calls = []
    history_calls = []
    entries = [
        {"name": "Anonymous", "lodestoneID": 1},
        {"name": "Missing ID"},
        {"name": "Hidden", "lodestoneID": 2},
        {"name": "Player", "lodestoneID": 3},
        {"name": "Player", "lodestoneID": 3},
    ]
    def rankings(encounter, **kwargs):
        calls.append(kwargs)
        return {"rankings": entries, "hasMorePages": False}
    def history(lodestone, encounter, **kwargs):
        history_calls.append((lodestone, kwargs))
        return {
            "name": "Hidden" if lodestone == 2 else "Player", "hidden": lodestone == 2,
            "encounterRankings": {"ranks": [
                _rank("BEST", 100), _rank("LOW", 5), _rank("LOW", 5),
                _rank("OTHERJOB", 20, spec="WhiteMage"),
                _rank("PRIVATE", 30, hidden=True),
            ]},
        }
    client = SimpleNamespace(_cancelled=False, get_encounter_rankings=rankings, get_character_history=history)
    records = list(_iter_historical_reports(client, 1079, spec_name="BlackMage", metric="rdps", partition=partition))
    assert [(record.code, record.percentile) for record in records] == [("BEST", 100), ("LOW", 5)]
    assert [item[0] for item in history_calls] == [2, 3]
    assert all(item[1]["partition"] == partition for item in history_calls)
    assert [call["partition"] for call in calls] == [partition]
    assert all(call["page"] == 1 for call in calls)


def test_history_discovery_honors_explicit_pagination_and_page_limit():
    calls = []
    def rankings(encounter, **kwargs):
        calls.append(kwargs["page"])
        return {"rankings": [{"name": "Player", "lodestoneID": kwargs["page"]}], "hasMorePages": True}
    client = SimpleNamespace(
        _cancelled=False, get_encounter_rankings=rankings,
        get_character_history=lambda *args, **kwargs: {"encounterRankings": {"ranks": []}},
    )
    assert list(_iter_historical_reports(client, 1079, spec_name="BlackMage", metric="rdps", partition=1, max_pages=2)) == []
    assert calls == [1, 2]


def test_cancelled_discovery_does_not_query_api():
    client = SimpleNamespace(_cancelled=True)
    assert list(_iter_historical_reports(client, 1079, spec_name="BlackMage", metric="rdps", partition=1)) == []


def test_unlinked_public_character_is_preserved_but_anonymous_report_is_excluded():
    calls = []
    histories = []
    def resolve(code, name, server_id):
        calls.append((code, name, server_id))
        return 1234
    def history(lodestone_id, encounter_id, **kwargs):
        histories.append((lodestone_id, kwargs))
        return {"id": 1234, "name": "Public", "hidden": False, "encounterRankings": {"ranks": [_rank("CLEAR", 5)]}}
    client = SimpleNamespace(
        _cancelled=False, resolve_ranking_character_id=resolve, get_character_history=history,
        get_encounter_rankings=lambda *args, **kwargs: {"hasMorePages": False, "rankings": [
            {"name": "Public", "lodestoneID": 0, "server": {"id": 81}, "report": {"code": "SEED"}},
            {"name": "Public", "lodestoneID": 0, "server": {"id": 81}, "report": {"code": "SEED"}},
            {"name": "Masked", "lodestoneID": 0, "server": {"id": 81}, "report": {"code": "a:DfrP27RKwgqBkQGA"}},
        ]},
    )
    records = list(_iter_historical_reports(client, 1079, spec_name="BlackMage", metric="rdps", partition=29))
    assert [record.code for record in records] == ["CLEAR"]
    assert records[0].character_id == 1234
    assert records[0].lodestone_id == 0
    assert calls == [("SEED", "Public", 81)]
    assert histories[0][0] is None
    assert histories[0][1]["character_id"] == 1234

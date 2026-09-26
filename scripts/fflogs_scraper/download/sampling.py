"""历史记录发现与十档百分位配额，不负责报告文件下载。"""

import logging
from collections.abc import Iterator

from scripts.common.dataset_layout import PERCENTILE_BUCKETS

from ..api.client import FFLogsV2Client
from ..config.validation import _validate_integer
from ..contracts.rankings import (
    HistoricalReport,
    _historical_report_from_rank,
    _is_anonymous_name,
    _is_anonymous_report,
)

logger = logging.getLogger(__name__)


def _allocate_percentile_quotas(total: int) -> dict[str, int]:
    """均分总量；不能整除十时，余数按高档到低档每档加一。"""
    total = _validate_integer(total, "count", minimum=1)
    base, remainder = divmod(total, 10)
    return {
        bucket: base + (index < remainder)
        for index, bucket in enumerate(PERCENTILE_BUCKETS)
    }


def _iter_historical_reports(
    client: FFLogsV2Client,
    encounter_id: int,
    *,
    spec_name: str,
    metric: str,
    partition: int | None = None,
    bracket: int = 0,
    max_pages: int = 10,
) -> Iterator[HistoricalReport]:
    """在 API 默认或指定分区发现公开角色，再读取同分区的历史击杀。

    页数上限作用于角色发现榜单，不限制某个角色的历史记录数量。
    调用方配额完成即可停止，不提前收集所有报告。
    """
    max_pages = _validate_integer(max_pages, "max_pages", minimum=1)
    seen_characters = set()
    seen_character_ids = set()
    seen_reports = set()
    for page in range(1, max_pages + 1):
        if client._cancelled:
            return
        logger.info("发现历史角色: encounter=%d partition=%s page=%d", encounter_id, partition, page)
        try:
            result = client.get_encounter_rankings(
                encounter_id, spec_name=spec_name, metric=metric,
                bracket=bracket, partition=partition, page=page,
            )
        except Exception as error:  # noqa: BLE001 -- 榜单失败时保留已完成的下载并报告配额缺口
            logger.warning("分区 %s 查询失败，停止发现: %s", partition, error)
            return
        entries = result.get("rankings", [])
        has_more = bool(entries) and (
            result.get("hasMorePages") is True
            or ("hasMorePages" not in result and len(entries) >= 100)
        )
        for entry in entries:
            if client._cancelled:
                return
            if not isinstance(entry, dict) or entry.get("hidden") or entry.get("anonymous"):
                continue
            if _is_anonymous_name(entry.get("name")):
                logger.debug("跳过匿名角色")
                continue
            report = entry.get("report") or {}
            code = report.get("code", "")
            if _is_anonymous_report(code):
                logger.debug("跳过匿名报告")
                continue
            try:
                lodestone_id = _validate_integer(entry.get("lodestoneID", 0), "lodestone_id", minimum=0)
            except ValueError:
                continue
            server_id = (entry.get("server") or {}).get("id")
            identity = ("lodestone", lodestone_id) if lodestone_id else ("server", server_id, entry["name"])
            if identity in seen_characters:
                continue
            try:
                character_id = None
                if not lodestone_id:
                    if not code or not server_id:
                        continue
                    character_id = client.resolve_ranking_character_id(code, entry["name"], server_id)
                    if not character_id or character_id in seen_character_ids:
                        seen_characters.add(identity)
                        continue
                character = client.get_character_history(
                    lodestone_id or None, encounter_id, spec_name=spec_name,
                    metric=metric, partition=partition,
                    **({"character_id": character_id} if character_id else {}),
                )
            except Exception as error:  # noqa: BLE001 -- 一个角色失败不应阻止发现其他历史记录
                logger.warning("角色 %d 历史查询失败: %s", lodestone_id, error)
                continue
            seen_characters.add(identity)
            canonical_id = character.get("id") or character_id
            if canonical_id is not None:
                if canonical_id in seen_character_ids:
                    continue
                seen_character_ids.add(canonical_id)
            name = character.get("name") or entry["name"]
            if character.get("hidden") or _is_anonymous_name(name):
                continue
            for rank in character.get("encounterRankings", {}).get("ranks", []):
                record = _historical_report_from_rank(
                    rank, player_name=name, lodestone_id=lodestone_id, spec_name=spec_name,
                    character_id=canonical_id,
                )
                if record is None:
                    continue
                key = (record.code, record.fight_id, record.player_name)
                if key in seen_reports:
                    continue
                seen_reports.add(key)
                yield record
        if not has_more:
            return
    logger.warning("角色发现达到 %d 页上限；不足档位可提高 --max-pages", max_pages)

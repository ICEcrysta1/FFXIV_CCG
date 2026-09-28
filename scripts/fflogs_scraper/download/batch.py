"""按历史百分位配额编排批量下载，复用单份报告保存流程。"""

import json
import logging
import os
import sys

from common.dataset_layout import percentile_directory
from scripts.common.json_io import atomic_write_json

from ..api.client import FFLogsV2Client
from ..config.constants import DOWNLOAD_SCHEMA_VERSION
from ..contracts.events import _attach_analysis_events
from ..contracts.rankings import _is_anonymous_name, _is_anonymous_report
from ..contracts.report import _build_download_payload, _find_fight
from ..io.filenames import _build_batch_output_filename
from .sampling import _allocate_percentile_quotas, _iter_historical_reports

logger = logging.getLogger(__name__)


def _cmd_batch(client: FFLogsV2Client, args) -> None:
    """把总量均分到十个历史百分位区间，失败时继续同档补位。"""
    if args.zone and not args.encounter:
        encounters = client.get_zone_encounters(args.zone)
        print(f"Zone {args.zone} encounters:")
        for encounter in encounters:
            print(f"  id={encounter['id']:<5} {encounter['name']}")
        print("\n用法: batch -e <id>")
        return
    if not args.encounter:
        print("错误: 需要 --encounter 参数", file=sys.stderr)
        sys.exit(1)

    quotas = _allocate_percentile_quotas(args.count)
    details = client.get_encounter_details(args.encounter)
    print(f"\n{details['name']}：总目标 {args.count} 份，按历史百分位分配：")
    for bucket, target in quotas.items():
        print(f"  {bucket}: {target}")
    reports = _iter_historical_reports(
        client, args.encounter, spec_name=args.spec_name, metric=args.metric,
        partition=args.partition,
        bracket=args.bracket, max_pages=args.max_pages,
    )
    _stratified_batch_download(client, reports, quotas, args.output, args.mode, args.metric)


def _existing_download_matches(path, code, fight_id, name, mode, ranking_metadata) -> bool:
    """只有身份、分档口径和当前下载模式完整的旧文件才计入配额。"""
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return False
        if (
            payload.get("download_schema_version") != DOWNLOAD_SCHEMA_VERSION
            or payload.get("report_code") != code
            or payload.get("fight_id") != fight_id
            or payload.get("player_name") != name
            or payload.get("ranking") != ranking_metadata
            or type(payload.get("source_id")) is not int
            or payload["source_id"] <= 0
        ):
            return False
        if mode == "damage-only":
            return isinstance(payload.get("damage_table"), dict)
        events = payload.get("events")
        return (
            payload.get("events_complete") is True
            and payload.get("event_scope") == "fight"
            and isinstance(events, list)
            and payload.get("event_count") == len(events)
        )
    except (OSError, ValueError):
        return False


def _download_report(
    client: FFLogsV2Client, code: str, fid: int, name: str, amount: float,
    output_dir: str, mode: str, *, ranking_metadata: dict | None = None,
) -> str:
    """下载一名玩家的一场报告，返回成功、已有或匿名状态；失败交给编排层。"""
    if _is_anonymous_name(name) or _is_anonymous_report(code):
        logger.info("  -> 匿名记录，跳过")
        return "anonymous"
    output_path = os.path.join(output_dir, _build_batch_output_filename(code, fid, name))
    if os.path.exists(output_path):
        if ranking_metadata is None or _existing_download_matches(
            output_path, code, fid, name, mode, ranking_metadata,
        ):
            logger.info("  -> 已存在，计入配额")
            return "existing"
        logger.warning("  旧文件不符合本次分档或完整性要求，重新下载: %s", output_path)

    sid = client.resolve_source_id(code, name, fid)
    if not sid:
        raise ValueError(f"未找到玩家 {name} 的 source ID")
    meta = client.get_report_fights(code)
    fight = _find_fight(meta, fid)
    if not fight:
        raise ValueError(f"未找到 fight={fid}")
    result = _build_download_payload(meta, fight, sid)
    result.update(player_name=name, player_dps=amount)
    if ranking_metadata is not None:
        result["ranking"] = ranking_metadata
    if mode == "damage-only":
        result["damage_table"] = client.get_damage_table(code, fight, sid)
    else:
        if mode != "events-only":
            try:
                result["damage_table"] = client.get_damage_table(code, fight, sid)
            except Exception as error:  # noqa: BLE001 -- 伤害表失败时仍保留完整事件
                logger.warning("  伤害表获取失败: %s", error)
        events = client.get_fight_events(code, fight, require_complete=True)
        _attach_analysis_events(result, events)
    atomic_write_json(output_path, result)
    logger.info("  -> %s", output_path)
    return "success"


def _stratified_batch_download(client, reports, quotas, output_dir, mode="default", metric="rdps"):
    """分档下载；失败不占配额，不挪用其他区间填补缺额。"""
    counts = dict.fromkeys(quotas, 0)
    statistics = {"success": 0, "existing": 0, "failed": 0, "anonymous": 0}
    seen = set()
    for bucket in quotas:
        percentile_directory(output_dir, bucket).mkdir(parents=True, exist_ok=True)
    for record in reports:
        if client._cancelled or all(counts[bucket] >= quota for bucket, quota in quotas.items()):
            break
        bucket = record.bucket
        key = (record.code, record.fight_id, record.player_name)
        if counts[bucket] >= quotas[bucket] or key in seen:
            continue
        seen.add(key)
        logger.info("[%s %d/%d] %s f=%d name=%s historical=%.2f",
                    bucket, counts[bucket] + 1, quotas[bucket], record.code,
                    record.fight_id, record.player_name, record.percentile)
        try:
            status = _download_report(
                client, record.code, record.fight_id, record.player_name, record.amount,
                str(percentile_directory(output_dir, bucket)), mode,
                ranking_metadata=record.ranking_metadata(metric),
            )
            statistics[status] += 1
            if status in ("success", "existing"):
                counts[bucket] += 1
        except Exception as error:  # noqa: BLE001 -- 单份失败后继续同档补位
            statistics["failed"] += 1
            logger.warning("  下载失败，继续同档补位: %s", error)
        if all(counts[bucket] >= quota for bucket, quota in quotas.items()):
            break
    print("\n历史分档下载结果：")
    for bucket, target in quotas.items():
        missing = target - counts[bucket]
        print(f"  {bucket}: {counts[bucket]}/{target}" + (f"，缺 {missing} 份" if missing else ""))
    print(f"新下载={statistics['success']} 已有={statistics['existing']} 失败={statistics['failed']}")
    if any(counts[bucket] < quota for bucket, quota in quotas.items()):
        logger.warning("分档配额未完成；公开历史记录不足、分页上限、下载失败或中断均可能导致缺额")
    return {"counts": counts, **statistics}


def _batch_download(
    client: FFLogsV2Client, reports: list[tuple], output_dir: str = "data", mode: str = "default",
):
    """兼容已有调用方的指定报告列表下载，不做历史分档。"""
    os.makedirs(output_dir, exist_ok=True)
    counts = {"success": 0, "existing": 0, "failed": 0, "anonymous": 0}
    for index, (code, fid, name, amount) in enumerate(reports, 1):
        if client._cancelled:
            break
        logger.info("[%d/%d] %s f=%d name=%s dps=%.0f", index, len(reports), code, fid, name, amount)
        try:
            counts[_download_report(client, code, fid, name, amount, output_dir, mode)] += 1
        except Exception as error:  # noqa: BLE001 -- 兼容原批量下载逐份隔离失败的行为
            logger.error("  -> 失败: %s", error)
            counts["failed"] += 1
    print(f"\n批量下载完成: 成功={counts['success']} 跳过={counts['existing'] + counts['anonymous']} "
          f"失败={counts['failed']} / 总计={len(reports)}")
    return counts

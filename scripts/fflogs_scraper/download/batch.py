"""排行报告筛选与批量下载编排。"""

import logging
import os
import sys

from ..api.client import FFLogsV2Client
from ..contracts.events import _attach_analysis_events
from ..contracts.report import _build_download_payload, _find_fight
from ..io.filenames import _build_batch_output_filename
from ..io.json_io import _write_download_json

logger = logging.getLogger(__name__)


def _cmd_batch(client: FFLogsV2Client, args) -> None:
    """V2 API 批量下载高分报告。"""
    if args.zone and not args.encounter:
        encounters = client.get_zone_encounters(args.zone)
        print(f"Zone {args.zone} encounters:")
        for e in encounters:
            print(f"  id={e['id']:<5} {e['name']}")
        print(f"\n用法: batch -e <id>")
        return

    if not args.encounter:
        print("错误: 需要 --encounter 参数", file=sys.stderr)
        sys.exit(1)

    logger.info("批量: encounter=%d spec=%s bracket=%d pages=%d metric=%s",
                 args.encounter, args.spec_name, args.bracket, args.max_pages, args.metric)

    reports = client.get_high_score_reports(
        encounter_id=args.encounter,
        spec_name=args.spec_name,
        bracket=args.bracket,
        max_pages=args.max_pages,
        metric=args.metric,
    )

    if not reports:
        print("未找到符合条件的报告")
        return

    print(f"\n找到 {len(reports)} 份报告，开始批量下载...\n")
    _batch_download(client, reports, output_dir=args.output, mode=args.mode)


def _batch_download(
    client: FFLogsV2Client,
    reports: list[tuple],
    output_dir: str = "data",
    mode: str = "default",
):
    """批量下载报告。

    Parameters
    ----------
    client: FFLogsV2Client 实例
    reports: [(report_code, fight_id, player_name, dps), ...]
    output_dir: 输出目录
    mode: "default" | "events-only" | "damage-only"
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(reports)
    success = 0
    skipped = 0
    failed = 0

    for i, (code, fid, name, dps) in enumerate(reports, 1):
        if client._cancelled:
            logger.info("收到中断信号，停止批量下载")
            break

        logger.info("[%d/%d] %s f=%d name=%s dps=%.0f",
                     i, total, code, fid, name, dps)

        try:
            output_filename = _build_batch_output_filename(code, fid, name)
        except (TypeError, ValueError) as e:
            logger.warning("  报告编号无效，跳过: %s", e)
            failed += 1
            continue

        output_path = os.path.join(output_dir, output_filename)
        if os.path.exists(output_path):
            logger.info("  -> 已存在，跳过")
            skipped += 1
            continue

        try:
            sid = client.resolve_source_id(code, name, fid)
            if not sid:
                logger.warning("  未找到玩家 %s 的 source ID，跳过", name)
                failed += 1
                continue

            meta = client.get_report_fights(code)
            fight = _find_fight(meta, fid)
            if not fight:
                logger.warning("  未找到 fight=%d，跳过", fid)
                failed += 1
                continue

            result = _build_download_payload(meta, fight, sid)
            result.update(player_name=name, player_dps=dps)

            if mode == "damage-only":
                table = client.get_damage_table(code, fight, sid)
                result["damage_table"] = table
            else:
                if mode != "events-only":
                    try:
                        table = client.get_damage_table(code, fight, sid)
                        result["damage_table"] = table
                    except Exception as e:
                        logger.warning("  伤害表获取失败: %s", e)
                events = client.get_fight_events(code, fight, require_complete=True)
                _attach_analysis_events(result, events)

            _write_download_json(output_path, result)
            logger.info("  -> %s", output_path)
            success += 1

        except Exception as e:
            logger.error("  -> 失败: %s", e)
            failed += 1

    logger.info("批量下载完成: 成功=%d 跳过=%d 失败=%d / 总计=%d",
                 success, skipped, failed, total)
    print(f"\n批量下载完成: 成功={success} 跳过={skipped} 失败={failed} / 总计={total}")

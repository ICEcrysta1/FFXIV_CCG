"""单报告下载编排。"""

import logging
import os
import sys

from ..api.client import FFLogsV2Client
from ..config.validation import _validate_report_code
from ..contracts.events import _attach_analysis_events
from ..contracts.report import _build_download_payload, _find_fight
from ..io.filenames import _build_output_filename
from ..io.json_io import _write_download_json
from ..io.urls import parse_fflogs_url

logger = logging.getLogger(__name__)


def _cmd_single(client: FFLogsV2Client, args) -> None:
    """单报告下载。"""
    if args.url:
        parsed = parse_fflogs_url(args.url)
        report_code = parsed["report_code"]
        fight_id = parsed.get("fight_id", args.fight)
        source_id = parsed.get("source_id", args.source)
    elif args.report:
        report_code = _validate_report_code(args.report)
        fight_id = args.fight
        source_id = args.source
    else:
        print("错误: 需要 report URL 或 --report 参数", file=sys.stderr)
        sys.exit(1)

    logger.info("目标: report=%s fight=%s source=%s", report_code, fight_id, source_id)

    logger.info("拉取战斗列表...")
    meta = client.get_report_fights(report_code)
    fight = _find_fight(meta, fight_id) if fight_id else None
    if fight_id and fight is None:
        raise ValueError(f"报告中不存在 fight={fight_id}")
    result = _build_download_payload(meta, fight, source_id)

    if fight:
        logger.info("选定战斗: [%d] %s", fight.id, fight.name)
    else:
        logger.info("共 %d 场战斗, 全部保留", len(meta.fights))

    if args.damage_only:
        if not fight or not source_id:
            print("错误: 拉取伤害表需要 --fight 和 --source", file=sys.stderr)
            sys.exit(1)
        logger.info("拉取伤害表 (fight=%d, source=%d)...", fight.id, source_id)
        table = client.get_damage_table(report_code, fight, source_id)
        result["damage_table"] = table
    else:
        if not args.events_only and source_id and fight:
            logger.info("拉取伤害表...")
            try:
                table = client.get_damage_table(report_code, fight, source_id)
                result["damage_table"] = table
            except Exception as e:
                logger.warning("获取伤害表失败: %s", e)
        # 分析器需要整场事实；选定玩家只用于训练索引与伤害表。
        if fight:
            events = client.get_fight_events(report_code, fight, require_complete=True)
        else:
            events = client.get_report_events(
                report_code, end_time=meta.report["end"] - meta.report["start"],
                require_complete=True,
            )
        _attach_analysis_events(result, events)

    output_path = args.output or os.path.join(
        args.output_dir or "data",
        _build_output_filename(report_code, result["fight_id"], source_id),
    )
    _write_download_json(output_path, result)

    logger.info("已写入 -> %s", output_path)
    print(f"输出文件: {output_path}")

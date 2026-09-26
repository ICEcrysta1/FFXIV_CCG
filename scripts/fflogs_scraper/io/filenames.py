"""单场与批量下载文件名构造。"""

from datetime import datetime
from typing import Optional

from ..config.validation import _validate_report_code


def _build_output_filename(report_code: str, fight_id: int, source_id: Optional[int]) -> str:
    report_code = _validate_report_code(report_code)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_source{source_id}" if source_id else ""
    return f"fflogs_{report_code}_f{fight_id}{suffix}_{ts}.json"


def _build_batch_output_filename(
    report_code: str,
    fight_id: int,
    player_name: str,
) -> str:
    """构造批量下载文件名，并隔离报告编号和角色名中的路径字符。"""
    report_code = _validate_report_code(report_code)
    safe_name = (
        str(player_name)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
    )
    return f"fflogs_{report_code}_f{fight_id}_{safe_name}.json"

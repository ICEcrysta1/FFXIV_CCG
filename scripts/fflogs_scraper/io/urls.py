"""FFLogs 报告链接解析。"""

import re
from urllib.parse import parse_qs, urlparse

from ..config.validation import _validate_report_code


def parse_fflogs_url(url: str) -> dict:
    """解析 FFLogs URL，提取 report code / fight / source 等。"""
    parsed = urlparse(url)
    path_match = re.match(r"/reports/([A-Za-z0-9]+)", parsed.path)
    if not path_match:
        raise ValueError(f"无法从 URL 中提取 report code: {url}")

    report_code = _validate_report_code(path_match.group(1))
    params = parse_qs(parsed.query)

    result: dict = {"report_code": report_code}

    if "fight" in params:
        fight_val = params["fight"][0]
        if fight_val.isdigit():
            result["fight_id"] = int(fight_val)
        elif fight_val.lower() == "last":
            result["fight_id"] = -1

    if "source" in params:
        source_val = params["source"][0]
        if source_val.isdigit():
            result["source_id"] = int(source_val)

    return result

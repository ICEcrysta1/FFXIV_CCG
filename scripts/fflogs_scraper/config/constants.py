"""下载请求、输入校验、格式版本和补丁日期常量。"""

import re
from datetime import datetime

DEFAULT_REQUEST_TIMEOUT = 60
BROWSER_FINGERPRINT = "chrome124"
REPORT_CODE_PATTERN = re.compile(r"^[A-Za-z0-9]+$")
INTEGER_PATTERN = re.compile(r"^-?\d+$")
RANKING_METRICS = frozenset({"dps", "rdps", "ndps", "adps"})
MAX_EVENTS = 100_000
DOWNLOAD_SCHEMA_VERSION = 2


# 7.x 大版本补丁发布时间（Unix 毫秒）
PATCH_RELEASE_MS: dict[str, int] = {}
_patch_dates = {
    "7.0": "2024-07-02", "7.01": "2024-07-16", "7.05": "2024-07-30",
    "7.08": "2024-08-06", "7.1": "2024-11-12", "7.11": "2024-11-26",
    "7.15": "2024-12-17", "7.16": "2025-01-21", "7.18": "2025-02-25",
    "7.2": "2025-03-25", "7.21": "2025-04-01", "7.25": "2025-04-22",
    "7.3": "2025-05-20", "7.4": "2025-07-08", "7.5": "2025-09-30",
}
for _p, _d in _patch_dates.items():
    PATCH_RELEASE_MS[_p] = int(datetime.strptime(_d, "%Y-%m-%d").timestamp() * 1000)

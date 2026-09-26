"""历史排名记录、匿名筛选与百分位区间契约。"""

import math
from dataclasses import dataclass

from ..config.validation import _validate_integer, _validate_report_code


def _is_anonymous_report(code: object) -> bool:
    """匿名链接使用 a: 报告前缀，不尝试解析其真实身份。"""
    return isinstance(code, str) and code.startswith("a:")


def _is_anonymous_name(name: object) -> bool:
    """排除缺失名称及接口使用的匿名占位名称。"""
    if not isinstance(name, str) or not name.strip():
        return True
    normalized = name.strip().casefold()
    return normalized in {"anonymous", "匿名"}


def _percentile_bucket(percentile: float) -> str:
    """区间左闭右开，最高档同时包含 100；最低档固定写作 00-10。"""
    if isinstance(percentile, bool) or not isinstance(percentile, (int, float)):
        raise TypeError(f"invalid historical percentile: {percentile!r}")
    if not math.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError(f"invalid historical percentile: {percentile!r}")
    lower = min(int(percentile // 10) * 10, 90)
    return f"{lower:02d}-{lower + 10}"


@dataclass(frozen=True)
class HistoricalReport:
    """一名公开角色的一次历史击杀，不限于个人最佳记录。"""

    code: str
    fight_id: int
    player_name: str
    amount: float
    percentile: float
    lodestone_id: int
    bracket_data: object = None
    locked_in: bool = False
    character_id: int | None = None

    @property
    def bucket(self) -> str:
        return _percentile_bucket(self.percentile)

    def ranking_metadata(self, metric: str) -> dict:
        """记录 FFLogs 历史排名口径；此百分位不代表训练 PPG 分位数。"""
        return {
            "metric": metric,
            "comparison": "rankings",
            "timeframe": "historical",
            "percentile": self.percentile,
            "percentile_bucket": self.bucket,
            "bracket_data": self.bracket_data,
            "locked_in": self.locked_in,
            "lodestone_id": self.lodestone_id,
            "character_id": self.character_id,
        }


def _historical_report_from_rank(
    rank: dict, *, player_name: str, lodestone_id: int, spec_name: str,
    character_id: int | None = None,
) -> HistoricalReport | None:
    """只接受有效、可下载且属于指定职业的历史记录，不猜测缺失百分位。"""
    if not isinstance(rank, dict) or rank.get("spec") != spec_name:
        return None
    if rank.get("hidden") or rank.get("anonymous") or _is_anonymous_name(player_name):
        return None
    try:
        report = rank["report"]
        code = _validate_report_code(report["code"])
        fight_id = _validate_integer(report["fightID"], "fight_id", minimum=1)
        percentile = rank["historicalPercent"]
        _percentile_bucket(percentile)
        amount = float(rank["amount"])
        if not math.isfinite(amount) or amount < 0:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return HistoricalReport(
        code, fight_id, player_name, amount, float(percentile), lodestone_id,
        rank.get("bracketData"), rank.get("lockedIn") is True, character_id,
    )

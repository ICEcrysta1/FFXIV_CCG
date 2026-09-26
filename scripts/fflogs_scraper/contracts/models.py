"""报告和战斗元数据容器。"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FightInfo:
    """一场战斗的元信息。"""
    id: int
    name: str
    start_time: int
    end_time: int
    difficulty: Optional[int] = None
    kill: bool = False
    size: Optional[int] = None
    zone_name: str = ""


@dataclass
class ReportMeta:
    """一份报告的元信息。"""
    code: str
    fights: list[FightInfo] = field(default_factory=list)
    # FFLogs V1 报告元数据，训练选定玩家另存于下载包装字段。
    report: dict = field(default_factory=dict)

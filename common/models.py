"""Python 配置与转换共用的静态技能、状态定义和动作类型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionKind(str, Enum):
    """技能类型：GCD 或 oGCD。"""

    GCD = "gcd"
    OGCD = "ogcd"


@dataclass(frozen=True)
class StatusDefinition:
    key: str
    game_id: int
    duration: float
    max_stacks: int = 1

    def __post_init__(self):
        if self.max_stacks < 1:
            raise ValueError(
                f"StatusDefinition {self.key}: max_stacks must be >= 1, got {self.max_stacks}"
            )


@dataclass(frozen=True)
class SkillDefinition:
    key: str
    game_id: int
    name: str
    kind: ActionKind
    behavior: str
    potency: int = 0
    value: float = 1.0
    # value 是训练辅助值；职业状态机可以基于运行时状态解析动作 value，但不改写共享技能定义。
    cast_time: float = 0.0
    recast_time: float = 2.5
    mp_cost: int | str = 0
    mp_cost_floor: int = 0
    cooldown: float = 0.0
    charges: int = 1
    enabled: bool = True
    max_targets: int = 1
    aoe_secondary_reduction: float = 1.0
    dot_potency: int = 0
    dot_duration: float = 0.0
    dot_key: str | None = None
    requires_target: bool = False
    applies_statuses: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

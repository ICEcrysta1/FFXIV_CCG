"""战斗模拟使用的数据模型。
这个文件只放纯数据结构，例如技能定义、状态定义、战斗状态和执行结果。
它不放具体规则，避免数据结构和状态转移逻辑缠在一起。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math


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


@dataclass
class CooldownState:
    available_charges: int
    recharge_timers: list[float] = field(default_factory=list)

    def clone(self) -> "CooldownState":
        return CooldownState(
            available_charges=self.available_charges,
            recharge_timers=list(self.recharge_timers),
        )


@dataclass
class StatusState:
    remaining: float
    stacks: int = 1

    def clone(self) -> "StatusState":
        return StatusState(
            remaining=self.remaining,
            stacks=self.stacks,
        )


@dataclass
class DotState:
    remaining: float
    potency_per_tick: float
    tick_interval: float = 3.0
    next_tick_in_seconds: float = 3.0

    def clone(self) -> "DotState":
        return DotState(
            remaining=self.remaining,
            potency_per_tick=self.potency_per_tick,
            tick_interval=self.tick_interval,
            next_tick_in_seconds=self.next_tick_in_seconds,
        )


@dataclass(frozen=True)
class JobResourceDefinition:
    """职业专属资源定义。

    系统层只负责注册、初始化和基础读写，
    具体规则仍由职业状态机自行实现。

    `vector_group` 只服务于训练侧输出分组：
    - `resource`: 进入职业量谱向量。
    - `buff`: 进入 Buff 向量。
    """

    key: str
    resource_type: str
    default_value: int | float | bool
    vector_group: str = "resource"
    display_unit: str = "count"
    max_value: float | None = None

    def __post_init__(self):
        if self.resource_type not in {"int", "float", "bool"}:
            raise ValueError(
                f"JobResourceDefinition {self.key}: unsupported resource_type "
                f"{self.resource_type!r}"
            )
        if self.max_value is None:
            return
        if not math.isfinite(float(self.max_value)) or float(self.max_value) <= 0.0:
            raise ValueError(
                f"JobResourceDefinition {self.key}: max_value must be positive and finite, "
                f"got {self.max_value!r}"
            )
        if self.resource_type == "int" and not float(self.max_value).is_integer():
            raise ValueError(
                f"JobResourceDefinition {self.key}: int resource max_value must be integral, "
                f"got {self.max_value!r}"
            )


@dataclass(frozen=True)
class JobStateRegistration:
    """职业向系统层注册的资源与状态定义。"""

    job_tag: str
    resources: dict[str, JobResourceDefinition]
    statuses: dict[str, StatusDefinition]


@dataclass(frozen=True)
class ActionHistoryEntry:
    """单次动作的历史快照。

    这里只保存系统层通用的历史信息，方便后续输出层按职业需要做展示。
    """

    skill_key: str
    skill_id: int
    skill_name: str
    skill_kind: str
    potency: int | float
    value: float
    mp_before: int
    mp_after: int
    cast_time_seconds: float
    cast_time_gcds: float
    gcd_window_seconds: float
    gcd_window_gcds: float
    is_legal: bool
    invalid_reason: str
    next_cooldown_seconds: float
    available_charges: int
    max_charges: int
    job_resources_before: dict[str, int | float | bool]
    job_resources_after: dict[str, int | float | bool]
    job_resources_consumed: dict[str, int | float | bool]
    time_seconds: float
    gcd_index: int
    state_before: dict[str, object]
    state_after: dict[str, object]


@dataclass
class CombatState:
    """单个时刻的战斗快照。"""

    time: float = 0.0
    gcd_index: int = 0
    fight_remaining: float = 600.0
    next_downtime_eta: float | None = None
    downtime_remaining: float = 0.0
    mp: int = 10000
    max_mp: int = 10000
    natural_mp_tick_progress: float = 0.0
    gcd_remaining: float = 0.0
    weave_window_remaining: float = 0.0
    ogcds_weaved: int = 0
    max_ogcd_per_window: int = 3
    is_moving: bool = False
    boss_targetable: bool = True
    target_count: int = 1
    cumulative_potency: float = 0.0
    cumulative_dot_potency: float = 0.0
    current_potency: float = 0.0
    current_gcd_dot_potency: float = 0.0
    job_resources: dict[str, int | float | bool] = field(default_factory=dict)
    cooldowns: dict[str, CooldownState] = field(default_factory=dict)
    statuses: dict[str, StatusState] = field(default_factory=dict)
    dots: dict[str, DotState] = field(default_factory=dict)
    history: list[ActionHistoryEntry] = field(default_factory=list)

    def clone(self, *, copy_history: bool = True) -> "CombatState":
        """显式复制战斗状态，避免递归 `deepcopy` 的巨大开销。

        这里只有几类可变字段需要真正复制：
        - 玩家公共标量：直接值拷贝。
        - `job_resources` / `cooldowns` / `statuses` / `dots`：复制容器和元素。
        - `history`：默认复制列表本身，但复用只读历史条目对象。

        `copy_history=False` 只适用于“明确不会改写 history 列表”的只读预演场景，
        这样可以进一步减少候选预演时的复制成本。
        """
        return CombatState(
            time=self.time,
            gcd_index=self.gcd_index,
            fight_remaining=self.fight_remaining,
            next_downtime_eta=self.next_downtime_eta,
            downtime_remaining=self.downtime_remaining,
            mp=self.mp,
            max_mp=self.max_mp,
            natural_mp_tick_progress=self.natural_mp_tick_progress,
            gcd_remaining=self.gcd_remaining,
            weave_window_remaining=self.weave_window_remaining,
            ogcds_weaved=self.ogcds_weaved,
            max_ogcd_per_window=self.max_ogcd_per_window,
            is_moving=self.is_moving,
            boss_targetable=self.boss_targetable,
            target_count=self.target_count,
            cumulative_potency=self.cumulative_potency,
            cumulative_dot_potency=self.cumulative_dot_potency,
            current_potency=self.current_potency,
            current_gcd_dot_potency=self.current_gcd_dot_potency,
            job_resources=dict(self.job_resources),
            cooldowns={
                key: cooldown.clone()
                for key, cooldown in self.cooldowns.items()
            },
            statuses={
                key: status.clone()
                for key, status in self.statuses.items()
            },
            dots={
                key: dot.clone()
                for key, dot in self.dots.items()
            },
            history=(list(self.history) if copy_history else self.history),
        )

    def job_resource(self, key: str, default: int | float | bool | None = None) -> int | float | bool | None:
        return self.job_resources.get(key, default)

    def set_job_resource(self, key: str, value: int | float | bool) -> None:
        self.job_resources[key] = value

    def has_status(self, key: str) -> bool:
        status = self.statuses.get(key)
        return status is not None and status.remaining > 0 and status.stacks > 0

    def status_stacks(self, key: str) -> int:
        status = self.statuses.get(key)
        if status is None or status.remaining <= 0:
            return 0
        return status.stacks

    def dot_remaining(self, key: str) -> float:
        dot = self.dots.get(key)
        if dot is None:
            return 0.0
        return max(dot.remaining, 0.0)


@dataclass(frozen=True)
class ValidationResult:
    """动作合法性检查结果。"""

    ok: bool
    reason: str = ""


@dataclass
class StepResult:
    """一次动作执行后的前后状态与公共动作元信息。"""

    skill: SkillDefinition
    previous_state: CombatState
    next_state: CombatState
    validation: ValidationResult
    gcd_unit_seconds: float = 0.0
    actual_occupancy_seconds: float = 0.0
    actual_occupancy_gcd: float = 0.0
    actual_occupancy_source: str = ""
    gcd_window_seconds: float = 0.0
    gcd_window_gcd: float = 0.0
    next_gcd_window_seconds: float = 0.0
    next_gcd_window_gcd: float = 0.0

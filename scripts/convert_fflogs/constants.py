"""FFLogs 转换模块级常量定义。"""

from __future__ import annotations

import logging

from common.contracts import SCENE_EPSILON

from common.scene_context_schema import (
    FORCED_MOVEMENT_CONTEXT_KEY,
    FORCED_MOVEMENT_FEATURE_KEYS,
    RAID_BUFF_WINDOW_CONTEXT_KEY,
    RAID_BUFF_WINDOW_FEATURE_KEYS,
    RAID_BUFF_WINDOW_SOURCE_KEYS,
    SCENE_CONTEXT_KEYS,
    TARGET_COUNT_WINDOW_CONTEXT_KEY,
    TARGET_COUNT_WINDOW_FEATURE_KEYS,
    TARGETABLE_SEGMENT_KINDS,
    TARGETABLE_WINDOW_CONTEXT_KEY,
    TARGETABLE_WINDOW_FEATURE_KEYS,
)

logger = logging.getLogger("convert_fflogs")

# 位移判定：前后坐标欧氏距离大于此阈值则视为移动
MOVE_DIST_THRESHOLD = 3.0

# 爆发药 FFLogs buff ID 和内部 skill ID
POTION_BUFF_ID = 1000049
POTION_SKILL_ID = 99999

# 团辅窗口标记技能来源
RAID_BUFF_WINDOW_MARKER_KEYS = RAID_BUFF_WINDOW_SOURCE_KEYS

# 强制移动窗口合并的最大间隔
# 滑步修正后每个时间戳代表一次独立的滑步移动（约每 GCD 一次），
# 不应跨 GCD 合并，仅合并几乎同时发生的检测点。
MOVEMENT_MERGE_GAP = 0.1

# Boss 停手判定的默认伤害间隔阈值
DEFAULT_DOWNTIME_GAP_SECONDS = 6.0

# 停手窗口按开区间建模：端点动作（停手前最后一击、恢复后第一击）发生在 Boss
# 可选中时，所以窗口边界要内缩。内缩量必须覆盖"端点动作因 GCD/动画锁偏差被
# 状态机推迟执行"的最坏情况——状态机会把就绪前不超过动作队列窗口的请求接受
# 并排队，因此端点动作最晚在日志时刻 + 该窗口之后才真正执行。
# 调用方应传入 engine_timing.action_queue_window_seconds；此值为缺省回退。
DOWNTIME_BOUNDARY_MARGIN = 0.05

# 多目标标记：双目标事件超过该间隔未刷新则开始新的标记区间，
# 最后一次双目标事件后保留一个短暂窗口，覆盖动作生效时刻的回放误差。
MULTI_TARGET_OBSERVATION_TIMEOUT_SECONDS = 2.0
MULTI_TARGET_MARKER_GRACE_SECONDS = 1.5


def _round_time(value: float) -> float:
    """四舍五入到 4 位小数，统一时间精度。"""
    return round(float(value), 4)

"""模型读取侧的分块历史窗口；完整原始 bank 不受窗口参数影响。"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
import math


def scene_window_end_times(values, types, schema) -> tuple[float, ...]:
    """从原始 FP32 场景提取有效窗口终点，与 ContextEncoder 的裁剪条件一致。"""
    windows = {window.scene_type_id: window for window in schema.scene_windows}
    ends = []
    for row, type_id in zip(values.tolist(), types.tolist(), strict=True):
        window = windows[type_id]
        start, end = row[window.start_offset_index], row[window.end_offset_index]
        if end > start:
            ends.append(end)
    return tuple(sorted(ends))


def history_window_length(
    total_history: int, context_capacity: int, reset_keep: int, *,
    request_times: Sequence[float], scene_end_times: Sequence[float], current_request_time: float,
    max_history: int | None = None,
) -> int:
    """按完整历史重建分块重置；场景、状态和动作共用一个 token 预算。

    起点固定时场景锚点也固定，可以直接跳到该块首次超限的位置。
    不依赖样本读取顺序，也不以当前场景数量反推过去的重置周期。
    """
    for name, value in (("total_history", total_history), ("context_capacity", context_capacity),
                        ("reset_keep", reset_keep)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if reset_keep < 1 or 2 * reset_keep + 1 > context_capacity:
        raise ValueError("context_capacity must hold reset_keep complete state/action pairs and the current state")
    if total_history > len(request_times):
        raise ValueError("complete history request times are required for context planning")
    if max_history is not None and (type(max_history) is not int or max_history < 0):
        raise ValueError("max_history must be a non-negative integer")
    start = 0
    while True:
        anchor = request_times[start] if start < total_history else current_request_time
        if not math.isfinite(anchor):
            raise ValueError("context request time must be finite")
        scene_count = len(scene_end_times) - bisect_right(scene_end_times, anchor)
        capacity = (context_capacity - scene_count - 1) // 2
        if capacity < 0:
            raise ValueError("model.context_capacity cannot hold the scene and current state")
        overflow = start + capacity + 1
        if total_history < overflow:
            length = total_history - start
            if max_history is None:
                return length
            length = min(length, max_history)
            # 冻结请求时间可能倒序；额外裁短后锚点回退，场景数量反而可能增加。
            while True:
                anchor = request_times[total_history - length] if length else current_request_time
                scene_count = len(scene_end_times) - bisect_right(scene_end_times, anchor)
                if scene_count + 2 * length + 1 <= context_capacity:
                    return length
                if length == 0:
                    raise ValueError("model.context_capacity cannot hold the scene and current state")
                length -= 1
        if capacity < reset_keep:
            raise ValueError("model.context_capacity cannot hold the scene and history_reset_keep actions")
        start = overflow - reset_keep

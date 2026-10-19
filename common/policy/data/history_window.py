"""模型读取侧的分块历史窗口；完整原始 bank 不受窗口参数影响。"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
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
    history_markers: Sequence[Mapping[str, object]] = (),
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
    marker_positions = _history_marker_positions(history_markers, total_history)
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
        if marker_positions:
            marker_start = bisect_left(marker_positions, start)

            def token_count(cursor: int) -> int:
                return scene_count + 2 * (cursor - start) + 1 + bisect_right(marker_positions, cursor) - marker_start

            # 死亡不推进动作 cursor；按最早超限的动作边界重放，不能只扣除当前标记总数。
            if token_count(total_history) > context_capacity:
                low, high = start, total_history
                while low < high:
                    middle = (low + high) // 2
                    if token_count(middle) > context_capacity:
                        high = middle
                    else:
                        low = middle + 1
                overflow = low
            else:
                overflow = total_history + 1
        if total_history < overflow:
            length = total_history - start
            if max_history is None:
                return length
            return fit_history_suffix_length(
                min(length, max_history), context_capacity,
                total_history=total_history, request_times=request_times,
                scene_end_times=scene_end_times, current_request_time=current_request_time,
                history_markers=history_markers,
            )
        if capacity < reset_keep:
            raise ValueError("model.context_capacity cannot hold the scene and history_reset_keep actions")
        next_start = max(0, overflow - reset_keep)
        if next_start <= start:
            raise ValueError("model.context_capacity cannot hold the scene, history_reset_keep actions and death markers")
        start = next_start


def fit_history_suffix_length(
    length: int, context_capacity: int, *, total_history: int,
    request_times: Sequence[float], scene_end_times: Sequence[float], current_request_time: float,
    history_markers: Sequence[Mapping[str, object]] = (),
) -> int:
    """只检查已选历史后缀的预算，不重新规划正式分块窗口。"""
    marker_positions = _history_marker_positions(history_markers, total_history)
    # 冻结请求时间可能倒序；额外裁短后锚点回退，场景数量反而可能增加。
    while True:
        anchor = request_times[total_history - length] if length else current_request_time
        scene_count = len(scene_end_times) - bisect_right(scene_end_times, anchor)
        marker_count = len(marker_positions) - bisect_left(marker_positions, total_history - length)
        if scene_count + 2 * length + marker_count + 1 <= context_capacity:
            return length
        if length == 0:
            raise ValueError("model.context_capacity cannot hold the scene and current state")
        length -= 1


def _history_marker_positions(
    markers: Sequence[Mapping[str, object]], total_history: int,
) -> tuple[int, ...]:
    """可见性由调用方按事件步裁定；这里仅校验动作边界顺序。"""
    positions = []
    previous = 0
    for marker in markers:
        position = marker.get("after_history_length")
        if type(position) is not int or not previous <= position <= total_history:
            raise ValueError("history marker positions must be ordered within the visible action history")
        positions.append(position)
        previous = position
    return tuple(positions)

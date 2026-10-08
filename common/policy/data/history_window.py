"""模型读取侧的分块历史窗口；完整原始 bank 不受窗口参数影响。"""

from __future__ import annotations


def history_window_length(total_history: int, capacity: int, reset_keep: int) -> int:
    """超出容量时保留最近 K 条，再随真实历史行增长到下一次重置。"""
    for name, value in (("total_history", total_history), ("capacity", capacity),
                        ("reset_keep", reset_keep)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if capacity == 0:
        if reset_keep != 0:
            raise ValueError("reset_keep must be zero when capacity is zero")
        return 0
    if not 1 <= reset_keep <= capacity:
        raise ValueError("reset_keep must be between one and capacity")
    if total_history <= capacity:
        return total_history
    return reset_keep + (total_history - capacity - 1) % (capacity - reset_keep + 1)

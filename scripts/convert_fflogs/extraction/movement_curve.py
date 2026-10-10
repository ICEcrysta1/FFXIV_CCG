"""移动提取内部的连续贝塞尔速度曲线与水平阈值截取。"""

from __future__ import annotations

import math

import numpy as np


def sample_bezier_speed(
    positions: list[tuple[float, float, float]],
    *,
    sample_step_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """以首个坐标为原点，拟合经过邻接区间平均速度的三次贝塞尔。

    首末半个观测间隔保持最近速度；不向首末有效坐标之外推断位移。
    极值点切线为零，其余切线受相邻弦斜率限制，控制点不产生额外峰值。
    """
    observations = np.asarray(positions, dtype=np.float64)
    relative_time = observations[:, 0] - observations[0, 0]
    gaps = np.diff(relative_time)
    velocities = np.hypot(*np.diff(observations[:, 1:], axis=0).T) / gaps
    knots = np.r_[0.0, (relative_time[:-1] + relative_time[1:]) / 2, relative_time[-1]]
    values = np.r_[velocities[0], velocities, velocities[-1]]
    widths = np.diff(knots)
    slopes = np.diff(values) / widths
    tangents = np.zeros_like(values)
    tangents[0], tangents[-1] = slopes[0], slopes[-1]
    chord = (values[2:] - values[:-2]) / (knots[2:] - knots[:-2])
    same_direction = ((slopes[:-1] > 0) & (slopes[1:] > 0)) | (
        (slopes[:-1] < 0) & (slopes[1:] < 0)
    )
    tangents[1:-1] = np.where(
        same_direction,
        np.copysign(np.minimum(np.abs(chord), 3 * np.minimum(
            np.abs(slopes[:-1]), np.abs(slopes[1:]),
        )), chord),
        0.0,
    )
    controls = np.column_stack((
        values[:-1], values[:-1] + tangents[:-1] * widths / 3,
        values[1:] - tangents[1:] * widths / 3, values[1:],
    ))
    # 均匀网格以有效坐标为原点，整场时间平移不改变阈值交点。
    times = np.arange(math.ceil(relative_time[-1] / sample_step_seconds) + 1) * sample_step_seconds
    indices = np.clip(np.searchsorted(knots, times, side="right") - 1, 0, len(widths) - 1)
    u = (np.minimum(times, relative_time[-1]) - knots[indices]) / widths[indices]
    c = controls[indices]
    speed = (
        (1-u)**3*c[:, 0] + 3*(1-u)**2*u*c[:, 1]
        + 3*(1-u)*u**2*c[:, 2] + u**3*c[:, 3]
    )
    # 常速段的浮点运算可能略低于阈值，按贝塞尔控制范围恢复数值边界。
    speed = np.clip(speed, np.minimum(values[indices], values[indices+1]),
                    np.maximum(values[indices], values[indices+1]))
    return times, speed


def threshold_windows(
    times: np.ndarray,
    speed: np.ndarray,
    *,
    threshold: float,
    duration: float,
) -> list[tuple[float, float]]:
    """通过水平阈值截取连续曲线，交点在相邻网格之间线性精化。"""
    active = speed >= threshold
    changes = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    windows = []
    for first, last in zip(starts, ends):
        start, end = times[first], times[last]
        if first > 0:
            start = times[first-1] + (threshold-speed[first-1]) / (
                speed[first]-speed[first-1]
            ) * (times[first]-times[first-1])
        if last+1 < len(speed):
            end = times[last] + (threshold-speed[last]) / (
                speed[last+1]-speed[last]
            ) * (times[last+1]-times[last])
        start, end = max(0.0, float(start)), min(duration, float(end))
        if end > start:
            windows.append((start, end))
    return windows

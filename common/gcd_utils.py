"""输出层共用的 GCD 单位换算工具。"""

from __future__ import annotations


def to_gcd_units(seconds: float, gcd_unit_seconds: float) -> float:
    if gcd_unit_seconds <= 0:
        return 0.0
    return float(seconds) / float(gcd_unit_seconds)


def to_optional_gcd_units(seconds: float | None, gcd_unit_seconds: float) -> float | None:
    if seconds is None:
        return None
    return to_gcd_units(seconds, gcd_unit_seconds)

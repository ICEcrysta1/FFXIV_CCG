// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Common;

/// <summary>公共 GCD 单位换算（对照 outputs/gcd_utils.py 的 to_gcd_units）。</summary>
public static class GcdUnits
{
    /// <summary>秒数换算成 GCD 数量；基准秒数非正时返回 0。</summary>
    public static double ToGcdUnits(double seconds, double gcdUnitSeconds) =>
        gcdUnitSeconds <= 0 ? 0.0 : seconds / gcdUnitSeconds;

    /// <summary>可空秒数换算成 GCD 数量；输入为 null 时保持 null（对照 to_optional_gcd_units）。</summary>
    public static double? ToOptionalGcdUnits(double? seconds, double gcdUnitSeconds) =>
        seconds is null ? null : ToGcdUnits(seconds.Value, gcdUnitSeconds);
}

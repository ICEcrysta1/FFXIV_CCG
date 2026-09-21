// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Outputs;

/// <summary>输出层共用的数值转换工具（对照 outputs/value_utils.py 的 to_float）。</summary>
public static class ValueUtils
{
    /// <summary>把布尔、整数或浮点值统一转换成输出浮点数。</summary>
    public static double ToFloat(object value) => value switch
    {
        bool boolValue => boolValue ? 1.0 : 0.0,
        int intValue => intValue,
        double doubleValue => doubleValue,
        _ => Convert.ToDouble(value),
    };
}

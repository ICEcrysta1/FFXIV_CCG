// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.System;

/// <summary>系统层共享常量（对照 system/constants.py）。</summary>
public static class SystemConstants
{
    /// <summary>零值判定阈值，所有时间推进与状态比较共用。</summary>
    public const double ZeroEpsilon = 0.0001;
}

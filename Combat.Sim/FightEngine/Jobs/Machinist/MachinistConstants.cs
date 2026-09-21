// Copyright (C) 2026 ICE_crystal
// Copyright (C) 2026 SpikeHS (original Machinist implementation)
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Jobs.Machinist;

/// <summary>机工职业规则常量。</summary>
public static class MachinistConstants
{
    /// <summary>过热状态下单体武器技能额外威力。</summary>
    public const double OverheatedSingleTargetPotencyBonus = 20.0;

    public const int ComboNone = 0;
    public const int ComboSplit = 1;
    public const int ComboSlug = 2;
}

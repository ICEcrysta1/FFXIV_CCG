// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Definitions;

/// <summary>技能类型：GCD 或 oGCD（对照 models.ActionKind）。</summary>
public enum ActionKind
{
    Gcd,
    Ogcd,
}

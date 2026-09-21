// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;

namespace Combat.Sim.Models.Policy;

/// <summary>一次纯策略决策及其调用方选定的后续观测状态。</summary>
public sealed record PolicyDecision(
    PolicyActionDefinition Action,
    double Timestamp,
    int GcdIndex,
    CombatState StateBefore,
    CombatState StateAfter)
{
    internal PolicyDecision DeepClone() => this with
    {
        StateBefore = StateBefore.Clone(),
        StateAfter = StateAfter.Clone(),
    };
}

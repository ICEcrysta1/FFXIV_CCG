// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Outputs;

namespace Combat.Sim.Models.Policy;

/// <summary>一次纯策略决策；真实动作后快照在决策落实时保存，模型状态在请求时冻结。</summary>
public sealed record PolicyDecision(
    PolicyActionDefinition Action,
    double Timestamp,
    int GcdIndex,
    CombatState StateBefore,
    CombatState StateAfter,
    ModelStateSnapshot ModelState)
{
    internal PolicyDecision DeepClone() => this with
    {
        StateBefore = StateBefore.CloneWithoutHistory(),
        StateAfter = StateAfter.CloneWithoutHistory(),
    };
}

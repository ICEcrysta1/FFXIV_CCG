// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;

namespace Combat.Sim.Models.Timeline;

/// <summary>
/// 时间线的完整可克隆快照，包含战斗状态、待处理事件、待结算事实和 sequence 游标。
/// </summary>
public sealed record SimulationSnapshot(
    CombatState State,
    IReadOnlyList<TimelineEvent> PendingEvents,
    IReadOnlyList<TimelineEvent> PendingSettlements,
    long NextSequence)
{
    public SimulationSnapshot DeepClone() => new(
        State.Clone(),
        PendingEvents.Select(item => item.DeepClone()).ToArray(),
        PendingSettlements.Select(item => item.DeepClone()).ToArray(),
        NextSequence);
}

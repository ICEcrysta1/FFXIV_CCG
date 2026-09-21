// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;

namespace Combat.Sim.Models.Timeline;

/// <summary>
/// 领域处理器返回给中心时间线的声明式变更。
/// 处理器可以描述状态变更和后续事件，但不直接操作时钟或事件队列。
/// </summary>
public sealed record TimelineMutation(
    Action<CombatState>? ApplyState = null,
    IReadOnlyList<TimelineEvent>? EventsToSchedule = null,
    IReadOnlyList<long>? SequencesToCancel = null)
{
    public static TimelineMutation Empty { get; } = new();
}

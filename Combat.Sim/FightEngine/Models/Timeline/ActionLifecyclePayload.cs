// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;

namespace Combat.Sim.Models.Timeline;

/// <summary>
/// 动作事件携带的不可变生命周期数据。状态快照必须深拷贝，避免 fork 共享可变战斗状态。
/// </summary>
internal sealed record ActionLifecyclePayload(
    Guid ActionInstanceId,
    ActionRequest Request,
    SkillDefinition Skill,
    CombatState RequestState,
    ActionTimingPlan Timing,
    double AcceptedTimestamp) : ITimelineEventPayload
{
    public object DeepClone() => this with { RequestState = RequestState.Clone() };
}

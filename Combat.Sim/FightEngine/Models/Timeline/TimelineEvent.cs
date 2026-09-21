// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Timeline;

/// <summary>时间线事件类型。事件只描述事实，具体效果由注册的处理器负责。</summary>
public enum TimelineEventKind
{
    ActionAccepted,
    CastCompleted,
    ActionEffect,
    GcdReady,
    CooldownChargeReady,
    StatusExpired,
    DotTick,
    MpTick,
    JobPeriodicTick,
    JobTimerExpired,
    SceneChanged,
    DecisionBoundary,
}

/// <summary>
/// 同一时间戳下的稳定处理优先级。
/// 数值越小越早处理；Sequence 作为最终稳定排序键。
/// </summary>
public enum TimelineEventPriority
{
    ExternalScene = 10,
    ConfirmedActionEffect = 20,
    PeriodicSettlement = 30,
    ExpirationAndCooldown = 40,
    ActionAccepted = 50,
    DecisionBoundary = 50,
}

/// <summary>
/// 时间线中的不可变事件。
/// Sequence 为 0 时表示尚未由 CombatTimelineRuntime 编号。
/// </summary>
public sealed record TimelineEvent(
    double Timestamp,
    TimelineEventPriority Priority,
    TimelineEventKind Kind,
    string? OwnerKey = null,
    Guid? ActionInstanceId = null,
    object? Payload = null,
    long Sequence = 0)
{
    internal TimelineEvent DeepClone() => this with
    {
        Payload = Payload is ITimelineEventPayload cloneable ? cloneable.DeepClone() : Payload,
    };
}

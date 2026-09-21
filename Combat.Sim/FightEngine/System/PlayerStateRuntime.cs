// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Timeline;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using static Combat.Sim.System.SystemConstants;

namespace Combat.Sim.System;

/// <summary>
/// 系统层玩家公共状态：读条 / GCD / weave 窗口 / Boss 停手机制与公共时间推进。
/// 动画间隔由调用方安排，不属于状态机的时间锁。
/// （对照 player_state.PlayerStateRuntime）。
/// </summary>
public sealed class PlayerStateRuntime
{

    /// <summary>校验所有职业共享的动作合法性。</summary>
    public ValidationResult ValidateCommonAction(
        CombatState state,
        SkillDefinition skill,
        bool isInstant,
        bool hasAvailableCharge,
        bool includeTemporalLocks = true)
    {
        if (!state.BossTargetable && skill.RequiresTarget)
        {
            return new ValidationResult(false, "boss_untargetable");
        }
        if (includeTemporalLocks && state.CastRemaining > ZeroEpsilon)
        {
            return new ValidationResult(false, "cast_locked");
        }
        if (includeTemporalLocks && skill.Kind == ActionKind.Gcd && state.GcdRemaining > ZeroEpsilon)
        {
            return new ValidationResult(false, "gcd_locked");
        }
        if (skill.Kind == ActionKind.Ogcd)
        {
            if (state.GcdRemaining > ZeroEpsilon && state.OgcdsWeaved >= state.MaxOgcdPerWindow)
            {
                return new ValidationResult(false, "ogcd_limit");
            }
        }
        if (state.IsMoving && !isInstant)
        {
            return new ValidationResult(false, "movement_locked");
        }
        if (includeTemporalLocks && !hasAvailableCharge)
        {
            return new ValidationResult(false, "cooldown_locked");
        }
        return new ValidationResult(true);
    }

    /// <summary>应用公共时序推进（空转等跳过时序的系统动作除外）。</summary>
    public void ApplyActionTiming(
        CombatState state,
        SkillDefinition skill,
        double effectiveGcd,
        double actualCastSeconds,
        double? gcdStartTimestamp = null)
    {
        if (skill.Kind == ActionKind.Gcd)
        {
            state.GcdIndex += 1;
            state.CastRemaining = actualCastSeconds;
            // 队列中的 GCD 从请求时刻开始计算 recast，避免连续排队逐发向后漂移。
            var gcdReadyAt = (gcdStartTimestamp ?? state.Time) + effectiveGcd;
            state.GcdReadyAt = Math.Max(state.Time, gcdReadyAt);
            state.WeaveWindowRemaining = Math.Max(0.0, effectiveGcd - actualCastSeconds);
            state.OgcdsWeaved = 0;
            return;
        }

        state.OgcdsWeaved += 1;
    }

    /// <summary>玩家锁与场景只声明截止时间，不递减剩余秒数。</summary>
    public IEnumerable<TimelineEvent> DescribeEvents(CombatState state)
    {
        if (state.GcdReadyAt > state.Time || state.OgcdsWeaved > 0 || state.WeaveWindowRemaining > 0)
            yield return new(Math.Max(state.Time, state.GcdReadyAt), TimelineEventPriority.DecisionBoundary,
                TimelineEventKind.GcdReady);
        if (state.BossTargetable && state.NextDowntimeStartsAt is double start && state.DowntimeRemaining > 0)
            yield return new(Math.Max(state.Time, start), TimelineEventPriority.ExternalScene,
                TimelineEventKind.SceneChanged, "downtime_start");
        else if (!state.BossTargetable && state.DowntimeEndsAt > state.Time + ZeroEpsilon)
            yield return new(Math.Max(state.Time, state.DowntimeEndsAt), TimelineEventPriority.ExternalScene,
                TimelineEventKind.SceneChanged, "downtime_end");
    }

    public TimelineMutation HandleGcdReady(TimelineEvent item, CombatState state)
    {
        if (state.GcdReadyAt > item.Timestamp + ZeroEpsilon)
        {
            return TimelineMutation.Empty;
        }

        return new TimelineMutation(ApplyState: target =>
        {
            target.OgcdsWeaved = 0;
            target.WeaveWindowRemaining = 0;
        });
    }

    public TimelineMutation HandleSceneChanged(TimelineEvent item, CombatState state) => new(ApplyState: target =>
    {
        if (item.OwnerKey == "downtime_start") target.BossTargetable = false;
        else if (item.OwnerKey == "downtime_end")
        {
            target.BossTargetable = true;
            target.DowntimeRemaining = 0;
            target.NextDowntimeEta = null;
        }
    });
}

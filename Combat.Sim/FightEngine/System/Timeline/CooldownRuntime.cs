// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Timeline;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;

namespace Combat.Sim.System.Timeline;

/// <summary>
/// 系统层公共冷却运行时：充能校验、消耗、回充、主动减少与快照
/// （对照 cooldown_runtime.CooldownRuntime）。
/// </summary>
public sealed class CooldownRuntime
{
    public IEnumerable<TimelineEvent> DescribeEvents(CombatState state) => state.Cooldowns.SelectMany(pair =>
        pair.Value.RechargeReadyAt.Select(time => new TimelineEvent(Math.Max(state.Time, time),
            TimelineEventPriority.ExpirationAndCooldown, TimelineEventKind.CooldownChargeReady, pair.Key)));

    public TimelineMutation HandleReady(TimelineEvent item, CombatState state, Func<string, int> getMaxCharges) =>
        new(ApplyState: target =>
        {
            if (!target.Cooldowns.TryGetValue(item.OwnerKey!, out var bucket)) return;
            var index = bucket.RechargeReadyAt.FindIndex(time => time <= target.Time);
            if (index < 0) return;
            bucket.RechargeReadyAt.RemoveAt(index);
            var max = getMaxCharges(item.OwnerKey!);
            bucket.AvailableCharges = Math.Min(max, bucket.AvailableCharges + 1);
            if (bucket.AvailableCharges >= max && bucket.RechargeReadyAt.Count == 0)
                target.Cooldowns.Remove(item.OwnerKey!);
        });

    public bool HasAvailableCharge(CombatState state, SkillDefinition skill)
    {
        if (skill.Cooldown <= 0 && skill.Charges <= 1)
        {
            return true;
        }
        if (!state.Cooldowns.TryGetValue(skill.Key, out var bucket))
        {
            return true;
        }
        bucket.BindTime(state.Time);
        return bucket.AvailableCharges > 0;
    }

    public double? NextChargeReadyAt(CombatState state, SkillDefinition skill)
    {
        if (!state.Cooldowns.TryGetValue(skill.Key, out var bucket)
            || bucket.RechargeReadyAt.Count == 0)
        {
            return null;
        }

        return bucket.RechargeReadyAt.Min();
    }

    public (double NextCooldownSeconds, int AvailableCharges, int MaxCharges) BuildCooldownSnapshot(
        CombatState state,
        SkillDefinition skill)
    {
        if (!state.Cooldowns.TryGetValue(skill.Key, out var bucket))
        {
            return (0.0, skill.Charges, skill.Charges);
        }
        bucket.BindTime(state.Time);
        double nextCooldownSeconds = 0.0;
        if (bucket.RechargeTimers.Count > 0)
        {
            nextCooldownSeconds = Math.Max(0.0, bucket.RechargeTimers.Min());
        }
        return (Math.Round(nextCooldownSeconds, 4), bucket.AvailableCharges, skill.Charges);
    }

    public void ConsumeCooldown(CombatState state, SkillDefinition skill)
    {
        if (skill.Cooldown <= 0 && skill.Charges <= 1)
        {
            return;
        }
        var bucket = CooldownBucket(state, skill);
        bucket.BindTime(state.Time);
        ReleaseReadyCharges(state, skill, bucket);
        if (bucket.AvailableCharges <= 0)
        {
            return;
        }
        bucket.AvailableCharges -= 1;
        bucket.RechargeReadyAt.Add(state.Time + skill.Cooldown);
        bucket.RechargeReadyAt.Sort();
    }

    /// <summary>减少指定技能所有尚未完成的回充计时。</summary>
    internal void ApplyReduction(CombatState state, SkillDefinition skill, double seconds)
    {
        if (!state.Cooldowns.TryGetValue(skill.Key, out var bucket))
        {
            return;
        }
        bucket.BindTime(state.Time);
        var reduction = Math.Max(0.0, seconds);
        bucket.RechargeReadyAt = bucket.RechargeReadyAt
            .Select(timer => Math.Max(state.Time, timer - reduction))
            .ToList();
        bucket.RechargeReadyAt.Sort();
        ReleaseReadyCharges(state, skill, bucket);
        if (bucket.AvailableCharges >= skill.Charges && bucket.RechargeTimers.Count == 0)
        {
            state.Cooldowns.Remove(skill.Key);
        }
    }

    private static void ReleaseReadyCharges(
        CombatState state,
        SkillDefinition skill,
        CooldownState bucket)
    {
        var ready = bucket.RechargeReadyAt.RemoveAll(time => time <= state.Time);
        bucket.AvailableCharges = Math.Min(skill.Charges, bucket.AvailableCharges + ready);
    }

    private static CooldownState CooldownBucket(CombatState state, SkillDefinition skill)
    {
        if (!state.Cooldowns.TryGetValue(skill.Key, out var bucket))
        {
            bucket = new CooldownState(availableCharges: skill.Charges);
            state.Cooldowns[skill.Key] = bucket;
        }
        bucket.BindTime(state.Time);
        return bucket;
    }
}

// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Timeline;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using static Combat.Sim.System.SystemConstants;

namespace Combat.Sim.System.Timeline;

/// <summary>
/// 系统层 DoT 时间推进：注册、挂载、按 tick 结算、剩余时间推进与移除
/// （对照 dot_timeline.DotTimelineRuntime）。
/// </summary>
public sealed class DotTimelineRuntime
{
    private IReadOnlyList<string> _registeredDotKeys = Array.Empty<string>();

    public IReadOnlyList<string> RegisteredDotKeys => _registeredDotKeys;

    /// <summary>同类 DoT 通过共享 dot_key 复用一个目标槽位。</summary>
    public void RegisterDots(IEnumerable<SkillDefinition> skills)
    {
        _registeredDotKeys = skills
            .Where(skill => skill.DotDuration > 0)
            .Select(skill => skill.DotKey ?? skill.Key)
            .Distinct(StringComparer.Ordinal)
            .OrderBy(key => key, StringComparer.Ordinal)
            .ToList();
    }

    public void GrantRegisteredDot(
        CombatState state,
        string key,
        double remaining,
        double potencyPerTick,
        double tickInterval = 3.0)
    {
        if (!_registeredDotKeys.Contains(key))
        {
            throw new InvalidOperationException($"unregistered dot key: {key}");
        }
        if (!double.IsFinite(tickInterval) || tickInterval <= 0)
            throw new ArgumentOutOfRangeException(nameof(tickInterval));
        state.Dots[key] = new DotState(
            remaining: remaining,
            potencyPerTick: potencyPerTick,
            tickInterval: tickInterval,
            nextTickInSeconds: tickInterval);
        state.Dots[key].BindTime(state.Time);
    }

    public IEnumerable<TimelineEvent> DescribeEvents(CombatState state) => state.Dots.Select(pair =>
        new TimelineEvent(Math.Max(state.Time, Math.Min(pair.Value.NextTickAt, pair.Value.ExpiresAt)),
            TimelineEventPriority.PeriodicSettlement, TimelineEventKind.DotTick, pair.Key));

    /// <summary>单次 tick 或到期清理；后续事件由中心根据新 deadline 续排。</summary>
    public TimelineMutation HandleTick(TimelineEvent item, CombatState state) => new(ApplyState: target =>
    {
        if (!target.Dots.TryGetValue(item.OwnerKey!, out var dot)) return;
        if (dot.NextTickAt <= target.Time && dot.NextTickAt <= dot.ExpiresAt)
        {
            if (target.BossTargetable)
            {
                target.CurrentGcdDotPotency += dot.PotencyPerTick;
                target.CumulativeDotPotency += dot.PotencyPerTick;
            }
            dot.NextTickAt += dot.TickInterval;
        }
        if (dot.ExpiresAt <= target.Time) target.Dots.Remove(item.OwnerKey!);
    });
}

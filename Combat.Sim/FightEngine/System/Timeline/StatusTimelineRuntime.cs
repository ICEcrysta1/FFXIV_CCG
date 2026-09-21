// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Timeline;

namespace Combat.Sim.System.Timeline;

/// <summary>声明状态截止时间、持有职业状态到期钩子并处理单次过期，不推进时钟。</summary>
public sealed class StatusTimelineRuntime
{
    private readonly Dictionary<string, Action<CombatState>> _expiryHooks = new(StringComparer.Ordinal);

    /// <summary>
    /// 注册职业状态到期处理。处理器在状态被移除之前调用，因此仍能读到即将过期的残留层数；
    /// 需要统计或结算"过期时的残留"的职业规则走这里，而不是另排一个倒计时事件去争优先级。
    /// 处理器只能修改状态，不能排布或取消事件。
    /// </summary>
    public void RegisterExpiryHook(string statusKey, Action<CombatState> handler)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(statusKey);
        ArgumentNullException.ThrowIfNull(handler);
        if (!_expiryHooks.TryAdd(statusKey, handler))
        {
            throw new InvalidOperationException($"status expiry hook already registered: {statusKey}");
        }
    }

    public IEnumerable<TimelineEvent> DescribeEvents(CombatState state) => state.Statuses.Select(pair =>
        new TimelineEvent(Math.Max(state.Time, pair.Value.ExpiresAt), TimelineEventPriority.ExpirationAndCooldown,
            TimelineEventKind.StatusExpired, pair.Key));

    public TimelineMutation HandleExpired(TimelineEvent item, CombatState state)
    {
        var statusKey = item.OwnerKey!;
        if (!state.Statuses.TryGetValue(statusKey, out var status) || status.ExpiresAt > state.Time)
        {
            return TimelineMutation.Empty;
        }

        return new TimelineMutation(ApplyState: target =>
        {
            if (!target.Statuses.TryGetValue(statusKey, out var current) || current.ExpiresAt > target.Time)
            {
                return;
            }

            // 职业到期处理与状态移除必须落在同一次结算内：处理器需要读取尚未移除的残留层数。
            if (_expiryHooks.TryGetValue(statusKey, out var hook))
            {
                hook(target);
            }

            target.Statuses.Remove(statusKey);
        });
    }
}

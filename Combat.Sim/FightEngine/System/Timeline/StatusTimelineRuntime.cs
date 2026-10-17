// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Timeline;

namespace Combat.Sim.System.Timeline;

/// <summary>声明所属容器的状态截止时间、持有到期钩子并处理单次过期，不推进时钟。</summary>
public sealed class StatusTimelineRuntime
{
    private readonly TimelineEventScope _scope;
    private readonly Func<CombatState, Dictionary<string, StatusState>> _statuses;
    private readonly Dictionary<string, Action<CombatState>> _expiryHooks = new(StringComparer.Ordinal);

    public StatusTimelineRuntime(TimelineEventScope scope = TimelineEventScope.System)
    {
        _scope = scope;
        _statuses = scope switch
        {
            TimelineEventScope.System => state => state.Statuses,
            TimelineEventScope.Job => state => state.Job.Statuses,
            _ => throw new ArgumentOutOfRangeException(nameof(scope)),
        };
    }

    /// <summary>
    /// 注册所属状态的到期处理。处理器在状态被移除之前调用，因此仍能读到即将过期的残留层数；
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

    public IEnumerable<TimelineEvent> DescribeEvents(CombatState state) => _statuses(state).Select(pair =>
        new TimelineEvent(Math.Max(state.Time, pair.Value.ExpiresAt), TimelineEventPriority.ExpirationAndCooldown,
            TimelineEventKind.StatusExpired, pair.Key, Scope: _scope));

    public TimelineMutation HandleExpired(TimelineEvent item, CombatState state)
    {
        if (item.Scope != _scope)
            throw new ArgumentException("status expiration belongs to another scope", nameof(item));

        var statusKey = item.OwnerKey!;
        if (!_statuses(state).TryGetValue(statusKey, out var status) || status.ExpiresAt > state.Time)
        {
            return TimelineMutation.Empty;
        }

        return new TimelineMutation(ApplyState: target =>
        {
            if (!_statuses(target).TryGetValue(statusKey, out var current) || current.ExpiresAt > target.Time)
            {
                return;
            }

            // 职业到期处理与状态移除必须落在同一次结算内：处理器需要读取尚未移除的残留层数。
            if (_expiryHooks.TryGetValue(statusKey, out var hook))
            {
                hook(target);
            }

            _statuses(target).Remove(statusKey);
        });
    }
}

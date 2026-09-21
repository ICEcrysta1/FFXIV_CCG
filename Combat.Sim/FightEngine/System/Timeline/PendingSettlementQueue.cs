// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Timeline;

namespace Combat.Sim.System.Timeline;

/// <summary>
/// 待结算队列：统一承接已经从资源描述里"毕业"的到期事实。
///
/// 领域模块只声明"什么时候到期"。一旦时钟越过到期点，这件事就从资源描述里毕业进入本队列，
/// 之后不再参与重新描述，因此不会被同刻其他事件触发的资源同步取消——用剩余量这类派生量去反推
/// 只能得出"没有到期事实"，会把已经成立的结算抹掉。
///
/// 队列只保存事实本身和它原有的排序键，不承载任何领域规则：结算仍由 <see cref="CombatTimelineRuntime"/>
/// 按 Kind 派发给对应处理器，并且与普通时间线事件共用同一套排序键，所以"事实落在哪个队列"
/// 不影响结算顺序。同一目标（类型 + owner + 到期时刻）只保留一条，重复登记会被忽略。
/// </summary>
public sealed class PendingSettlementQueue
{
    private readonly PriorityQueue<TimelineEvent, TimelineEventOrder> _queue = new();
    private readonly Dictionary<SettlementTarget, TimelineEvent> _byTarget = new();

    /// <summary>当前待结算事实数量。</summary>
    public int Count => _byTarget.Count;

    /// <summary>
    /// 登记一条待结算事实。同一目标已经登记时返回 false 并忽略本次登记，
    /// 避免同一个到期事实被重新描述多次而重复结算。
    /// </summary>
    public bool Enqueue(TimelineEvent settlement)
    {
        ArgumentNullException.ThrowIfNull(settlement);
        if (settlement.Sequence <= 0)
        {
            throw new ArgumentException("pending settlement must carry a positive sequence", nameof(settlement));
        }

        if (!_byTarget.TryAdd(SettlementTarget.From(settlement), settlement))
        {
            return false;
        }

        _queue.Enqueue(settlement, TimelineEventOrder.From(settlement));
        return true;
    }

    /// <summary>查看最早的一条待结算事实，不改变队列内容。</summary>
    internal bool TryPeek(out TimelineEvent settlement, out TimelineEventOrder order)
    {
        while (_queue.TryPeek(out var candidate, out var candidateOrder))
        {
            if (_byTarget.TryGetValue(SettlementTarget.From(candidate), out var current) &&
                ReferenceEquals(current, candidate))
            {
                settlement = candidate;
                order = candidateOrder;
                return true;
            }

            // 已被替换或重复入队的陈旧条目，丢弃后继续。
            _queue.Dequeue();
        }

        settlement = null!;
        order = default;
        return false;
    }

    /// <summary>取出不晚于 <paramref name="timestamp"/> 的最早一条待结算事实。</summary>
    public bool TryDequeue(double timestamp, out TimelineEvent settlement)
    {
        while (_queue.TryDequeue(out var candidate, out var order))
        {
            var target = SettlementTarget.From(candidate);
            if (!_byTarget.TryGetValue(target, out var current) || !ReferenceEquals(current, candidate))
            {
                continue;
            }

            if (order.Timestamp > timestamp)
            {
                _queue.Enqueue(candidate, order);
                settlement = null!;
                return false;
            }

            _byTarget.Remove(target);
            settlement = candidate;
            return true;
        }

        settlement = null!;
        return false;
    }

    public IReadOnlyList<TimelineEvent> Snapshot() =>
        _byTarget.Values.OrderBy(item => item, TimelineEventOrderComparer.Instance).ToArray();

    public void Restore(IEnumerable<TimelineEvent> settlements)
    {
        ArgumentNullException.ThrowIfNull(settlements);
        Clear();
        foreach (var settlement in settlements)
        {
            Enqueue(settlement);
        }
    }

    public void Clear()
    {
        _queue.Clear();
        _byTarget.Clear();
    }

    /// <summary>待结算事实的同一性：类型 + owner + 到期时刻。</summary>
    private readonly record struct SettlementTarget(
        TimelineEventKind Kind,
        string? OwnerKey,
        double DueAt)
    {
        public static SettlementTarget From(TimelineEvent settlement) =>
            new(settlement.Kind, settlement.OwnerKey, settlement.Timestamp);
    }
}

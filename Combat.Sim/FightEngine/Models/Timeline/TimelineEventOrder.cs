// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Timeline;

/// <summary>
/// 时间线派发顺序：绝对时间、稳定优先级，最后用 sequence 兜底。
/// 事件队列与待结算队列共用同一套排序键，保证"事实在哪个队列里"不影响结算顺序。
/// </summary>
internal readonly record struct TimelineEventOrder(
    double Timestamp,
    TimelineEventPriority Priority,
    long Sequence) : IComparable<TimelineEventOrder>
{
    public static TimelineEventOrder From(TimelineEvent timelineEvent) =>
        new(timelineEvent.Timestamp, timelineEvent.Priority, timelineEvent.Sequence);

    public int CompareTo(TimelineEventOrder other)
    {
        var timestampComparison = Timestamp.CompareTo(other.Timestamp);
        if (timestampComparison != 0)
        {
            return timestampComparison;
        }

        var priorityComparison = Priority.CompareTo(other.Priority);
        return priorityComparison != 0 ? priorityComparison : Sequence.CompareTo(other.Sequence);
    }
}

internal sealed class TimelineEventOrderComparer : IComparer<TimelineEvent>
{
    public static TimelineEventOrderComparer Instance { get; } = new();

    public int Compare(TimelineEvent? x, TimelineEvent? y)
    {
        if (ReferenceEquals(x, y))
        {
            return 0;
        }
        if (x is null)
        {
            return -1;
        }
        if (y is null)
        {
            return 1;
        }

        return TimelineEventOrder.From(x).CompareTo(TimelineEventOrder.From(y));
    }
}

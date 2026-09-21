// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Timeline;

namespace Combat.Sim.System.Timeline;

/// <summary>
/// 时间线内核：唯一负责移动逻辑时钟、排序并排空到期事件。
///
/// MP、冷却、Buff、DoT 和职业模块只能注册处理器，并通过
/// <see cref="TimelineMutation"/> 描述状态变更或后续事件；它们不拥有时钟和队列。
/// </summary>
public sealed class CombatTimelineRuntime
{
    public const double TimeEpsilon = 0.0000001;

    private readonly CombatState _state;
    private readonly PriorityQueue<TimelineEvent, TimelineEventOrder> _queue = new();
    private readonly Dictionary<long, TimelineEvent> _scheduledEvents = new();
    private readonly Dictionary<TimelineEventKind, Func<TimelineEvent, CombatState, TimelineMutation>> _handlers = new();
    private readonly PendingSettlementQueue _pendingSettlements = new();
    private long _nextSequence;
    private Func<CombatState, IReadOnlyList<TimelineEvent>>? _resourceEvents;
    private readonly HashSet<long> _resourceSequences = new();

    /// <summary>领域只声明绝对截止时刻；中心统一取消旧事件、注册和续排。</summary>
    internal void ConfigureResources(Func<CombatState, IReadOnlyList<TimelineEvent>> describe)
    {
        _resourceEvents = describe;
        SynchronizeResources();
    }

    private void SynchronizeResources()
    {
        if (_resourceEvents is null) return;
        _state.BindResourceTimes();

        // 先让时钟已经到达的资源事件毕业进待结算队列，再描述未来事件：资源描述对已经过去的
        // 截止时间会 clamp 到当前时刻，若不先毕业就会重新生成同一个已到期事件，与毕业出去的那份重复。
        HarvestDueResourceEvents();

        var requested = _resourceEvents(_state).ToList();
        foreach (var sequence in _resourceSequences.ToArray())
        {
            if (!_scheduledEvents.TryGetValue(sequence, out var existing))
            {
                _resourceSequences.Remove(sequence);
                continue;
            }

            var index = requested.FindIndex(e => e == (existing with { Sequence = 0 }));
            if (index >= 0)
            {
                // 描述仍然产生同一事件：保留既有 sequence，并把描述里的这一份消费掉，避免重复排程。
                requested.RemoveAt(index);
                continue;
            }

            Cancel(sequence);
            _resourceSequences.Remove(sequence);
        }

        // 描述里新出现的已到期项直接毕业，不必先排程再取消。
        for (var index = requested.Count - 1; index >= 0; index--)
        {
            if (requested[index].Timestamp > _state.Time + TimeEpsilon)
            {
                continue;
            }

            _pendingSettlements.Enqueue(requested[index] with { Sequence = NextSettlementSequence() });
            requested.RemoveAt(index);
        }

        foreach (var item in requested) _resourceSequences.Add(Schedule(item).Sequence);
    }

    /// <summary>
    /// 把时钟已经到达的资源事件毕业进待结算队列：它们脱离资源重新描述，因此不会被同刻
    /// 其他事件触发的同步取消，也不会因为重新描述把截止时刻 clamp 到当前时刻而丢掉原始精度。
    /// </summary>
    private void HarvestDueResourceEvents()
    {
        foreach (var sequence in _resourceSequences.ToArray())
        {
            if (!_scheduledEvents.TryGetValue(sequence, out var scheduled))
            {
                _resourceSequences.Remove(sequence);
                continue;
            }

            if (scheduled.Timestamp > _state.Time + TimeEpsilon)
            {
                continue;
            }

            Cancel(sequence);
            _resourceSequences.Remove(sequence);
            _pendingSettlements.Enqueue(scheduled);
        }
    }

    /// <summary>为只进待结算队列、不进事件队列的事实分配稳定 sequence。</summary>
    private long NextSettlementSequence() => _nextSequence++;

    public CombatTimelineRuntime(CombatState? initialState = null)
    {
        _state = (initialState ?? new CombatState()).Clone();
        _state.BindResourceTimes();
        ValidateTime(_state.Time);
        _nextSequence = 1;
    }

    private CombatTimelineRuntime(SimulationSnapshot snapshot,
        IReadOnlyDictionary<TimelineEventKind, Func<TimelineEvent, CombatState, TimelineMutation>> handlers)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        _state = snapshot.State.Clone();
        ValidateTime(_state.Time);
        _nextSequence = snapshot.NextSequence;

        foreach (var pendingEvent in snapshot.PendingEvents)
        {
            AddRestoredEvent(pendingEvent);
        }

        foreach (var settlement in snapshot.PendingSettlements)
        {
            ValidateTime(settlement.Timestamp);
            _pendingSettlements.Enqueue(settlement);
            if (settlement.Sequence >= _nextSequence)
            {
                _nextSequence = settlement.Sequence + 1;
            }
        }

        foreach (var pair in handlers)
        {
            _handlers[pair.Key] = pair.Value;
        }
    }

    /// <summary>当前逻辑时间。该属性没有 setter，只有 AdvanceTo 可以修改它。</summary>
    public double CurrentTime => _state.Time;

    public IReadOnlyList<TimelineEvent> PendingEvents =>
        _scheduledEvents.Values
            .OrderBy(item => item, TimelineEventOrderComparer.Instance)
            .Select(item => item.DeepClone())
            .ToArray();

    public CombatState GetState() => _state.Clone();

    /// <summary>
    /// 注册一个按事件类型分派的领域处理器。
    /// 注册只改变路由表，不推进时钟，也不写入事件队列。
    /// </summary>
    public void RegisterHandler(
        TimelineEventKind kind,
        Func<TimelineEvent, CombatState, TimelineMutation> handler)
    {
        ArgumentNullException.ThrowIfNull(handler);
        if (!_handlers.TryAdd(kind, handler))
        {
            throw new InvalidOperationException($"timeline handler already registered: {kind}");
        }
    }

    /// <summary>由中心时间线为新事件分配稳定 sequence 并入队。</summary>
    public TimelineEvent Schedule(TimelineEvent timelineEvent)
    {
        ArgumentNullException.ThrowIfNull(timelineEvent);
        ValidateTime(timelineEvent.Timestamp);
        if (timelineEvent.Timestamp < CurrentTime - TimeEpsilon)
        {
            throw new ArgumentOutOfRangeException(
                nameof(timelineEvent),
                timelineEvent.Timestamp,
                "cannot schedule an event in the past");
        }
        if (timelineEvent.Sequence != 0)
        {
            throw new ArgumentException("new events must not provide a sequence", nameof(timelineEvent));
        }

        var scheduled = timelineEvent with { Sequence = _nextSequence++ };
        AddScheduledEvent(scheduled);
        return scheduled;
    }

    /// <summary>取消尚未处理的事件；已处理或不存在的 sequence 返回 false。</summary>
    public bool Cancel(long sequence) => _scheduledEvents.Remove(sequence);

    public double? GetNextScheduledEventTime()
    {
        RemoveStaleQueueEntries();
        return _queue.TryPeek(out _, out var order) ? order.Timestamp : null;
    }

    /// <summary>
    /// 将逻辑时钟单调推进到绝对时间戳，并处理所有不晚于该时刻的事件。
    /// 这是生产代码唯一的时间推进和事件排空入口。
    /// </summary>
    public CombatState AdvanceTo(double timestamp)
    {
        ValidateTime(timestamp);
        if (timestamp < CurrentTime - TimeEpsilon)
        {
            throw new ArgumentOutOfRangeException(
                nameof(timestamp),
                timestamp,
                $"timeline cannot move backwards from {CurrentTime:R}");
        }

        TimelineEvent? previousDispatch = null;
        var repeatedDispatchCount = 0;
        while (TryDequeueNextDispatch(timestamp, out var timelineEvent))
        {
            repeatedDispatchCount = IsSameDispatch(previousDispatch, timelineEvent) ? repeatedDispatchCount + 1 : 0;
            if (repeatedDispatchCount > MaxRepeatedDispatch)
            {
                throw new InvalidOperationException(
                    $"timeline event {timelineEvent.Kind}('{timelineEvent.OwnerKey}') at {timelineEvent.Timestamp:R} "
                    + $"was dispatched more than {MaxRepeatedDispatch} times in a row; "
                    + "its handler must consume the fact or advance the deadline it declared");
            }

            previousDispatch = timelineEvent;
            _state.SetTimelineTime(Math.Max(_state.Time, timelineEvent.Timestamp));
            if (_handlers.TryGetValue(timelineEvent.Kind, out var handler))
            {
                // 处理器只能观察当前状态；所有真实写入必须通过中心应用 mutation。
                ApplyMutation(handler(timelineEvent, _state.Clone()));
            }
        }

        _state.SetTimelineTime(Math.Max(_state.Time, timestamp));
        SynchronizeResources();
        return GetState();
    }

    /// <summary>
    /// 取出下一个待派发项。待结算事实与事件队列共用同一套排序键，因此"事实落在哪个队列"
    /// 不影响结算顺序：先同步一次，把时钟已到达的资源事件毕业进待结算队列，再比较两者队首。
    /// </summary>
    private bool TryDequeueNextDispatch(double timestamp, out TimelineEvent timelineEvent)
    {
        SynchronizeResources();

        var hasSettlement = _pendingSettlements.TryPeek(out _, out var settlementOrder);
        RemoveStaleQueueEntries();
        var hasEvent = _queue.TryPeek(out _, out var eventOrder);

        if (hasSettlement && (!hasEvent || settlementOrder.CompareTo(eventOrder) <= 0))
        {
            return _pendingSettlements.TryDequeue(timestamp, out timelineEvent);
        }

        if (hasEvent)
        {
            return TryDequeueDueEvent(timestamp, out timelineEvent);
        }

        timelineEvent = null!;
        return false;
    }

    public SimulationSnapshot CreateSnapshot() => new(
        _state.Clone(),
        PendingEvents,
        _pendingSettlements.Snapshot().Select(item => item.DeepClone()).ToArray(),
        _nextSequence);

    public void RestoreSnapshot(SimulationSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        ValidateTime(snapshot.State.Time);
        if (snapshot.NextSequence <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(snapshot), "next sequence must be positive");
        }
        if (snapshot.PendingEvents.Any(item => item.Sequence <= 0))
        {
            throw new ArgumentException("snapshot events must have positive sequences", nameof(snapshot));
        }
        if (snapshot.PendingEvents.Select(item => item.Sequence).Distinct().Count() != snapshot.PendingEvents.Count)
        {
            throw new ArgumentException("snapshot event sequences must be unique", nameof(snapshot));
        }
        if (snapshot.PendingEvents.Any(item => item.Timestamp < snapshot.State.Time - TimeEpsilon))
        {
            throw new ArgumentException("snapshot cannot contain an event in the past", nameof(snapshot));
        }
        if (snapshot.PendingSettlements.Any(item => item.Sequence <= 0))
        {
            throw new ArgumentException("snapshot settlements must have positive sequences", nameof(snapshot));
        }
        if (snapshot.PendingSettlements.Any(item => item.Timestamp < snapshot.State.Time - TimeEpsilon))
        {
            throw new ArgumentException("snapshot cannot contain a settlement in the past", nameof(snapshot));
        }

        _queue.Clear();
        _scheduledEvents.Clear();
        _pendingSettlements.Clear();
        CopyState(snapshot.State, _state);
        _nextSequence = snapshot.NextSequence;
        foreach (var pendingEvent in snapshot.PendingEvents)
        {
            AddRestoredEvent(pendingEvent);
        }
        foreach (var settlement in snapshot.PendingSettlements)
        {
            _pendingSettlements.Enqueue(settlement);
            if (settlement.Sequence >= _nextSequence)
            {
                _nextSequence = settlement.Sequence + 1;
            }
        }
        _resourceSequences.Clear();
        if (_resourceEvents is not null)
        {
            var requested = _resourceEvents(_state);
            foreach (var item in PendingEvents)
                if (requested.Contains(item with { Sequence = 0 })) _resourceSequences.Add(item.Sequence);
        }
    }

    public CombatTimelineRuntime Fork()
    {
        var fork = new CombatTimelineRuntime(CreateSnapshot(), _handlers);
        fork._resourceEvents = _resourceEvents;
        fork._resourceSequences.UnionWith(_resourceSequences);
        return fork;
    }

    public void ApplyMutation(TimelineMutation? mutation)
    {
        if (mutation is null)
        {
            return;
        }

        if (mutation.ApplyState is not null)
        {
            var changed = _state.Clone();
            mutation.ApplyState(changed);
            if (changed.Time != _state.Time)
            {
                throw new InvalidOperationException(
                    "timeline mutation cannot change logical time; schedule a timeline event instead");
            }
            CopyState(changed, _state);
        }

        if (mutation.SequencesToCancel is not null)
        {
            foreach (var sequence in mutation.SequencesToCancel)
            {
                Cancel(sequence);
            }
        }

        if (mutation.EventsToSchedule is not null)
        {
            foreach (var timelineEvent in mutation.EventsToSchedule)
            {
                Schedule(timelineEvent);
            }
        }
        SynchronizeResources();
    }

    private bool TryDequeueDueEvent(double timestamp, out TimelineEvent timelineEvent)
    {
        while (_queue.TryDequeue(out var candidate, out var order))
        {
            if (!_scheduledEvents.Remove(candidate.Sequence))
            {
                continue;
            }

            if (order.Timestamp > timestamp)
            {
                AddScheduledEvent(candidate);
                timelineEvent = null!;
                return false;
            }

            timelineEvent = candidate;
            return true;
        }

        timelineEvent = null!;
        return false;
    }

    private void AddScheduledEvent(TimelineEvent timelineEvent)
    {
        if (!_scheduledEvents.TryAdd(timelineEvent.Sequence, timelineEvent))
        {
            throw new InvalidOperationException($"duplicate timeline sequence: {timelineEvent.Sequence}");
        }
        _queue.Enqueue(timelineEvent, TimelineEventOrder.From(timelineEvent));
    }

    private void AddRestoredEvent(TimelineEvent timelineEvent)
    {
        ValidateTime(timelineEvent.Timestamp);
        AddScheduledEvent(timelineEvent);
        if (timelineEvent.Sequence >= _nextSequence)
        {
            _nextSequence = timelineEvent.Sequence + 1;
        }
    }

    private void RemoveStaleQueueEntries()
    {
        while (_queue.TryPeek(out var timelineEvent, out _)
            && !_scheduledEvents.ContainsKey(timelineEvent.Sequence))
        {
            _queue.Dequeue();
        }
    }

    private static void CopyState(CombatState source, CombatState target)
    {
        var clone = source.Clone();
        target.SetTimelineTime(clone.Time);
        target.GcdIndex = clone.GcdIndex;
        target.FightEndsAt = clone.FightEndsAt;
        target.NextDowntimeStartsAt = clone.NextDowntimeStartsAt;
        target.DowntimeRemaining = clone.DowntimeRemaining;
        target.Mp = clone.Mp;
        target.MaxMp = clone.MaxMp;
        target.NaturalMpLastTickAt = clone.NaturalMpLastTickAt;
        target.CastEndsAt = clone.CastEndsAt;
        target.GcdReadyAt = clone.GcdReadyAt;
        target.WeaveEndsAt = clone.WeaveEndsAt;
        target.OgcdsWeaved = clone.OgcdsWeaved;
        target.MaxOgcdPerWindow = clone.MaxOgcdPerWindow;
        target.IsMoving = clone.IsMoving;
        target.BossTargetable = clone.BossTargetable;
        target.DowntimeEndsAt = clone.DowntimeEndsAt;
        target.TargetCount = clone.TargetCount;
        target.CumulativePotency = clone.CumulativePotency;
        target.CumulativeDotPotency = clone.CumulativeDotPotency;
        target.CurrentPotency = clone.CurrentPotency;
        target.CurrentGcdDotPotency = clone.CurrentGcdDotPotency;
        target.JobResources = clone.JobResources;
        target.JobTimelineDeadlines = clone.JobTimelineDeadlines;
        target.Cooldowns = clone.Cooldowns;
        target.Statuses = clone.Statuses;
        target.Dots = clone.Dots;
        target.History = clone.History;
    }

    private static void ValidateTime(double timestamp)
    {
        if (double.IsNaN(timestamp) || double.IsInfinity(timestamp))
        {
            throw new ArgumentOutOfRangeException(nameof(timestamp), timestamp, "timeline time must be finite");
        }
    }

    /// <summary>
    /// 同一个到期事实允许的连续派发次数。多个充能在同一时刻到点、同一刻多个 DoT 结算
    /// 这类合法场景会连续派发相同事实，但次数有界；超过上限说明处理器既没有消费这个事实、
    /// 也没有推进它声明的截止时刻，内核会因此在原地无限循环。这里报错而不是挂死。
    /// </summary>
    private const int MaxRepeatedDispatch = 5;

    private static bool IsSameDispatch(TimelineEvent? previous, TimelineEvent current) =>
        previous is not null
        && previous.Kind == current.Kind
        && string.Equals(previous.OwnerKey, current.OwnerKey, StringComparison.Ordinal)
        && Math.Abs(previous.Timestamp - current.Timestamp) <= TimeEpsilon;
}

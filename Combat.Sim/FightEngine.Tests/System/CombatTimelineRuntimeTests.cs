using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Timeline;
using Combat.Sim.System.Timeline;

namespace FightEngine.Tests.System;

/// <summary>阶段 1 时间线内核：绝对时间、稳定事件顺序和完整快照。</summary>
public sealed class CombatTimelineRuntimeTests
{
    [Fact]
    public void 单次推进与分段推进结果一致()
    {
        var direct = CreateRuntime();
        var split = CreateRuntime();

        direct.AdvanceTo(10.0);
        split.AdvanceTo(3.0);
        split.AdvanceTo(10.0);

        Assert.Equal(direct.CurrentTime, split.CurrentTime);
        Assert.Equal(direct.GetState().Mp, split.GetState().Mp);
        Assert.Equal(direct.GetState().JobResources, split.GetState().JobResources);
        Assert.Equal(direct.PendingEvents, split.PendingEvents);
    }

    [Fact]
    public void 回拨时间被拒绝且不会改变状态和队列()
    {
        var runtime = CreateRuntime();
        runtime.AdvanceTo(5.0);
        var before = runtime.CreateSnapshot();

        Assert.Throws<ArgumentOutOfRangeException>(() => runtime.AdvanceTo(4.999));

        var after = runtime.CreateSnapshot();
        Assert.Equal(before.State.Time, after.State.Time);
        Assert.Equal(before.PendingEvents, after.PendingEvents);
        Assert.Equal(before.NextSequence, after.NextSequence);
    }

    [Fact]
    public void 同时间戳按优先级再按sequence稳定处理()
    {
        var runtime = new CombatTimelineRuntime();
        var handled = new List<string>();
        runtime.RegisterHandler(
            TimelineEventKind.SceneChanged,
            (timelineEvent, _) =>
            {
                handled.Add(timelineEvent.OwnerKey!);
                return TimelineMutation.Empty;
            });

        runtime.Schedule(new TimelineEvent(2.0, TimelineEventPriority.DecisionBoundary, TimelineEventKind.SceneChanged, "late"));
        runtime.Schedule(new TimelineEvent(2.0, TimelineEventPriority.ExternalScene, TimelineEventKind.SceneChanged, "external"));
        runtime.Schedule(new TimelineEvent(2.0, TimelineEventPriority.ExternalScene, TimelineEventKind.SceneChanged, "external-2"));

        runtime.AdvanceTo(2.0);

        Assert.Equal(new[] { "external", "external-2", "late" }, handled);
    }

    [Fact]
    public void 处理器只能通过mutation安排后续事件()
    {
        var runtime = new CombatTimelineRuntime();
        runtime.RegisterHandler(
            TimelineEventKind.JobPeriodicTick,
            (timelineEvent, _) => new TimelineMutation(
                ApplyState: state => state.Mp += 100,
                EventsToSchedule: new[]
                {
                    new TimelineEvent(
                        timelineEvent.Timestamp + 1.0,
                        TimelineEventPriority.PeriodicSettlement,
                        TimelineEventKind.JobPeriodicTick),
                }));

        runtime.Schedule(new TimelineEvent(
            1.0,
            TimelineEventPriority.PeriodicSettlement,
            TimelineEventKind.JobPeriodicTick));
        runtime.AdvanceTo(2.0);

        Assert.Equal(10200, runtime.GetState().Mp);
        Assert.Equal(3.0, runtime.GetNextScheduledEventTime());
    }

    [Fact]
    public void 处理器不能直接修改真实状态时钟()
    {
        var runtime = new CombatTimelineRuntime();
        runtime.RegisterHandler(
            TimelineEventKind.DecisionBoundary,
            (_, state) =>
            {
                state.SetTimelineTime(99.0);
                return TimelineMutation.Empty;
            });
        runtime.Schedule(new TimelineEvent(
            1.0,
            TimelineEventPriority.DecisionBoundary,
            TimelineEventKind.DecisionBoundary));

        runtime.AdvanceTo(2.0);

        Assert.Equal(2.0, runtime.CurrentTime);
    }

    [Fact]
    public void mutation不能修改逻辑时钟()
    {
        var runtime = new CombatTimelineRuntime();
        runtime.RegisterHandler(
            TimelineEventKind.DecisionBoundary,
            (_, _) => new TimelineMutation(ApplyState: state => state.SetTimelineTime(99.0)));
        runtime.Schedule(new TimelineEvent(
            1.0,
            TimelineEventPriority.DecisionBoundary,
            TimelineEventKind.DecisionBoundary));

        Assert.Throws<InvalidOperationException>(() => runtime.AdvanceTo(2.0));
        Assert.Equal(1.0, runtime.CurrentTime);
    }

    [Fact]
    public void 快照恢复和fork不污染原时间线()
    {
        var runtime = CreateRuntime();
        runtime.AdvanceTo(2.0);
        var snapshot = runtime.CreateSnapshot();
        var fork = runtime.Fork();

        fork.AdvanceTo(10.0);

        Assert.Equal(2.0, runtime.CurrentTime);
        Assert.Equal(snapshot.PendingEvents, runtime.PendingEvents);
        Assert.Equal(10.0, fork.CurrentTime);

        runtime.AdvanceTo(10.0);
        Assert.Equal(runtime.GetState().Mp, fork.GetState().Mp);
        Assert.Equal(runtime.PendingEvents, fork.PendingEvents);
    }

    [Fact]
    public void 取消事件后不会被处理且不会影响后续sequence()
    {
        var runtime = new CombatTimelineRuntime();
        var handled = new List<long>();
        runtime.RegisterHandler(
            TimelineEventKind.DecisionBoundary,
            (timelineEvent, _) =>
            {
                handled.Add(timelineEvent.Sequence);
                return TimelineMutation.Empty;
            });

        var canceled = runtime.Schedule(new TimelineEvent(
            1.0,
            TimelineEventPriority.DecisionBoundary,
            TimelineEventKind.DecisionBoundary));
        var kept = runtime.Schedule(new TimelineEvent(
            2.0,
            TimelineEventPriority.DecisionBoundary,
            TimelineEventKind.DecisionBoundary));

        Assert.True(runtime.Cancel(canceled.Sequence));
        runtime.AdvanceTo(2.0);

        Assert.Equal(new[] { kept.Sequence }, handled);
        Assert.Equal(kept.Sequence + 1, runtime.CreateSnapshot().NextSequence);
    }

    private static CombatTimelineRuntime CreateRuntime()
    {
        var runtime = new CombatTimelineRuntime(new CombatState { Mp = 10000 });
        runtime.RegisterHandler(
            TimelineEventKind.MpTick,
            (_, _) => new TimelineMutation(ApplyState: state => state.Mp += 100));
        runtime.Schedule(new TimelineEvent(
            1.0,
            TimelineEventPriority.PeriodicSettlement,
            TimelineEventKind.MpTick));
        runtime.Schedule(new TimelineEvent(
            6.0,
            TimelineEventPriority.PeriodicSettlement,
            TimelineEventKind.MpTick));
        return runtime;
    }

    [Fact]
    public void 待结算队列按目标去重并在到期后取出()
    {
        var queue = new PendingSettlementQueue();
        var first = new TimelineEvent(
            5.0, TimelineEventPriority.PeriodicSettlement, TimelineEventKind.JobTimerExpired, "job.a", Sequence: 1);
        var duplicate = first with { Sequence = 2 };
        var later = new TimelineEvent(
            9.0, TimelineEventPriority.PeriodicSettlement, TimelineEventKind.JobTimerExpired, "job.b", Sequence: 3);

        Assert.True(queue.Enqueue(first));
        // 同一个到期事实被重新描述时不能重复登记，否则会重复结算。
        Assert.False(queue.Enqueue(duplicate));
        Assert.True(queue.Enqueue(later));
        Assert.Equal(2, queue.Count);

        Assert.False(queue.TryDequeue(4.0, out _));
        Assert.True(queue.TryDequeue(5.0, out var taken));
        Assert.Equal("job.a", taken.OwnerKey);
        Assert.Equal(1, queue.Count);

        var snapshot = queue.Snapshot();
        queue.Clear();
        Assert.Equal(0, queue.Count);

        queue.Restore(snapshot);
        Assert.True(queue.TryDequeue(9.0, out var restored));
        Assert.Equal("job.b", restored.OwnerKey);
    }

    [Fact]
    public void 恢复快照会带回待结算事实()
    {
        var runtime = new CombatTimelineRuntime(new CombatState { Mp = 0 });
        runtime.RegisterHandler(
            TimelineEventKind.MpTick,
            (_, _) => new TimelineMutation(ApplyState: state => state.Mp += 100));

        var snapshot = runtime.CreateSnapshot();
        var settlement = new TimelineEvent(
            0.0,
            TimelineEventPriority.PeriodicSettlement,
            TimelineEventKind.MpTick,
            OwnerKey: "restored",
            Sequence: snapshot.NextSequence);

        runtime.RestoreSnapshot(snapshot with { PendingSettlements = new[] { settlement } });
        var end = runtime.AdvanceTo(0.0);

        Assert.Equal(100, end.Mp);
        Assert.Equal(settlement.Sequence + 1, runtime.CreateSnapshot().NextSequence);
    }
}

using Combat.Sim.Facade;
using Combat.Sim.Models.Timeline;

namespace FightEngine.Tests.Facade;

public sealed class JobSimulatorTests
{
    [Fact]
    public void 同戳连续非公共技能不生成动画锁或排队等待()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        var first = simulator.SubmitAction(0.0, "ogcd_punch");
        var second = simulator.SubmitAction(0.0, "ogcd_punch");
        Assert.True(first.Accepted);
        Assert.True(second.Accepted);
        Assert.False(second.Queued);
        Assert.Equal(0.0, second.EffectTimestamp);
        Assert.Equal(2, simulator.GetState().History.Count);
        Assert.DoesNotContain(simulator.CreateSnapshot().PendingEvents,
            item => item.Kind.ToString().Contains("Animation"));
    }

    [Fact]
    public void 状态快照不能改写内部游标()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        var snapshot = simulator.GetState();
        snapshot.Mp = 1;

        Assert.Equal(FacadeKit.JobTag, simulator.JobTag);
        Assert.Equal(0.0, simulator.Time, 9);
        Assert.Equal(10000, simulator.GetState().Mp);
    }

    [Fact]
    public void 瞬发动作通过接受和效果事件原子提交()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());

        var result = simulator.SubmitAction(0.0, "gcd_strike");

        Assert.True(result.Accepted);
        Assert.False(result.Queued);
        Assert.Equal(0.0, result.AcceptedTimestamp!.Value, 9);
        Assert.Equal(0.0, result.EffectTimestamp!.Value, 9);
        Assert.Equal(2.5, simulator.GetState().GcdRemaining, 9);
        Assert.Single(simulator.GetState().History);
    }

    [Fact]
    public void 读条动作在滑步结算时生效但完整读条结束后才解锁()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        var accepted = simulator.SubmitAction(0.0, "cast_skill");

        Assert.True(accepted.Accepted);
        Assert.Equal(2.3, accepted.EffectTimestamp!.Value, 9);
        Assert.Empty(simulator.GetState().History);
        simulator.ObserveAt(2.299);
        Assert.Empty(simulator.GetState().History);
        simulator.ObserveAt(2.3);

        var history = Assert.Single(simulator.GetState().History);
        Assert.Equal(0.0, history.RequestTimestamp!.Value, 9);
        Assert.Equal(2.8, history.CastCompletedTimestamp!.Value, 9);
        Assert.Equal(2.3, history.EffectTimestamp!.Value, 9);
        Assert.Equal(accepted.ActionInstanceId, history.ActionInstanceId);
        Assert.Equal(0.5, simulator.GetState().CastRemaining, 9);

        simulator.ObserveAt(2.8);
        Assert.Equal(0.0, simulator.GetState().CastRemaining, 9);
    }

    [Fact]
    public void 两秒实际读条固定提前零点五秒结算()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        var accepted = simulator.SubmitAction(0.0, "cast_skill", actualCastSeconds: 2.0);

        Assert.True(accepted.Accepted);
        Assert.Equal(1.5, accepted.EffectTimestamp!.Value, 9);

        simulator.AdvanceTo(1.5);
        var history = Assert.Single(simulator.GetState().History);
        Assert.Equal(1.5, history.EffectTimestamp!.Value, 9);
        Assert.Equal(2.0, history.CastCompletedTimestamp!.Value, 9);
        Assert.Equal(0.5, simulator.GetState().CastRemaining, 9);
    }

    [Fact]
    public void Gcd缩放不缩放固定滑步提前量()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine(baseGcd: 2.17));
        var accepted = simulator.SubmitAction(0.0, "cast_skill");

        Assert.True(accepted.Accepted);
        Assert.Equal(2.8 * 2.17 / 2.5 - 0.5, accepted.EffectTimestamp!.Value, 9);

        simulator.AdvanceTo(accepted.EffectTimestamp!.Value);
        var history = Assert.Single(simulator.GetState().History);
        Assert.Equal(2.8 * 2.17 / 2.5, history.CastCompletedTimestamp!.Value, 9);
        Assert.Equal(0.5, simulator.GetState().CastRemaining, 9);
    }

    [Fact]
    public void 已在移动前开始的读条不会被移动事实取消()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        var accepted = simulator.SubmitAction(0.0, "cast_skill", actualCastSeconds: 2.0);

        Assert.True(accepted.Accepted);
        simulator.ApplyExternalEvent(new ExternalCombatEvent(
            1.6,
            ExternalCombatEventKinds.MovementChanged,
            true));

        simulator.AdvanceTo(1.6);
        Assert.Single(simulator.GetState().History);
        Assert.True(simulator.GetState().IsMoving);

        simulator.AdvanceTo(2.5);
        Assert.Equal("movement_locked", simulator.ValidateActionAt(2.5, "cast_skill").Reason);
    }

    [Fact]
    public void 读条期间拒绝插入动作且末尾允许容量一排队()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        Assert.True(simulator.SubmitAction(0.0, "cast_skill").Accepted);

        var duringCast = simulator.SubmitAction(1.0, "ogcd_punch");
        Assert.False(duringCast.Accepted);
        Assert.Equal("cast_locked", duringCast.Reason);
        Assert.Empty(simulator.GetState().History);

        var queued = simulator.SubmitAction(2.76, "gcd_strike");
        Assert.True(queued.Accepted);
        Assert.True(queued.Queued);
        Assert.Equal(2.8, queued.AcceptedTimestamp!.Value, 9);
        Assert.Equal("action_queue_occupied", simulator.SubmitAction(2.77, "gcd_strike").Reason);
        Assert.Single(simulator.GetState().History);

        simulator.AdvanceTo(2.8);
        var state = simulator.GetState();
        Assert.Equal(2, state.History.Count);
        Assert.Equal("cast_skill", state.History[0].SkillKey);
        Assert.Equal("gcd_strike", state.History[1].SkillKey);
        Assert.Equal(2.46, state.GcdRemaining, 9);
    }

    [Fact]
    public void 容量一队列在锁结束时自动执行并随快照恢复()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        Assert.True(simulator.SubmitAction(0.0, "gcd_strike").Accepted);

        var queued = simulator.SubmitAction(2.46, "gcd_strike");
        Assert.True(queued.Accepted);
        Assert.True(queued.Queued);
        Assert.Equal(2.5, queued.AcceptedTimestamp!.Value, 9);
        Assert.Equal("action_queue_occupied", simulator.SubmitAction(2.47, "gcd_strike").Reason);

        var snapshot = simulator.CreateSnapshot();
        var fork = simulator.Fork();
        simulator.AdvanceTo(2.5);
        fork.AdvanceTo(2.5);
        Assert.Equal(2, simulator.GetState().History.Count);
        Assert.Equal(2, fork.GetState().History.Count);
        Assert.Equal(2.46, simulator.GetState().GcdRemaining, 9);
        Assert.Equal(2.5, simulator.GetState().WeaveWindowRemaining, 9);

        simulator.RestoreSnapshot(snapshot);
        simulator.AdvanceTo(2.5);
        Assert.Equal(2, simulator.GetState().History.Count);
    }

    [Fact]
    public void QueuedGcdDoesNotBlockImmediateOgcd()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        Assert.True(simulator.SubmitAction(0.0, "gcd_strike").Accepted);

        var queued = simulator.SubmitAction(2.1, "gcd_strike");
        Assert.True(queued.Accepted);
        Assert.True(queued.Queued);
        Assert.Equal(2.5, queued.AcceptedTimestamp!.Value, 9);

        var ogcd = simulator.SubmitAction(2.2, "ogcd_punch");
        Assert.True(ogcd.Accepted);
        Assert.False(ogcd.Queued);
        Assert.Equal(2.2, ogcd.AcceptedTimestamp!.Value, 9);
    }

    [Fact]
    public void 冷却队列在回充事实结算后消耗新充能()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        Assert.True(simulator.SubmitAction(0.0, "cd_skill").Accepted);
        simulator.AdvanceTo(29.97);

        var queued = simulator.SubmitAction(29.97, "cd_skill");
        Assert.True(queued.Accepted);
        Assert.True(queued.Queued);
        Assert.Equal(30.0, queued.AcceptedTimestamp!.Value, 9);

        simulator.AdvanceTo(30.0);
        var state = simulator.GetState();
        Assert.Equal(2, state.History.Count);
        Assert.Equal("cooldown_locked", simulator.ValidateActionAt(32.5, "cd_skill").Reason);
        Assert.Equal(27.5, simulator.GetState().Cooldowns["cd_skill"].RechargeTimers.Single(), 9);
    }

    [Fact]
    public void 超出队列窗口的动作保持拒绝且不安排接受事件()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        Assert.True(simulator.SubmitAction(0.0, "gcd_strike").Accepted);

        var rejected = simulator.SubmitAction(1.99, "gcd_strike");

        Assert.False(rejected.Accepted);
        Assert.Equal("gcd_locked", rejected.Reason);
        Assert.DoesNotContain(simulator.CreateSnapshot().PendingEvents,
            item => item.Kind == TimelineEventKind.ActionAccepted);
    }

    [Fact]
    public void 队列不会绕过职业资源校验()
    {
        var root = FindRepoRoot();
        var machine = CombatStateMachine.FromDefaultConfig(root, "black_mage");
        var state = machine.InitialState();
        state.GcdRemaining = 0.04;
        var simulator = new JobSimulator(machine, state);

        var result = simulator.SubmitAction(0.0, "xenoglossy");

        Assert.False(result.Accepted);
        Assert.Equal("requires_polyglot", result.Reason);
        Assert.Empty(simulator.CreateSnapshot().PendingEvents
            .Where(item => item.Kind == TimelineEventKind.ActionAccepted));
    }

    [Fact]
    public void 候选预演从完整时间线分支且不污染主状态()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        var before = simulator.CreateSnapshot();

        var candidates = simulator.BuildCandidatePreviews();

        var gcd = candidates.Single(item => item.Skill.Key == "gcd_strike");
        Assert.True(gcd.IsLegal);
        Assert.NotNull(gcd.CandidateAfterState);
        Assert.DoesNotContain(candidates, item => item.Skill.Key == "ogcd_wait");
        var after = simulator.CreateSnapshot();
        Assert.Equal(before.State.Time, after.State.Time);
        Assert.Equal(before.State.History.Count, after.State.History.Count);
        Assert.Equal(before.NextSequence, after.NextSequence);
        Assert.Equal(before.PendingEvents.Select(item => (item.Timestamp, item.Kind, item.Sequence)),
            after.PendingEvents.Select(item => (item.Timestamp, item.Kind, item.Sequence)));
    }

    [Fact]
    public void 外部事实按时间戳进入同一时间线且分支隔离()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        simulator.ApplyExternalEvent(new ExternalCombatEvent(
            1.0,
            ExternalCombatEventKinds.BossTargetableChanged,
            false));

        var fork = simulator.Fork();
        fork.ApplyExternalEvent(new ExternalCombatEvent(
            2.0,
            ExternalCombatEventKinds.BossTargetableChanged,
            true));

        Assert.True(fork.ValidateActionAt(2.0, "gcd_strike").Ok);
        Assert.Equal("boss_untargetable", simulator.ValidateActionAt(2.0, "gcd_strike").Reason);
    }

    [Fact]
    public void 外部事实覆盖移动和目标数且非法载荷不推进()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        simulator.ApplyExternalEvent(new ExternalCombatEvent(
            1.0,
            ExternalCombatEventKinds.MovementChanged,
            true));
        simulator.ApplyExternalEvent(new ExternalCombatEvent(
            1.0,
            ExternalCombatEventKinds.TargetCountChanged,
            TargetCount: 3));
        var state = simulator.GetState();
        Assert.True(state.IsMoving);
        Assert.Equal(3, state.TargetCount);

        Assert.Throws<ArgumentException>(() => simulator.ApplyExternalEvent(new ExternalCombatEvent(
            2.0,
            ExternalCombatEventKinds.TargetCountChanged,
            TargetCount: -1)));
        Assert.Equal(1.0, simulator.Time, 9);
    }

    [Fact]
    public void 团辅外部事实使用注册状态并按绝对时间到期()
    {
        var simulator = JobSimulator.Create(FindRepoRoot(), "black_mage");
        simulator.ApplyExternalEvent(new ExternalCombatEvent(
            1.0,
            ExternalCombatEventKinds.RaidBuffWindowChanged,
            true,
            RemainingSeconds: 5.0));

        Assert.Equal(5.0, simulator.GetState().Statuses["raid_buff_window"].Remaining, 9);

        simulator.AdvanceTo(6.0);
        Assert.False(simulator.GetState().HasStatus("raid_buff_window"));
    }

    [Fact]
    public void 未知动作和非法外部事实不会先推进时钟()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());

        Assert.Throws<KeyNotFoundException>(() => simulator.SubmitAction(3.0, "missing_action"));
        Assert.Equal(0.0, simulator.Time, 9);

        Assert.Throws<KeyNotFoundException>(() => simulator.ValidateActionAt(4.0, "missing_action"));
        Assert.Equal(0.0, simulator.Time, 9);

        Assert.Throws<ArgumentException>(() => simulator.ApplyExternalEvent(
            new ExternalCombatEvent(5.0, "unsupported")));
        Assert.Equal(0.0, simulator.Time, 9);
    }

    private static string FindRepoRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
            if (File.Exists(Path.Combine(dir.FullName, "config", "default.yaml"))) return dir.FullName;
        throw new InvalidOperationException("repo root not found");
    }
}

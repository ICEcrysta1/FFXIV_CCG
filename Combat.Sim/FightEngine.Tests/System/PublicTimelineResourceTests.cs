using System.Text.Json;
using Combat.Sim.Facade;
using Combat.Sim.Jobs;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.Models.Timeline;
using Combat.Sim.System;
using Combat.Sim.System.Timeline;
using static FightEngine.Tests.System.SystemTestKit;

namespace FightEngine.Tests.System;

/// <summary>公共资源迁移：真实领域事件、截止时间、刷新和快照确定性。</summary>
public sealed class PublicTimelineResourceTests
{
    private static SystemStateMachine CreateSystem() => new(Timing, Potency, MpRecovery);

    private static CombatTimelineRuntime CreateTimeline(SystemStateMachine system, CombatState state) =>
        system.CreateTimeline(state, _ => 2);

    private static void AssertSame(CombatTimelineRuntime expected, CombatTimelineRuntime actual)
    {
        Assert.Equal(JsonSerializer.Serialize(expected.GetState()), JsonSerializer.Serialize(actual.GetState()));
        Assert.Equal(expected.PendingEvents, actual.PendingEvents);
        Assert.Equal(expected.CreateSnapshot().NextSequence, actual.CreateSnapshot().NextSequence);
    }

    [Fact]
    public void 公共资源任意分段推进与一次推进逐字段一致()
    {
        var system = CreateSystem();
        var state = new CombatState { Mp = 0, GcdRemaining = 2.5,
            WeaveWindowRemaining = 2, OgcdsWeaved = 1, NextDowntimeEta = 5, DowntimeRemaining = 2 };
        state.Statuses["buff"] = new StatusState(8);
        state.Dots["dot"] = new DotState(12, 100);
        system.ConsumeCooldown(state, Skill("charge", cooldown: 5, charges: 2));
        system.ConsumeCooldown(state, Skill("charge", cooldown: 5, charges: 2));
        var direct = CreateTimeline(system, state);
        var split = direct.Fork();
        direct.AdvanceTo(15);
        foreach (var time in new[] { 0.1, 0.4, 2.5, 3, 4.999, 5, 6, 7, 8, 9, 12, 15 })
            split.AdvanceTo(time);
        AssertSame(direct, split);
        Assert.Equal(1000, direct.GetState().Mp);
        Assert.Equal(300, direct.GetState().CumulativeDotPotency);
        Assert.Empty(direct.GetState().Cooldowns);
        Assert.Empty(direct.GetState().Statuses);
        Assert.Empty(direct.GetState().Dots);
    }

    [Fact]
    public void 公共资源快照恢复和fork保留截止时间及事件编号()
    {
        var system = CreateSystem();
        var state = new CombatState { Mp = 0, NextDowntimeEta = 2, DowntimeRemaining = 5 };
        state.Statuses["buff"] = new StatusState(10);
        state.Dots["dot"] = new DotState(12, 100);
        system.ConsumeCooldown(state, Skill("charge", cooldown: 10, charges: 2));
        var timeline = CreateTimeline(system, state);
        timeline.AdvanceTo(3);
        var saved = timeline.CreateSnapshot();
        var fork = timeline.Fork();
        fork.AdvanceTo(12);
        Assert.Equal(3, timeline.CurrentTime);
        timeline.AdvanceTo(20);
        timeline.RestoreSnapshot(saved);
        timeline.AdvanceTo(12);
        AssertSame(fork, timeline);
    }

    [Fact]
    public void 刷新状态取消旧过期事件并在新时刻过期()
    {
        var state = new CombatState();
        state.Statuses["buff"] = new StatusState(5);
        var timeline = CreateTimeline(CreateSystem(), state);
        var old = Assert.Single(timeline.PendingEvents.Where(e => e.Kind == TimelineEventKind.StatusExpired));
        timeline.AdvanceTo(2);
        timeline.ApplyMutation(new(ApplyState: target => target.Statuses["buff"] = new StatusState(10)));
        Assert.DoesNotContain(timeline.PendingEvents, e => e.Sequence == old.Sequence);
        Assert.Equal(12, timeline.GetState().Statuses["buff"].ExpiresAt);
        Assert.True(timeline.AdvanceTo(5).HasStatus("buff"));
        Assert.False(timeline.AdvanceTo(12).HasStatus("buff"));
    }

    [Fact]
    public void 主动缩短冷却取消旧事件且不重复回充()
    {
        var system = CreateSystem();
        var skill = Skill("charge", cooldown: 10, charges: 2);
        var state = new CombatState();
        system.ConsumeCooldown(state, skill);
        var timeline = CreateTimeline(system, state);
        timeline.AdvanceTo(2);
        timeline.ApplyMutation(new(ApplyState: target => system.ApplyCooldownReduction(target, skill, 5)));
        Assert.DoesNotContain(timeline.PendingEvents, e => e.Kind == TimelineEventKind.CooldownChargeReady && e.Timestamp == 10);
        Assert.Equal(1, timeline.AdvanceTo(4.999).Cooldowns["charge"].AvailableCharges);
        Assert.Empty(timeline.AdvanceTo(5).Cooldowns);
        timeline.ApplyMutation(new(ApplyState: target => system.ConsumeCooldown(target, skill)));
        Assert.Equal(1, timeline.AdvanceTo(10).Cooldowns["charge"].AvailableCharges);
        Assert.Empty(timeline.AdvanceTo(15).Cooldowns);
    }

    [Fact]
    public void 星灵移位五秒前严格禁止再次使用()
    {
        var machine = CombatStateMachine.FromDefaultConfig(Combat.Sim.Common.RepoRootLocator.Find(), "black_mage");
        var state = machine.InitialState();
        state.SetJobResource("astral_fire", 1);
        state = TimelineTestDriver.Execute(machine, state, "transpose").NextState;
        foreach (var elapsed in new[] { 1.0, 2.0, 1.95, 0.049999 })
        {
            state = TimelineTestDriver.AdvanceBy(machine, state, elapsed);
            Assert.Equal("cooldown_locked", machine.ValidateAction(state, "transpose").Reason);
        }
        state = TimelineTestDriver.AdvanceBy(machine, state, 5 - state.Time);
        Assert.True(machine.ValidateAction(state, "transpose").Ok);
    }

    [Theory]
    [InlineData(0, 0, 400)]
    [InlineData(3, 0, 0)]
    [InlineData(0, 3, 400)]
    public void 黑魔MP修正在精确周期生效(int fire, int ice, int expectedMp)
    {
        var machine = CombatStateMachine.FromDefaultConfig(Combat.Sim.Common.RepoRootLocator.Find(), "black_mage");
        var state = machine.InitialState();
        state.Mp = 0;
        state.SetJobResource("astral_fire", fire);
        state.SetJobResource("umbral_ice", ice);
        var direct = TimelineTestDriver.AdvanceBy(machine, state, 6);
        var split = TimelineTestDriver.AdvanceBy(machine, TimelineTestDriver.AdvanceBy(machine, state, 2), 4);
        Assert.Equal(expectedMp, direct.Mp);
        Assert.Equal(direct.Mp, split.Mp);
        Assert.Equal(direct.NaturalMpTickProgress, split.NaturalMpTickProgress);
    }

    [Fact]
    public void 刷新DoT取消旧tick并按新持续时间结算()
    {
        var state = new CombatState();
        state.Dots["dot"] = new DotState(6, 100);
        var timeline = CreateTimeline(CreateSystem(), state);
        timeline.AdvanceTo(2);
        timeline.ApplyMutation(new(ApplyState: target => target.Dots["dot"] = new DotState(6, 200)));
        Assert.Equal(0, timeline.AdvanceTo(3).CumulativeDotPotency);
        Assert.Equal(200, timeline.AdvanceTo(5).CumulativeDotPotency);
        var end = timeline.AdvanceTo(8);
        Assert.Equal(400, end.CumulativeDotPotency);
        Assert.Empty(end.Dots);
    }

    [Fact]
    public void 时间容差不会提前结算未来事件()
    {
        var state = new CombatState();
        state.Statuses["buff"] = new StatusState(5);
        var timeline = CreateTimeline(CreateSystem(), state);
        var observed = timeline.AdvanceTo(5 - CombatTimelineRuntime.TimeEpsilon / 2);
        Assert.True(observed.HasStatus("buff"));
        Assert.True(observed.Time < 5);
        Assert.False(timeline.AdvanceTo(5).HasStatus("buff"));
    }

    [Fact]
    public void 同刻场景先切换然后结算DoT并在到期时清理()
    {
        var state = new CombatState { NextDowntimeEta = 3, DowntimeRemaining = 3 };
        state.Dots["dot"] = new DotState(6, 100);
        var timeline = CreateTimeline(CreateSystem(), state);
        Assert.Equal(0, timeline.AdvanceTo(3).CumulativeDotPotency);
        var end = timeline.AdvanceTo(6);
        Assert.True(end.BossTargetable);
        Assert.Equal(100, end.CumulativeDotPotency);
        Assert.Empty(end.Dots);
    }

    [Fact]
    public void MP回调读取精确tick状态且过期边界不再获得Buff加成()
    {
        var state = new CombatState { Mp = 0 };
        state.Statuses["bonus"] = new StatusState(6);
        var observations = new List<(double Time, double Remaining)>();
        var system = CreateSystem();
        system.JobTimeline.RegisterMpTickModifier((tickState, amount) =>
        {
            var remaining = tickState.Statuses.TryGetValue("bonus", out var buff) ? buff.Remaining : 0;
            observations.Add((tickState.Time, remaining));
            return amount + (tickState.HasStatus("bonus") ? 100 : 0);
        });
        var timeline = system.CreateTimeline(state, _ => 1);
        var end = timeline.AdvanceTo(9);
        Assert.Equal(new[] { (3.0, 3.0), (6.0, 0.0), (9.0, 0.0) }, observations);
        Assert.Equal(700, end.Mp);
        Assert.False(end.HasStatus("bonus"));
    }

    [Fact]
    public void 观测剩余时间不改变绝对截止时刻()
    {
        var state = new CombatState { GcdRemaining = 5 };
        state.Statuses["buff"] = new StatusState(10);
        state.Dots["dot"] = new DotState(12, 100);
        var timeline = CreateTimeline(CreateSystem(), state);
        var observed = timeline.AdvanceTo(1);
        Assert.Equal(5, observed.GcdReadyAt);
        Assert.Equal(4, observed.GcdRemaining);
        Assert.Equal(10, observed.Statuses["buff"].ExpiresAt);
        Assert.Equal(9, observed.Statuses["buff"].Remaining);
        Assert.Equal(3, observed.Dots["dot"].NextTickAt);
        Assert.Equal(2, observed.Dots["dot"].NextTickInSeconds);
    }

    [Fact]
    public void 公共领域不再提供秒数推进入口()
    {
        foreach (var type in new[] { typeof(PlayerStateRuntime), typeof(CooldownRuntime),
            typeof(StatusTimelineRuntime), typeof(DotTimelineRuntime), typeof(MpRecoveryRuntime),
            typeof(JobTimelineRegistry) })
            Assert.DoesNotContain(type.GetMethods(), method => method.Name is "AdvanceTime" or "AdvanceTo");
        Assert.DoesNotContain(typeof(IJobStateMachine).GetMethods(), method =>
            method.Name is "AdvanceTime" or "AdvanceTimeBeforeSystem" or "ResolveMpRecoveryAmount");
    }

    [Fact]
    public void 职业注册的周期和倒计时由公共时间线驱动()
    {
        var system = CreateSystem();
        system.JobTimeline.RegisterPeriodicResource(
            "test.periodic",
            2.0,
            "periodic_elapsed",
            state => state.JobTimelineDeadlines.TryGetValue("test.periodic", out var deadline) ? deadline : null,
            (item, state) => new TimelineMutation(ApplyState: target =>
            {
                target.SetJobResource(
                    "periodic_ticks",
                    Convert.ToInt32(target.GetJobResource("periodic_ticks", 0)) + 1);
                // 结算后把下一次结算推进一个完整周期，由中心据此续排。
                target.JobTimelineDeadlines["test.periodic"] = item.Timestamp + 2.0;
            }));
        system.JobTimeline.RegisterCountdown(
            "test.countdown",
            "countdown_remaining",
            state => state.JobTimelineDeadlines.TryGetValue("test.countdown", out var deadline) ? deadline : null,
            (_, state, _) => new TimelineMutation(ApplyState: target =>
            {
                target.SetJobResource("countdown_expired", true);
                target.JobTimelineDeadlines.Remove("test.countdown");
            }));

        var state = new CombatState();
        state.JobTimelineDeadlines["test.periodic"] = 2.0;
        state.JobTimelineDeadlines["test.countdown"] = 5.0;
        var timeline = system.CreateTimeline(state, _ => 1);

        var end = timeline.AdvanceTo(7.0);

        Assert.Equal(3, Convert.ToInt32(end.GetJobResource("periodic_ticks")));
        Assert.True(Convert.ToBoolean(end.GetJobResource("countdown_expired")));
        Assert.Equal(0.0, Convert.ToDouble(end.GetJobResource("countdown_remaining")));
    }

    [Fact]
    public void 职业周期资源的结算时刻不依赖推进被切成几段()
    {
        // 周期资源由绝对截止时刻驱动：无论时钟一次推进还是分成多少段，
        // 结算时刻都必须由"起点 + 周期"唯一确定，而不是由推进的切分方式决定。
        static (int Ticks, List<double> FireTimes) Run(IEnumerable<double> advancePlan)
        {
            var system = CreateSystem();
            var firedAt = new List<double>();
            system.JobTimeline.RegisterPeriodicResource(
                "test.periodic",
                2.0,
                "periodic_view",
                state => state.JobTimelineDeadlines.TryGetValue("test.periodic", out var deadline) ? deadline : null,
                (item, state) => new TimelineMutation(ApplyState: target =>
                {
                    firedAt.Add(item.Timestamp);
                    target.JobTimelineDeadlines["test.periodic"] = item.Timestamp + 2.0;
                }));

            var state = new CombatState();
            state.JobTimelineDeadlines["test.periodic"] = 2.0;
            var timeline = system.CreateTimeline(state, _ => 1);
            foreach (var seconds in advancePlan)
            {
                timeline.AdvanceTo(timeline.CurrentTime + seconds);
            }

            return (firedAt.Count, firedAt);
        }

        var direct = Run(new[] { 10.0 });
        var split = Run(new[] { 0.5, 4.5, 5.0 });
        var fine = Run(Enumerable.Repeat(0.25, 40));

        Assert.Equal(5, direct.Ticks);
        Assert.Equal(direct.FireTimes, split.FireTimes);
        Assert.Equal(direct.FireTimes, fine.FireTimes);
    }

    [Fact]
    public void 职业倒计时与状态同刻到期时结算不会被取消()
    {
        var system = CreateSystem();
        var fired = 0;
        system.JobTimeline.RegisterCountdown(
            "test.countdown",
            "countdown_remaining",
            state => state.JobTimelineDeadlines.TryGetValue("test.countdown", out var deadline) ? deadline : null,
            (_, _, _) => new TimelineMutation(ApplyState: target =>
            {
                // 处理器消费这次到期事实：不清除截止时刻内核会重复结算。
                target.JobTimelineDeadlines.Remove("test.countdown");
                fired++;
            }));

        var state = new CombatState();
        state.JobTimelineDeadlines["test.countdown"] = 5.0;
        // 让一个状态与倒计时在同一时刻过期：状态过期会触发资源重新同步，
        // 描述不再产出到期事实，稍晚派发就会被取消掉。
        state.Statuses["buff"] = new StatusState(5);
        var timeline = system.CreateTimeline(state, _ => 1);

        var end = timeline.AdvanceTo(5);

        Assert.Equal(1, fired);
        Assert.False(end.HasStatus("buff"));
    }

    [Fact]
    public void 职业状态到期处理在状态移除前执行且同刻只结算一次()
    {
        var system = CreateSystem();
        var observedStacks = new List<int>();
        system.StatusTimeline.RegisterExpiryHook("buff", state => observedStacks.Add(
            state.Statuses.TryGetValue("buff", out var expired) ? expired.Stacks : -1));
        var state = new CombatState();
        state.Statuses["buff"] = new StatusState(5, stacks: 3);
        var timeline = system.CreateTimeline(state, _ => 1);

        var end = timeline.AdvanceTo(5);

        // 处理器必须还能读到尚未移除的层数（-1 表示状态已经被提前移除），
        // 且不能因为时间线分段或重复同步而累计两次。
        Assert.Equal(new[] { 3 }, observedStacks);
        Assert.False(end.HasStatus("buff"));
        Assert.Empty(end.Statuses);
    }

    [Fact]
    public void 状态到期钩子重复注册直接报错()
    {
        var system = CreateSystem();
        system.StatusTimeline.RegisterExpiryHook("buff", _ => { });

        Assert.Throws<InvalidOperationException>(() => system.StatusTimeline.RegisterExpiryHook("buff", _ => { }));
    }

    [Fact]
    public void 职业时间资源注册键跨类型唯一且禁止重复MP修正()
    {
        var registry = new JobTimelineRegistry();
        registry.RegisterPeriodicResource(
            "job.timer",
            1.0,
            "job_timer_view",
            _ => null,
            (_, _) => TimelineMutation.Empty);

        Assert.Throws<InvalidOperationException>(() => registry.RegisterCountdown(
            "job.timer",
            "job_timer_view",
            _ => null,
            (_, _, _) => TimelineMutation.Empty));

        registry.RegisterMpTickModifier((_, amount) => amount);
        Assert.Throws<InvalidOperationException>(() => registry.RegisterMpTickModifier((_, amount) => amount));
    }

    [Fact]
    public void 时间原点只能由时间线设置且正式推进消耗剩余量()
    {
        var state = new CombatState { Mp = 0 };
        state.SetTimelineTime(-3);
        state.NaturalMpTickProgress = 0;
        state.GcdRemaining = 2;
        state.Statuses["buff"] = new StatusState(5);
        state.Dots["dot"] = new DotState(6, 100);
        var system = CreateSystem();
        system.ConsumeCooldown(state, Skill("charge", cooldown: 5, charges: 2));
        Assert.Equal(2, state.GcdRemaining);
        Assert.Equal(5, state.Statuses["buff"].Remaining);
        Assert.Equal(5, state.Cooldowns["charge"].RechargeTimers[0]);
        var timeline = CreateTimeline(system, state);
        var end = timeline.AdvanceTo(0);
        Assert.Equal(0, end.GcdRemaining);
        Assert.Equal(2, end.Statuses["buff"].Remaining);
        Assert.Equal(2, end.Cooldowns["charge"].RechargeTimers[0]);
        Assert.Equal(200, end.Mp);
        Assert.Equal(100, end.CumulativeDotPotency);
    }

    [Fact]
    public void 公开时间线不持有调用方状态别名且拒绝mutation改时钟()
    {
        var source = new CombatState();
        var timeline = CreateTimeline(CreateSystem(), source);
        source.SetTimelineTime(99);
        Assert.Equal(0, timeline.CurrentTime);
        var saved = timeline.CreateSnapshot();
        Assert.Throws<InvalidOperationException>(() => timeline.ApplyMutation(new(ApplyState: target =>
        {
            target.Mp = 0;
            target.SetTimelineTime(target.Time + CombatTimelineRuntime.TimeEpsilon / 2);
        })));
        Assert.Equal(saved.State.Mp, timeline.GetState().Mp);
        Assert.Equal(saved.PendingEvents, timeline.PendingEvents);
        Assert.Equal(saved.NextSequence, timeline.CreateSnapshot().NextSequence);
        Assert.Equal(0, timeline.CurrentTime);
    }
}

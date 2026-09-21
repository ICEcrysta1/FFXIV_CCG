using Combat.Sim.Config;
using Combat.Sim.Facade;
using Combat.Sim.Models.Timeline;
using Combat.Sim.Models.Definitions;
using FightEngine.Tests.System;
using Xunit;

namespace FightEngine.Tests.Facade;

public class CombatStateMachineTests
{
    [Fact]
    public void 未注册职业标签抛错()
    {
        var config = FacadeKit.BuildProjectConfig();
        var ex = Assert.Throws<InvalidOperationException>(() => new CombatStateMachine(config, "unknown_job"));
        Assert.Contains("unsupported job tag: unknown_job", ex.Message);
    }

    [Fact]
    public void 配置职业与请求标签不匹配抛错()
    {
        var config = FacadeKit.BuildProjectConfig();
        config = config with { Job = config.Job with { Key = "other_job" } };
        var ex = Assert.Throws<InvalidOperationException>(() => new CombatStateMachine(config, FacadeKit.JobTag));
        Assert.Contains("loaded config job does not match requested tag", ex.Message);
    }

    [Fact]
    public void 技能行为无归属抛错()
    {
        var config = FacadeKit.BuildProjectConfig();
        var mystery = SystemTestKit.Skill(
            "mystery_skill",
            ActionKind.Gcd,
            behavior: "no_owner",
            gameId: 98);
        config = config with { Job = config.Job with { Skills = config.Job.Skills.Append(mystery).ToList() } };
        var ex = Assert.Throws<InvalidOperationException>(() => new CombatStateMachine(config, FacadeKit.JobTag));
        Assert.Contains("unsupported behavior for skill mystery_skill", ex.Message);
    }

    [Fact]
    public void 初始状态使用职业默认战斗剩余与引擎最大蓝量()
    {
        var machine = FacadeKit.BuildMachine();
        var state = machine.InitialState();
        Assert.Equal(600.0, state.FightRemaining);
        Assert.Equal(10000, state.Mp);
        Assert.Equal(10000, state.MaxMp);
        Assert.Equal(0.0, state.Time);
        Assert.Equal(1, state.TargetCount);

        var custom = machine.InitialState(fightRemaining: 120.0, maxMp: 5000);
        Assert.Equal(120.0, custom.FightRemaining);
        Assert.Equal(5000, custom.Mp);
    }

    [Fact]
    public void 完整决策循环_时序与历史()
    {
        var machine = FacadeKit.BuildMachine();
        var state = machine.InitialState();

        // 瞬发 GCD：占用 0，GCD 窗口 = 基准 GCD 2.5，下一窗口 = max(0, 2.5)
        var gcdResult = TimelineTestDriver.Execute(machine, state, "gcd_strike");
        Assert.True(gcdResult.Validation.Ok);
        Assert.Equal(0.0, gcdResult.ActualOccupancySeconds, 9);
        Assert.Equal("instant", gcdResult.ActualOccupancySource);
        Assert.Equal(2.5, gcdResult.GcdWindowSeconds, 9);
        Assert.Equal(2.5, gcdResult.NextGcdWindowSeconds, 9);
        Assert.Equal(1, gcdResult.NextState.GcdIndex);
        Assert.Equal(2.5, gcdResult.NextState.GcdRemaining, 9);
        Assert.Equal(300.0, gcdResult.NextState.CurrentPotency, 9);
        Assert.Single(gcdResult.NextState.History);
        state = gcdResult.NextState;

        // GCD 锁定中再次 GCD 非法（严格模式抛错）
        Assert.Equal("gcd_locked", machine.ValidateAction(state, "gcd_strike").Reason);
        Assert.Throws<InvalidOperationException>(() => TimelineTestDriver.Execute(machine, state, "gcd_strike"));

        // 推进到 GCD 窗口结束
        state = TimelineTestDriver.AdvanceBy(machine, state, 2.5);
        Assert.Equal(0.0, state.GcdRemaining, 9);
        Assert.Equal(2.5, state.Time, 9);

        // oGCD 技能即时生效，调用方负责安排操作间隔。
        var ogcdResult = TimelineTestDriver.Execute(machine, state, "ogcd_punch");
        Assert.True(ogcdResult.Validation.Ok);
        Assert.Equal(0.0, ogcdResult.ActualOccupancySeconds, 9);
        Assert.Equal("instant", ogcdResult.ActualOccupancySource);
        Assert.Equal(200.0, ogcdResult.NextState.CurrentPotency, 9);
        state = ogcdResult.NextState;

        // 状态机不模拟动画锁，同戳 oGCD 不因动画间隔被拒绝。
        Assert.True(machine.ValidateAction(state, "ogcd_punch").Ok);

        // 读条技能：实际读条 = cast_time 2.8（基准 2.5 等比），下一窗口 = max(2.8, 2.5)
        state = TimelineTestDriver.AdvanceBy(machine, state, 0.4);
        var castResult = TimelineTestDriver.Execute(machine, state, "cast_skill");
        Assert.True(castResult.Validation.Ok);
        Assert.Equal(2.8, castResult.ActualOccupancySeconds, 9);
        Assert.Equal("cast_time", castResult.ActualOccupancySource);
        Assert.Equal(1.12, castResult.ActualOccupancyGcd, 9);
        Assert.Equal(5.2, castResult.NextState.Time, 9);
        Assert.Equal(0.2, castResult.NextState.GcdRemaining, 9);
        Assert.Equal(0.5, castResult.NextState.CastRemaining, 9);
        Assert.Equal(500.0, castResult.NextState.CurrentPotency, 9);
    }

    [Fact]
    public void 冷却技能按冷却时间锁定期限()
    {
        var machine = FacadeKit.BuildMachine();
        var state = machine.InitialState();

        Assert.True(machine.ValidateAction(state, "cd_skill").Ok);
        state = TimelineTestDriver.Execute(machine, state, "cd_skill").NextState;

        // GCD 锁定优先于冷却报告
        Assert.Equal("gcd_locked", machine.ValidateAction(state, "cd_skill").Reason);

        // 推进 GCD 后冷却锁定
        state = TimelineTestDriver.AdvanceBy(machine, state, 2.5);
        Assert.Equal("cooldown_locked", machine.ValidateAction(state, "cd_skill").Reason);

        // 再推进 27.4 秒（共 29.9 秒）仍锁定
        state = TimelineTestDriver.AdvanceBy(machine, state, 27.4);
        Assert.Equal("cooldown_locked", machine.ValidateAction(state, "cd_skill").Reason);

        // 最后 0.1 秒解锁（容差 0.05）
        state = TimelineTestDriver.AdvanceBy(machine, state, 0.1);
        Assert.True(machine.ValidateAction(state, "cd_skill").Ok);
    }

    [Fact]
    public void 可用技能随状态变化()
    {
        var machine = FacadeKit.BuildMachine();
        var state = machine.InitialState();

        // 初始状态：空转需要 GCD 产生的 weave 窗口，因此不可用
        var keys = machine.AvailableActionKeys(state);
        Assert.Contains("gcd_strike", keys);
        Assert.Contains("ogcd_punch", keys);
        Assert.Contains("cd_skill", keys);
        Assert.DoesNotContain("ogcd_wait", keys);
        Assert.Contains("potion", keys);

        // cd_skill 用过后推进 GCD，冷却锁定使其不可用
        state = TimelineTestDriver.Execute(machine, state, "cd_skill").NextState;
        state = TimelineTestDriver.AdvanceBy(machine, state, 2.5);
        Assert.DoesNotContain("cd_skill", machine.AvailableActionKeys(state));

        // GCD 锁定中 GCD 技能不可用，真实 oGCD 仍由状态机校验
        state = TimelineTestDriver.Execute(machine, state, "gcd_strike").NextState;
        var lockedKeys = machine.AvailableActionKeys(state);
        Assert.DoesNotContain("gcd_strike", lockedKeys);
        Assert.Contains("ogcd_punch", lockedKeys);
        Assert.DoesNotContain("ogcd_wait", lockedKeys);
    }

    [Fact]
    public void 目标数量由带时间戳外部事实更新并拒绝负数()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());

        simulator.ApplyExternalEvent(new(
            1.0,
            ExternalCombatEventKinds.TargetCountChanged,
            TargetCount: 2));
        Assert.Equal(2, simulator.GetState().TargetCount);
        Assert.Throws<ArgumentException>(() => simulator.ApplyExternalEvent(new(
            2.0,
            ExternalCombatEventKinds.TargetCountChanged,
            TargetCount: -3)));
        Assert.Equal(1.0, simulator.Time);
        simulator.ApplyExternalEvent(new(
            3.0,
            ExternalCombatEventKinds.TargetCountChanged,
            TargetCount: 4));
        Assert.Equal(4, simulator.GetState().TargetCount);
    }

    [Fact]
    public void 非法动作提交返回拒绝结果且不改变状态()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        Assert.True(simulator.SubmitAction(0.0, "gcd_strike").Accepted);
        var before = simulator.CreateSnapshot();

        var result = simulator.SubmitAction(0.0, "gcd_strike");

        Assert.False(result.Accepted);
        Assert.Equal("gcd_locked", result.Reason);
        Assert.Equal(before.State.Time, simulator.Time);
        Assert.Equal(before.PendingEvents.Count, simulator.CreateSnapshot().PendingEvents.Count);
    }

    [Fact]
    public void 负时间推进抛错()
    {
        var machine = FacadeKit.BuildMachine();
        var state = machine.InitialState();
        Assert.Throws<ArgumentOutOfRangeException>(() => TimelineTestDriver.AdvanceBy(machine, state, -1.0));
    }

    [Fact]
    public void 禁用技能校验返回disabled()
    {
        var config = FacadeKit.BuildProjectConfig();
        var disabled = SystemTestKit.Skill("disabled_skill", ActionKind.Gcd, gameId: 99) with { Enabled = false };
        config = config with { Job = config.Job with { Skills = config.Job.Skills.Append(disabled).ToList() } };
        var machine = new CombatStateMachine(config, FacadeKit.JobTag);
        var state = machine.InitialState();
        Assert.Equal("disabled", machine.ValidateAction(state, "disabled_skill").Reason);
    }

    [Fact]
    public void 回放缓存支持增量执行与快照()
    {
        var machine = FacadeKit.BuildMachine();
        var cache = new ReplayStateCache(new JobSimulator(machine));
        Assert.Equal(600.0, cache.CurrentState.FightRemaining);

        var result = cache.SubmitAction(new(0.0, "gcd_strike"));
        Assert.True(result.Accepted);
        Assert.Equal(0.0, cache.CurrentState.Time, 9);
        Assert.Equal(2.5, cache.CurrentState.GcdRemaining, 9);
        Assert.Single(cache.CurrentState.History);

        var snapshot = cache.SnapshotAt(0);
        Assert.Equal(0.0, snapshot.State.Time, 9);

        cache.AdvanceTo(2.5);
        Assert.Equal(2.5, cache.CurrentState.Time, 9);
        Assert.Equal(0.0, cache.CurrentState.GcdRemaining, 9);

        cache.RestoreSnapshot(snapshot);
        Assert.Equal(0.0, cache.CurrentState.Time, 9);
        Assert.Equal(2.5, cache.CurrentState.GcdRemaining, 9);
    }

    [Fact]
    public void 按游戏id解析技能()
    {
        var machine = FacadeKit.BuildMachine();
        var skill = machine.ResolveSkill(1);
        Assert.Equal("gcd_strike", skill.Key);
        Assert.Throws<KeyNotFoundException>(() => machine.ResolveSkill(9999));
    }

    [Fact]
    public void 历史条目记录完整动作元信息()
    {
        var machine = FacadeKit.BuildMachine();
        var state = machine.InitialState();
        state = TimelineTestDriver.Execute(machine, state, "gcd_strike").NextState;

        var entry = state.History[0];
        Assert.Equal("gcd_strike", entry.SkillKey);
        Assert.Equal(1, entry.SkillId);
        Assert.Equal("gcd", entry.SkillKind);
        Assert.Equal(300.0, entry.Potency, 9);
        Assert.Equal(0.0, entry.CastTimeSeconds, 9);
        Assert.Equal(0.0, entry.CastTimeGcds, 9);
        Assert.Equal(2.5, entry.GcdWindowSeconds, 9);
        Assert.True(entry.IsLegal);
        Assert.Equal(1, entry.GcdIndex);
        Assert.Equal(0.0, entry.TimeSeconds, 9);
        Assert.Equal(10000, entry.MpBefore);
        Assert.Equal(10000, entry.MpAfter);
    }
}

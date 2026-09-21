using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.System.Timeline;
using static FightEngine.Tests.System.SystemTestKit;

namespace FightEngine.Tests.System;

/// <summary>时间推进测试：状态时间轴、DoT 时间轴与冷却运行时。</summary>
public class TimelineTests
{
    // ---------- StatusTimelineRuntime ----------

    [Fact]
    public void 状态时间推进并在耗尽时移除()
    {
        var timeline = new StatusTimelineRuntime();
        var state = new CombatState();
        state.Statuses["buff"] = new StatusState(5.0, 1);

        state = AdvancePublic(state, 3.0);
        Assert.Equal(2.0, state.Statuses["buff"].Remaining);

        state = AdvancePublic(state, 2.0);
        Assert.False(state.Statuses.ContainsKey("buff"));
    }

    // ---------- DotTimelineRuntime ----------

    [Fact]
    public void DoT注册取dot_key并按序排序()
    {
        var timeline = new DotTimelineRuntime();
        var skills = new[]
        {
            Skill("high_thunder", potency: 0, requiresTarget: true, dotDuration: 18.0, dotKey: "high_thunder_dot"),
            Skill("thunder", potency: 0, requiresTarget: true, dotDuration: 21.0, dotKey: "high_thunder_dot"),
            Skill("plain", potency: 100),
        };

        timeline.RegisterDots(skills);

        Assert.Equal(new[] { "high_thunder_dot" }, timeline.RegisteredDotKeys);
    }

    [Fact]
    public void DoT未注册挂载抛错()
    {
        var timeline = new DotTimelineRuntime();

        Assert.Throws<InvalidOperationException>(
            () => timeline.GrantRegisteredDot(new CombatState(), "nope", remaining: 10.0, potencyPerTick: 100));
    }

    [Fact]
    public void DoT在单个推进窗口内多次tick结算()
    {
        var timeline = new DotTimelineRuntime();
        timeline.RegisterDots(new[] { Skill("thunder", potency: 0, requiresTarget: true, dotDuration: 18.0, dotKey: "dot") });
        var state = new CombatState();
        timeline.GrantRegisteredDot(state, "dot", remaining: 10.0, potencyPerTick: 100);

        state = AdvancePublic(state, 7.0);

        // 7 秒窗口：t=3 与 t=6 各结算一次
        Assert.Equal(200.0, state.CurrentGcdDotPotency);
        Assert.Equal(200.0, state.CumulativeDotPotency);
        Assert.Equal(3.0, state.Dots["dot"].Remaining);
        Assert.Equal(2.0, state.Dots["dot"].NextTickInSeconds); // 余 1 秒：3 - 1 = 2
    }

    [Fact]
    public void DoT剩余归零时移除且跨调用不累加窗口威力()
    {
        var timeline = new DotTimelineRuntime();
        timeline.RegisterDots(new[] { Skill("thunder", potency: 0, requiresTarget: true, dotDuration: 18.0, dotKey: "dot") });
        var state = new CombatState();
        timeline.GrantRegisteredDot(state, "dot", remaining: 4.0, potencyPerTick: 100);

        state = AdvancePublic(state, 3.0);
        Assert.Equal(100.0, state.CurrentGcdDotPotency);

        state = AdvancePublic(state, 2.0);
        Assert.False(state.Dots.ContainsKey("dot"));
        // 观测分段不再重置动作窗口统计；它只在下一次真实动作接受时重置。
        Assert.Equal(100.0, state.CurrentGcdDotPotency);
        Assert.Equal(100.0, state.CumulativeDotPotency);
    }

    // ---------- CooldownRuntime ----------

    [Fact]
    public void 冷却消耗后回充计时并恢复充能()
    {
        var runtime = new CooldownRuntime();
        var skill = Skill("potion", kind: ActionKind.Ogcd, cooldown: 60.0);
        var state = new CombatState();

        runtime.ConsumeCooldown(state, skill);
        Assert.Equal(0, state.Cooldowns["potion"].AvailableCharges);
        Assert.Equal(new[] { 60.0 }, state.Cooldowns["potion"].RechargeTimers);

        state = AdvancePublic(state, 60.0, key => 1);

        Assert.False(state.Cooldowns.ContainsKey("potion")); // 满充自动清桶
        Assert.True(runtime.HasAvailableCharge(state, skill));
    }

    [Fact]
    public void 多充能技能按序回充()
    {
        var runtime = new CooldownRuntime();
        var skill = Skill("charge_skill", kind: ActionKind.Ogcd, cooldown: 20.0, charges: 2);
        var state = new CombatState();

        runtime.ConsumeCooldown(state, skill);              // timer [20]，可用 1
        state = AdvancePublic(state, 10.0, key => 2);         // timer [10]，可用 1
        runtime.ConsumeCooldown(state, skill);              // timer [10, 20]，可用 0
        Assert.Equal(0, state.Cooldowns["charge_skill"].AvailableCharges);

        state = AdvancePublic(state, 10.0, key => 2);         // 第一个计时到点回充

        Assert.Equal(1, state.Cooldowns["charge_skill"].AvailableCharges);
        Assert.Equal(new[] { 10.0 }, state.Cooldowns["charge_skill"].RechargeTimers.ToArray()); // 第二个计时已走 10 秒
    }

    [Fact]
    public void 多充能同时到点满充后清理冷却桶()
    {
        var runtime = new CooldownRuntime();
        var skill = Skill("charge_skill", kind: ActionKind.Ogcd, cooldown: 20.0, charges: 2);
        var state = new CombatState();

        runtime.ConsumeCooldown(state, skill);
        runtime.ConsumeCooldown(state, skill);              // timer [20, 20]，可用 0

        state = AdvancePublic(state, 20.0, key => 2);         // 两个计时同时到点

        Assert.False(state.Cooldowns.ContainsKey("charge_skill"));
        Assert.True(runtime.HasAvailableCharge(state, skill));
    }

    [Fact]
    public void 冷却快照不提前回充并输出round4()
    {
        var runtime = new CooldownRuntime();
        var skill = Skill("potion", kind: ActionKind.Ogcd, cooldown: 60.0);
        var state = new CombatState();
        runtime.ConsumeCooldown(state, skill);
        state.Cooldowns["potion"].RechargeReadyAt[0] = 0.123456;

        var snapshot = runtime.BuildCooldownSnapshot(state, skill);

        Assert.Equal(0.1235, snapshot.NextCooldownSeconds);
        Assert.Equal(0, snapshot.AvailableCharges);

        // 即使在旧容差内，观测也不能提前恢复充能。
        state.Cooldowns["potion"].RechargeReadyAt[0] = 0.01;
        var ready = runtime.BuildCooldownSnapshot(state, skill);
        Assert.Equal(0, ready.AvailableCharges);
        Assert.Equal(0.01, ready.NextCooldownSeconds);
    }

    [Fact]
    public void 主动减少冷却并清理满充桶()
    {
        var runtime = new CooldownRuntime();
        var skill = Skill("potion", kind: ActionKind.Ogcd, cooldown: 60.0);
        var state = new CombatState();
        runtime.ConsumeCooldown(state, skill);

        runtime.ApplyReduction(state, skill, 60.0);

        Assert.False(state.Cooldowns.ContainsKey("potion"));
    }

    [Fact]
    public void 无冷却技能无桶且恒有充能()
    {
        var runtime = new CooldownRuntime();
        var skill = Skill("plain_gcd");
        var state = new CombatState();

        Assert.True(runtime.HasAvailableCharge(state, skill));
        var snapshot = runtime.BuildCooldownSnapshot(state, skill);
        Assert.Equal((0.0, 1, 1), (snapshot.NextCooldownSeconds, snapshot.AvailableCharges, snapshot.MaxCharges));
        runtime.ConsumeCooldown(state, skill);
        Assert.False(state.Cooldowns.ContainsKey("plain_gcd"));
    }
}

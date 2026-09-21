using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.Outputs;
using Combat.Sim.System;
using static FightEngine.Tests.System.SystemTestKit;

namespace FightEngine.Tests.System;

/// <summary>系统状态机测试：门面编排、MP 回蓝、系统技能与威力解析。</summary>
public class SystemStateMachineTests
{
    private static SystemStateMachine CreateMachine(int? maxHistory = null)
    {
        var machine = new SystemStateMachine(Timing, Potency, MpRecovery, maxHistory: maxHistory);
        machine.RegisterSystemStatuses(new Dictionary<string, StatusDefinition>
        {
            ["burst_potion"] = new("burst_potion", GameId: 1, Duration: 30.0),
            ["raid_buff_window"] = new("raid_buff_window", GameId: 2, Duration: 20.0),
        });
        return machine;
    }

    // ---------- advance_time 五段编排 ----------

    [Fact]
    public void 时间推进按五段顺序编排()
    {
        var machine = CreateMachine();
        var state = machine.InitialState(fightRemaining: 600.0, maxMp: 10000);
        state.Mp = 9000; // 留出回蓝空间（满蓝会被上限钳回）
        // 挂一个 DoT，并制造一个冷却桶
        machine.RegisterJobTargetDots(new[] { Skill("thunder", dotKey: "dot", dotDuration: 18.0) });
        machine.GrantRegisteredTargetDot(state, "dot", remaining: 9.0, potencyPerTick: 100);
        var skill = Skill("potion", kind: ActionKind.Ogcd, cooldown: 60.0);
        machine.ConsumeCooldown(state, skill);
        state.Statuses["buff"] = new StatusState(5.0, 1);

        var next = SystemTimelineTestDriver.AdvanceBy(machine, state, 3.0, key => 1);

        Assert.Equal(2.0, next.Statuses["buff"].Remaining);      // status 推进
        Assert.Equal(100.0, next.CurrentGcdDotPotency);          // DoT t=3 结算
        Assert.Equal(9200, next.Mp);                             // MP tick 200
        Assert.Equal(57.0, next.Cooldowns["potion"].RechargeTimers[0]); // 冷却推进
    }

    [Fact]
    public void 初始状态装配量谱默认值()
    {
        var machine = CreateMachine();
        machine.RegisterJobState(new JobStateRegistration(
            "black_mage",
            new Dictionary<string, JobResourceDefinition>
            {
                ["astral_fire"] = new("astral_fire", "int", DefaultValue: 0, MaxValue: 3),
            },
            new Dictionary<string, StatusDefinition>()));
        machine.RegisterSystemStatuses(new Dictionary<string, StatusDefinition>
        {
            ["burst_potion"] = new("burst_potion", GameId: 1, Duration: 30.0),
        });

        var state = machine.InitialState(fightRemaining: 600.0, maxMp: 10000);

        Assert.Equal(0, state.GetJobResource("astral_fire"));
        Assert.Equal(10000, state.Mp);
        Assert.Equal(600.0, state.FightRemaining);
    }

    // ---------- MP 回蓝 ----------

    [Fact]
    public void 回蓝按tick间隔推进并读取回调()
    {
        var machine = CreateMachine();
        var state = machine.InitialState(fightRemaining: 600.0, maxMp: 10000);
        state.Mp = 8000; // 留出回蓝空间
        var observedNaturalAmounts = new List<int>();
        machine.JobTimeline.RegisterMpTickModifier((_, naturalAmount) =>
        {
            observedNaturalAmounts.Add(naturalAmount);
            return naturalAmount;
        });

        var next = SystemTimelineTestDriver.AdvanceBy(machine, state, 7.0, key => 1);

        Assert.Equal(8400, next.Mp); // 8000 + 2 tick × 200
        Assert.Equal(new[] { 200, 200 }, observedNaturalAmounts);
        Assert.Equal(1.0, next.NaturalMpTickProgress);
    }

    [Fact]
    public void 回蓝回调返回负值时按零处理()
    {
        var machine = CreateMachine();
        var state = machine.InitialState(fightRemaining: 600.0, maxMp: 10000);
        machine.JobTimeline.RegisterMpTickModifier((_, _) => -50);

        var next = SystemTimelineTestDriver.AdvanceBy(machine, state, 3.0, key => 1);

        Assert.Equal(10000, next.Mp);
    }

    // ---------- 威力解析 ----------

    [Fact]
    public void 基础威力倍率叠加爆发药与团辅()
    {
        var machine = CreateMachine();
        var state = machine.InitialState(fightRemaining: 600.0, maxMp: 10000);
        machine.GrantRegisteredStatus(state, "burst_potion");
        machine.GrantRegisteredStatus(state, "raid_buff_window");

        var resolved = machine.ResolveAppliedPotency(state, 100);

        Assert.Equal(100 * 1.08 * 1.10, resolved);
    }

    [Fact]
    public void AoE目标衰减按技能额外目标计算()
    {
        var machine = CreateMachine();
        var aoe = Skill("aoe", potency: 100, maxTargets: 8, aoeSecondaryReduction: 0.7);
        var state = machine.InitialState(fightRemaining: 600.0, maxMp: 10000);
        state.TargetCount = 3;

        // 1 + (1-0.7) × (3-1) = 1.6
        Assert.Equal(1.6, SystemStateMachine.ResolveAoeTargetMultiplier(aoe, 3));
        Assert.Equal(100 * 1.6, machine.ResolveAppliedPotency(state, 100, aoe));

        // 单目标不衰减
        Assert.Equal(1.0, SystemStateMachine.ResolveAoeTargetMultiplier(aoe, 1));
    }

    [Fact]
    public void 事件时刻倍率按offset判定状态剩余()
    {
        var machine = CreateMachine();
        var state = machine.InitialState(fightRemaining: 600.0, maxMp: 10000);
        machine.GrantRegisteredStatus(state, "burst_potion", remaining: 5.0);

        Assert.Equal(100 * 1.08, machine.ResolveAppliedPotencyAtOffset(state, 100, 3.0));
        Assert.Equal(100.0, machine.ResolveAppliedPotencyAtOffset(state, 100, 6.0)); // 已过期
    }

    [Fact]
    public void 目标威力记录累计并清零当前GCDdot()
    {
        var machine = CreateMachine();
        var state = machine.InitialState(fightRemaining: 600.0, maxMp: 10000);
        var skill = Skill("fire", potency: 300);
        state.CurrentGcdDotPotency = 200.0;

        machine.RecordTargetPotency(state, state, skill, 300);

        Assert.Equal(300.0, state.CurrentPotency);
        Assert.Equal(0.0, state.CurrentGcdDotPotency);
        Assert.Equal(300.0, state.CumulativePotency);
    }

    [Fact]
    public void 延迟威力记录到累计()
    {
        var machine = CreateMachine();
        var state = machine.InitialState(fightRemaining: 600.0, maxMp: 10000);

        var resolved = machine.RecordDelayedPotency(state, 150.0);

        Assert.Equal(150.0, resolved);
        Assert.Equal(150.0, state.CumulativePotency);
    }

    // ---------- 历史记录 ----------

    [Fact]
    public void 历史记录挂载量谱差分并限制长度()
    {
        var machine = CreateMachine(maxHistory: 2);
        machine.RegisterJobState(new JobStateRegistration(
            "black_mage",
            new Dictionary<string, JobResourceDefinition>
            {
                ["astral_fire"] = new("astral_fire", "int", DefaultValue: 0, MaxValue: 3),
            },
            new Dictionary<string, StatusDefinition>()));
        var state = machine.InitialState(fightRemaining: 600.0, maxMp: 10000);
        var skill = Skill("fire", potency: 300);
        machine.SetJobResource(state, "astral_fire", 3);

        for (var i = 0; i < 3; i++)
        {
            machine.RecordActionHistory(
                state, state, skill,
                potency: 300,
                value: 1.0,
                castTimeSeconds: 0.0, castTimeGcds: 0.0,
                gcdWindowSeconds: 2.5, gcdWindowGcds: 1.0,
                isLegal: true, invalidReason: "",
                nextCooldownSeconds: 0.0, availableCharges: 1, maxCharges: 1,
                stateBefore: StateContext.Empty(), stateAfter: StateContext.Empty());
        }

        Assert.Equal(2, state.History.Count);
        Assert.Equal(0, state.History[0].JobResourcesConsumed["astral_fire"]); // 前后相同
    }
}

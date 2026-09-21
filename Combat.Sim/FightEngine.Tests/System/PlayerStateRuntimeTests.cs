using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.System;
using static FightEngine.Tests.System.SystemTestKit;

namespace FightEngine.Tests.System;

/// <summary>玩家公共态测试：时序应用、时间推进、公共校验（对照 test_system.py 玩家态部分）。</summary>
public class PlayerStateRuntimeTests
{
    private static PlayerStateRuntime CreateMachine() =>
        new();

    [Fact]
    public void GCD时序应用推进gcd与weave窗口()
    {
        var machine = CreateMachine();
        var state = new CombatState();
        var skill = Skill("fire", potency: 300, castTime: 3.5);

        machine.ApplyActionTiming(state, skill, effectiveGcd: 2.5, actualCastSeconds: 3.5);

        Assert.Equal(1, state.GcdIndex);
        Assert.Equal(2.5, state.GcdRemaining);
        Assert.Equal(0.0, state.WeaveWindowRemaining); // 读条 > gcd，无 weave 窗口
        Assert.Equal(0, state.OgcdsWeaved);
    }

    [Fact]
    public void oGCD只累计weave而不扣除调用方动画间隔()
    {
        var machine = CreateMachine();
        var state = new CombatState { GcdRemaining = 2.0, WeaveWindowRemaining = 2.0 };
        var skill = Skill("oGCD", kind: ActionKind.Ogcd);

        machine.ApplyActionTiming(state, skill, effectiveGcd: 0.0, actualCastSeconds: 0.0);

        Assert.Equal(2.0, state.WeaveWindowRemaining);
        Assert.Equal(1, state.OgcdsWeaved);
    }

    [Fact]
    public void 时间推进推进玩家时间轴并在gcd结束后重置weave()
    {
        var machine = CreateMachine();
        var state = new CombatState();
        state.SetTimelineTime(5.0);
        state.FightRemaining = 600.0;
        state.GcdRemaining = 2.0;
        state.WeaveWindowRemaining = 1.0;
        state.OgcdsWeaved = 2;

        var next = AdvancePublic(state, 2.5);

        Assert.Equal(7.5, next.Time);
        Assert.Equal(597.5, next.FightRemaining);
        Assert.Equal(0.0, next.GcdRemaining);
        Assert.Equal(0.0, next.WeaveWindowRemaining);
        Assert.Equal(0, next.OgcdsWeaved);
        // 输入状态未被修改（clone 语义）
        Assert.Equal(5.0, state.Time);
    }

    [Fact]
    public void 停手窗口到点切换为不可选中()
    {
        var machine = CreateMachine();
        var state = new CombatState { NextDowntimeEta = 2.0, DowntimeRemaining = 5.0 };

        var next = AdvancePublic(state, 2.5);

        Assert.False(next.BossTargetable);
        Assert.Equal(0.0, next.NextDowntimeEta);
        Assert.Equal(4.5, next.DowntimeRemaining);
    }

    [Fact]
    public void 停手结束恢复可选中并清除时间戳()
    {
        var machine = CreateMachine();
        var state = new CombatState
        {
            BossTargetable = false,
            DowntimeRemaining = 3.0,
            NextDowntimeEta = 0.0,
        };

        var next = AdvancePublic(state, 3.5);

        Assert.True(next.BossTargetable);
        Assert.Equal(0.0, next.DowntimeRemaining);
        Assert.Null(next.NextDowntimeEta);
    }

    [Theory]
    [InlineData("gcd_locked")]
    [InlineData("ogcd_limit")]
    [InlineData("boss_untargetable")]
    [InlineData("movement_locked")]
    [InlineData("cooldown_locked")]
    public void 公共校验拒绝原因(string expectedReason)
    {
        var machine = CreateMachine();
        var state = new CombatState
        {
            GcdRemaining = expectedReason is "gcd_locked" or "ogcd_limit" ? 1.0 : 0.0,
            OgcdsWeaved = expectedReason == "ogcd_limit" ? 3 : 0,
            BossTargetable = expectedReason != "boss_untargetable",
            IsMoving = expectedReason == "movement_locked",
        };
        var kind = expectedReason is "ogcd_limit" ? ActionKind.Ogcd : ActionKind.Gcd;
        var skill = Skill("skill", kind: kind);

        var result = machine.ValidateCommonAction(
            state,
            skill,
            isInstant: false,
            hasAvailableCharge: expectedReason != "cooldown_locked");

        Assert.False(result.Ok);
        Assert.Equal(expectedReason, result.Reason);
    }

    [Fact]
    public void 公共校验通过返回ok()
    {
        var machine = CreateMachine();
        var state = new CombatState();
        var skill = Skill("skill");

        var result = machine.ValidateCommonAction(state, skill, isInstant: false, hasAvailableCharge: true);

        Assert.True(result.Ok);
    }
}

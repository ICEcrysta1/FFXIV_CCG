using Combat.Sim.Config;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.System;

namespace FightEngine.Tests.System;

/// <summary>测试共享配置与技能构造辅助。</summary>
public static class SystemTestKit
{
    public static readonly EngineTimingConfig Timing = new(
        ActionQueueWindowSeconds: 0.05);

    public static readonly SystemPotencyConfig Potency = new(
        BurstPotionMultiplier: 1.08,
        RaidBuffWindowMultiplier: 1.10);

    public static readonly SystemMpRecoveryConfig MpRecovery = new(
        TickIntervalSeconds: 3.0,
        InCombatAmount: 200);

    public static CombatState AdvancePublic(CombatState state, double seconds, Func<string, int>? getMaxCharges = null)
    {
        var machine = new SystemStateMachine(Timing, Potency, MpRecovery);
        var timeline = machine.CreateTimeline(state, getMaxCharges ?? (_ => 1));
        return timeline.AdvanceTo(state.Time + seconds);
    }

    /// <summary>构造最小技能定义（默认 gcd、无冷却、无耗蓝）。</summary>
    public static SkillDefinition Skill(
        string key,
        ActionKind kind = ActionKind.Gcd,
        int potency = 0,
        double cooldown = 0.0,
        int charges = 1,
        bool requiresTarget = true,
        int maxTargets = 1,
        double aoeSecondaryReduction = 1.0,
        string behavior = "standard",
        IReadOnlyList<string>? appliesStatuses = null,
        int gameId = 0,
        double castTime = 0.0,
        string? dotKey = null,
        double dotDuration = 0.0) => new(
        Key: key,
        GameId: gameId,
        Name: key,
        Kind: kind,
        Behavior: behavior,
        Potency: potency,
        RequiresTarget: requiresTarget,
        Cooldown: cooldown,
        Charges: charges,
        MaxTargets: maxTargets,
        AoeSecondaryReduction: aoeSecondaryReduction,
        AppliesStatuses: appliesStatuses ?? Array.Empty<string>(),
        Tags: Array.Empty<string>(),
        CastTime: castTime,
        DotKey: dotKey,
        DotDuration: dotDuration);
}

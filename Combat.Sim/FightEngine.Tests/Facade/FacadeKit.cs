using Combat.Sim.Config;
using Combat.Sim.Facade;
using Combat.Sim.Jobs;
using Combat.Sim.Models.Definitions;
using FightEngine.Tests.System;

namespace FightEngine.Tests.Facade;

/// <summary>门面测试共享的最小内嵌配置与技能构造辅助。</summary>
public static class FacadeKit
{
    public const string JobTag = "test_job";

    public static readonly EngineTimingConfig Timing = new(
        ActionQueueWindowSeconds: 0.5);

    public static readonly SystemPotencyConfig Potency = SystemTestKit.Potency;

    public static readonly SystemMpRecoveryConfig MpRecovery = SystemTestKit.MpRecovery;

    /// <summary>构造最小项目配置：真实系统技能（爆发药）+ 四个测试技能。</summary>
    public static ProjectConfig BuildProjectConfig(double baseGcd = 2.5)
    {        var statuses = new Dictionary<string, StatusDefinition>
        {
            ["burst_potion"] = new("burst_potion", 1000049, 30.0, MaxStacks: 1),
        };
        var systemSkills = new List<SkillDefinition>
        {
            SystemTestKit.Skill(
                "potion",
                ActionKind.Ogcd,
                cooldown: 270.0,
                behavior: "grant_status",
                appliesStatuses: new[] { "burst_potion" },
                gameId: 99999),
        };
        var jobSkills = new List<SkillDefinition>
        {
            SystemTestKit.Skill("gcd_strike", ActionKind.Gcd, potency: 300, gameId: 1),
            SystemTestKit.Skill("ogcd_punch", ActionKind.Ogcd, potency: 200, gameId: 2),
            SystemTestKit.Skill("cd_skill", ActionKind.Gcd, potency: 400, cooldown: 30.0, gameId: 3),
            SystemTestKit.Skill("cast_skill", ActionKind.Gcd, potency: 500, castTime: 2.8, gameId: 4),
        };
        return new ProjectConfig(
            Name: "test",
            Version: "0.0.0",
            Runtime: new RuntimeConfig(JobTag, "config/system.yaml", "config/skills/test_job.yaml"),
            EngineTiming: Timing,
            EngineResources: new EngineResourceConfig(MaxMp: 10000),
            System: new SystemConfig(
                BaseGcd: baseGcd,
                SkillTableBaseGcd: 2.5,
                MpRecovery: MpRecovery,
                Potency: Potency,
                RaidBuffWindowMarkerSkills: new[] { "gcd_strike" },
                RaidBuffWindowDuration: 20.0,
                Statuses: statuses,
                Skills: systemSkills),
            Job: new JobConfig(
                Key: JobTag,
                Name: "Test Job",
                Timing: new Dictionary<string, double> { ["default_fight_remaining"] = 600.0 },
                ResourceLimits: new Dictionary<string, double>(),
                Statuses: new Dictionary<string, StatusDefinition>(),
                Skills: jobSkills));
    }

    static FacadeKit()
    {
        JobMachineRegistry.Register(JobTag, () => new TestJobStateMachine());
    }

    /// <summary>创建测试门面（测试职业已在静态构造中注册）。</summary>
    public static CombatStateMachine BuildMachine(double baseGcd = 2.5) =>
        new(BuildProjectConfig(baseGcd), JobTag);


}

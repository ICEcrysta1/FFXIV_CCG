using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.Config;
using Combat.Sim.Jobs;
using Combat.Sim.System;

namespace FightEngine.Tests.Facade;

/// <summary>
/// 测试专用最小职业子状态机：不承载任何职业规则，全部行为走系统层默认。
/// 行为固定，用于门面组合逻辑的场景与回归验证。
/// </summary>
public sealed class TestJobStateMachine : IJobStateMachine
{
    private double _baseGcd;

    public TestJobStateMachine(double baseGcd = 2.5)
    {
        _baseGcd = baseGcd;
    }

    public void Bind(ProjectConfig project, SystemStateMachine system)
    {
        _baseGcd = project.System.BaseGcd;
    }

    public static string Tag => "test_job";

    public double DefaultFightRemaining => 600.0;

    public JobStateRegistration BuildStateRegistration() =>
        new(Tag,
            new Dictionary<string, JobResourceDefinition>(),
            new Dictionary<string, StatusDefinition>());

    public double CurrentGcdDuration(CombatState state) => _baseGcd;

    /// <summary>oGCD 或无读条时间视为瞬发。</summary>
    public bool IsInstant(CombatState state, SkillDefinition skill) =>
        skill.Kind == ActionKind.Ogcd || skill.CastTime <= 0;

    /// <summary>测试职业只认领标准技能行为；系统行为（空转/爆发药）由系统层处理。</summary>
    public bool SupportsBehavior(string behavior) => behavior == "standard";

    public ValidationResult ValidateAction(CombatState state, SkillDefinition skill) => new(true);

    public void ApplyAction(CombatState previousState, CombatState nextState, SkillDefinition skill)
    {
    }

    public void ConsumeTimingResources(CombatState state, SkillDefinition skill)
    {
    }

}

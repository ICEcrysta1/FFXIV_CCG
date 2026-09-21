using Combat.Sim.Jobs;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Xunit;

namespace FightEngine.Tests.Jobs;

/// <summary>注册中心测试专用职业 A（tag = registry_job_a）。</summary>
public class RegistryJobA : IJobStateMachine
{
    public static string Tag => "registry_job_a";

    public double DefaultFightRemaining => 600.0;

    public JobStateRegistration BuildStateRegistration() =>
        new(Tag, new Dictionary<string, JobResourceDefinition>(), new Dictionary<string, StatusDefinition>());

    public double CurrentGcdDuration(CombatState state) => 2.5;

    public bool IsInstant(CombatState state, SkillDefinition skill) => true;

    public bool SupportsBehavior(string behavior) => false;

    public ValidationResult ValidateAction(CombatState state, SkillDefinition skill) => new(true);

    public void ApplyAction(CombatState previousState, CombatState nextState, SkillDefinition skill)
    {
    }

    public void ConsumeTimingResources(CombatState state, SkillDefinition skill)
    {
    }

}

/// <summary>注册中心测试专用职业 B（与 A 同 tag、不同类型，用于冲突校验）。</summary>
public sealed class RegistryJobB : RegistryJobA
{
    // static 成员不继承，派生类必须自行实现静态 Tag
    public static new string Tag => "registry_job_a";
}

/// <summary>
/// 工厂先占标签冲突测试专用职业（独立实现接口，避免派生类静态成员
/// 不参与接口实现解析导致 T.Tag 绑定到基类）。
/// </summary>
public sealed class PreemptedJob : IJobStateMachine
{
    public static string Tag => "registry_preempted_job";

    public double DefaultFightRemaining => 600.0;

    public JobStateRegistration BuildStateRegistration() =>
        new(Tag, new Dictionary<string, JobResourceDefinition>(), new Dictionary<string, StatusDefinition>());

    public double CurrentGcdDuration(CombatState state) => 2.5;

    public bool IsInstant(CombatState state, SkillDefinition skill) => true;

    public bool SupportsBehavior(string behavior) => false;

    public ValidationResult ValidateAction(CombatState state, SkillDefinition skill) => new(true);

    public void ApplyAction(CombatState previousState, CombatState nextState, SkillDefinition skill)
    {
    }

    public void ConsumeTimingResources(CombatState state, SkillDefinition skill)
    {
    }

}

/// <summary>带构造计数的注册测试职业：锁定注册路径不构造实例。</summary>
public sealed class CountingJob : IJobStateMachine
{
    public static int Constructions;

    public CountingJob()
    {
        Constructions++;
    }

    public static string Tag => "counting_job";

    public double DefaultFightRemaining => 600.0;

    public JobStateRegistration BuildStateRegistration() =>
        new(Tag, new Dictionary<string, JobResourceDefinition>(), new Dictionary<string, StatusDefinition>());

    public double CurrentGcdDuration(CombatState state) => 2.5;

    public bool IsInstant(CombatState state, SkillDefinition skill) => true;

    public bool SupportsBehavior(string behavior) => false;

    public ValidationResult ValidateAction(CombatState state, SkillDefinition skill) => new(true);

    public void ApplyAction(CombatState previousState, CombatState nextState, SkillDefinition skill)
    {
    }

    public void ConsumeTimingResources(CombatState state, SkillDefinition skill)
    {
    }

}

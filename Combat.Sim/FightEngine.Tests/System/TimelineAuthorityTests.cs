using System.Reflection;
using Combat.Sim.Facade;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Timeline;
using Combat.Sim.System.Timeline;

namespace FightEngine.Tests.System;

/// <summary>防止新时间线 API 在领域模块中重新出现第二套推进入口。</summary>
public sealed class TimelineAuthorityTests
{
    // 公开绝对时间门面只能委托 CombatTimelineRuntime.AdvanceTo，不属于第二套内核。
    private static readonly HashSet<string> AllowedDelegatingEntryPoints = new(StringComparer.Ordinal)
    {
        "Combat.Sim.Facade.JobSimulator.AdvanceTo",
        "Combat.Sim.Facade.ReplayStateCache.AdvanceTo",
    };

    [Fact]
    public void 时间线内核是唯一公开的绝对时间推进入口()
    {
        var assembly = typeof(CombatTimelineRuntime).Assembly;
        var advanceToMethods = assembly.GetTypes()
            .Where(type => type.Namespace?.StartsWith("Combat.Sim", StringComparison.Ordinal) == true)
            .Select(type => (Type: type, Method: type.GetMethod("AdvanceTo")))
            .Where(item => item.Method is not null)
            .ToArray();

        Assert.Contains(advanceToMethods,
            item => item.Type == typeof(CombatTimelineRuntime)
                && item.Method!.ReturnType == typeof(CombatState));
        Assert.All(advanceToMethods, item =>
            Assert.True(item.Type == typeof(CombatTimelineRuntime)
                || AllowedDelegatingEntryPoints.Contains($"{item.Type.FullName}.{item.Method!.Name}")));
    }

    /// <summary>
    /// 内核之外新出现的推进入口直接失败。只覆盖具名方法：
    /// 注册给内核的 lambda 回调由阶段 3 收尾任务单独清理，其编译器生成名不适合做白名单。
    /// </summary>
    [Fact]
    public void 内核之外不存在额外的推进入口()
    {
        var detected = DetectAdvanceEntryPoints();

        var unregistered = detected
            .Where(name => !AllowedDelegatingEntryPoints.Contains(name))
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();
        Assert.True(unregistered.Length == 0,
            "内核之外不允许自行推进时钟或自行结算到期事实：应返回 TimelineMutation，交由 "
            + "CombatTimelineRuntime.AdvanceTo 统一派发。\n"
            + string.Join("\n", unregistered));
    }

    [Fact]
    public void 时间线事件使用公开稳定优先级契约()
    {
        Assert.True((int)TimelineEventPriority.ExternalScene < (int)TimelineEventPriority.ConfirmedActionEffect);
        Assert.True((int)TimelineEventPriority.ConfirmedActionEffect < (int)TimelineEventPriority.PeriodicSettlement);
        Assert.True((int)TimelineEventPriority.PeriodicSettlement < (int)TimelineEventPriority.ExpirationAndCooldown);
        Assert.True((int)TimelineEventPriority.ExpirationAndCooldown < (int)TimelineEventPriority.ActionAccepted);
        Assert.Equal(TimelineEventPriority.DecisionBoundary, TimelineEventPriority.ActionAccepted);
    }

    [Fact]
    public void 阶段四旧协议和兼容类型已从生产程序集删除()
    {
        var forbiddenMethods = new HashSet<string>(StringComparer.Ordinal)
        {
            "Step", "StepInPlace", "StepInternal",
            "Advance", "AdvanceTime", "AdvanceTimeInPlace", "AdvanceTimeInternal",
            "SetScene", "SetTargetCount", "PreviewCandidates", "FormatStep",
        };
        foreach (var type in new[]
        {
            typeof(CombatStateMachine),
            typeof(JobSimulator),
            typeof(ReplayStateCache),
        })
        {
            Assert.DoesNotContain(type.GetMethods(
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance
                | BindingFlags.Static | BindingFlags.DeclaredOnly),
                method => forbiddenMethods.Contains(method.Name));
        }

        Assert.Null(typeof(CombatState).GetProperty(nameof(CombatState.Time))!.SetMethod);
        var assembly = typeof(CombatStateMachine).Assembly;
        Assert.Null(assembly.GetType("Combat.Sim.Models.Combat.StepResult"));
        Assert.Null(assembly.GetType("Combat.Sim.System.SystemSkillRuntime"));
        Assert.Null(assembly.GetType("Combat.Sim.System.Timeline.TempJobTimerLegacyBridge"));
    }

    [Fact]
    public void 动作协议只保留提交输入和轻量结果()
    {
        Assert.Equal(
            new[] { "SkillKey", "Timestamp" },
            typeof(ActionRequest).GetProperties()
                .Select(property => property.Name)
                .OrderBy(name => name, StringComparer.Ordinal));
        Assert.DoesNotContain(
            typeof(ActionSubmissionResult).GetProperties(),
            property => property.PropertyType == typeof(CombatState));
        Assert.DoesNotContain(
            typeof(ExternalEventResult).GetProperties(),
            property => property.PropertyType == typeof(CombatState));
    }

    /// <summary>按方法名与参数形状识别"自行推进 / 自行结算"的入口，时间线内核本身除外。</summary>
    private static HashSet<string> DetectAdvanceEntryPoints()
    {
        var found = new HashSet<string>(StringComparer.Ordinal);
        foreach (var type in typeof(CombatTimelineRuntime).Assembly.GetTypes())
        {
            if (type.Namespace?.StartsWith("Combat.Sim", StringComparison.Ordinal) != true
                || type == typeof(CombatTimelineRuntime)
                || type.FullName is null)
            {
                continue;
            }

            foreach (var method in type.GetMethods(
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance
                | BindingFlags.Static | BindingFlags.DeclaredOnly))
            {
                if (IsAdvanceEntryPoint(method))
                {
                    found.Add($"{type.FullName}.{method.Name}");
                }
            }
        }

        return found;
    }

    private static bool IsAdvanceEntryPoint(MethodInfo method)
    {
        if (method.Name.StartsWith("Advance", StringComparison.Ordinal))
        {
            return true;
        }

        return false;
    }
}

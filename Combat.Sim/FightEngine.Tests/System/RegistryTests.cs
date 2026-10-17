using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.Models.Timeline;
using Combat.Sim.Jobs;
using Combat.Sim.State;

namespace FightEngine.Tests.System;

/// <summary>注册中心测试：Buff 注册中心与职业量谱注册中心。</summary>
public class RegistryTests
{
    // ---------- BuffStateRegistry ----------

    [Fact]
    public void Buff授予默认使用定义时长与层数()
    {
        var registry = new BuffStateRegistry();
        registry.RegisterStatuses(new Dictionary<string, StatusDefinition>
        {
            ["burst_potion"] = new("burst_potion", GameId: 1, Duration: 30.0, MaxStacks: 1),
        });
        var state = new CombatState();

        registry.GrantRegisteredStatus(state, "burst_potion");

        Assert.True(state.HasStatus("burst_potion"));
        Assert.Equal(30.0, state.Statuses["burst_potion"].Remaining);
        Assert.Equal(1, state.Statuses["burst_potion"].Stacks);
    }

    [Fact]
    public void Buff授予覆盖剩余时间与层数并钳制()
    {
        var registry = new BuffStateRegistry();
        registry.RegisterStatuses(new Dictionary<string, StatusDefinition>
        {
            ["triplecast"] = new("triplecast", GameId: 2, Duration: 15.0, MaxStacks: 3),
        });
        var state = new CombatState();

        registry.GrantRegisteredStatus(state, "triplecast", remaining: 10.0, stacks: 5);

        Assert.Equal(10.0, state.Statuses["triplecast"].Remaining);
        Assert.Equal(3, state.Statuses["triplecast"].Stacks); // 钳到 max_stacks
    }

    [Fact]
    public void Buff剩余时间归零时清除()
    {
        var registry = new BuffStateRegistry();
        registry.RegisterStatuses(new Dictionary<string, StatusDefinition>
        {
            ["burst_potion"] = new("burst_potion", GameId: 1, Duration: 30.0),
        });
        var state = new CombatState();
        registry.GrantRegisteredStatus(state, "burst_potion");

        registry.GrantRegisteredStatus(state, "burst_potion", remaining: 0.0);

        Assert.False(state.HasStatus("burst_potion"));
    }

    [Fact]
    public void 消耗层数到零清除状态()
    {
        var registry = new BuffStateRegistry();
        registry.RegisterStatuses(new Dictionary<string, StatusDefinition>
        {
            ["triplecast"] = new("triplecast", GameId: 2, Duration: 15.0, MaxStacks: 3),
        });
        var state = new CombatState();
        registry.GrantRegisteredStatus(state, "triplecast", stacks: 2);

        registry.ConsumeRegisteredStatusStack(state, "triplecast");
        Assert.Equal(1, state.StatusStacks("triplecast"));
        registry.ConsumeRegisteredStatusStack(state, "triplecast");
        Assert.False(state.HasStatus("triplecast"));
    }

    [Fact]
    public void 已激活状态满层校验拒绝()
    {
        var registry = new BuffStateRegistry();
        registry.RegisterStatuses(new Dictionary<string, StatusDefinition>
        {
            ["burst_potion"] = new("burst_potion", GameId: 1, Duration: 30.0),
        });
        var state = new CombatState();
        registry.GrantRegisteredStatus(state, "burst_potion");

        var result = registry.ValidateRegisteredStatus(state, "burst_potion");

        Assert.False(result.Ok);
        Assert.Equal("status_already_active", result.Reason);
    }

    [Fact]
    public void 未注册状态查询抛错()
    {
        var registry = new BuffStateRegistry();

        Assert.Throws<KeyNotFoundException>(() => registry.StatusDefinitionOf("nope"));
    }

    [Fact]
    public void 同名公共与职业Buff的授予和消耗各自独立()
    {
        var system = new BuffStateRegistry();
        var job = new BuffStateRegistry(state => state.Job.Statuses);
        system.RegisterStatuses(new Dictionary<string, StatusDefinition>
        {
            ["buff"] = new("buff", GameId: 1, Duration: 30, MaxStacks: 2),
        });
        job.RegisterStatuses(new Dictionary<string, StatusDefinition>
        {
            ["buff"] = new("buff", GameId: 2, Duration: 10, MaxStacks: 3),
        });
        var state = new CombatState();

        system.GrantRegisteredStatus(state, "buff");
        Assert.False(state.Job.HasStatus("buff"));
        job.GrantRegisteredStatus(state, "buff");
        job.ConsumeRegisteredStatusStack(state, "buff");

        Assert.Equal(2, state.StatusStacks("buff"));
        Assert.Equal(2, state.Job.StatusStacks("buff"));
        Assert.Equal(30, state.Statuses["buff"].Remaining);
        Assert.Equal(10, state.Job.Statuses["buff"].Remaining);
        Assert.Equal(1, system.Definitions["buff"].GameId);
        Assert.Equal(2, job.Definitions["buff"].GameId);
    }

    [Fact]
    public void 清理公共Buff保留职业状态DoT与MP()
    {
        var system = new BuffStateRegistry();
        var state = new CombatState { Mp = 1234 };
        state.Statuses["buff"] = new StatusState(30);
        state.Job.Statuses["buff"] = new StatusState(10, 2);
        state.Job.SetResource("gauge", 42);
        state.Job.TimelineDeadlines["job.timer"] = 25;
        state.Dots["dot"] = new DotState(18, 100);

        system.ClearRegisteredStatus(state, "buff");

        Assert.Empty(state.Statuses);
        Assert.Equal(2, state.Job.StatusStacks("buff"));
        Assert.Equal(42, state.Job.GetResource("gauge"));
        Assert.Equal(25, state.Job.TimelineDeadlines["job.timer"]);
        Assert.Equal(18, state.Dots["dot"].Remaining);
        Assert.Equal(1234, state.Mp);
    }

    [Fact]
    public void 职业Buff处理不改变公共BuffDoT与MP()
    {
        var job = new BuffStateRegistry(state => state.Job.Statuses);
        var state = new CombatState { Mp = 1234 };
        state.Statuses["buff"] = new StatusState(30, 2);
        state.Job.Statuses["buff"] = new StatusState(10, 2);
        state.Dots["dot"] = new DotState(18, 100);

        job.ClearRegisteredStatus(state, "buff");

        Assert.Empty(state.Job.Statuses);
        Assert.Equal(2, state.StatusStacks("buff"));
        Assert.Equal(30, state.Statuses["buff"].Remaining);
        Assert.Equal(18, state.Dots["dot"].Remaining);
        Assert.Equal(1234, state.Mp);
    }

    [Fact]
    public void 战斗克隆独立复制职业容器及Buff实例()
    {
        var original = new CombatState();
        original.Statuses["public"] = new StatusState(30, 2);
        original.Job.Statuses["job"] = new StatusState(10, 3);
        original.Job.SetResource("gauge", 42);
        original.Job.TimelineDeadlines["job.timer"] = 25;
        original.Dots["dot"] = new DotState(18, 100);

        var clone = original.Clone();
        clone.Statuses["public"].Stacks = 1;
        clone.Job.Statuses["job"].Stacks = 1;
        clone.Job.SetResource("gauge", 0);
        clone.Job.TimelineDeadlines.Clear();
        clone.Dots.Clear();

        Assert.NotSame(original.Job, clone.Job);
        Assert.Equal(2, original.StatusStacks("public"));
        Assert.Equal(3, original.Job.StatusStacks("job"));
        Assert.Equal(42, original.Job.GetResource("gauge"));
        Assert.Equal(25, original.Job.TimelineDeadlines["job.timer"]);
        Assert.Single(original.Dots);
    }

    // ---------- ResourceStateRegistry ----------

    [Fact]
    public void 量谱注册用配置上限覆盖定义上限()
    {
        var registry = new ResourceStateRegistry();
        registry.RegisterJobResources(
            "black_mage",
            new Dictionary<string, JobResourceDefinition>
            {
                ["astral_fire"] = new("astral_fire", "int", DefaultValue: 0, MaxValue: 3),
                ["polyglot_timer"] = new("polyglot_timer", "float", DefaultValue: 0.0, MaxValue: 30.0),
            },
            resourceLimits: new Dictionary<string, double> { ["astral_fire"] = 6 });

        Assert.Equal(6.0, registry.ResourceMaxValue("astral_fire"));
        Assert.Equal(30.0, registry.ResourceMaxValue("polyglot_timer"));
    }

    [Fact]
    public void 量谱注册未知上限抛错()
    {
        var registry = new ResourceStateRegistry();

        Assert.Throws<InvalidOperationException>(() => registry.RegisterJobResources(
            "black_mage",
            new Dictionary<string, JobResourceDefinition>
            {
                ["astral_fire"] = new("astral_fire", "int", DefaultValue: 0, MaxValue: 3),
            },
            resourceLimits: new Dictionary<string, double> { ["unknown"] = 1 }));
    }

    [Fact]
    public void 量谱归一化钳制到上限()
    {
        var registry = new ResourceStateRegistry();
        registry.RegisterJobResources(
            "black_mage",
            new Dictionary<string, JobResourceDefinition>
            {
                ["astral_fire"] = new("astral_fire", "int", DefaultValue: 0, MaxValue: 3),
            });
        var state = new CombatState();

        registry.SetJobResource(state, "astral_fire", 99);

        Assert.Equal(3, registry.GetJobResource(state, "astral_fire"));
    }

    [Fact]
    public void 量谱快照float值四舍五入到4位()
    {
        var registry = new ResourceStateRegistry();
        registry.RegisterJobResources(
            "black_mage",
            new Dictionary<string, JobResourceDefinition>
            {
                ["polyglot_timer"] = new("polyglot_timer", "float", DefaultValue: 0.0, MaxValue: 30.0),
            });
        var state = new CombatState();
        registry.SetJobResource(state, "polyglot_timer", 12.345678);

        var snapshot = registry.BuildJobResourceSnapshot(state);

        Assert.Equal(12.3457, snapshot["polyglot_timer"]);
    }

    [Fact]
    public void 量谱前后差分导出consumed()
    {
        var job = new JobStateRuntime();
        job.Register(new JobStateRegistration(
            "black_mage",
            new Dictionary<string, JobResourceDefinition>
            {
                ["astral_fire"] = new("astral_fire", "int", DefaultValue: 0, MaxValue: 3),
                ["thundercloud_ready"] = new("thundercloud_ready", "bool", DefaultValue: false, MaxValue: 1),
            }, new Dictionary<string, StatusDefinition>()));
        var registry = job.Resources;
        var before = new CombatState();
        registry.SetJobResource(before, "astral_fire", 3);
        registry.SetJobResource(before, "thundercloud_ready", true);
        var after = new CombatState();
        registry.SetJobResource(after, "astral_fire", 1);
        registry.SetJobResource(after, "thundercloud_ready", false);

        var (_, _, consumed) = job.BuildResourceTransition(before, after);

        Assert.Equal(2, consumed["astral_fire"]);
        Assert.Equal(true, consumed["thundercloud_ready"]); // bool: before && !after
    }

    [Fact]
    public void 职业量谱差分使用权威截止时刻投影()
    {
        var job = new JobStateRuntime();
        job.Register(new JobStateRegistration("test_job",
            new Dictionary<string, JobResourceDefinition>
            {
                ["timer_remaining"] = new("timer_remaining", "float", 0.0, MaxValue: 30),
            }, new Dictionary<string, StatusDefinition>()));
        job.Timeline.RegisterCountdown("test.timer", "timer_remaining",
            state => state.Job.TimelineDeadlines.TryGetValue("test.timer", out var deadline) ? deadline : null,
            (_, _, _) => TimelineMutation.Empty);
        var before = new CombatState();
        before.Job.SetResource("timer_remaining", 29.0);
        before.Job.TimelineDeadlines["test.timer"] = 20;
        var after = before.Clone();
        after.Job.SetResource("timer_remaining", 2.0);
        after.Job.TimelineDeadlines["test.timer"] = 10;

        var (previous, current, consumed) = job.BuildResourceTransition(before, after);

        Assert.Equal(20.0, previous["timer_remaining"]);
        Assert.Equal(10.0, current["timer_remaining"]);
        Assert.Equal(10.0, consumed["timer_remaining"]);
    }

    [Fact]
    public void 快照排除internal分组而向量视图按组排序()
    {
        var registry = new ResourceStateRegistry();
        registry.RegisterJobResources(
            "black_mage",
            new Dictionary<string, JobResourceDefinition>
            {
                ["firestarter_ready"] = new("firestarter_ready", "bool", DefaultValue: false, VectorGroup: "buff", MaxValue: 1),
                ["astral_fire"] = new("astral_fire", "int", DefaultValue: 0, MaxValue: 3),
                ["internal_thing"] = new("internal_thing", "int", DefaultValue: 0, VectorGroup: "internal", MaxValue: 3),
            });

        Assert.Equal(new[] { "firestarter_ready", "astral_fire" }, registry.SnapshotResourceKeys);
        Assert.Equal(new[] { "astral_fire" }, registry.VectorResourceKeys("resource"));
        Assert.Equal(new[] { "firestarter_ready" }, registry.VectorResourceKeys("buff"));
    }
}

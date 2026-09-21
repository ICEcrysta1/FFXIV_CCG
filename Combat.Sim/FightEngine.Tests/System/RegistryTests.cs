using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.Models.Definitions;
using Combat.Sim.System.Registries;
using static FightEngine.Tests.System.SystemTestKit;

namespace FightEngine.Tests.System;

/// <summary>注册中心测试：Buff 注册中心与职业量谱注册中心。</summary>
public class RegistryTests
{
    // ---------- BuffStateRegistry ----------

    [Fact]
    public void Buff授予默认使用定义时长与层数()
    {
        var registry = new BuffStateRegistry();
        registry.RegisterSystemStatuses(new Dictionary<string, StatusDefinition>
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
        registry.RegisterSystemStatuses(new Dictionary<string, StatusDefinition>
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
        registry.RegisterSystemStatuses(new Dictionary<string, StatusDefinition>
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
        registry.RegisterSystemStatuses(new Dictionary<string, StatusDefinition>
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
    public void grant状态满层校验拒绝()
    {
        var registry = new BuffStateRegistry();
        registry.RegisterSystemStatuses(new Dictionary<string, StatusDefinition>
        {
            ["burst_potion"] = new("burst_potion", GameId: 1, Duration: 30.0),
        });
        var skill = Skill("potion", kind: ActionKind.Ogcd, behavior: "grant_status",
            appliesStatuses: new[] { "burst_potion" });
        var state = new CombatState();
        registry.GrantRegisteredStatus(state, "burst_potion");

        var result = registry.ValidateSharedBehavior(state, skill);

        Assert.False(result.Ok);
        Assert.Equal("status_already_active", result.Reason);
    }

    [Fact]
    public void grant行为应用授予全部状态()
    {
        var registry = new BuffStateRegistry();
        registry.RegisterSystemStatuses(new Dictionary<string, StatusDefinition>
        {
            ["burst_potion"] = new("burst_potion", GameId: 1, Duration: 30.0),
        });
        var skill = Skill("potion", kind: ActionKind.Ogcd, behavior: "grant_status",
            appliesStatuses: new[] { "burst_potion" });
        var state = new CombatState();

        Assert.True(registry.ApplySharedBehavior(state, skill));
        Assert.True(state.HasStatus("burst_potion"));
    }

    [Fact]
    public void 未注册状态查询抛错()
    {
        var registry = new BuffStateRegistry();

        Assert.Throws<KeyNotFoundException>(() => registry.StatusDefinitionOf("nope"));
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
        var registry = new ResourceStateRegistry();
        registry.RegisterJobResources(
            "black_mage",
            new Dictionary<string, JobResourceDefinition>
            {
                ["astral_fire"] = new("astral_fire", "int", DefaultValue: 0, MaxValue: 3),
                ["thundercloud_ready"] = new("thundercloud_ready", "bool", DefaultValue: false, MaxValue: 1),
            });
        var before = new CombatState();
        registry.SetJobResource(before, "astral_fire", 3);
        registry.SetJobResource(before, "thundercloud_ready", true);
        var after = new CombatState();
        registry.SetJobResource(after, "astral_fire", 1);
        registry.SetJobResource(after, "thundercloud_ready", false);

        var (_, _, consumed) = registry.BuildJobResourceTransition(before, after);

        Assert.Equal(2, consumed["astral_fire"]);
        Assert.Equal(true, consumed["thundercloud_ready"]); // bool: before && !after
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

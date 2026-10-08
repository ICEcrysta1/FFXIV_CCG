using System.Text.Json;
using Combat.Sim.Common;
using Combat.Sim.Config;
using Combat.Sim.Facade;
using Combat.Sim.Models.Timeline;
using Combat.Sim.Outputs;
using Combat.Sim.Policy;

namespace FightEngine.Tests.Facade;

/// <summary>统一接受规则、捕获时刻和同刻事实是状态表的权威，输出不能重新计算历史。</summary>
public sealed class SkillAvailabilityBuilderTests
{
    [Theory]
    [InlineData("black_mage")]
    [InlineData("machinist")]
    public void 真实输出随职业动作布局生成两段字段(string job)
    {
        var simulator = new JobSimulator(CombatStateMachine.FromDefaultConfig(RepoRootLocator.Find(), job));
        var output = simulator.FormatVectorState();
        var keys = Assert.IsType<List<string>>(output["action_keys"]);
        var context = Assert.IsType<Dictionary<string, object?>>(output["current_state_context"]);
        var featureKeys = Assert.IsType<List<string>>(context["skill_availability_feature_keys"]);
        var token = Assert.Single(Assert.IsType<List<Dictionary<string, double[]>>>(context["tokens"]));
        Assert.Equal(keys.Count * 2, token["skill_availability"].Length);
        Assert.Equal(keys.Select(key => $"previous_action_after.{key}")
            .Concat(keys.Select(key => $"request_state.{key}")), featureKeys);
        Assert.Equal(token["skill_availability"].Take(keys.Count), token["skill_availability"].Skip(keys.Count));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void 纯查询与提交在空槽和满槽下逐动作一致(bool occupied)
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        Assert.True(simulator.SubmitAction(0, "gcd_strike").Accepted);
        simulator.AdvanceTo(2.1);
        if (occupied) Assert.True(simulator.SubmitAction(2.1, "gcd_strike").Queued);
        simulator.AdvanceTo(2.2);
        var before = JsonSerializer.Serialize(simulator.CreateSnapshot());
        var output = simulator.FormatVectorState();
        var keys = Assert.IsType<List<string>>(output["action_keys"]);
        var mask = Assert.IsType<List<bool>>(output["action_legal_mask"]);
        var frame = simulator.CaptureModelStateFrame(simulator.GetState());
        var available = simulator.AvailableActionKeysAt(simulator.Time);
        foreach (var (key, index) in keys.Select((key, index) => (key, index)))
        {
            var validation = simulator.ValidateActionAt(simulator.Time, key);
            var result = simulator.Fork().SubmitAction(simulator.Time, key);
            Assert.Equal(result.Accepted, validation.Ok);
            Assert.Equal(result.Reason, validation.Reason);
            Assert.Equal(result.Accepted, mask[index]);
            Assert.Equal(result.Accepted, frame.SkillAvailability[key]);
            Assert.Equal(result.Accepted, available.Contains(key));
        }
        Assert.True(frame.SkillAvailability["ogcd_punch"]);
        Assert.Equal(!occupied, frame.SkillAvailability["gcd_strike"]);
        Assert.Equal(before, JsonSerializer.Serialize(simulator.CreateSnapshot()));
    }

    [Fact]
    public void 资源不足与移动读条冷却限制来自正式接受规则()
    {
        var machine = CombatStateMachine.FromDefaultConfig(RepoRootLocator.Find(), "black_mage");
        var simulator = new JobSimulator(machine);
        var snapshot = simulator.CreateSnapshot();
        snapshot.State.Mp = 0;
        simulator.RestoreSnapshot(snapshot);
        Assert.False(simulator.CaptureModelStateFrame(simulator.GetState()).SkillAvailability["fire_iii"]);
        Assert.Equal("not_enough_mp", simulator.ValidateActionAt(0, "fire_iii").Reason);
        foreach (var key in simulator.ActionKeys)
        {
            var expected = simulator.Fork().SubmitAction(0, key);
            Assert.Equal(expected.Accepted, simulator.ValidateActionAt(0, key).Ok);
        }
        simulator.ApplyExternalEvent(new(0, ExternalCombatEventKinds.MovementChanged, true));
        Assert.False(simulator.CaptureModelStateFrame(simulator.GetState()).SkillAvailability["fire_iii"]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void 移动开始和结束与效果同刻时整批事实先于冻结(bool moving)
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        var action = simulator.SubmitAction(0, "cast_skill");
        if (!moving) simulator.ApplyExternalEvent(new(1, ExternalCombatEventKinds.MovementChanged, true));
        var time = action.EffectTimestamp!.Value;
        simulator.ApplyExternalEvents(new[]
        {
            new ExternalCombatEvent(time, ExternalCombatEventKinds.BossTargetableChanged, true),
            new ExternalCombatEvent(time, ExternalCombatEventKinds.MovementChanged, moving),
            new ExternalCombatEvent(time, ExternalCombatEventKinds.TargetCountChanged, TargetCount: 2),
        });
        var after = simulator.GetState().LastDecisionAfter!;
        Assert.Equal(moving, after.State.Player.IsMoving);
        Assert.Equal(!moving, after.SkillAvailability["cast_skill"]);
        Assert.Equal(2, simulator.GetState().TargetCount);
        Assert.Single(simulator.GetState().History);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void 非法批次不会部分入队或推进(int invalidKind)
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        simulator.SubmitAction(0, "cast_skill");
        var before = JsonSerializer.Serialize(simulator.CreateSnapshot());
        var invalid = invalidKind switch
        {
            0 => new ExternalCombatEvent(2.3, ExternalCombatEventKinds.TargetCountChanged, TargetCount: -1),
            1 => new ExternalCombatEvent(double.NaN, ExternalCombatEventKinds.MovementChanged, true),
            2 => new ExternalCombatEvent(2.4, ExternalCombatEventKinds.MovementChanged, true),
            _ => new ExternalCombatEvent(2.3, ExternalCombatEventKinds.BossTargetableChanged, false),
        };
        Assert.Throws<ArgumentException>(() => simulator.ApplyExternalEvents(new[]
        {
            new ExternalCombatEvent(2.3, ExternalCombatEventKinds.BossTargetableChanged, true), invalid,
        }));
        Assert.Equal(before, JsonSerializer.Serialize(simulator.CreateSnapshot()));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void 子分支生效捕获只读取自己的队列并覆盖恢复及二次分支(bool withoutHistory)
    {
        var parent = new JobSimulator(FacadeKit.BuildMachine());
        parent.SubmitAction(0, "cast_skill");
        var child = withoutHistory ? parent.ForkWithoutHistory() : parent.Fork();
        var restored = new JobSimulator(FacadeKit.BuildMachine());
        restored.RestoreSnapshot(child.CreateSnapshot());
        var grandchild = child.Fork();
        Assert.True(parent.SubmitAction(2.3, "gcd_strike").Queued);
        Assert.True(parent.HasQueuedAction());
        foreach (var branch in new[] { child, restored, grandchild })
        {
            branch.AdvanceTo(2.3);
            Assert.False(branch.HasQueuedAction());
            Assert.True(branch.GetState().LastDecisionAfter!.SkillAvailability["gcd_strike"]);
        }
    }

    [Fact]
    public void 冻结包含请求前状态且后续冷却恢复不回填历史()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        simulator.SubmitAction(0, "cd_skill");
        var first = Assert.Single(simulator.GetState().History).ModelState!;
        Assert.True(first.RequestState.SkillAvailability["cd_skill"]);
        Assert.False(simulator.GetState().LastDecisionAfter!.SkillAvailability["cd_skill"]);
        Assert.Throws<NotSupportedException>(() =>
            ((IDictionary<string, bool>)first.RequestState.SkillAvailability)["cd_skill"] = false);
        simulator.AdvanceTo(31);
        Assert.True(simulator.CaptureModelStateFrame(simulator.GetState()).SkillAvailability["cd_skill"]);
        Assert.True(first.RequestState.SkillAvailability["cd_skill"]);
        Assert.False(simulator.GetState().LastDecisionAfter!.SkillAvailability["cd_skill"]);
    }

    [Fact]
    public void 完整策略布局扩为136且真实和策略输出缓存不串列()
    {
        var simulator = new JobSimulator(CombatStateMachine.FromDefaultConfig(RepoRootLocator.Find(), "black_mage"));
        var policy = PolicySession.Create(RepoRootLocator.Find(), simulator);
        simulator.SubmitAction(0, "fire_iii");
        simulator.AdvanceTo(4);
        policy.Record(simulator, 4, "ogcd_wait", 5);
        var real = simulator.FormatVectorState();
        var frozenReal = JsonSerializer.Serialize(real);
        var full = policy.BuildVectorContext(simulator, 5);
        var keys = Assert.IsType<List<string>>(full["action_keys"]);
        Assert.Equal(25, keys.Count);
        foreach (var name in new[] { "current_state_context", "state_history_context" })
        {
            var context = Assert.IsType<Dictionary<string, object?>>(full[name]);
            var featureKeys = Assert.IsType<List<string>>(context["skill_availability_feature_keys"]);
            Assert.Equal(50, featureKeys.Count);
            Assert.Equal(SchemaConfigLoader.Instance.StateSnapshots.SelectMany(prefix => keys.Select(key => $"{prefix}.{key}")), featureKeys);
            foreach (var token in Assert.IsType<List<Dictionary<string, double[]>>>(context["tokens"]))
            {
                Assert.Equal(136, token.Values.Sum(values => values.Length));
                Assert.All(token["skill_availability"], value => Assert.True(value is 0 or 1));
                Assert.Equal(1, token["skill_availability"][featureKeys.IndexOf("request_state.ogcd_wait")]);
            }
        }
        Assert.Equal(frozenReal, JsonSerializer.Serialize(real));
        Assert.Equal(frozenReal, JsonSerializer.Serialize(simulator.FormatVectorState()));
        Assert.Equal(JsonSerializer.Serialize(full), JsonSerializer.Serialize(policy.BuildVectorContext(simulator, 5)));
        var realSkill = Assert.IsType<List<Dictionary<string, object?>>>(real["skill_history_context"]);
        var fullSkill = Assert.IsType<List<Dictionary<string, object?>>>(full["skill_history_context"]);
        Assert.Equal(JsonSerializer.Serialize(realSkill[0]), JsonSerializer.Serialize(fullSkill[0]));
        Assert.Equal(JsonSerializer.Serialize(real["scene_context"]), JsonSerializer.Serialize(full["scene_context"]));
    }
}

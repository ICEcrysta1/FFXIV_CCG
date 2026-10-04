using System.Text.Json;
using Combat.Sim.Common;
using Combat.Sim.Facade;
using Combat.Sim.Outputs;
using Combat.Sim.Policy;
using FightEngine.Tests.Facade;

namespace FightEngine.Tests.Outputs;

/// <summary>跨步快照、请求冻结、统一 wait、窗口和快照恢复的真实时间线回归。</summary>
public sealed class ModelStateSemanticsTests
{
    [Fact]
    public void 首步回退且执行后不回填历史请求状态()
    {
        var simulator = Create();
        EqualSegments(Context(simulator.FormatVectorState(), "current_state_context"));
        var action = simulator.SubmitAction(0, "fire_iii");
        Assert.True(action.Accepted);
        simulator.AdvanceTo(action.EffectTimestamp!.Value);
        var output = simulator.FormatVectorState();
        var history = Context(output, "state_history_context");
        EqualSegments(history);
        Assert.Equal(0, Value(history, "request_state.time_seconds"));
        Assert.Equal(10000, Value(history, "request_state.mp"));
        Assert.Equal(8000, simulator.GetState().Mp);
        var metrics = Assert.Single((List<Dictionary<string, double>>)history["execution_metrics"]!);
        Assert.True(metrics["cumulative_potency"] > 0);
        var frozen = JsonSerializer.Serialize(history);
        simulator.AdvanceTo(10);
        Assert.Equal(frozen, JsonSerializer.Serialize(Context(simulator.FormatVectorState(), "state_history_context")));
    }

    [Fact]
    public void Wait与真实动作使用相同两段状态和真实时间锚点()
    {
        var simulator = Create();
        var policy = PolicySession.Create(RepoRootLocator.Find(), simulator);
        var action = simulator.SubmitAction(0, "fire_iii");
        simulator.AdvanceTo(4);
        var decision = policy.Record(simulator, 4, "ogcd_wait", 100);
        Assert.Equal(4, decision.StateAfter.Time);
        Assert.Equal(decision.StateBefore.Mp, decision.StateAfter.Mp);
        Assert.Equal(4, simulator.Time);
        var before = policy.BuildVectorContext(simulator, 100);
        var history = Context(before, "state_history_context");
        Assert.Equal(action.EffectTimestamp!.Value, Value(history, "previous_action_after.time_seconds", 1));
        Assert.Equal(4, Value(history, "request_state.time_seconds", 1));
        var frozen = JsonSerializer.Serialize(history);
        simulator.AdvanceTo(5);
        var next = policy.BuildVectorContext(simulator, 200);
        var current = Context(next, "current_state_context");
        Assert.Equal(4, Value(current, "previous_action_after.time_seconds"));
        Assert.Equal(5, Value(current, "request_state.time_seconds"));
        Assert.Equal(history["player_state_feature_keys"], current["player_state_feature_keys"]);
        Assert.Equal(frozen, JsonSerializer.Serialize(Context(next, "state_history_context")));
        Assert.True(simulator.SubmitAction(5, "blizzard_iii").Accepted);
        simulator.AdvanceTo(9);
        var final = policy.BuildVectorContext(simulator, 100);
        var third = Context(final, "state_history_context");
        Assert.Equal(4, Value(third, "previous_action_after.time_seconds", 2));
        Assert.Equal(5, Value(third, "request_state.time_seconds", 2));
    }

    [Fact]
    public void 前一步未生效时回退且旧效果不能覆盖后续Wait基准()
    {
        var simulator = Create();
        var policy = PolicySession.Create(RepoRootLocator.Find(), simulator);
        simulator.SubmitAction(0, "fire_iii");
        simulator.AdvanceTo(0.1);
        var decision = policy.Record(simulator, 0.1, "ogcd_wait", 10);
        Assert.Equal(decision.ModelState.PreviousActionAfter, decision.ModelState.RequestState);
        simulator.AdvanceTo(4);
        var output = policy.BuildVectorContext(simulator, 10);
        var current = Context(output, "current_state_context");
        Assert.Equal(0.1, Value(current, "previous_action_after.time_seconds"));
        Assert.Equal(10000, Value(current, "previous_action_after.mp"));
        Assert.Equal(4, Value(current, "request_state.time_seconds"));
        Assert.Equal(8000, Value(current, "request_state.mp"));
        var history = Context(output, "state_history_context");
        // effect 排序仍保留，但模型状态按请求时冻结，wait 没有读到后来生效的技能结果。
        EqualSegments(history, 0);
        Assert.Equal(0.1, Value(history, "request_state.time_seconds", 0));
    }

    [Fact]
    public void 排队动作在请求时冻结而非接受或生效时重新取值()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        simulator.SubmitAction(0, "cast_skill");
        var queued = simulator.SubmitAction(2.76, "gcd_strike");
        Assert.True(queued.Queued);
        simulator.AdvanceTo(2.8);
        var entry = simulator.GetState().History[1];
        Assert.Equal(2.3, entry.ModelState!.PreviousActionAfter.Player.TimeSeconds, 8);
        Assert.Equal(2.76, entry.ModelState.RequestState.Player.TimeSeconds, 8);
        Assert.Equal(2.8, entry.StateAfter.Player.TimeSeconds, 8);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void 窗口外快照仍可用且Fork与恢复保持相同输入(int historyLimit)
    {
        var simulator = Create(historyLimit);
        var policy = PolicySession.Create(RepoRootLocator.Find(), simulator);
        simulator.SubmitAction(0, "fire_iii");
        simulator.AdvanceTo(4);
        policy.Record(simulator, 4, "ogcd_wait", 5);
        var snapshot = simulator.CreateSnapshot();
        var fork = simulator.Fork();
        simulator.AdvanceTo(5);
        fork.AdvanceTo(5);
        Assert.Equal(JsonSerializer.Serialize(simulator.FormatVectorState()), JsonSerializer.Serialize(fork.FormatVectorState()));
        var expected = simulator.FormatVectorState();
        Assert.Equal(4, Value(Context(expected, "current_state_context"), "previous_action_after.time_seconds"));
        simulator.RestoreSnapshot(snapshot);
        simulator.AdvanceTo(5);
        Assert.Equal(JsonSerializer.Serialize(expected), JsonSerializer.Serialize(simulator.FormatVectorState()));
        Assert.True(simulator.GetState().History.Count <= historyLimit);
    }

    [Fact]
    public void 同戳请求按身份关联而非时间戳或数组偏移()
    {
        var simulator = new JobSimulator(FacadeKit.BuildMachine());
        simulator.SubmitAction(0, "ogcd_punch");
        simulator.SubmitAction(0, "ogcd_punch");
        var entries = simulator.GetState().History;
        Assert.Equal(0, entries[0].ModelState!.PreviousActionAfter.Target.CumulativePotency);
        Assert.Equal(200, entries[1].ModelState!.PreviousActionAfter.Target.CumulativePotency);
        Assert.Equal(200, entries[1].ModelState!.RequestState.Target.CumulativePotency);
        Assert.Equal(400, entries[1].StateAfter.Target.CumulativePotency);
    }

    [Fact]
    public void Wait期间回蓝只改变下一请求段并保留动作后MP()
    {
        var simulator = Create();
        var initial = simulator.CreateSnapshot();
        initial.State.Mp = 6000;
        simulator.RestoreSnapshot(initial);
        var policy = PolicySession.Create(RepoRootLocator.Find(), simulator);
        policy.Record(simulator, 0, "ogcd_wait", 100);
        simulator.AdvanceTo(3);
        var current = Context(policy.BuildVectorContext(simulator, 100), "current_state_context");
        Assert.Equal(6000, Value(current, "previous_action_after.mp"));
        Assert.True(Value(current, "request_state.mp") > 6000);
        Assert.Equal(0, Value(current, "previous_action_after.time_seconds"));
        Assert.Equal(3, Value(current, "request_state.time_seconds"));
    }

    [Fact]
    public void Wait期间DoT量谱与Buff变化不回填动作后快照()
    {
        var simulator = Create();
        var policy = PolicySession.Create(RepoRootLocator.Find(), simulator);
        simulator.SubmitAction(0, "fire_iii");
        simulator.AdvanceTo(4);
        Assert.True(simulator.SubmitAction(4, "high_thunder").Accepted);
        Assert.True(simulator.SubmitAction(4, "potion").Accepted);
        simulator.AdvanceTo(4.25);
        policy.Record(simulator, 4.25, "ogcd_wait", 100);
        var before = policy.BuildVectorContext(simulator, 100);
        var frozen = JsonSerializer.Serialize(Context(before, "state_history_context"));
        simulator.AdvanceTo(8);
        var after = policy.BuildVectorContext(simulator, 100);
        var current = Context(after, "current_state_context");
        double Read(string group, string key) => OutputsTestKit.HistoryVectorValue(current, group, key);
        Assert.True(Read("resource_state", "request_state.polyglot_timer") > Read("resource_state", "previous_action_after.polyglot_timer"));
        Assert.True(Read("buff_state", "request_state.system.burst_potion.remaining_seconds") < Read("buff_state", "previous_action_after.system.burst_potion.remaining_seconds"));
        Assert.True(Read("target_buff_state", "request_state.target.cumulative_dot_potency") > Read("target_buff_state", "previous_action_after.target.cumulative_dot_potency"));
        Assert.Equal(frozen, JsonSerializer.Serialize(Context(after, "state_history_context")));
    }

    private static JobSimulator Create(int? historyLimit = null) => new(
        CombatStateMachine.FromDefaultConfig(RepoRootLocator.Find(), "black_mage", maxHistory: historyLimit));

    private static Dictionary<string, object?> Context(Dictionary<string, object?> output, string key) =>
        (Dictionary<string, object?>)output[key]!;

    private static double Value(Dictionary<string, object?> context, string key, int token = 0) =>
        OutputsTestKit.HistoryVectorValue(context, "player_state", key, token);

    private static void EqualSegments(Dictionary<string, object?> context, int token = 0)
    {
        var tokens = (List<Dictionary<string, double[]>>)context["tokens"]!;
        foreach (var values in tokens[token].Values)
            Assert.Equal(values.Take(values.Length / 2), values.Skip(values.Length / 2));
    }
}

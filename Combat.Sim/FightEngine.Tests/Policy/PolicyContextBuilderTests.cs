using Combat.Sim.Common;
using Combat.Sim.Models.Policy;
using Combat.Sim.Outputs;
using Combat.Sim.Policy;

namespace FightEngine.Tests.Policy;

public sealed class PolicyContextBuilderTests
{
    [Fact]
    public void 等待输出缓存随窗口淘汰并在同刻恢复新快照后重新生成()
    {
        var (simulator, registry) = CreateRuntime();
        var history = new PolicyDecisionHistory(registry, new HistoryRetention(2));
        var builder = new PolicyContextBuilder(registry);
        for (var index = 0; index < 20; index++)
        {
            simulator.AdvanceTo(index);
            history.Record(simulator, index, "ogcd_wait", index + 1);
            var actual = builder.BuildVectorContext(simulator, history, index + 1);
            var expected = new PolicyContextBuilder(registry).BuildVectorContext(simulator, history, index + 1);
            Assert.Equal(global::System.Text.Json.JsonSerializer.Serialize(expected),
                global::System.Text.Json.JsonSerializer.Serialize(actual));
        }
        var restored = history.CreateSnapshot();
        restored[0].StateBefore.Mp = 123;
        restored[0].StateAfter.Mp = 456;
        history.RestoreSnapshot(restored);
        var rebuilt = builder.BuildVectorContext(simulator, history, simulator.Time + 1);
        var fresh = new PolicyContextBuilder(registry).BuildVectorContext(simulator, history, simulator.Time + 1);
        Assert.Equal(global::System.Text.Json.JsonSerializer.Serialize(fresh),
            global::System.Text.Json.JsonSerializer.Serialize(rebuilt));
        Assert.Equal(2, Assert.IsType<List<Dictionary<string, object?>>>(
            rebuilt[OutputContextSchema.SkillHistoryContextKey]).Count);
    }

    private static (JobSimulator Simulator, PolicyActionRegistry Registry) CreateRuntime()
    {
        var root = RepoRootLocator.Find();
        var machine = CombatStateMachine.FromDefaultConfig(root, "black_mage");
        return (new JobSimulator(machine), PolicyActionRegistry.Load(root, machine.SkillBook));
    }

    [Fact]
    public void Wait只存在于Policy层且上下文仍有25个候选()
    {
        var (simulator, registry) = CreateRuntime();
        Assert.False(simulator.Rules.SkillBook.Contains("ogcd_wait"));
        Assert.Throws<KeyNotFoundException>(() => simulator.SubmitAction(0.0, "ogcd_wait"));

        var history = new PolicyDecisionHistory(registry);
        var before = simulator.CreateSnapshot();
        history.Record(simulator, 0.0, "ogcd_wait", 2.5);

        var afterRecord = simulator.CreateSnapshot();
        Assert.Equal(before.State.Time, afterRecord.State.Time);
        Assert.Equal(before.NextSequence, afterRecord.NextSequence);
        Assert.Equal(before.PendingEvents.Select(item => (item.Timestamp, item.Kind, item.Sequence)),
            afterRecord.PendingEvents.Select(item => (item.Timestamp, item.Kind, item.Sequence)));
        Assert.Empty(afterRecord.State.History);

        var output = new PolicyContextBuilder(registry).BuildVectorContext(simulator, history, 2.5);
        var candidates = Assert.IsType<List<Dictionary<string, object?>>>(
            output[OutputContextSchema.CandidateSkillContextKey]);
        Assert.Equal(25, candidates.Count);
        Assert.Equal("ogcd_wait", candidates[0]["skill_key"]);

        var candidateStates = Assert.IsType<Dictionary<string, object?>>(
            output[OutputContextSchema.CandidateStateContextKey]);
        Assert.Equal(25, Assert.IsAssignableFrom<IReadOnlyList<object>>(candidateStates["tokens"]).Count);
        var skillHistory = Assert.IsType<List<Dictionary<string, object?>>>(
            output[OutputContextSchema.SkillHistoryContextKey]);
        Assert.Single(skillHistory);
        Assert.Equal("ogcd_wait", skillHistory[0]["skill_key"]);
        Assert.Equal(0.0, simulator.Time);
        Assert.Empty(simulator.GetState().History);
    }

    [Fact]
    public void Policy注册拒绝与真实技能冲突()
    {
        var root = RepoRootLocator.Find();
        var machine = CombatStateMachine.FromDefaultConfig(root, "black_mage");
        var collision = new PolicyActionDefinition(
            "potion", 0, "冲突", "ogcd", "end_weave_window", 1.0, Array.Empty<string>());

        Assert.Throws<InvalidOperationException>(() =>
            new PolicyActionRegistry(new[] { collision }, machine.SkillBook));
    }

    [Fact]
    public void Policy决策只能记录当前观测时刻()
    {
        var (simulator, registry) = CreateRuntime();
        var history = new PolicyDecisionHistory(registry);

        Assert.Throws<InvalidOperationException>(() =>
            history.Record(simulator, 0.1, "ogcd_wait", 2.5));
        Assert.Empty(history.Entries);
        Assert.Equal(0.0, simulator.Time);
    }

    [Fact]
    public void Policy记录与快照不嵌套持有动作历史且保留最新状态()
    {
        var (simulator, registry) = CreateRuntime();
        simulator.SubmitAction(0, "blizzard_iii");
        simulator.AdvanceTo(4);
        Assert.Single(simulator.GetState().History);
        var history = new PolicyDecisionHistory(registry, new HistoryRetention(2));
        for (var index = 0; index < 10; index++)
        {
            simulator.AdvanceTo(4 + index);
            var decision = history.Record(simulator, 4 + index, "ogcd_wait", 5 + index);
            Assert.Empty(decision.StateBefore.History);
            Assert.Empty(decision.StateAfter.History);
        }
        Assert.Equal(new[] { 12.0, 13.0 }, history.Entries.Select(item => item.Timestamp));
        var fork = history.Fork();
        Assert.Equal(2, fork.Entries.Count);
        Assert.All(fork.Entries, item => Assert.Empty(item.StateBefore.History));
        Assert.Single(simulator.GetState().History);
    }
}

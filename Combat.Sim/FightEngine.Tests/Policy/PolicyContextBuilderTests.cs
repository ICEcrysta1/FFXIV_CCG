using Combat.Sim.Common;
using Combat.Sim.Models.Policy;
using Combat.Sim.Outputs;
using Combat.Sim.Policy;

namespace FightEngine.Tests.Policy;

public sealed class PolicyContextBuilderTests
{
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
}

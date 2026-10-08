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
            Assert.Equal((long)index + 1, actual[OutputContextSchema.HistoryCursorKey]);
            Assert.Equal(actual[OutputContextSchema.HistoryCursorKey],
                builder.BuildVectorContext(simulator, history, index + 1)[OutputContextSchema.HistoryCursorKey]);
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

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void 同刻真实技能与等待保留记录顺序且历史前缀不变(bool waitFirst)
    {
        var (simulator, registry) = CreateRuntime();
        Assert.True(simulator.SubmitAction(0, "fire_iii", actualCastSeconds: 0).Accepted);
        const double timestamp = 0.16049999999999998;
        simulator.AdvanceTo(timestamp);
        var history = new PolicyDecisionHistory(registry);
        var builder = new PolicyContextBuilder(registry);

        if (waitFirst) history.Record(simulator, timestamp, "ogcd_wait", 1);
        else Assert.True(simulator.SubmitAction(timestamp, "lucid_dreaming").Accepted);
        var prefix = HistoryRows(builder.BuildVectorContext(simulator, history, 1));

        if (waitFirst) Assert.True(simulator.SubmitAction(timestamp, "lucid_dreaming").Accepted);
        else history.Record(simulator, timestamp, "ogcd_wait", 1);
        var output = builder.BuildVectorContext(simulator, history, 1);
        Assert.Equal(3L, output[OutputContextSchema.HistoryCursorKey]);
        var rows = HistoryRows(output);
        Assert.Equal(prefix, rows.Take(prefix.Length));
        var skills = Assert.IsType<List<Dictionary<string, object?>>>(output[OutputContextSchema.SkillHistoryContextKey]);
        Assert.Equal(waitFirst
            ? new[] { "fire_iii", "ogcd_wait", "lucid_dreaming" }
            : new[] { "fire_iii", "lucid_dreaming", "ogcd_wait" }, skills.Select(skill => (string)skill["skill_key"]!));
        Assert.All(skills, skill => Assert.DoesNotContain("time_seconds", skill.Keys));
        Assert.All(skills, skill => Assert.Equal(skills[0].Keys, skill.Keys));
        var realEntries = simulator.GetState().History;
        Assert.Equal(new[] { 1L, waitFirst ? 3L : 2L }, realEntries.Select(entry => entry.HistorySequence));
        Assert.Equal(waitFirst ? 2L : 3L, Assert.Single(history.Entries).HistorySequence);
        // 技能时间字段删除后，同刻顺序仍由内部序号决定，真实请求时刻保留在状态中。
        var context = Assert.IsType<Dictionary<string, object?>>(output[OutputContextSchema.StateHistoryContextKey]);
        var states = Assert.IsType<List<Dictionary<string, double[]>>>(context["tokens"]);
        var timeIndex = Assert.IsType<List<string>>(context["player_state_feature_keys"]).IndexOf("request_state.time_seconds");
        Assert.True(timeIndex >= 0);
        Assert.Equal(timestamp, states[1]["player_state"][timeIndex]);
        Assert.Equal(timestamp, states[2]["player_state"][timeIndex]);
    }

    [Fact]
    public void 历史序号随Fork与恢复保存且重新记录后不重复()
    {
        var (simulator, registry) = CreateRuntime();
        Assert.True(simulator.SubmitAction(0, "fire_iii", actualCastSeconds: 0).Accepted);
        simulator.AdvanceTo(0.5);
        var history = new PolicyDecisionHistory(registry);
        var wait = history.Record(simulator, 0.5, "ogcd_wait", 1);
        Assert.Equal(2, wait.HistorySequence);
        var snapshot = simulator.CreateSnapshot();
        var historySnapshot = history.CreateSnapshot();
        var fork = simulator.Fork();
        var forkHistory = history.Fork();
        var builder = new PolicyContextBuilder(registry);
        Assert.Equal(2L, builder.BuildVectorContext(simulator, history, 1)[OutputContextSchema.HistoryCursorKey]);
        Assert.Equal(2L, builder.BuildVectorContext(fork, forkHistory, 1)[OutputContextSchema.HistoryCursorKey]);

        Assert.True(simulator.SubmitAction(0.5, "lucid_dreaming").Accepted);
        var expected = HistoryRows(builder.BuildVectorContext(simulator, history, 1));
        Assert.Equal(3, simulator.GetState().History[^1].HistorySequence);
        Assert.True(fork.SubmitAction(0.5, "lucid_dreaming").Accepted);
        Assert.Equal(3, fork.GetState().History[^1].HistorySequence);
        Assert.Equal(expected, HistoryRows(new PolicyContextBuilder(registry).BuildVectorContext(fork, forkHistory, 1)));

        simulator.RestoreSnapshot(snapshot);
        history.RestoreSnapshot(historySnapshot);
        Assert.Equal(2L, builder.BuildVectorContext(simulator, history, 1)[OutputContextSchema.HistoryCursorKey]);
        Assert.True(simulator.SubmitAction(0.5, "lucid_dreaming").Accepted);
        Assert.Equal(3, simulator.GetState().History[^1].HistorySequence);
        Assert.Equal(expected, HistoryRows(builder.BuildVectorContext(simulator, history, 1)));
    }

    [Fact]
    public void 同刻历史裁剪保留最后记录的真实技能()
    {
        var root = RepoRootLocator.Find();
        var machine = CombatStateMachine.FromDefaultConfig(root, "black_mage", maxHistory: 1);
        var simulator = new JobSimulator(machine);
        var registry = PolicyActionRegistry.Load(root, machine.SkillBook);
        var history = new PolicyDecisionHistory(registry, new HistoryRetention(1));
        Assert.True(simulator.SubmitAction(0, "fire_iii", actualCastSeconds: 0).Accepted);
        simulator.AdvanceTo(0.5);
        history.Record(simulator, 0.5, "ogcd_wait", 1);
        Assert.True(simulator.SubmitAction(0.5, "lucid_dreaming").Accepted);

        var output = new PolicyContextBuilder(registry).BuildVectorContext(simulator, history, 1);
        var skill = Assert.Single(Assert.IsType<List<Dictionary<string, object?>>>(output[OutputContextSchema.SkillHistoryContextKey]));
        // 留存窗口只有一条，累计 canonical 游标仍为三；读取不会增加游标。
        Assert.Equal(3L, output[OutputContextSchema.HistoryCursorKey]);
        Assert.Equal("lucid_dreaming", skill["skill_key"]);
        Assert.Equal(3, Assert.Single(simulator.GetState().History).HistorySequence);
    }

    private static string[] HistoryRows(Dictionary<string, object?> output)
    {
        var skills = (List<Dictionary<string, object?>>)output[OutputContextSchema.SkillHistoryContextKey]!;
        var context = (Dictionary<string, object?>)output[OutputContextSchema.StateHistoryContextKey]!;
        var states = (List<Dictionary<string, double[]>>)context["tokens"]!;
        var metrics = (List<Dictionary<string, double>>)context["execution_metrics"]!;
        return skills.Select((skill, index) => global::System.Text.Json.JsonSerializer.Serialize(
            new object[] { skill, states[index], metrics[index] })).ToArray();
    }

    [Fact]
    public void Wait与真实技能使用同一字段且保持零消耗和独立状态时间()
    {
        var (simulator, registry) = CreateRuntime();
        Assert.True(simulator.SubmitAction(0, "fire_iii", actualCastSeconds: 0).Accepted);
        simulator.AdvanceTo(0.5);
        var history = new PolicyDecisionHistory(registry);
        history.Record(simulator, 0.5, "ogcd_wait", 1);
        var output = new PolicyContextBuilder(registry).BuildVectorContext(simulator, history, 1);
        var skills = Assert.IsType<List<Dictionary<string, object?>>>(output[OutputContextSchema.SkillHistoryContextKey]);
        Assert.Equal(2, skills.Count);
        Assert.Equal(skills[0].Keys, skills[1].Keys);
        Assert.All(skills, skill => Assert.DoesNotContain("time_seconds", skill.Keys));
        var wait = skills[1];
        Assert.Equal("ogcd_wait", wait["skill_key"]);
        Assert.Equal(0, wait["kind"]);
        Assert.Equal(0, wait["actual_mp_cost"]);
        Assert.Equal(0.0, wait["potency"]);
        Assert.True(Assert.IsType<bool>(wait["is_legal"]));
        Assert.Equal("", wait["invalid_reason"]);
        Assert.Equal(0.0, Assert.IsType<Dictionary<string, object?>>(wait["cast_time"])["seconds"]);
        Assert.Equal(0.0, Assert.IsType<Dictionary<string, object?>>(wait["gcd_window"])["seconds"]);
        var resources = Assert.IsType<Dictionary<string, object?>>(wait["job_resources_consumed"]);
        Assert.Equal(Assert.IsType<Dictionary<string, object?>>(skills[0]["job_resources_consumed"]).Keys, resources.Keys);
        Assert.All(resources.Values, value => Assert.Equal(0.0, Convert.ToDouble(value)));
        Assert.Equal(0.5, Assert.Single(history.Entries).Timestamp);

        var context = Assert.IsType<Dictionary<string, object?>>(output[OutputContextSchema.StateHistoryContextKey]);
        var states = Assert.IsType<List<Dictionary<string, double[]>>>(context["tokens"]);
        var keys = Assert.IsType<List<string>>(context["player_state_feature_keys"]);
        Assert.Equal(0.0, states[1]["player_state"][keys.IndexOf("previous_action_after.time_seconds")]);
        Assert.Equal(0.5, states[1]["player_state"][keys.IndexOf("request_state.time_seconds")]);
    }

    [Fact]
    public void Wait只存在于Policy层且固定词表包含25个动作()
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
        var keys = Assert.IsType<List<string>>(output[OutputContextSchema.ActionKeysKey]);
        Assert.Equal(25, keys.Count);
        Assert.Equal(keys.OrderBy(key => key, StringComparer.Ordinal), keys);
        Assert.Contains("ogcd_wait", keys);
        var legalMask = Assert.IsType<List<bool>>(output[OutputContextSchema.ActionLegalMaskKey]);
        Assert.True(legalMask[keys.IndexOf("ogcd_wait")]);

        var currentState = Assert.IsType<Dictionary<string, object?>>(
            output[OutputContextSchema.CurrentStateContextKey]);
        Assert.Single(Assert.IsAssignableFrom<IReadOnlyList<object>>(currentState["tokens"]));
        var laterOutput = new PolicyContextBuilder(registry).BuildVectorContext(simulator, history, 100);
        Assert.Equal(global::System.Text.Json.JsonSerializer.Serialize(currentState),
            global::System.Text.Json.JsonSerializer.Serialize(laterOutput[OutputContextSchema.CurrentStateContextKey]));
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

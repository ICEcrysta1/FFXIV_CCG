using System.Text.Json;
using Combat.Sim.Common;
using Combat.Sim.Outputs;
using Combat.Sim.Sessions;

namespace FightEngine.Tests.Sessions;

public sealed class SimulationDeltaObservationTests
{
    private static string Root => RepoRootLocator.Find();

    [Theory]
    [InlineData("black_mage", "blizzard_iii")]
    [InlineData("machinist", "heated_split_shot")]
    public void 多次增量复原完整观测且等长和多条新增均保持三通道对齐(string job, string action)
    {
        using var engine = new SimulationEngine(Root, job);
        using var session = engine.CreateSession(null, fightRemaining: 60);
        var history = new ReconstructedHistory();
        var cursor = history.AssertAndAppend(session, 0, 1, 0);
        Assert.Equal(0, cursor);
        for (var step = 0; step < 4; step++)
        {
            var timestamp = step * 6.0;
            Assert.True(session.SubmitAction(timestamp, action, actualCastSeconds: 0).Value.Accepted);
            var afterAction = history.AssertAndAppend(session, timestamp, timestamp + 1, cursor);
            Assert.Equal(cursor + 1, afterAction);
            cursor = history.AssertAndAppend(session, timestamp, timestamp + 1, afterAction);
            Assert.Equal(afterAction, cursor);

            session.RecordPolicyAction(timestamp, "ogcd_wait", timestamp + 1);
            session.RecordPolicyAction(timestamp, "ogcd_wait", timestamp + 1);
            var afterWaits = history.AssertAndAppend(session, timestamp, timestamp + 1, cursor);
            Assert.Equal(cursor + 2, afterWaits);
            cursor = afterWaits;
        }
        Assert.Equal(12, cursor);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void 同刻真实技能和等待按照实际记录顺序输出增量(bool waitFirst)
    {
        using var engine = new SimulationEngine(Root, "black_mage");
        using var session = engine.CreateSession(null);
        var history = new ReconstructedHistory();
        Assert.True(session.SubmitAction(0, "fire_iii", actualCastSeconds: 0).Value.Accepted);
        var cursor = history.AssertAndAppend(session, 0, 1, 0);
        const double timestamp = 0.16049999999999998;
        session.AdvanceTo(timestamp);

        if (waitFirst) session.RecordPolicyAction(timestamp, "ogcd_wait", 1);
        else Assert.True(session.SubmitAction(timestamp, "lucid_dreaming").Value.Accepted);
        cursor = history.AssertAndAppend(session, timestamp, 1, cursor);
        if (waitFirst) Assert.True(session.SubmitAction(timestamp, "lucid_dreaming").Value.Accepted);
        else session.RecordPolicyAction(timestamp, "ogcd_wait", 1);
        cursor = history.AssertAndAppend(session, timestamp, 1, cursor);

        Assert.Equal(3, cursor);
        Assert.Equal(waitFirst
            ? new[] { "fire_iii", "ogcd_wait", "lucid_dreaming" }
            : new[] { "fire_iii", "lucid_dreaming", "ogcd_wait" },
            history.Skills.Select(skill => (string)skill["skill_key"]!));
    }

    [Theory]
    [InlineData("black_mage", "blizzard_iii")]
    [InlineData("machinist", "heated_split_shot")]
    public void 并发多队列增量游标和历史内容独立(string job, string action)
    {
        using var engine = new SimulationEngine(Root, job, 4);
        var sessions = Enumerable.Range(0, 4)
            .Select(index => engine.CreateSession(null, initialTimestamp: index * 100, fightRemaining: 60)).ToArray();
        Parallel.For(0, sessions.Length, index =>
        {
            var session = sessions[index];
            var history = new ReconstructedHistory();
            long cursor = 0;
            for (var step = 0; step < 6; step++)
            {
                var timestamp = index * 100 + step * 6.0;
                Assert.True(session.SubmitAction(timestamp, action, actualCastSeconds: 0).Value.Accepted);
                for (var wait = 0; wait < index; wait++)
                    session.RecordPolicyAction(timestamp, "ogcd_wait", timestamp + 1);
                cursor = history.AssertAndAppend(session, timestamp, timestamp + 1, cursor);
                Assert.Equal((long)(step + 1) * (index + 1), cursor);
            }
            Assert.Equal(6 * (index + 1), history.Skills.Count);
        });
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(8)]
    public void 有限历史会话拒绝增量但普通观测不变(int historyLimit)
    {
        using var engine = new SimulationEngine(Root, "machinist");
        using var session = engine.CreateSession(historyLimit);
        Assert.Throws<InvalidOperationException>(() => session.ObserveVectorDeltaAt(0, 1, 0));
        var ordinary = session.ObserveAt(0, "vector", 1).Value;
        Assert.DoesNotContain("history_start", ordinary.Keys);
        Assert.Equal(0L, ordinary["history_cursor"]);
    }

    [Theory]
    [InlineData(-1L)]
    [InlineData(1L)]
    public void 空历史拒绝负游标和未来游标(long afterSequence)
    {
        using var engine = new SimulationEngine(Root, "machinist");
        using var session = engine.CreateSession(null);
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => session.ObserveVectorDeltaAt(0, 1, afterSequence));
        Assert.Equal("afterSequence", error.ParamName);
        Assert.Equal(0L, session.ObserveVectorDeltaAt(0, 1, 0).Value["history_cursor"]);
    }

    [Fact]
    public void 未来游标在推进或结算前拒绝且状态零变更()
    {
        using var engine = new SimulationEngine(Root, "black_mage");
        using var session = engine.CreateSession(null);
        Assert.True(session.SubmitAction(0, "fire_iii").Value.Accepted);
        var before = JsonSerializer.Serialize(session.ObserveAt(0, "vector", 10).Value);
        var statistics = session.GetStatistics();

        Assert.Throws<ArgumentOutOfRangeException>(() => session.ObserveVectorDeltaAt(10, 11, 1));

        Assert.Equal(statistics, session.GetStatistics());
        Assert.Equal(before, JsonSerializer.Serialize(session.ObserveAt(0, "vector", 10).Value));
        Assert.Equal(1L, session.ObserveVectorDeltaAt(10, 11, 0).Value["history_cursor"]);
    }

    [Fact]
    public void 重置重新开始游标并拒绝重置前的游标和有限历史()
    {
        using var engine = new SimulationEngine(Root, "machinist");
        using var session = engine.CreateSession(null);
        Assert.True(session.SubmitAction(0, "heated_split_shot").Value.Accepted);
        session.AdvanceTo(0.7);
        session.RecordPolicyAction(0.7, "ogcd_wait", 1);
        Assert.Equal(2L, session.ObserveVectorDeltaAt(0.7, 1, 0).Value["history_cursor"]);

        session.Reset(null, initialTimestamp: 20);
        Assert.Throws<ArgumentOutOfRangeException>(() => session.ObserveVectorDeltaAt(20, 21, 2));
        var history = new ReconstructedHistory();
        Assert.Equal(0, history.AssertAndAppend(session, 20, 21, 0));
        Assert.True(session.SubmitAction(20, "heated_split_shot").Value.Accepted);
        Assert.Equal(1, history.AssertAndAppend(session, 20, 21, 0));

        session.Reset(1, initialTimestamp: 30);
        Assert.Throws<InvalidOperationException>(() => session.ObserveVectorDeltaAt(30, 31, 0));
        session.Reset(null, initialTimestamp: 40);
        Assert.Equal(0L, session.ObserveVectorDeltaAt(40, 41, 0).Value["history_cursor"]);
    }

    private sealed class ReconstructedHistory
    {
        public List<Dictionary<string, object?>> Skills { get; } = new();
        private readonly List<Dictionary<string, double[]>> _states = new();
        private readonly List<Dictionary<string, double>> _metrics = new();

        public long AssertAndAppend(SimulationSession session, double timestamp, double next, long afterSequence)
        {
            var expected = session.ObserveAt(timestamp, "vector", next);
            var observed = session.ObserveVectorDeltaAt(timestamp, next, afterSequence);
            var delta = observed.Value;
            Assert.Equal(expected.Timestamp, observed.Timestamp);
            Assert.Equal(expected.NextScheduledEventTime, observed.NextScheduledEventTime);
            Assert.Equal(afterSequence, delta["history_start"]);
            var cursor = Assert.IsType<long>(delta["history_cursor"]);
            var skills = Assert.IsType<List<Dictionary<string, object?>>>(delta["skill_history_context"]);
            var stateContext = Assert.IsType<Dictionary<string, object?>>(delta["state_history_context"]);
            var states = Assert.IsType<List<Dictionary<string, double[]>>>(stateContext["tokens"]);
            var metrics = Assert.IsType<List<Dictionary<string, double>>>(stateContext["execution_metrics"]);
            var added = checked((int)(cursor - afterSequence));
            Assert.Equal(added, skills.Count);
            Assert.Equal(added, states.Count);
            Assert.Equal(added, metrics.Count);
            Skills.AddRange(skills);
            _states.AddRange(states);
            _metrics.AddRange(metrics);

            // 替换三段增量为累计历史后，整个 canonical context（含当前状态和元数据）应逐值相同。
            var reconstructed = new Dictionary<string, object?>(delta);
            reconstructed.Remove("history_start");
            reconstructed["skill_history_context"] = Skills;
            reconstructed["state_history_context"] = new Dictionary<string, object?>(stateContext)
            {
                ["tokens"] = _states,
                ["execution_metrics"] = _metrics,
            };
            Assert.Equal(JsonSerializer.Serialize(expected.Value), JsonSerializer.Serialize(reconstructed));
            Assert.Equal(JsonSerializer.Serialize(OutputContextSchema.BuildOutputContextSchemaMetadata(expected.Value)),
                JsonSerializer.Serialize(OutputContextSchema.BuildOutputContextSchemaMetadata(reconstructed)));
            return cursor;
        }
    }
}

using System.Text.Json;
using Combat.Sim.Common;
using Combat.Sim.Models.Timeline;
using Combat.Sim.Sessions;

namespace FightEngine.Tests.Policy;

public sealed class PolicySequenceReplayTests
{
    private static string Root => RepoRootLocator.Find();

    [Theory]
    [InlineData("black_mage", "blizzard_iii")]
    [InlineData("machinist", "heated_split_shot")]
    public void 批量与手工逐步回放的等待观测提交及同刻事实完全一致(string job, string action)
    {
        using var engine = new SimulationEngine(Root, job, 2);
        using var batch = engine.CreateSession(null, fightRemaining: 60);
        using var stepwise = engine.CreateSession(null, fightRemaining: 60);
        var facts = new ExternalCombatEvent[]
        {
            new(0.5, ExternalCombatEventKinds.MovementChanged, true),
            new(0.5, ExternalCombatEventKinds.TargetCountChanged, TargetCount: 2),
            new(1, ExternalCombatEventKinds.MovementChanged, false),
            new(1, ExternalCombatEventKinds.RaidBuffWindowChanged, true, RemainingSeconds: 5),
            new(6, ExternalCombatEventKinds.TargetCountChanged, TargetCount: 3),
            new(6, ExternalCombatEventKinds.RaidBuffWindowChanged, false),
            // 紧邻请求后的事实不能提前提交到请求 6 的观测或技能效果中。
            new(Math.BitIncrement(6.0), ExternalCombatEventKinds.TargetCountChanged, TargetCount: 4),
        };
        var requests = new[] { Request(0, action, 0), Request(6, action, 0), Request(12, action, 0) };
        var actual = batch.ReplayPolicySequence(JsonSerializer.Serialize(requests), FactsJson(facts), 60, "ogcd_wait");

        var expected = new List<Dictionary<string, object?>>();
        long cursor = 0;
        expected.Add(Real(stepwise, 0, 0, 6, action, 0, ref cursor));
        expected.Add(Wait(stepwise, 1, 0, 6, ref cursor));
        stepwise.ApplyExternalEvents(facts[..2]);
        stepwise.ApplyExternalEvents(facts[2..4]);
        stepwise.ApplyExternalEvents(facts[4..6]);
        expected.Add(Real(stepwise, 1, 6, 12, action, 0, ref cursor));
        expected.Add(Wait(stepwise, 2, 6, 12, ref cursor));
        stepwise.ApplyExternalEvents(facts[6..]);
        expected.Add(Real(stepwise, 2, 12, 60, action, 0, ref cursor));

        Assert.Equal(JsonSerializer.Serialize(expected), JsonSerializer.Serialize(Observations(actual)));
        Assert.Equal(stepwise.GetStatistics(), batch.GetStatistics());
        Assert.Equal(JsonSerializer.Serialize(stepwise.ObserveAt(12, "vector", 60).Value),
            JsonSerializer.Serialize(batch.ObserveAt(12, "vector", 60).Value));
        Assert.Equal(12, actual.Timestamp);
        Assert.All(Observations(actual), frame => Assert.Contains("history_start", Context(frame).Keys));
        var waits = Observations(actual).Where(frame => (bool)frame["is_wait"]!).ToArray();
        Assert.Equal(2, waits.Length);
        Assert.All(waits, frame => Assert.Null(frame["submission"]));
    }

    [Fact]
    public void 排队请求早于前次读条结束且末条生效时刻不会被额外推进()
    {
        using var engine = new SimulationEngine(Root, "black_mage", 2);
        using var batch = engine.CreateSession(null, actualBaseGcd: 2.46, fightRemaining: 12);
        using var stepwise = engine.CreateSession(null, actualBaseGcd: 2.46, fightRemaining: 12);
        var requests = new[] { Request(0, "fire_iii"), Request(2.96, "blizzard_iii") };
        var actual = batch.ReplayPolicySequence(JsonSerializer.Serialize(requests), "[]", 12, "ogcd_wait");
        var expected = new List<Dictionary<string, object?>>();
        long cursor = 0;
        expected.Add(Real(stepwise, 0, 0, 2.96, "fire_iii", null, ref cursor));
        expected.Add(Real(stepwise, 1, 2.96, 12, "blizzard_iii", null, ref cursor));

        Assert.Equal(JsonSerializer.Serialize(expected), JsonSerializer.Serialize(Observations(actual)));
        Assert.True((bool)Assert.IsType<Dictionary<string, object?>>(Observations(actual)[1]["submission"])["queued"]!);
        Assert.Equal(2.96, actual.Timestamp);
        Assert.Equal(stepwise.GetStatistics(), batch.GetStatistics());
        Assert.True(batch.GetStatistics().PendingEventCount > 0);
    }

    [Theory]
    [InlineData("black_mage", "blizzard_iii")]
    [InlineData("machinist", "heated_split_shot")]
    public void 真实请求拒绝后返回失败帧且后续请求与事实不执行(string job, string action)
    {
        using var engine = new SimulationEngine(Root, job, 2);
        using var batch = engine.CreateSession(null);
        using var stepwise = engine.CreateSession(null);
        var requests = new[] { Request(0, action, 0), Request(0, action, 0), Request(6, action, 0) };
        var facts = new[] { new ExternalCombatEvent(3, ExternalCombatEventKinds.MovementChanged, true) };
        var actual = batch.ReplayPolicySequence(JsonSerializer.Serialize(requests), FactsJson(facts), 12, "ogcd_wait");
        long cursor = 0;
        var expected = new[]
        {
            Real(stepwise, 0, 0, 0, action, 0, ref cursor),
            Real(stepwise, 1, 0, 6, action, 0, ref cursor),
        };

        Assert.Equal(JsonSerializer.Serialize(expected), JsonSerializer.Serialize(Observations(actual)));
        Assert.Equal(2, Observations(actual).Count);
        var failed = Assert.IsType<Dictionary<string, object?>>(Observations(actual)[1]["submission"]);
        Assert.False((bool)failed["accepted"]!);
        Assert.NotEmpty((string)failed["reason"]!);
        Assert.Null(failed["accepted_timestamp"]);
        Assert.Null(failed["effect_timestamp"]);
        Assert.Equal(0, actual.Timestamp);
        Assert.Equal(stepwise.GetStatistics(), batch.GetStatistics());
        Assert.Equal(JsonSerializer.Serialize(stepwise.ObserveAt(0, "vector", 12).Value),
            JsonSerializer.Serialize(batch.ObserveAt(0, "vector", 12).Value));
    }

    [Fact]
    public void 同一窗口内的真实非GCD技能不提前插入等待()
    {
        using var engine = new SimulationEngine(Root, "black_mage", 2);
        using var batch = engine.CreateSession(null);
        using var stepwise = engine.CreateSession(null);
        var requests = new[]
        {
            Request(0, "fire_iii", 0), Request(0.1, "lucid_dreaming"), Request(6, "blizzard_iii", 0),
        };
        var actual = batch.ReplayPolicySequence(JsonSerializer.Serialize(requests), "[]", 12, "ogcd_wait");
        long cursor = 0;
        var expected = new[]
        {
            Real(stepwise, 0, 0, 0.1, "fire_iii", 0, ref cursor),
            Real(stepwise, 1, 0.1, 6, "lucid_dreaming", null, ref cursor),
            Wait(stepwise, 2, 0.1, 6, ref cursor),
            Real(stepwise, 2, 6, 12, "blizzard_iii", 0, ref cursor),
        };
        Assert.Equal(JsonSerializer.Serialize(expected), JsonSerializer.Serialize(Observations(actual)));
    }

    [Fact]
    public void 重置后的批量回放从新时刻和零游标开始且有限历史拒绝()
    {
        using var engine = new SimulationEngine(Root, "machinist");
        using var session = engine.CreateSession(null);
        var firstRequests = JsonSerializer.Serialize(new[] { Request(0, "heated_split_shot", 0) });
        session.ReplayPolicySequence(firstRequests, "[]", 60, "ogcd_wait");
        session.Reset(null, initialTimestamp: 20);
        var secondRequests = JsonSerializer.Serialize(new[] { Request(20, "heated_split_shot", 0) });
        var afterReset = session.ReplayPolicySequence(secondRequests, "[]", 60, "ogcd_wait");
        var firstContext = Context(Assert.Single(Observations(afterReset)));
        Assert.Equal(0L, firstContext["history_start"]);
        Assert.Equal(0L, firstContext["history_cursor"]);
        Assert.Equal(20, afterReset.Timestamp);
        Assert.Equal(1, session.GetStatistics().ActionHistoryCount);

        session.Reset(1);
        var before = session.GetStatistics();
        Assert.Throws<InvalidOperationException>(() => session.ReplayPolicySequence(firstRequests, "[]", 60, "ogcd_wait"));
        Assert.Equal(before, session.GetStatistics());
    }

    private static Dictionary<string, object?> Request(double timestamp, string key, double? cast = null) => new()
    {
        ["timestamp"] = timestamp, ["skill_key"] = key, ["actual_cast_seconds"] = cast,
    };

    private static string FactsJson(IEnumerable<ExternalCombatEvent> facts) => JsonSerializer.Serialize(
        facts.Select(fact => new Dictionary<string, object?>
        {
            ["timestamp"] = fact.Timestamp, ["event_kind"] = fact.Kind, ["value"] = fact.Value,
            ["target_count"] = fact.TargetCount, ["remaining_seconds"] = fact.RemainingSeconds,
        }));

    private static List<Dictionary<string, object?>> Observations(SessionResult<Dictionary<string, object?>> result) =>
        Assert.IsType<List<Dictionary<string, object?>>>(result.Value["observations"]);

    private static Dictionary<string, object?> Context(Dictionary<string, object?> frame) =>
        Assert.IsType<Dictionary<string, object?>>(frame["context"]);

    private static Dictionary<string, object?> Real(
        SimulationSession session, int sourceIndex, double timestamp, double next, string key,
        double? cast, ref long cursor)
    {
        session.AdvanceTo(timestamp);
        var context = session.ObserveVectorDeltaAt(timestamp, next, cursor).Value;
        cursor = (long)context["history_cursor"]!;
        var submission = session.SubmitAction(timestamp, key, cast).Value;
        return Frame(sourceIndex, false, timestamp, context, new Dictionary<string, object?>
        {
            ["accepted"] = submission.Accepted, ["queued"] = submission.Queued, ["reason"] = submission.Reason,
            ["accepted_timestamp"] = submission.AcceptedTimestamp, ["effect_timestamp"] = submission.EffectTimestamp,
        });
    }

    private static Dictionary<string, object?> Wait(
        SimulationSession session, int sourceIndex, double timestamp, double next, ref long cursor)
    {
        session.AdvanceTo(timestamp);
        var context = session.ObserveVectorDeltaAt(timestamp, next, cursor).Value;
        cursor = (long)context["history_cursor"]!;
        session.RecordPolicyAction(timestamp, "ogcd_wait", next);
        return Frame(sourceIndex, true, timestamp, context, null);
    }

    private static Dictionary<string, object?> Frame(
        int sourceIndex, bool isWait, double timestamp, Dictionary<string, object?> context,
        Dictionary<string, object?>? submission) => new()
        {
            ["source_index"] = sourceIndex, ["is_wait"] = isWait, ["timestamp"] = timestamp,
            ["context"] = context, ["submission"] = submission,
        };
}

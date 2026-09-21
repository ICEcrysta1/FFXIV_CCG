using Combat.Sim.Facade;
using Combat.Sim.Models.Timeline;

namespace FightEngine.Tests.Facade;

/// <summary>
/// 序列回放辅助测试。
/// </summary>
public class SequenceRunnerTests
{
    [Fact]
    public void 绝对时间序列只保留最终候选快照()
    {
        var machine = FacadeKit.BuildMachine();
        ActionRequest[] requests =
        {
            new(0.0, "gcd_strike"),
            new(0.4, "ogcd_punch"),
            new(2.5, "gcd_strike"),
        };
        var summary = SequenceRunner.RunActionSequence(
            machine,
            requests);

        var requested = Assert.IsType<List<Dictionary<string, object?>>>(summary["requested_sequence"]);
        Assert.Equal(new[] { 0.0, 0.4, 2.5 }, requested.Select(item => (double)item["timestamp"]!));
        var appliedActions = Assert.IsType<List<Dictionary<string, object?>>>(summary["applied_actions"]);
        Assert.Equal(3, appliedActions.Count);
        Assert.All(appliedActions, action => Assert.DoesNotContain("output", action));
        Assert.Equal("gcd_strike", appliedActions[0]["action"]);
        Assert.Equal(0.4, appliedActions[1]["request_time_seconds"]);

        var finalOutput = Assert.IsType<Dictionary<string, object?>>(summary["final_output"]);
        var skillHistory = Assert.IsType<List<Dictionary<string, object?>>>(finalOutput["skill_history_context"]);
        Assert.Equal("gcd_strike", skillHistory[^1]["skill_key"]);
        var candidateSkills = Assert.IsType<List<Dictionary<string, object?>>>(finalOutput["candidate_skill_context"]);
        Assert.DoesNotContain(candidateSkills, item => Equals(item["skill_key"], "ogcd_wait"));
    }

    [Fact]
    public void 输出历史按绝对请求序列完整记录()
    {
        var machine = FacadeKit.BuildMachine();
        var requests = Enumerable.Range(0, 8)
            .Select(index => new ActionRequest(index * 2.5, "gcd_strike"))
            .ToArray();
        var summary = SequenceRunner.RunActionSequence(machine, requests);

        var appliedActions = Assert.IsType<List<Dictionary<string, object?>>>(summary["applied_actions"]);
        Assert.Equal(requests.Length, appliedActions.Count);

        var finalOutput = Assert.IsType<Dictionary<string, object?>>(summary["final_output"]);
        var skillHistory = Assert.IsType<List<Dictionary<string, object?>>>(finalOutput["skill_history_context"]);
        Assert.Equal(requests.Length, skillHistory.Count);
        Assert.All(skillHistory, item => Assert.Equal("gcd_strike", item["skill_key"]));

        var stateHistory = Assert.IsType<Dictionary<string, object?>>(finalOutput["state_history_context"]);
        var stateTokens = Assert.IsAssignableFrom<IReadOnlyList<object>>(stateHistory["tokens"]);
        Assert.Equal(requests.Length, stateTokens.Count);

        var candidateSkills = Assert.IsType<List<Dictionary<string, object?>>>(finalOutput["candidate_skill_context"]);
        var candidateStates = Assert.IsType<Dictionary<string, object?>>(finalOutput["candidate_state_context"]);
        var candidateStateTokens = Assert.IsAssignableFrom<IReadOnlyList<object>>(candidateStates["tokens"]);
        Assert.Equal(candidateSkills.Count, candidateStateTokens.Count);
    }
}

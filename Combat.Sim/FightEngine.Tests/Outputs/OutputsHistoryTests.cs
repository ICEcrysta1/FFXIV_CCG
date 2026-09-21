using Combat.Sim.Models.Combat;
using Xunit;

namespace FightEngine.Tests.Outputs;

/// <summary>
/// 输出层历史上下文测试。
/// </summary>
public class OutputsHistoryTests
{
    [Fact]
    public void VectorStateHistoryContextPutsMpBeforeAfterInPlayerAndConsumedInResource()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("paradox_ready", true);
        state.SetJobResource("astral_fire", 3);
        state.Mp = 2000;

        var nextState = TimelineTestDriver.Execute(machine, state, "paradox").NextState;
        var payload = TimelineTestDriver.FormatVectorState(machine, nextState);
        var historyState = (Dictionary<string, object?>)payload["state_history_context"];

        var skillHistory = (List<Dictionary<string, object?>>)payload["skill_history_context"];
        Assert.Equal("paradox", skillHistory[^1]["skill_key"]);
        Assert.Equal(1600, skillHistory[^1]["actual_mp_cost"]);
        Assert.Equal(2000.0, OutputsTestKit.HistoryVectorValue(historyState, "player_state", "before.mp"), 5);
        Assert.Equal(400.0, OutputsTestKit.HistoryVectorValue(historyState, "player_state", "after.mp"), 5);
        Assert.Equal(1.0, OutputsTestKit.HistoryVectorValue(historyState, "resource_state", "before.paradox_ready"), 5);
        Assert.Equal(0.0, OutputsTestKit.HistoryVectorValue(historyState, "resource_state", "after.paradox_ready"), 5);
        Assert.Equal(1.0, OutputsTestKit.HistoryVectorValue(historyState, "resource_state", "consumed.paradox_ready"), 5);
    }

    [Fact]
    public void VectorStateOutputExposesSystemAndJobStatusBuffs()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "potion").NextState;
        state = TimelineTestDriver.Execute(machine, state, "ley_lines").NextState;
        var payload = TimelineTestDriver.FormatVectorState(machine, state);
        var historyState = (Dictionary<string, object?>)payload["state_history_context"];
        var tokenIndex = ((List<Dictionary<string, double[]>>)historyState["tokens"]).Count - 1;

        Assert.Equal(1.0, OutputsTestKit.HistoryVectorValue(historyState, "buff_state",
            "after.system.burst_potion.active", tokenIndex), 5);
        Assert.True(OutputsTestKit.HistoryVectorValue(historyState, "buff_state",
            "after.system.burst_potion.remaining_seconds", tokenIndex) > 0.0);
        Assert.Equal(1.0, OutputsTestKit.HistoryVectorValue(historyState, "buff_state",
            "after.job.ley_lines.active", tokenIndex), 5);
        Assert.True(OutputsTestKit.HistoryVectorValue(historyState, "buff_state",
            "after.job.ley_lines.remaining_seconds", tokenIndex) > 0.0);
        var buffFeatureKeys = (List<string>)historyState["buff_state_feature_keys"];
        Assert.True(buffFeatureKeys.IndexOf("after.system.burst_potion.active") <
                    buffFeatureKeys.IndexOf("after.job.ley_lines.active"));
    }

    [Fact]
    public void VectorStateOutputExposesUnifiedRaidBuffWindow()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = machine.InitialState();
        machine.SystemMachine.GrantRegisteredStatus(state, "raid_buff_window", remaining: 18.0);
        var payload = TimelineTestDriver.FormatVectorState(machine, state);

        Assert.Equal(1.0,
            (double)OutputsTestKit.CandidateStateVectorValue(payload, "fire_iii", "buff_state",
                "before.system.raid_buff_window.active")!, 5);
        Assert.Equal(18.0,
            (double)OutputsTestKit.CandidateStateVectorValue(payload, "fire_iii", "buff_state",
                "before.system.raid_buff_window.remaining_seconds")!, 5);
    }

    [Fact]
    public void VectorStateOutputKeepsTriplecastAndSwiftcastRealBuffs()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = OutputsTestKit.ReadyAfterGcd(machine, TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState);
        state = TimelineTestDriver.Execute(machine, state, "triplecast").NextState;
        var payload = TimelineTestDriver.FormatVectorState(machine, state);
        var historyState = (Dictionary<string, object?>)payload["state_history_context"];

        Assert.Equal(1.0, OutputsTestKit.HistoryVectorValue(historyState, "buff_state",
            "after.job.triplecast.active", tokenIndex: 1), 5);
        Assert.Equal(3.0, OutputsTestKit.HistoryVectorValue(historyState, "buff_state",
            "after.job.triplecast.stacks", tokenIndex: 1), 5);
    }

    [Fact]
    public void VectorStateOutputExposesLucidDreamingJobBuff()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "lucid_dreaming").NextState;
        var payload = TimelineTestDriver.FormatVectorState(machine, state);
        var historyState = (Dictionary<string, object?>)payload["state_history_context"];

        Assert.Equal(1.0,
            OutputsTestKit.HistoryVectorValue(historyState, "buff_state", "after.job.lucid_dreaming.active", tokenIndex: 0), 5);
        Assert.Equal(21.0,
            OutputsTestKit.HistoryVectorValue(historyState, "buff_state", "after.job.lucid_dreaming.remaining_seconds", tokenIndex: 0), 5);
    }

    [Fact]
    public void VectorStateOutputEncodesTargetDotInTargetBuffState()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("thundercloud_ready", true);
        var result = TimelineTestDriver.Execute(machine, state, "high_thunder");
        var payload = TimelineTestDriver.FormatResult(machine, result);
        var historyState = (Dictionary<string, object?>)payload["state_history_context"];

        Assert.Equal(1.0, OutputsTestKit.HistoryVectorValue(historyState, "target_buff_state",
            "after.target.high_thunder.active"), 5);
        // 动作历史在 effect 时刻记录；瞬发 DoT 此时拥有完整持续时间。
        Assert.Equal(30.0, OutputsTestKit.HistoryVectorValue(historyState, "target_buff_state",
            "after.target.high_thunder.remaining_seconds"), 5);
        Assert.Equal(1.0, OutputsTestKit.HistoryVectorValue(historyState, "target_buff_state",
            "after.target.high_thunder.stacks"), 5);
    }

    [Fact]
    public void SkillTokensDisplayElementalPotencyAndKeepParadoxUnscaled()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("astral_fire", 3);
        var payload = TimelineTestDriver.FormatVectorState(machine, state);
        var candidateSkill = (List<Dictionary<string, object?>>)payload["candidate_skill_context"];

        var fireIv = candidateSkill.First(token => (string)token["skill_key"] == "fire_iv");
        var paradox = candidateSkill.First(token => (string)token["skill_key"] == "paradox");
        Assert.Equal(300.0 * 1.8, (double)fireIv["potency"]!, 5);
        Assert.Equal(540.0, (double)paradox["potency"]!, 5);

        var result = TimelineTestDriver.Execute(machine, state, "fire_iv");
        var afterPayload = TimelineTestDriver.FormatVectorState(machine, result.NextState);
        var skillHistory = (List<Dictionary<string, object?>>)afterPayload["skill_history_context"];
        Assert.Equal(300.0 * 1.8, (double)skillHistory[^1]["potency"]!, 5);
    }

    [Fact]
    public void HistorySkillPotencyUsesStateBeforeCast()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("umbral_ice", 3);

        var result = TimelineTestDriver.Execute(machine, state, "fire_iii");
        var payload = TimelineTestDriver.FormatVectorState(machine, result.NextState);
        var skillHistory = (List<Dictionary<string, object?>>)payload["skill_history_context"];

        // UI3 下火属性威力 ×0.7，威力取动作前状态
        Assert.Equal(290.0 * 0.7, (double)skillHistory[^1]["potency"]!, 5);
    }

    [Fact]
    public void SecondsAndGcdSnapshotsExposeMpAndTimers()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState;
        var seconds = TimelineTestDriver.FormatState(machine, state, "seconds");
        var gcd = TimelineTestDriver.FormatState(machine, state, "gcd");

        Assert.Equal("seconds", seconds["mode"]);
        Assert.Equal(8000, seconds["mp"]);
        // 读条动作在 effect 时刻写历史并返回该绝对时刻的状态。
        Assert.Equal(3.0, (double)seconds["time_seconds"]!, 5);
        Assert.Equal("gcd", gcd["mode"]);
        Assert.Equal(2.5, (double)gcd["current_gcd_seconds"]!, 5);
        // polyglot_timer 是 seconds 量谱，GCD 制下转为 *_gcd
        Assert.DoesNotContain("polyglot_timer", gcd.Keys);
        Assert.True(gcd.ContainsKey("polyglot_timer_gcd"));
        Assert.True(seconds.ContainsKey("polyglot_timer"));
    }
}

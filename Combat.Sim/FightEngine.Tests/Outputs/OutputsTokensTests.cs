using System.Text.Encodings.Web;
using System.Text.Json;
using Combat.Sim.Outputs.TokenBuilders;
using Xunit;

namespace FightEngine.Tests.Outputs;

/// <summary>
/// 输出层候选上下文测试。
/// </summary>
public class OutputsTokensTests
{
    [Fact]
    public void VectorStateOutputContainsSplitCandidateContexts()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState;
        var payload = TimelineTestDriver.FormatVectorState(machine, state);

        var candidateSkill = (List<Dictionary<string, object?>>)payload["candidate_skill_context"];
        var candidateState = (Dictionary<string, object?>)payload["candidate_state_context"];
        var expectedSkillKeys = machine.SkillBook.EnabledSkills().Select(skill => skill.Key).ToHashSet();

        Assert.Equal(expectedSkillKeys, candidateSkill.Select(token => (string)token["skill_key"]!).ToHashSet());
        Assert.Equal(candidateSkill.Count, ((List<object>)candidateState["tokens"]).Count);
        var historyState = (Dictionary<string, object?>)payload["state_history_context"];
        foreach (var groupKey in new[] { "player_state", "buff_state", "target_buff_state", "resource_state" })
        {
            Assert.Equal(historyState[$"{groupKey}_feature_keys"], candidateState[$"{groupKey}_feature_keys"]);
        }
    }

    [Fact]
    public void CandidatePreviewPayloadDoesNotMutateSourceState()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState;
        var before = state.Clone();

        TimelineTestDriver.BuildCandidatePreviews(machine, state);

        Assert.Equal(
            JsonSerializer.Serialize(before, Options),
            JsonSerializer.Serialize(state, Options));
    }

    [Fact]
    public void CandidatePreviewChecksOgcdCooldownAtCurrentTime()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("astral_fire", 3);

        state = TimelineTestDriver.Execute(machine, state, "despair").NextState;
        var transpose = TimelineTestDriver.Execute(machine, state, "transpose");
        state = TimelineTestDriver.AdvanceBy(machine, transpose.NextState, transpose.ActualOccupancySeconds);
        state = TimelineTestDriver.AdvanceBy(machine, state, state.GcdRemaining);
        state = TimelineTestDriver.Execute(machine, state, "paradox").NextState;

        var actualValidation = machine.ValidateAction(state, "transpose");
        Assert.False(actualValidation.Ok);
        Assert.Equal("cooldown_locked", actualValidation.Reason);

        var preview = TimelineTestDriver.BuildCandidatePreviews(machine, state)
            .Single(candidate => candidate.Skill.Key == "transpose");

        Assert.False(preview.IsLegal);
        Assert.Equal("cooldown_locked", preview.InvalidReason);
        Assert.Equal(2.5, preview.NextCooldownSeconds, 5);
    }

    [Fact]
    public void CandidateSkillAndStateTokensReuseSameSchema()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState;
        var payload = TimelineTestDriver.FormatVectorState(machine, state);

        var skillHistory = (List<Dictionary<string, object?>>)payload["skill_history_context"];
        var candidateSkill = (List<Dictionary<string, object?>>)payload["candidate_skill_context"];
        var historyToken = skillHistory[^1];
        var highThunder = candidateSkill.First(token => (string)token["skill_key"] == "high_thunder");

        Assert.Equal(historyToken.Keys.OrderBy(k => k), highThunder.Keys.OrderBy(k => k));
        Assert.True((bool)historyToken["is_legal"]!);
        Assert.Equal(1, historyToken["kind"]);
        Assert.Equal("", historyToken["invalid_reason"]);
        Assert.Equal(0.0, (double)historyToken["next_cooldown_seconds"]!, 5);
        Assert.True((bool)highThunder["is_legal"]!);
        Assert.Equal(1, highThunder["kind"]);
        Assert.Equal("", highThunder["invalid_reason"]);
        Assert.Equal(0.0, (double)highThunder["next_cooldown_seconds"]!, 5);
        Assert.Equal(1, highThunder["available_charges"]);
        Assert.Equal(1, highThunder["max_charges"]);
        Assert.Equal(0, highThunder["actual_mp_cost"]);
        Assert.Equal(8000.0,
            (double)OutputsTestKit.CandidateStateVectorValue(payload, "high_thunder", "player_state", "before.mp")!, 5);
        Assert.Equal(8000.0,
            (double)OutputsTestKit.CandidateStateVectorValue(payload, "high_thunder", "player_state", "after.mp")!, 5);
        Assert.Equal(1.0,
            (double)OutputsTestKit.CandidateStateVectorValue(payload, "high_thunder", "buff_state",
                "before.resource.thundercloud_ready")!, 5);
    }

    [Fact]
    public void IllegalCandidateKeepsBeforeStateAndMarksAfterWithNull()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState;
        var payload = TimelineTestDriver.FormatVectorState(machine, state);

        var candidateSkill = (List<Dictionary<string, object?>>)payload["candidate_skill_context"];
        var candidateState = (Dictionary<string, object?>)payload["candidate_state_context"];
        var tokens = (List<object>)candidateState["tokens"];
        var index = candidateSkill.FindIndex(token => (string)token["skill_key"] == "flare_star");
        var flareStar = candidateSkill[index];

        Assert.False((bool)flareStar["is_legal"]!);
        Assert.Equal("insufficient_astral_soul", flareStar["invalid_reason"]);
        Assert.Equal(8000.0,
            (double)OutputsTestKit.CandidateStateVectorValue(payload, "flare_star", "player_state", "before.mp")!, 5);

        var playerAfterStart = ((List<string>)candidateState["player_state_feature_keys"]).Count / 2;
        var buffAfterStart = ((List<string>)candidateState["buff_state_feature_keys"]).Count / 2;
        var targetAfterStart = ((List<string>)candidateState["target_buff_state_feature_keys"]).Count / 2;
        var resourceBeforeSize = ((List<string>)candidateState["resource_state_feature_keys"]).Count / 3;
        var resourceAfterStart = resourceBeforeSize;
        var resourceConsumedStart = resourceBeforeSize * 2;

        Assert.All(OutputsTestKit.VectorOf(tokens[index], "player_state").Skip(playerAfterStart), value => Assert.Null(value));
        Assert.All(OutputsTestKit.VectorOf(tokens[index], "buff_state").Skip(buffAfterStart), value => Assert.Null(value));
        Assert.All(OutputsTestKit.VectorOf(tokens[index], "target_buff_state").Skip(targetAfterStart), value => Assert.Null(value));
        Assert.All(OutputsTestKit.VectorOf(tokens[index], "resource_state").Skip(resourceAfterStart), value => Assert.Null(value));
        Assert.All(OutputsTestKit.VectorOf(tokens[index], "resource_state").Skip(resourceConsumedStart), value => Assert.Null(value));
    }

    [Fact]
    public void SkillTokensExposeCooldownAndChargeSnapshot()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = OutputsTestKit.ReadyAfterGcd(machine, TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState);
        var result = TimelineTestDriver.Execute(machine, state, "ley_lines");
        var payload = TimelineTestDriver.FormatResult(machine, result);
        var skillHistory = (List<Dictionary<string, object?>>)payload["skill_history_context"];
        var historyToken = skillHistory[^1];

        Assert.Equal("ley_lines", historyToken["skill_key"]);
        Assert.Equal(0, historyToken["kind"]);
        Assert.True((bool)historyToken["is_legal"]!);
        Assert.Equal(120.0, (double)historyToken["next_cooldown_seconds"]!, 5);
        Assert.Equal(1, historyToken["available_charges"]);
        Assert.Equal(2, historyToken["max_charges"]);

        var readyState = result.NextState;
        var readyPayload = TimelineTestDriver.FormatVectorState(machine, readyState);
        var candidateSkill = (List<Dictionary<string, object?>>)readyPayload["candidate_skill_context"];
        var leyLines = candidateSkill.First(token => (string)token["skill_key"] == "ley_lines");

        Assert.False((bool)leyLines["is_legal"]!);
        Assert.Equal("status_already_active", leyLines["invalid_reason"]);
        Assert.Equal(120.0, (double)leyLines["next_cooldown_seconds"]!, 5);
        Assert.Equal(1, leyLines["available_charges"]);
        Assert.Equal(2, leyLines["max_charges"]);
    }

    [Fact]
    public void SkillTokenBuilderIsSingleReusableFunction()
    {
        // 技能历史与候选技能共用同一个 build 函数：相同输入产出相同 token，
        // 可选属性（time_seconds / gcd_index）按来源注入后才会出现。
        var token1 = SkillTokenBuilder.Build(
            skillId: 152, skillKey: "fire_iii", skillName: "爆炎", potency: 290,
            value: 1.0, kind: "gcd",
            actualMpCost: 2000, castTimeSeconds: 3.5, gcdWindowSeconds: 2.5,
            isLegal: true, invalidReason: "", nextCooldownSeconds: 0.0,
            availableCharges: 1, maxCharges: 1,
            jobResourcesConsumed: new Dictionary<string, object>());
        var tokenSame = SkillTokenBuilder.Build(
            skillId: 152, skillKey: "fire_iii", skillName: "爆炎", potency: 290,
            value: 1.0, kind: "gcd",
            actualMpCost: 2000, castTimeSeconds: 3.5, gcdWindowSeconds: 2.5,
            isLegal: true, invalidReason: "", nextCooldownSeconds: 0.0,
            availableCharges: 1, maxCharges: 1,
            jobResourcesConsumed: new Dictionary<string, object>());
        var tokenWithTiming = SkillTokenBuilder.Build(
            skillId: 152, skillKey: "fire_iii", skillName: "爆炎", potency: 290,
            value: 1.0, kind: "gcd",
            actualMpCost: 2000, castTimeSeconds: 3.5, gcdWindowSeconds: 2.5,
            isLegal: true, invalidReason: "", nextCooldownSeconds: 0.0,
            availableCharges: 1, maxCharges: 1,
            jobResourcesConsumed: new Dictionary<string, object>(),
            timeSeconds: 1.0, gcdIndex: 2);

        Assert.Equal(JsonSerializer.Serialize(token1), JsonSerializer.Serialize(tokenSame));
        Assert.DoesNotContain("time_seconds", token1.Keys);
        Assert.Equal(152, token1["skill_id"]);
        Assert.Equal(3.5, (double)((Dictionary<string, object?>)token1["cast_time"])["seconds"]!, 5);
        Assert.Equal(2, tokenWithTiming["gcd_index"]);
        Assert.Equal(1.0, (double)tokenWithTiming["time_seconds"]!, 5);
    }

    private static readonly JsonSerializerOptions Options = new()
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };
}

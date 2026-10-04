using System.Text.Encodings.Web;
using System.Text.Json;
using Combat.Sim.Outputs.TokenBuilders;
using Xunit;

namespace FightEngine.Tests.Outputs;

/// <summary>
/// 因果上下文与固定动作输出契约测试。
/// </summary>
public class OutputsTokensTests
{
    [Fact]
    public void 当前状态与历史共用字段且两段逐值相同()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState;
        var payload = TimelineTestDriver.FormatVectorState(machine, state);
        var current = (Dictionary<string, object?>)payload["current_state_context"]!;
        var history = (Dictionary<string, object?>)payload["state_history_context"]!;
        var token = Assert.Single((List<Dictionary<string, double[]>>)current["tokens"]!);
        foreach (var group in new[] { "player_state", "buff_state", "target_buff_state", "resource_state" })
        {
            Assert.Equal(history[$"{group}_feature_keys"], current[$"{group}_feature_keys"]);
            var values = token[group];
            Assert.Equal(values.Take(values.Length / 2), values.Skip(values.Length / 2));
        }
        Assert.Equal(8000, OutputsTestKit.CurrentStateVectorValue(payload, "player_state", "previous_action_after.mp"));
        Assert.Equal(state.Time, OutputsTestKit.CurrentStateVectorValue(payload, "player_state", "request_state.time_seconds"));
    }

    [Fact]
    public void 动作词表排序稳定且合法性等于真实提交()
    {
        var simulator = new Combat.Sim.Facade.JobSimulator(OutputsTestKit.BuildMachine());
        simulator.SubmitAction(0, "fire_iii");
        foreach (var time in new[] { 0.0, 0.2, 2.2, 3.0, 6.0 })
        {
            simulator.AdvanceTo(time);
            var output = simulator.FormatVectorState();
            var keys = (List<string>)output["action_keys"]!;
            var mask = (List<bool>)output["action_legal_mask"]!;
            var values = (List<double>)output["action_values"]!;
            Assert.Equal(keys.OrderBy(key => key, StringComparer.Ordinal), keys);
            Assert.Equal(keys.Count, mask.Count);
            Assert.Equal(keys.Count, values.Count);
            for (var index = 0; index < keys.Count; index++)
                Assert.Equal(simulator.Fork().SubmitAction(time, keys[index]).Accepted, mask[index]);
        }
    }

    [Fact]
    public void 输出不提交动作不推进未来也不污染主时间线()
    {
        var simulator = new Combat.Sim.Facade.JobSimulator(OutputsTestKit.BuildMachine());
        simulator.SubmitAction(0, "fire_iii");
        var before = simulator.CreateSnapshot();
        var output = simulator.FormatVectorState();
        Assert.DoesNotContain("candidate_skill_context", output.Keys);
        Assert.DoesNotContain("candidate_state_context", output.Keys);
        Assert.Empty((List<Dictionary<string, object?>>)output["skill_history_context"]!);
        Assert.Equal(JsonSerializer.Serialize(before), JsonSerializer.Serialize(simulator.CreateSnapshot()));
        Assert.Equal(0, OutputsTestKit.CurrentStateVectorValue(output, "player_state", "request_state.time_seconds"));
    }

    [Fact]
    public void 非法动作只影响合法性不再生成空状态段()
    {
        var payload = TimelineTestDriver.FormatVectorState(OutputsTestKit.BuildMachine(), OutputsTestKit.BuildMachine().InitialState());
        var keys = (List<string>)payload["action_keys"]!;
        var mask = (List<bool>)payload["action_legal_mask"]!;
        Assert.False(mask[keys.IndexOf("flare_star")]);
        var current = (Dictionary<string, object?>)payload["current_state_context"]!;
        Assert.Single((List<Dictionary<string, double[]>>)current["tokens"]!);
        Assert.All(((List<Dictionary<string, double[]>>)current["tokens"]!)[0].Values.SelectMany(vector => vector),
            value => Assert.True(double.IsFinite(value)));
    }

    [Fact]
    public void 历史技能继续保留技能种类冷却和充能字段()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = OutputsTestKit.ReadyAfterGcd(machine, TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState);
        var result = TimelineTestDriver.Execute(machine, state, "ley_lines");
        var payload = TimelineTestDriver.FormatResult(machine, result);
        var history = (List<Dictionary<string, object?>>)payload["skill_history_context"]!;
        var token = history[^1];
        Assert.Equal("ley_lines", token["skill_key"]);
        Assert.All(history, skill => Assert.DoesNotContain("time_seconds", skill.Keys));
        Assert.Equal(0, token["kind"]);
        Assert.True((bool)token["is_legal"]!);
        Assert.Equal(120, (double)token["next_cooldown_seconds"]!, 5);
        Assert.Equal(1, token["available_charges"]);
        Assert.Equal(2, token["max_charges"]);
        var keys = (List<string>)payload["action_keys"]!;
        var mask = (List<bool>)payload["action_legal_mask"]!;
        Assert.False(mask[keys.IndexOf("ley_lines")]);
    }

    [Fact]
    public void SkillTokenBuilderIsSingleReusableFunction()
    {
        // 技能历史与策略动作历史共用同一个 build 函数：相同输入产出相同 token，
        // 执行时刻与累计 GCD 索引都不进入技能 token。
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
        Assert.Equal(JsonSerializer.Serialize(token1), JsonSerializer.Serialize(tokenSame));
        Assert.DoesNotContain("time_seconds", token1.Keys);
        Assert.Equal(152, token1["skill_id"]);
        Assert.Equal(3.5, (double)((Dictionary<string, object?>)token1["cast_time"])["seconds"]!, 5);
        Assert.DoesNotContain("gcd_index", token1.Keys);
    }

    [Fact]
    public void 同一技能平移执行时刻不改变技能Token且保留状态时间()
    {
        Dictionary<string, object?>? firstToken = null;
        foreach (var requestTime in new[] { 0.0, 0.125 })
        {
            var simulator = new Combat.Sim.Facade.JobSimulator(OutputsTestKit.BuildMachine());
            simulator.AdvanceTo(requestTime);
            var submission = simulator.SubmitAction(requestTime, "fire_iii");
            Assert.True(submission.Accepted);
            simulator.AdvanceTo(submission.EffectTimestamp!.Value);
            var output = simulator.FormatVectorState();
            var token = Assert.Single((List<Dictionary<string, object?>>)output["skill_history_context"]!);

            Assert.DoesNotContain("time_seconds", token.Keys);
            if (firstToken is not null)
                Assert.Equal(JsonSerializer.Serialize(firstToken), JsonSerializer.Serialize(token));
            firstToken = token;

            // 内部执行统计和两段状态的时间仍按真实时间线保存，不随技能字段删除丢失。
            var entry = Assert.Single(simulator.GetState().History);
            Assert.Equal(requestTime, entry.RequestTimestamp!.Value, 8);
            Assert.Equal(submission.EffectTimestamp.Value, entry.TimeSeconds, 8);
            var history = (Dictionary<string, object?>)output["state_history_context"]!;
            Assert.Equal(requestTime, OutputsTestKit.HistoryVectorValue(history, "player_state", "request_state.time_seconds"), 8);
            Assert.Equal(submission.EffectTimestamp.Value,
                OutputsTestKit.CurrentStateVectorValue(output, "player_state", "previous_action_after.time_seconds"), 8);
            Assert.Equal(submission.EffectTimestamp.Value,
                OutputsTestKit.CurrentStateVectorValue(output, "player_state", "request_state.time_seconds"), 8);
        }
    }

    private static readonly JsonSerializerOptions Options = new()
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };
}

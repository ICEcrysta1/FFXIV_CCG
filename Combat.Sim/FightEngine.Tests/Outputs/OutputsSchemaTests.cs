using System.Text.Json;
using Combat.Sim.Config;
using Combat.Sim.Facade;
using Combat.Sim.Outputs;
using Xunit;

namespace FightEngine.Tests.Outputs;

/// <summary>
/// 输出层 schema 与打包测试。
/// </summary>
public class OutputsSchemaTests
{
    [Fact]
    public void VectorStateHistoryFreezesRequestSnapshotsBeforeEffects()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState;
        var payload = TimelineTestDriver.FormatVectorState(machine, state);

        Assert.Equal("black_mage", payload["job_tag"]);
        Assert.Equal(OutputContextSchema.CanonicalContextSchemaVersion, payload["schema_version"]);
        Assert.Equal(
            JsonSerializer.Serialize(SceneContextSchema.BuildEmptySceneContext()),
            JsonSerializer.Serialize(payload["scene_context"]));
        Assert.DoesNotContain("current_state", payload.Keys);
        Assert.DoesNotContain("current_state_vector_context", payload.Keys);

        var skillHistory = (List<Dictionary<string, object?>>)payload["skill_history_context"];
        Assert.Equal("fire_iii", skillHistory[^1]["skill_key"]);
        Assert.Equal(1.1, (double)skillHistory[^1]["value"]!, 5);
        Assert.DoesNotContain("gcds", ((Dictionary<string, object?>)skillHistory[^1]["cast_time"]).Keys);
        Assert.DoesNotContain("gcds", ((Dictionary<string, object?>)skillHistory[^1]["gcd_window"]).Keys);

        var historyState = (Dictionary<string, object?>)payload["state_history_context"];
        var historyTokens = (List<Dictionary<string, double[]>>)historyState["tokens"];
        Assert.Single(historyTokens);
        Assert.Contains("previous_action_after.mp", (List<string>)historyState["player_state_feature_keys"]);
        Assert.Contains("request_state.mp", (List<string>)historyState["player_state_feature_keys"]);
        Assert.Equal(10000.0, OutputsTestKit.HistoryVectorValue(historyState, "player_state", "request_state.mp"), 5);
        Assert.Equal(0.0, OutputsTestKit.HistoryVectorValue(historyState, "player_state", "previous_action_after.time_seconds"), 5);
        Assert.Equal(0.0, OutputsTestKit.HistoryVectorValue(historyState, "player_state", "request_state.time_seconds"), 5);
        Assert.Equal(2.5, OutputsTestKit.HistoryVectorValue(historyState, "player_state", "request_state.current_gcd_seconds"), 5);
        Assert.Equal(597.0, state.FightRemaining, 5);
        Assert.Equal(0.0, state.GcdRemaining, 5);
        Assert.Equal(1.0, OutputsTestKit.HistoryVectorValue(historyState, "player_state", "request_state.boss_targetable"), 5);
        Assert.Equal(0.0, OutputsTestKit.HistoryVectorValue(historyState, "buff_state", "request_state.resource.thundercloud_ready"), 5);
        Assert.Equal(new List<string>
        {
            "previous_action_after.target.high_thunder.active",
            "previous_action_after.target.high_thunder.remaining_seconds",
            "previous_action_after.target.high_thunder.stacks",
            "previous_action_after.target.cumulative_dot_potency",
            "previous_action_after.target.cumulative_potency",
            "previous_action_after.target.current_potency",
            "previous_action_after.target.current_gcd_dot_potency",
            "request_state.target.high_thunder.active",
            "request_state.target.high_thunder.remaining_seconds",
            "request_state.target.high_thunder.stacks",
            "request_state.target.cumulative_dot_potency",
            "request_state.target.cumulative_potency",
            "request_state.target.current_potency",
            "request_state.target.current_gcd_dot_potency",
        }, historyState["target_buff_state_feature_keys"]);
        Assert.Equal(0.0, OutputsTestKit.HistoryVectorValue(historyState, "resource_state", "request_state.astral_fire"), 5);
        Assert.Equal(3.0, OutputsTestKit.CurrentStateVectorValue(payload, "resource_state", "previous_action_after.astral_fire"), 5);
    }

    [Fact]
    public void ModelTokensExcludeProgressCountersSchedulingWindowsAndGcdTimeUnits()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState;
        var payload = TimelineTestDriver.FormatVectorState(machine, state);
        var removedFields = new[]
        {
            "gcd_index", "fight_remaining_seconds", "gcd_remaining_gcds",
            "weave_window_gcds", "ogcd_window_gcds", "downtime_remaining_gcds",
            "gcd_remaining_seconds", "weave_window_seconds", "ogcd_window_seconds",
            "next_untargetable_in_gcds", "remaining_gcds",
            "ogcds_weaved", "max_ogcd_per_window",
        };

        // 模拟器继续计数和推进结束边界，历史与当前向量只输出部署需要的字段。
        Assert.Equal(1, state.GcdIndex);
        foreach (var contextKey in new[] { "state_history_context", "current_state_context" })
        {
            var context = (Dictionary<string, object?>)payload[contextKey];
            var featureKeys = (List<string>)context["player_state_feature_keys"];
            Assert.Equal(18, featureKeys.Count);
            foreach (var field in removedFields)
            {
                Assert.DoesNotContain($"previous_action_after.{field}", featureKeys);
                Assert.DoesNotContain($"request_state.{field}", featureKeys);
            }
            Assert.Contains("previous_action_after.current_gcd_seconds", featureKeys);
            Assert.Contains("request_state.downtime_remaining_seconds", featureKeys);
            var allFeatureKeys = new[]
            {
                "player_state_feature_keys", "buff_state_feature_keys",
                "target_buff_state_feature_keys", "resource_state_feature_keys",
            }.SelectMany(key => (List<string>)context[key]).ToArray();
            Assert.Equal(86, allFeatureKeys.Length);
            Assert.DoesNotContain(allFeatureKeys, key => key.EndsWith("_gcds", StringComparison.Ordinal));
            Assert.DoesNotContain(allFeatureKeys, key => key.StartsWith("consumed.", StringComparison.Ordinal));
            Assert.DoesNotContain(allFeatureKeys, key => key.Contains(".manaward.", StringComparison.Ordinal)
                || key.Contains(".surecast.", StringComparison.Ordinal));
            Assert.Equal(40, ((List<string>)context["buff_state_feature_keys"]).Count);
            var resourceKeys = (List<string>)context["resource_state_feature_keys"];
            Assert.Equal(14, resourceKeys.Count);
            Assert.Equal(resourceKeys.Take(7).Select(key => key["previous_action_after.".Length..]),
                resourceKeys.Skip(7).Select(key => key["request_state.".Length..]));
            Assert.Contains("previous_action_after.job.triplecast.remaining_seconds", allFeatureKeys);
            Assert.Contains("request_state.target.high_thunder.remaining_seconds", allFeatureKeys);
        }
        foreach (var contextKey in new[] { "skill_history_context" })
        {
            var tokens = (List<Dictionary<string, object?>>)payload[contextKey];
            Assert.All(tokens, token => Assert.DoesNotContain("gcd_index", token.Keys));
        }
    }

    [Fact]
    public void StepOutputMatchesVectorStateCanonicalContext()
    {
        var machine = OutputsTestKit.BuildMachine();
        var state = OutputsTestKit.ReadyAfterGcd(machine, TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState);
        state = TimelineTestDriver.Execute(machine, state, "fire_iv").NextState;
        var result = TimelineTestDriver.Execute(machine, state, "ley_lines");

        Assert.Equal(
            JsonSerializer.Serialize(TimelineTestDriver.FormatVectorState(machine, result.NextState)),
            JsonSerializer.Serialize(TimelineTestDriver.FormatResult(machine, result)));
    }

    [Fact]
    public void VectorStateOutputReservesEmptySceneContext()
    {
        var machine = OutputsTestKit.BuildMachine();
        var payload = TimelineTestDriver.FormatVectorState(machine, machine.InitialState());

        Assert.Equal(OutputContextSchema.CanonicalContextSchemaVersion, payload["schema_version"]);
        Assert.Equal(
            JsonSerializer.Serialize(SceneContextSchema.BuildEmptySceneContext()),
            JsonSerializer.Serialize(payload["scene_context"]));
    }

    [Fact]
    public void TensorStateOutputPacksContextIntoTorchFriendlyPayload()
    {
        var machine = OutputsTestKit.BuildMachine();
        var payload = (Dictionary<string, object?>)TimelineTestDriver.FormatTensorState(machine, machine.InitialState())!;

        Assert.Equal("black_mage", payload["job_tag"]);
        Assert.Equal(OutputContextSchema.CanonicalContextSchemaVersion, payload["schema_version"]);

        var skillKeys = (List<object>)payload["action_keys"];
        Assert.True(skillKeys.Count > 0);
        Assert.Equal(skillKeys.Count, ((List<object>)((Dictionary<string, object?>)payload["action_values"])["values"]).Count);
        Assert.False((bool)((List<object>)payload["action_legal_mask"])[skillKeys.IndexOf("despair")]);
        var currentState = (Dictionary<string, object?>)payload["current_state_context"];
        // tensor 打包按列式结构重排：feature keys 以 object 装箱出现
        var playerFeatureKeys = (List<object>)currentState["player_state_feature_keys"];
        Assert.NotEmpty(playerFeatureKeys);
        var playerPayload = (Dictionary<string, object?>)((Dictionary<string, object?>)currentState["tokens"])["player_state"];
        var values = (List<object>)playerPayload["values"];
        var isNull = (List<object>)playerPayload["is_null"];

        // 只有一条真实请求状态，非法动作不制造未来状态或 null 段。
        Assert.Single(values);
        Assert.Single(isNull);
        Assert.All(((List<object>)isNull[0]).Cast<bool>(), flag => Assert.False(flag));
    }

    [Fact]
    public void OutputContextSchemaMetadataMatchesCanonicalOutput()
    {
        var machine = OutputsTestKit.BuildMachine();
        var payload = TimelineTestDriver.FormatVectorState(machine, machine.InitialState());
        var schemaMetadata = OutputContextSchema.BuildOutputContextSchemaMetadata(payload);

        Assert.Equal(OutputContextSchema.CanonicalContextSchemaVersion, schemaMetadata["schema_version"]);
        Assert.Equal(OutputContextSchema.CanonicalContextTopLevelKeys.ToList(), schemaMetadata["top_level_keys"]);
        Assert.Equal(new List<string> { "player_state", "buff_state", "target_buff_state", "resource_state", "skill_availability" },
            schemaMetadata["state_vector_group_keys"]);
        var historyState = (Dictionary<string, object?>)payload["state_history_context"];
        var currentState = (Dictionary<string, object?>)payload["current_state_context"];
        var metadataHistory = (Dictionary<string, List<string>>)schemaMetadata["state_history_feature_keys"];
        var metadataCurrent = (Dictionary<string, List<string>>)schemaMetadata["current_state_feature_keys"];
        Assert.Equal(historyState["player_state_feature_keys"], metadataHistory["player_state"]);
        Assert.Equal(currentState["target_buff_state_feature_keys"], metadataCurrent["target_buff_state"]);
    }

    [Fact]
    public void CanonicalOutputScalesCastAndGcdWithActualBaseGcd()
    {
        var machine = CombatStateMachine.FromDefaultConfig(OutputsTestKit.RepoRoot, "black_mage", actualBaseGcd: 2.17);
        var result = TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii");
        var payload = TimelineTestDriver.FormatResult(machine, result);

        var actualBaseGcd = 2.17;
        var referenceGcd = machine.Project.System.SkillTableBaseGcd;
        var expectedFireIiiCast = actualBaseGcd / referenceGcd * 3.5;
        var expectedFireIvCast = actualBaseGcd / referenceGcd * 2.0;

        var skillHistory = (List<Dictionary<string, object?>>)payload["skill_history_context"];
        Assert.Equal(expectedFireIiiCast,
            (double)((Dictionary<string, object?>)skillHistory[^1]["cast_time"])["seconds"]!, 5);
        var historyState = (Dictionary<string, object?>)payload["state_history_context"];
        Assert.Equal(actualBaseGcd,
            OutputsTestKit.HistoryVectorValue(historyState, "player_state", "request_state.current_gcd_seconds"), 5);

        var nextState = OutputsTestKit.ReadyAfterGcd(machine, result.NextState);
        var nextResult = TimelineTestDriver.Execute(machine, nextState, "fire_iv");
        var nextPayload = TimelineTestDriver.FormatResult(machine, nextResult);
        var fireIv = ((List<Dictionary<string, object?>>)nextPayload["skill_history_context"])[^1];
        Assert.Equal(expectedFireIvCast, (double)((Dictionary<string, object?>)fireIv["cast_time"])["seconds"]!, 5);
        Assert.Equal(actualBaseGcd, (double)((Dictionary<string, object?>)fireIv["gcd_window"])["seconds"]!, 5);
        Assert.DoesNotContain("gcds", ((Dictionary<string, object?>)fireIv["cast_time"]).Keys);
        Assert.DoesNotContain("gcds", ((Dictionary<string, object?>)fireIv["gcd_window"]).Keys);
    }

    [Fact]
    public void PrecisionConfigLoaderParsesDtypes()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"precision_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(Path.Combine(tempDir, "config"));
        File.WriteAllText(
            Path.Combine(tempDir, "config", "precision.yaml"),
            "precision:\n  int_dtype: int64\n  float_dtype: float32\n");

        try
        {
            var precision = PrecisionConfigLoader.Load(tempDir);
            Assert.Equal(TensorDtype.Int64, precision.IntDtype);
            Assert.Equal(TensorDtype.Float32, precision.FloatDtype);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }
}

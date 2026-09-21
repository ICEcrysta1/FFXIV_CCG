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
    public void VectorStateHistoryAfterAdvancesToNextDecisionTiming()
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
        Assert.Contains("before.mp", (List<string>)historyState["player_state_feature_keys"]);
        Assert.Contains("after.mp", (List<string>)historyState["player_state_feature_keys"]);
        Assert.Equal(8000.0, OutputsTestKit.HistoryVectorValue(historyState, "player_state", "after.mp"), 5);
        Assert.Equal(0.0, OutputsTestKit.HistoryVectorValue(historyState, "player_state", "before.time_seconds"), 5);
        Assert.Equal(3.0, OutputsTestKit.HistoryVectorValue(historyState, "player_state", "after.time_seconds"), 5);
        Assert.Equal(2.5, OutputsTestKit.HistoryVectorValue(historyState, "player_state", "after.current_gcd_seconds"), 5);
        Assert.Equal(597.0, OutputsTestKit.HistoryVectorValue(historyState, "player_state", "after.fight_remaining_seconds"), 5);
        Assert.Equal(0.0, OutputsTestKit.HistoryVectorValue(historyState, "player_state", "after.gcd_remaining_seconds"), 5);
        Assert.Equal(1.0, OutputsTestKit.HistoryVectorValue(historyState, "player_state", "after.boss_targetable"), 5);
        Assert.Equal(1.0, OutputsTestKit.HistoryVectorValue(historyState, "buff_state", "after.resource.thundercloud_ready"), 5);
        Assert.Equal(new List<string>
        {
            "before.target.high_thunder.active",
            "before.target.high_thunder.remaining_seconds",
            "before.target.high_thunder.remaining_gcds",
            "before.target.high_thunder.stacks",
            "before.target.cumulative_dot_potency",
            "before.target.cumulative_potency",
            "before.target.current_potency",
            "before.target.current_gcd_dot_potency",
            "after.target.high_thunder.active",
            "after.target.high_thunder.remaining_seconds",
            "after.target.high_thunder.remaining_gcds",
            "after.target.high_thunder.stacks",
            "after.target.cumulative_dot_potency",
            "after.target.cumulative_potency",
            "after.target.current_potency",
            "after.target.current_gcd_dot_potency",
        }, historyState["target_buff_state_feature_keys"]);
        Assert.Equal(3.0, OutputsTestKit.HistoryVectorValue(historyState, "resource_state", "after.astral_fire"), 5);
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

        var candidateSkill = (Dictionary<string, object?>)payload["candidate_skill_context"];
        var skillKeys = (List<object>)candidateSkill["skill_key"];
        Assert.True(skillKeys.Count > 0);
        Assert.Contains("value", candidateSkill.Keys);

        var despairIndex = skillKeys.IndexOf("despair");
        Assert.True(despairIndex >= 0);

        var candidateState = (Dictionary<string, object?>)payload["candidate_state_context"];
        // tensor 打包按列式结构重排：feature keys 以 object 装箱出现
        var playerFeatureKeys = (List<object>)candidateState["player_state_feature_keys"];
        Assert.NotEmpty(playerFeatureKeys);
        var playerPayload = (Dictionary<string, object?>)((Dictionary<string, object?>)candidateState["tokens"])["player_state"];
        var values = (List<object>)playerPayload["values"];
        var isNull = (List<object>)playerPayload["is_null"];

        // despair 初始非法：after 段全部 null（is_null 有 true）
        var afterStart = ((List<object>)candidateState["player_state_feature_keys"]).Count / 2;
        var despairNull = ((List<object>)isNull[despairIndex]).Cast<bool>().Skip(afterStart);
        Assert.Contains(true, despairNull);
        _ = values;
    }

    [Fact]
    public void OutputContextSchemaMetadataMatchesCanonicalOutput()
    {
        var machine = OutputsTestKit.BuildMachine();
        var payload = TimelineTestDriver.FormatVectorState(machine, machine.InitialState());
        var schemaMetadata = OutputContextSchema.BuildOutputContextSchemaMetadata(payload);

        Assert.Equal(OutputContextSchema.CanonicalContextSchemaVersion, schemaMetadata["schema_version"]);
        Assert.Equal(OutputContextSchema.CanonicalContextTopLevelKeys.ToList(), schemaMetadata["top_level_keys"]);
        Assert.Equal(new List<string> { "player_state", "buff_state", "target_buff_state", "resource_state" },
            schemaMetadata["state_vector_group_keys"]);
        var historyState = (Dictionary<string, object?>)payload["state_history_context"];
        var candidateState = (Dictionary<string, object?>)payload["candidate_state_context"];
        var metadataHistory = (Dictionary<string, List<string>>)schemaMetadata["state_history_feature_keys"];
        var metadataCandidate = (Dictionary<string, List<string>>)schemaMetadata["candidate_state_feature_keys"];
        Assert.Equal(historyState["player_state_feature_keys"], metadataHistory["player_state"]);
        Assert.Equal(candidateState["target_buff_state_feature_keys"], metadataCandidate["target_buff_state"]);
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
            OutputsTestKit.HistoryVectorValue(historyState, "player_state", "after.current_gcd_seconds"), 5);

        var candidateSkill = (List<Dictionary<string, object?>>)payload["candidate_skill_context"];
        var fireIv = candidateSkill.First(token => (string)token["skill_key"] == "fire_iv");
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

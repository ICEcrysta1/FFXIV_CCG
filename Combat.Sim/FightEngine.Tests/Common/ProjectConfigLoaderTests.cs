using Combat.Sim.Config;
using Combat.Sim.Models.Definitions;

namespace FightEngine.Tests.Common;

/// <summary>配置加载测试：真实仓库配置解析与校验错误路径。</summary>
public class ProjectConfigLoaderTests
{
    private const string BaseSystemYaml = """
        timing:
          base_gcd: 2.5
          skill_table_base_gcd: 2.5
        mp_recovery:
          tick_interval_seconds: 3.0
          in_combat_amount: 200
        potency:
          burst_potion_multiplier: 1.08
          raid_buff_window_multiplier: 1.10
        raid_buff_window:
          marker_skills: [amplifier]
          duration_seconds: 20.0
        statuses:
          burst_potion:
            game_id: 1000049
            duration: 30.0
        skills:
          potion:
            game_id: 99999
            name: 爆发药
            kind: ogcd
            behavior: grant_status
            cooldown: 270.0
            applies_statuses: [burst_potion]
        """;

    private const string BaseJobYaml = """
        job:
          key: black_mage
          name: 黑魔法师
        timing:
          default_fight_remaining: 600.0
        resources:
          astral_fire:
            max_value: 3
        statuses:
          ley_lines:
            game_id: 737
            duration: 20.0
            max_stacks: 1
        skills:
          fire_iii:
            game_id: 100
            name: 爆炎
            kind: gcd
            behavior: fire_iii
            potency: 290
            value: 1.10
            cast_time: 3.5
            mp_cost: 2000
            aoe_secondary_reduction: 1.00
        """;

    private const string WildfireDriftJobYaml = """
        job:
          key: machinist
          name: 机工士
        timing:
          default_fight_remaining: 600.0
          wildfire_duration: 10.0
        resources:
          wildfire_remaining:
            max_value: 9
        """;

    private const string BaseDefaultYaml = """
        project:
          name: Test
          version: 1.0.0
        runtime:
          strict_validation: true
          system_config: config/system.yaml
          job_configs:
            black_mage: config/jobs/black_mage.yaml
        engine_timing:
          animation_lock_ogcd: 0.4
          action_effect_settle_seconds: 0.05
          action_queue_window_seconds: 0.05
        engine_resources:
          max_mp: 10000
        """;

    /// <summary>写最小配置树到临时目录，返回 default.yaml 路径。</summary>
    private static string WriteTree(string root, string systemYaml, string jobYaml, string? defaultYaml = null)
    {
        Directory.CreateDirectory(Path.Combine(root, "config", "jobs"));
        File.WriteAllText(Path.Combine(root, "config", "system.yaml"), systemYaml);
        File.WriteAllText(Path.Combine(root, "config", "jobs", "black_mage.yaml"), jobYaml);
        var defaultPath = Path.Combine(root, "config", "default.yaml");
        File.WriteAllText(defaultPath, defaultYaml ?? BaseDefaultYaml);
        return defaultPath;
    }

    private static string CreateTempRoot() =>
        Path.Combine(Path.GetTempPath(), "fight_engine_tests_" + Guid.NewGuid().ToString("N"));

    /// <summary>断言指定文本加载失败且错误消息包含期望片段。</summary>
    private static void AssertLoadFails(
        string systemYaml,
        string jobYaml,
        string expectedMessagePart,
        string? jobTag = "black_mage")
    {
        var root = CreateTempRoot();
        try
        {
            WriteTree(root, systemYaml, jobYaml);
            var ex = Assert.Throws<InvalidOperationException>(
                () => ProjectConfigLoader.LoadProjectConfig(root, jobTag!));
            Assert.Contains(expectedMessagePart, ex.Message);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    // ---------- 真实仓库配置解析 ----------

    [Fact]
    public void 真实仓库配置可完整解析()
    {
        var root = FindRepoRoot();
        var config = ProjectConfigLoader.LoadProjectConfig(root, "black_mage");

        // project / runtime
        Assert.Equal("Context Combat Generator", config.Name);
        Assert.Equal("1.0.0", config.Version);
        Assert.Equal("black_mage", config.Runtime.JobTag);

        // engine_timing / engine_resources
        Assert.Equal(0.6, config.EngineTiming.ActionQueueWindowSeconds);
        Assert.Equal(10000, config.EngineResources.MaxMp);

        // system
        Assert.Equal(2.5, config.System.BaseGcd);
        Assert.Equal(2.5, config.System.SkillTableBaseGcd);
        Assert.Equal(3.0, config.System.MpRecovery.TickIntervalSeconds);
        Assert.Equal(200, config.System.MpRecovery.InCombatAmount);
        Assert.Equal(1.08, config.System.Potency.BurstPotionMultiplier);
        Assert.Equal(1.10, config.System.Potency.RaidBuffWindowMultiplier);
        Assert.Equal(new[] { "amplifier" }, config.System.RaidBuffWindowMarkerSkills);
        Assert.Equal(20.0, config.System.RaidBuffWindowDuration);
        Assert.Equal(2, config.System.Statuses.Count);
        Assert.Equal(1000049, config.System.Statuses["burst_potion"].GameId);
        Assert.Single(config.System.Skills);
        Assert.DoesNotContain(config.System.Skills, skill => skill.Key == "ogcd_wait");

        // 系统技能细节
        var potion = config.System.Skills.Single(s => s.Key == "potion");
        Assert.Equal(99999, potion.GameId);
        Assert.Equal("grant_status", potion.Behavior);
        Assert.Equal(270.0, potion.Cooldown);
        Assert.Equal(new[] { "burst_potion" }, potion.AppliesStatuses);
        Assert.Contains("burst", potion.Tags);

        // job 身份统一用 key 标识
        Assert.Equal("black_mage", config.Job.Key);
        Assert.Equal(600.0, config.Job.DefaultFightRemaining);
        Assert.Equal(9, config.Job.ResourceLimits.Count);
        Assert.Equal(3, config.Job.ResourceLimits["astral_fire"]);
        Assert.Equal(30, config.Job.ResourceLimits["polyglot_timer"]);
        Assert.Equal(6, config.Job.Statuses.Count);
        Assert.Equal(737, config.Job.Statuses["ley_lines"].GameId);
        Assert.True(config.Job.Skills.Count >= 20);

        // 职业技能字段（fire_iii）
        var fireIii = config.Job.Skills.Single(s => s.Key == "fire_iii");
        Assert.Equal(152, fireIii.GameId);
        Assert.Equal(ActionKind.Gcd, fireIii.Kind);
        Assert.Equal("fire_iii", fireIii.Behavior);
        Assert.Equal(290, fireIii.Potency);
        Assert.Equal(1.10, fireIii.Value);
        Assert.Equal(2000, fireIii.MpCost);
        Assert.False(fireIii.MpCostIsFull);
        Assert.Equal(2000, fireIii.MpCostFloor);
        Assert.Equal(3.5, fireIii.CastTime);
        Assert.Equal(2.5, fireIii.RecastTime);
        Assert.True(fireIii.RequiresTarget); // potency > 0 时默认推断
    }

    [Fact]
    public void 未注册职业标签抛错()
    {
        var root = CreateTempRoot();
        try
        {
            WriteTree(root, BaseSystemYaml, BaseJobYaml);
            var ex = Assert.Throws<InvalidOperationException>(
                () => ProjectConfigLoader.LoadProjectConfig(root, "machinist"));
            Assert.Contains("no job config registered for", ex.Message);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    // ---------- 校验错误路径 ----------

    [Fact]
    public void 团辅标记为空抛错() =>
        AssertLoadFails(
            BaseSystemYaml.Replace("[amplifier]", "[]"),
            BaseJobYaml,
            "marker_skills must not be empty");

    [Fact]
    public void 团辅标记重复抛错() =>
        AssertLoadFails(
            BaseSystemYaml.Replace("[amplifier]", "[amplifier, amplifier]"),
            BaseJobYaml,
            "marker_skills must not contain duplicates");

    [Fact]
    public void 团辅时长非正抛错() =>
        AssertLoadFails(
            BaseSystemYaml.Replace("duration_seconds: 20.0", "duration_seconds: 0"),
            BaseJobYaml,
            "duration_seconds must be positive");

    [Fact]
    public void AoE衰减越界抛错() =>
        AssertLoadFails(
            BaseSystemYaml,
            BaseJobYaml.Replace("aoe_secondary_reduction: 1.00", "aoe_secondary_reduction: 1.50"),
            "aoe_secondary_reduction must be between 0 and 1");

    [Fact]
    public void 技能价值非正抛错() =>
        AssertLoadFails(
            BaseSystemYaml,
            BaseJobYaml.Replace("value: 1.10", "value: 0"),
            "value must be a positive finite number");

    [Fact]
    public void 状态层数小于1抛错() =>
        AssertLoadFails(
            BaseSystemYaml,
            BaseJobYaml.Replace("max_stacks: 1", "max_stacks: 0"),
            "max_stacks must be >= 1");

    [Fact]
    public void 状态键跨配置重复抛错() =>
        AssertLoadFails(
            BaseSystemYaml,
            BaseJobYaml.Replace(
                "  ley_lines:\n    game_id: 737\n    duration: 20.0\n    max_stacks: 1",
                "  ley_lines:\n    game_id: 737\n    duration: 20.0\n    max_stacks: 1\n" +
                "  burst_potion:\n    game_id: 1000050\n    duration: 10.0"),
            "duplicate status keys across system/job config");

    [Fact]
    public void 状态game_id跨配置冲突抛错() =>
        AssertLoadFails(
            BaseSystemYaml,
            BaseJobYaml.Replace("game_id: 737", "game_id: 1000049"),
            "duplicate status game_id across system/job config");

    [Fact]
    public void 技能键跨配置重复抛错() =>
        AssertLoadFails(
            BaseSystemYaml,
            BaseJobYaml.Replace("  fire_iii:", "  potion:"),
            "duplicate skill keys across system/job config");

    [Fact]
    public void 技能game_id跨配置冲突抛错() =>
        AssertLoadFails(
            BaseSystemYaml,
            BaseJobYaml.Replace("game_id: 100", "game_id: 99999"),
            "duplicate skill game_id across system/job config");

    [Fact]
    public void 引用未注册状态抛错() =>
        AssertLoadFails(
            BaseSystemYaml.Replace("applies_statuses: [burst_potion]", "applies_statuses: [nope]"),
            BaseJobYaml,
            "references unregistered status");

    [Fact]
    public void grant状态技能无状态抛错() =>
        AssertLoadFails(
            BaseSystemYaml.Replace("applies_statuses: [burst_potion]", "applies_statuses: []"),
            BaseJobYaml,
            "uses grant_status but applies_statuses is empty");

    [Fact]
    public void 非法mp_cost语义抛错() =>
        AssertLoadFails(
            BaseSystemYaml,
            BaseJobYaml.Replace("mp_cost: 2000", "mp_cost: half"),
            "unsupported mp_cost semantic");

    [Fact]
    public void 资源缺max_value抛错() =>
        AssertLoadFails(
            BaseSystemYaml,
            BaseJobYaml.Replace("  astral_fire:\n    max_value: 3", "  astral_fire:\n    other: 1"),
            "is missing max_value");

    [Fact]
    public void 资源max_value非正抛错() =>
        AssertLoadFails(
            BaseSystemYaml,
            BaseJobYaml.Replace("max_value: 3", "max_value: 0"),
            "max_value must be positive and finite");

    [Fact]
    public void 野火持续时间与资源上限漂移抛错() =>
        AssertLoadFails(
            BaseSystemYaml,
            WildfireDriftJobYaml,
            "wildfire_duration and resources.wildfire_remaining.max_value must match");

    // ---------- 辅助 ----------

    private static string FindRepoRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "config", "default.yaml")))
            {
                return dir.FullName;
            }
        }
        throw new InvalidOperationException("repo root not found from test working directory");
    }
}

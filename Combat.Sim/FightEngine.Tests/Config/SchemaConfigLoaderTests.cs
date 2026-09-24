using System.Reflection;

using Combat.Sim.Config;
using Combat.Sim.Outputs;
using Combat.Sim.Outputs.TokenBuilders;

using Xunit;

namespace FightEngine.Tests.Config;

/// <summary>
/// 共享 schema 加载层测试：锁定「未 Load 时访问必须抛友好 InvalidOperationException」，
/// 避免输出层静态成员在类型初始化期访问 Instance 时被 TypeInitializationException 包装。
/// 本类通过反射重置进程级单例，依赖程序集级串行配置（见 TestSchemaInit.cs）。
/// </summary>
public class SchemaConfigLoaderStateTests
{
    [Fact]
    public void Instance_未Load时_抛友好错误()
    {
        ResetForTest();
        try
        {
            var ex = Assert.Throws<InvalidOperationException>(() => _ = SchemaConfigLoader.Instance);
            Assert.Contains("必须先于", ex.Message);
        }
        finally
        {
            SchemaConfigLoader.Load(TestRepoRoot.Find());
        }
    }

    [Fact]
    public void PlayerVectorTokenBuilder_未Load时_抛友好错误而非TypeInitializationException()
    {
        ResetForTest();
        try
        {
            var ex = Assert.Throws<InvalidOperationException>(
                () => _ = new PlayerVectorTokenBuilder().FeatureKeys);
            Assert.Contains("必须先于", ex.Message);
        }
        finally
        {
            SchemaConfigLoader.Load(TestRepoRoot.Find());
        }
    }

    private static void ResetForTest()
    {
        var gate = typeof(SchemaConfigLoader)
            .GetField("Gate", BindingFlags.NonPublic | BindingFlags.Static)!.GetValue(null);
        lock (gate!)
        {
            SetStatic("_instance", null);
            SetStatic("_projectRoot", null);
        }
    }

    private static void SetStatic(string fieldName, object? value) =>
        typeof(SchemaConfigLoader)
            .GetField(fieldName, BindingFlags.NonPublic | BindingFlags.Static)!
            .SetValue(null, value);
}

/// <summary>
/// 共享 schema 内容契约测试：严格字段组访问与团辅窗口模板展开
/// （不依赖单例重置，随程序集初始化直接可用）。
/// </summary>
public class SchemaConfigContentTests
{
    [Fact]
    public void SidecarContractVersion_来自共享Schema()
    {
        // 契约版本来自 schema，构建时同时嵌入 FightEngine 程序集供运行时识别旧 DLL。
        Assert.Equal(8, SchemaConfigLoader.Instance.SidecarContractVersion);
        Assert.Equal(
            SchemaConfigLoader.Instance.SidecarContractVersion,
            SchemaConfigLoader.AssemblySidecarContractVersion);
    }

    [Fact]
    public void Load_程序集契约版本与schema不一致时拒绝旧DLL()
    {
        var realRoot = TestRepoRoot.Find();
        var fakeRoot = Path.Combine(Path.GetTempPath(), $"ffxiv_schema_stale_{Guid.NewGuid():N}");
        var fakeConfigDirectory = Path.Combine(fakeRoot, "config");
        Directory.CreateDirectory(fakeConfigDirectory);
        var realSchemaPath = Path.Combine(realRoot, "config", "schema.yaml");
        var fakeSchemaPath = Path.Combine(fakeConfigDirectory, "schema.yaml");
        var schemaText = File.ReadAllText(realSchemaPath);
        var oldVersionLine = $"sidecar_contract_version: {SchemaConfigLoader.AssemblySidecarContractVersion}";
        var newVersionLine = $"sidecar_contract_version: {SchemaConfigLoader.AssemblySidecarContractVersion + 1}";
        var staleSchemaText = schemaText.Replace(oldVersionLine, newVersionLine, StringComparison.Ordinal);
        Assert.NotEqual(schemaText, staleSchemaText);
        File.WriteAllText(fakeSchemaPath, staleSchemaText);

        try
        {
            var exception = Assert.Throws<InvalidOperationException>(
                () => SchemaConfigLoader.Load(fakeRoot));
            Assert.Contains(newVersionLine.Split(':')[1].Trim(), exception.Message);
            Assert.Contains("FightEngine 程序集内嵌版本", exception.Message);
        }
        finally
        {
            try
            {
                SchemaConfigLoader.Load(realRoot);
            }
            finally
            {
                Directory.Delete(fakeRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void RequireStateVectorFields_缺失组_显式报错()
    {
        var ex = Assert.Throws<InvalidOperationException>(
            () => SchemaConfigLoader.RequireStateVectorFields("no_such_group"));
        Assert.Contains("no_such_group", ex.Message);
    }

    [Fact]
    public void RequireStateVectorFields_玩家字段组_与输出featureKeys一致()
    {
        var fields = SchemaConfigLoader.RequireStateVectorFields("player_state");
        Assert.NotEmpty(fields);
        var builder = new PlayerVectorTokenBuilder();
        Assert.Equal(fields, builder.FeatureKeys);
        Assert.Equal(fields.Count * 2, builder.HistoryFeatureKeys.Count);
    }

    [Fact]
    public void RaidBuffWindowFeatureKeys_按模板展开标记占位()
    {
        // 当前模板 window_extra_fields.raid_buff = ["source.{key}"]；
        // C# 消费模板展开，模板形态变化时此断言失败，提醒同步修改
        var keys = SceneContextSchema.RaidBuffWindowFeatureKeysFor(new[] { "amplifier" });
        Assert.Equal(
            SchemaConfigLoader.Instance.WindowTimeFields.Concat(new[] { "source.amplifier" }).ToArray(),
            keys);
    }

    [Fact]
    public void RaidBuffWindowFeatureKeys_多标记按注册序展开()
    {
        var keys = SceneContextSchema.RaidBuffWindowFeatureKeysFor(new[] { "amplifier", "technique" });
        var timeFieldCount = SchemaConfigLoader.Instance.WindowTimeFields.Count;
        Assert.Equal(new[] { "source.amplifier", "source.technique" }, keys.Skip(timeFieldCount));
    }

    [Fact]
    public void RaidBuffWindowFeatureKeys_空标记_抛错()
    {
        Assert.Throws<ArgumentException>(
            () => SceneContextSchema.RaidBuffWindowFeatureKeysFor(Array.Empty<string>()));
    }

    [Fact]
    public void Load_同根重复调用_返回同一实例()
    {
        var root = TestRepoRoot.Find();
        Assert.Same(SchemaConfigLoader.Load(root), SchemaConfigLoader.Load(root));
    }

    [Fact]
    public void Load_跨根重载_Instance返回最新实例()
    {
        var realRoot = TestRepoRoot.Find();
        // 构造最小假 root（只含 config/schema.yaml，内容与真实一致），模拟跨 root 重载
        var fakeRoot = Path.Combine(Path.GetTempPath(), $"ffxiv_schema_fake_{Guid.NewGuid():N}");
        Directory.CreateDirectory(Path.Combine(fakeRoot, "config"));
        File.Copy(
            Path.Combine(realRoot, "config", "schema.yaml"),
            Path.Combine(fakeRoot, "config", "schema.yaml"));
        try
        {
            var loaded = SchemaConfigLoader.Load(fakeRoot);
            Assert.Same(loaded, SchemaConfigLoader.Instance);
        }
        finally
        {
            // 恢复真实 root 即使抛错也必须清理临时目录（嵌套 finally 保证清理先于恢复失败扩散）
            try
            {
                SchemaConfigLoader.Load(realRoot);
            }
            finally
            {
                Directory.Delete(fakeRoot, recursive: true);
            }
        }
    }

    [Fact]
    public void TargetBuffWindowFeatureKeys_无占位模板_原样透出()
    {
        // targetable 窗口模板不含 {key} 占位：字段原样透出，不依赖任何 source keys
        var keys = SceneContextSchema.TargetableWindowFeatureKeys;
        Assert.Contains("targetable", keys);
        Assert.Contains("segment_kind.combat", keys);
        Assert.Contains(SchemaConfigLoader.Instance.WindowTimeFields[0], keys);
    }
}

/// <summary>本文件测试类共享的仓库根定位（测试运行目录向上找 config/schema.yaml）。</summary>
internal static class TestRepoRoot
{
    public static string Find()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "config", "schema.yaml")))
            {
                return dir.FullName;
            }
        }
        throw new InvalidOperationException("仓库根未找到（缺少 config/schema.yaml）");
    }
}

using System.Runtime.CompilerServices;

using Combat.Sim.Config;

// SchemaConfigLoader 单例状态测试（未 Load 错误路径）需要反射重置进程级单例。
// 不能改用 xUnit collection：CollectionDefinition(DisableParallelization) 只串行化集合内
// 测试，与其他集合并行时重置窗口内仍可能让别的测试访问到空单例，因此程序集级禁用并行
//（185 项测试串行约 0.5s，开销可忽略）。
[assembly: CollectionBehavior(DisableTestParallelization = true)]

namespace FightEngine.Tests;

/// <summary>
/// 测试程序集加载时统一初始化共享 schema（config/schema.yaml）。
/// 输出层静态类（SceneContracts/SceneContextSchema 等）依赖它，
/// 所有直接访问的测试无需各自 Load。
/// </summary>
internal static class TestSchemaInit
{
    [ModuleInitializer]
    internal static void Init()
    {
        SchemaConfigLoader.Load(FindRepoRoot());
    }

    /// <summary>从测试运行目录向上定位仓库根（含 config/schema.yaml）。</summary>
    private static string FindRepoRoot()
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

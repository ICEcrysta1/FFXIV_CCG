using Combat.Sim.Common;
using Xunit;

namespace FightEngine.Tests.Common;

public class SceneWindowTests
{
    [Fact]
    public void 契约常量与Python一致()
    {
        Assert.Equal("targetable_window_context", SceneContracts.TargetableWindowContextKey);
        Assert.Equal("forced_movement_context", SceneContracts.ForcedMovementContextKey);
        Assert.Equal("raid_buff_window_context", SceneContracts.RaidBuffWindowContextKey);
        Assert.Equal("target_count_window_context", SceneContracts.TargetCountWindowContextKey);
        Assert.Equal("absolute_from_fight_scene_context", SceneContracts.SceneContextAbsoluteMode);
        Assert.Equal(0.5, SceneContracts.SlidecastWindowSeconds, 9);
        Assert.Equal(1e-6, SceneContracts.SceneEpsilon, 12);
    }

    [Fact]
    public void 索引映射按枚举顺序()
    {
        var mapping = SceneWindow.FeatureIndexMap(new[] { "a", "b", "c" });
        Assert.Equal(0, mapping["a"]);
        Assert.Equal(1, mapping["b"]);
        Assert.Equal(2, mapping["c"]);
    }

    [Fact]
    public void 相同key列表复用缓存映射()
    {
        var keys = new[] { "x", "y" };
        var first = SceneWindow.FeatureIndexMap(keys);
        var second = SceneWindow.FeatureIndexMap(new[] { "x", "y" });
        Assert.Same(first, second);
        // 不同顺序是不同的 key 列表
        var reversed = SceneWindow.FeatureIndexMap(new[] { "y", "x" });
        Assert.NotSame(first, reversed);
    }

    [Fact]
    public void 解析窗口索引()
    {
        var (start, end, duration) = SceneWindow.ResolveSceneWindowIndices(
            new[] { "start_offset_seconds", "end_offset_seconds", "duration_seconds" },
            "targetable_window_context");
        Assert.Equal((0, 1, 2), (start, end, duration));

        var shuffled = SceneWindow.ResolveSceneWindowIndices(
            new[] { "duration_seconds", "start_offset_seconds", "end_offset_seconds", "extra" },
            "target_count_window_context");
        Assert.Equal((1, 2, 0), shuffled);
    }

    [Fact]
    public void 缺失必填字段抛错()
    {
        var ex = Assert.Throws<InvalidOperationException>(() =>
            SceneWindow.ResolveSceneWindowIndices(new[] { "start_offset_seconds" }, "forced_movement_context"));
        Assert.Contains("forced_movement_context", ex.Message);
        Assert.Contains("end_offset_seconds", ex.Message);
        Assert.Contains("duration_seconds", ex.Message);
    }

    [Fact]
    public void 必填字段清单与Python一致()
    {
        Assert.Equal(
            new[] { "start_offset_seconds", "end_offset_seconds", "duration_seconds" },
            SceneWindow.RequiredSceneWindowFeatureKeys.ToArray());
    }
}

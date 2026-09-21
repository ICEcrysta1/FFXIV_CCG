// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Common;
using Combat.Sim.Config;

namespace Combat.Sim.Outputs;

/// <summary>
/// scene_context 统一 schema 定义（对照 outputs/scene_context_schema.py）。
/// 字段模板与常量由 <c>config/schema.yaml</c> 提供（C# 与 Python 单一事实来源），
/// 使用前必须先调用 <see cref="SchemaConfigLoader.Load"/>。
/// </summary>
public static class SceneContextSchema
{
    public static string[] SceneContextKeys =>
        new[]
        {
            SceneContracts.TargetableWindowContextKey,
            SceneContracts.ForcedMovementContextKey,
            SceneContracts.RaidBuffWindowContextKey,
            SceneContracts.TargetCountWindowContextKey,
        };

    public static string[] TargetableSegmentKinds =>
        SchemaConfigLoader.Instance.TargetableSegmentKinds.ToArray();

    public static string[] RaidBuffWindowSourceKeys =>
        SchemaConfigLoader.Instance.RaidBuffWindowSourceKeys.ToArray();

    private static string[] WindowTimeFields =>
        SchemaConfigLoader.Instance.WindowTimeFields.ToArray();

    private static string[] WindowExtraFields(string windowKey) =>
        SchemaConfigLoader.Instance.WindowExtraFields.TryGetValue(windowKey, out var fields)
            ? fields.ToArray()
            : throw new InvalidOperationException(
                $"schema.yaml 缺少 scene_context.window_extra_fields.{windowKey}");

    /// <summary>按配置中的团辅标记技能构造 scene token feature keys（对照 raid_buff_window_feature_keys）。</summary>
    public static string[] RaidBuffWindowFeatureKeysFor(IReadOnlyList<string> sourceKeys)
    {
        if (sourceKeys.Count == 0)
        {
            throw new ArgumentException("raid buff window source keys must not be empty", nameof(sourceKeys));
        }

        return WindowTimeFields
            .Concat(ExpandWindowExtraFields(WindowExtraFields("raid_buff"), sourceKeys))
            .ToArray();
    }

    /// <summary>
    /// 按窗口额外字段模板展开 feature keys；`{key}` 占位由 source_keys 逐个填充，
    /// 其余字段原样透出（对照 Python _window_feature_keys 的 field.format(key=...)）。
    /// </summary>
    private static IEnumerable<string> ExpandWindowExtraFields(
        IReadOnlyList<string> extraFields,
        IReadOnlyList<string> sourceKeys) =>
        extraFields.SelectMany(field =>
            field.Contains("{key}")
                ? sourceKeys.Select(sourceKey => field.Replace("{key}", sourceKey))
                : new[] { field });

    public static string[] TargetableWindowFeatureKeys =>
        WindowTimeFields
            .Concat(WindowExtraFields("targetable"))
            .Concat(TargetableSegmentKinds.Select(kind => $"segment_kind.{kind}"))
            .ToArray();

    public static string[] ForcedMovementWindowFeatureKeys =>
        WindowTimeFields
            .Concat(WindowExtraFields("forced_movement"))
            .ToArray();

    public static string[] RaidBuffWindowFeatureKeys =>
        RaidBuffWindowFeatureKeysFor(RaidBuffWindowSourceKeys);

    public static string[] TargetCountWindowFeatureKeys =>
        WindowTimeFields
            .Concat(WindowExtraFields("target_count"))
            .ToArray();

    /// <summary>构造统一的窗口上下文结构（对照 build_window_context）。</summary>
    public static Dictionary<string, object?> BuildWindowContext(
        IReadOnlyList<string> featureKeys,
        List<List<double>>? tokens = null) =>
        new()
        {
            ["feature_keys"] = featureKeys.ToList(),
            ["tokens"] = tokens ?? new List<List<double>>(),
        };

    /// <summary>构造稳定空壳 scene_context（对照 build_empty_scene_context）。</summary>
    public static Dictionary<string, object?> BuildEmptySceneContext() =>
        new()
        {
            [SceneContracts.TargetableWindowContextKey] = BuildWindowContext(TargetableWindowFeatureKeys),
            [SceneContracts.ForcedMovementContextKey] = BuildWindowContext(ForcedMovementWindowFeatureKeys),
            [SceneContracts.RaidBuffWindowContextKey] = BuildWindowContext(RaidBuffWindowFeatureKeys),
            [SceneContracts.TargetCountWindowContextKey] = BuildWindowContext(TargetCountWindowFeatureKeys),
        };
}

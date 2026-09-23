// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using System.Globalization;
using System.Reflection;

using Combat.Sim.Common;

namespace Combat.Sim.Config;

/// <summary>
/// 状态向量分组 schema（对照 common/output_context_schema.py 的 StateVectorGroupSchema）。
/// </summary>
public sealed record StateVectorGroupSchema(string GroupKey, string FeatureKeysField, string? ContextKey);

/// <summary>
/// 共享 schema 定义（对照 common/schema_config.py 的 config/schema.yaml）。
/// 承载共享常量、scene context schema、canonical 输出 schema 与状态向量字段模板；
/// C# 与 Python 以 `config/schema.yaml` 为单一事实来源。
/// </summary>
public sealed class SchemaConfig
{
    public double SlidecastWindowSeconds { get; init; }
    public double SceneEpsilon { get; init; }
    public int SidecarContractVersion { get; init; }
    public string SceneContextAbsoluteMode { get; init; } = "";
    public IReadOnlyDictionary<string, string> SceneContextKeys { get; init; } = new Dictionary<string, string>();
    public IReadOnlyList<string> WindowTimeFields { get; init; } = Array.Empty<string>();
    public IReadOnlyList<string> TargetableSegmentKinds { get; init; } = Array.Empty<string>();
    public IReadOnlyList<string> RaidBuffWindowSourceKeys { get; init; } = Array.Empty<string>();
    public IReadOnlyDictionary<string, IReadOnlyList<string>> WindowExtraFields { get; init; } = new Dictionary<string, IReadOnlyList<string>>();
    public int CanonicalSchemaVersion { get; init; }
    public string TokenKey { get; init; } = "";
    public IReadOnlyList<string> CanonicalTopLevelKeys { get; init; } = Array.Empty<string>();
    public IReadOnlyList<StateVectorGroupSchema> StateVectorGroups { get; init; } = Array.Empty<StateVectorGroupSchema>();
    public IReadOnlyDictionary<string, IReadOnlyList<string>> StateVectorFields { get; init; } = new Dictionary<string, IReadOnlyList<string>>();
}

/// <summary>
/// 共享 schema 加载层（对照 common/schema_config.py）。
/// 进程内单例；调用方（状态机构造/导出器）先按项目根 Load 一次。
/// </summary>
public static class SchemaConfigLoader
{
    private const string SidecarContractMetadataKey = "Combat.Sim.SidecarContractVersion";

    // 进程内单例的装载锁：并行 Load 时保护 _instance/_projectRoot 两字段的一致性；
    // _instance 声明为 volatile，读端（Instance/RequireStateVectorFields）无锁也能读到
    // 完整加载完成的实例，不会在跨 root 重载窗口撕裂读
    private static readonly object Gate = new();
    private static volatile SchemaConfig? _instance;
    private static string? _projectRoot;

    /// <summary>从当前 FightEngine 程序集读取构建时嵌入的契约版本。</summary>
    public static int AssemblySidecarContractVersion
    {
        get
        {
            var value = typeof(SchemaConfigLoader).Assembly
                .GetCustomAttributes<AssemblyMetadataAttribute>()
                .SingleOrDefault(metadata => metadata.Key == SidecarContractMetadataKey)
                ?.Value;
            if (!int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out var version))
            {
                throw new InvalidOperationException(
                    $"FightEngine 程序集缺少有效的 {SidecarContractMetadataKey} 构建元数据");
            }
            return version;
        }
    }

    /// <summary>按项目根加载共享 schema（进程内单例，同根重复调用幂等）。</summary>
    public static SchemaConfig Load(string projectRoot)
    {
        lock (Gate)
        {
            if (_instance is not null && string.Equals(_projectRoot, projectRoot, StringComparison.Ordinal))
            {
                return _instance;
            }

            var configPath = Path.Combine(projectRoot, "config", "schema.yaml");
            var payload = YamlConfig.LoadYamlMapping(configPath);
            var schema = Build(payload, configPath);
            var assemblyContractVersion = AssemblySidecarContractVersion;
            if (schema.SidecarContractVersion != assemblyContractVersion)
            {
                throw new InvalidOperationException(
                    $"{configPath}: schema sidecar_contract_version={schema.SidecarContractVersion} " +
                    $"与 FightEngine 程序集内嵌版本={assemblyContractVersion} 不一致；请重新构建 SidecarHost。");
            }
            _instance = schema;
            _projectRoot = projectRoot;
            return _instance;
        }
    }

    /// <summary>已加载的共享 schema；未初始化时抛错（必须先 Load）。</summary>
    public static SchemaConfig Instance =>
        _instance ?? throw new InvalidOperationException(
            "SchemaConfigLoader.Load(projectRoot) 必须先于 SchemaConfigLoader.Instance 访问调用");

    /// <summary>严格读取状态向量字段组；schema 缺失该组时显式报错，避免静默返回空数组。</summary>
    public static IReadOnlyList<string> RequireStateVectorFields(string groupKey)
    {
        if (!Instance.StateVectorFields.TryGetValue(groupKey, out var fields))
        {
            throw new InvalidOperationException($"schema.yaml 缺少 state_vector_fields.{groupKey}");
        }
        return fields;
    }

    private static SchemaConfig Build(Dictionary<string, object?> payload, string configPath)
    {
        var contracts = RequireMapping(payload, "contracts", configPath);
        var scene = RequireMapping(payload, "scene_context", configPath);
        var output = RequireMapping(payload, "canonical_output", configPath);

        return new SchemaConfig
        {
            SlidecastWindowSeconds = YamlValues.ToDouble(
                YamlValues.RequireKey(contracts, "slidecast_window_seconds", configPath), configPath),
            SceneEpsilon = YamlValues.ToDouble(
                YamlValues.RequireKey(contracts, "scene_epsilon", configPath), configPath),
            SidecarContractVersion = YamlValues.ToInt(
                YamlValues.RequireKey(contracts, "sidecar_contract_version", configPath), configPath),
            SceneContextAbsoluteMode = YamlValues.ToText(
                YamlValues.RequireKey(contracts, "scene_context_absolute_mode", configPath)),
            SceneContextKeys = ToStringMapping(
                RequireMapping(contracts, "scene_context_keys", configPath), configPath),
            WindowTimeFields = ToStringList(
                YamlValues.RequireKey(scene, "window_time_fields", configPath)),
            TargetableSegmentKinds = ToStringList(
                YamlValues.RequireKey(scene, "targetable_segment_kinds", configPath)),
            RaidBuffWindowSourceKeys = ToStringList(
                YamlValues.RequireKey(scene, "raid_buff_window_source_keys", configPath)),
            WindowExtraFields = ToStringListMapping(
                RequireMapping(scene, "window_extra_fields", configPath), configPath),
            CanonicalSchemaVersion = YamlValues.ToInt(
                YamlValues.RequireKey(output, "schema_version", configPath), configPath),
            TokenKey = YamlValues.ToText(YamlValues.RequireKey(output, "token_key", configPath)),
            CanonicalTopLevelKeys = ToStringList(
                YamlValues.RequireKey(output, "top_level_keys", configPath)),
            StateVectorGroups = BuildStateVectorGroups(
                RequireMapping(output, "state_vector_groups", configPath), configPath),
            StateVectorFields = ToStringListMapping(
                RequireMapping(payload, "state_vector_fields", configPath), configPath),
        };
    }

    private static List<StateVectorGroupSchema> BuildStateVectorGroups(
        Dictionary<string, object?> groups, string configPath)
    {
        var result = new List<StateVectorGroupSchema>();
        foreach (var (groupKey, rawSpec) in groups)
        {
            var spec = RequireMappingValue(rawSpec, groupKey, configPath);
            var contextKey = YamlValues.Get(spec, "context_key", null);
            result.Add(new StateVectorGroupSchema(
                groupKey,
                YamlValues.ToText(YamlValues.RequireKey(spec, "feature_keys_field", configPath)),
                contextKey is null ? null : YamlValues.ToText(contextKey)));
        }

        return result;
    }

    private static Dictionary<string, object?> RequireMapping(
        Dictionary<string, object?> mapping, string key, string context) =>
        RequireMappingValue(YamlValues.RequireKey(mapping, key, context), key, context);

    private static Dictionary<string, object?> RequireMappingValue(
        object? value, string key, string context)
    {
        if (value is not Dictionary<string, object?> nested)
        {
            throw new InvalidOperationException($"{context}: {key} must be a mapping");
        }
        return nested;
    }

    private static Dictionary<string, string> ToStringMapping(
        Dictionary<string, object?> mapping, string context) =>
        mapping.ToDictionary(
            pair => pair.Key,
            pair => YamlValues.ToText(pair.Value),
            StringComparer.Ordinal);

    private static Dictionary<string, IReadOnlyList<string>> ToStringListMapping(
        Dictionary<string, object?> mapping, string context) =>
        mapping.ToDictionary(
            pair => pair.Key,
            pair => (IReadOnlyList<string>)ToStringList(pair.Value),
            StringComparer.Ordinal);

    private static List<string> ToStringList(object? value)
    {
        if (value is null)
        {
            return new List<string>();
        }
        if (value is not List<object?> items)
        {
            throw new InvalidOperationException($"expected a sequence, got {value}");
        }
        return items.Select(YamlValues.ToText).ToList();
    }
}

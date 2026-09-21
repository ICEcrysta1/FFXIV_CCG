// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;

namespace Combat.Sim.System.Registries;

/// <summary>
/// 系统层职业量谱注册中心：定义注册、量谱归一化读写、快照与前后差分输出
/// （对照 resource_state.ResourceStateRegistry）。
/// </summary>
public sealed class ResourceStateRegistry
{
    private Dictionary<string, JobResourceDefinition> _jobResourceDefinitions = new();
    private string? _registeredJobTag;
    private IReadOnlyList<string> _snapshotResourceKeys = Array.Empty<string>();
    private IReadOnlyDictionary<string, IReadOnlyList<string>> _vectorGroupResourceKeys =
        new Dictionary<string, IReadOnlyList<string>>();

    public IReadOnlyDictionary<string, JobResourceDefinition> JobResourceDefinitions =>
        _jobResourceDefinitions;

    public string? RegisteredJobTag => _registeredJobTag;

    /// <summary>快照输出使用的稳定量谱 key 顺序（注册顺序，排除 internal 分组）。</summary>
    public IReadOnlyList<string> SnapshotResourceKeys => _snapshotResourceKeys;

    public void RegisterJobResources(
        string jobTag,
        IReadOnlyDictionary<string, JobResourceDefinition> resources,
        IReadOnlyDictionary<string, double>? resourceLimits = null)
    {
        var configuredLimits = resourceLimits ?? new Dictionary<string, double>();
        var unknownKeys = configuredLimits.Keys.Except(resources.Keys).OrderBy(k => k).ToList();
        if (unknownKeys.Count > 0)
        {
            throw new InvalidOperationException(
                $"{jobTag} resources contain unknown definitions: {string.Join(", ", unknownKeys)}");
        }
        _registeredJobTag = jobTag;
        _jobResourceDefinitions = resources.ToDictionary(
            pair => pair.Key,
            pair => configuredLimits.TryGetValue(pair.Key, out var limit)
                ? pair.Value with { MaxValue = limit }
                : pair.Value,
            StringComparer.Ordinal);
        RebuildResourceKeyViews();
    }

    /// <summary>返回某个向量分组使用的稳定量谱 key 顺序（排序后）。</summary>
    public IReadOnlyList<string> VectorResourceKeys(string vectorGroup) =>
        _vectorGroupResourceKeys.TryGetValue(vectorGroup, out var keys) ? keys : Array.Empty<string>();

    public double? ResourceMaxValue(string key)
    {
        if (!_jobResourceDefinitions.TryGetValue(key, out var definition))
        {
            throw new KeyNotFoundException($"unregistered job resource: {key}");
        }
        return definition.MaxValue;
    }

    /// <summary>构造职业量谱默认初始值。</summary>
    public Dictionary<string, object> BuildInitialJobResources()
    {
        var resources = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var (key, definition) in _jobResourceDefinitions)
        {
            resources[key] = NormalizeJobResourceValue(definition, definition.DefaultValue);
        }
        return resources;
    }

    public object GetJobResource(CombatState state, string key)
    {
        var definition = RequireDefinition(key);
        return NormalizeJobResourceValue(definition, state.GetJobResource(key, definition.DefaultValue));
    }

    public void SetJobResource(CombatState state, string key, object value)
    {
        var definition = RequireDefinition(key);
        state.SetJobResource(key, NormalizeJobResourceValue(definition, value));
    }

    /// <summary>按资源定义钳制并序列化一个外部计算出的值（供计时器视图投影复用同一套规范化）。</summary>
    public object NormalizeResourceValue(string key, object value)
    {
        var definition = RequireDefinition(key);
        return SerializeJobResourceValue(definition, NormalizeJobResourceValue(definition, value));
    }

    /// <summary>按注册顺序导出当前职业量谱快照（float 值四舍五入到 4 位）。</summary>
    public Dictionary<string, object> BuildJobResourceSnapshot(CombatState state)
    {
        var snapshot = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var key in _snapshotResourceKeys)
        {
            var definition = _jobResourceDefinitions[key];
            snapshot[key] = SerializeJobResourceValue(definition, GetJobResource(state, key));
        }
        return snapshot;
    }

    /// <summary>导出一次动作前后的量谱快照与消耗量谱。</summary>
    public (IReadOnlyDictionary<string, object> Before, IReadOnlyDictionary<string, object> After,
        IReadOnlyDictionary<string, object> Consumed) BuildJobResourceTransition(
        CombatState previousState,
        CombatState nextState)
    {
        return BuildJobResourceTransition(
            BuildJobResourceSnapshot(previousState),
            BuildJobResourceSnapshot(nextState));
    }

    internal (IReadOnlyDictionary<string, object> Before, IReadOnlyDictionary<string, object> After,
        IReadOnlyDictionary<string, object> Consumed) BuildJobResourceTransition(
        IReadOnlyDictionary<string, object> before,
        IReadOnlyDictionary<string, object> after)
    {
        var consumed = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var key in _snapshotResourceKeys)
        {
            var definition = _jobResourceDefinitions[key];
            consumed[key] = BuildConsumedJobResourceValue(definition, before[key], after[key]);
        }
        return (before, after, consumed);
    }

    private JobResourceDefinition RequireDefinition(string key)
    {
        if (!_jobResourceDefinitions.TryGetValue(key, out var definition))
        {
            throw new KeyNotFoundException($"unregistered job resource: {key}");
        }
        return definition;
    }

    private static object NormalizeJobResourceValue(JobResourceDefinition definition, object? value)
    {
        switch (definition.ResourceType)
        {
            case "int":
            {
                var normalized = ToIntValue(value);
                if (definition.MaxValue is { } maxValue)
                {
                    normalized = Math.Max(0, Math.Min((int)maxValue, normalized));
                }
                return normalized;
            }
            case "float":
            {
                var normalized = ToDoubleValue(value);
                if (definition.MaxValue is { } maxValue)
                {
                    normalized = Math.Max(0.0, Math.Min(maxValue, normalized));
                }
                return normalized;
            }
            case "bool":
                return ToBoolValue(value);
            default:
                throw new InvalidOperationException($"unsupported job resource type: {definition.ResourceType}");
        }
    }

    private static object SerializeJobResourceValue(JobResourceDefinition definition, object value)
    {
        switch (definition.ResourceType)
        {
            case "int":
                return ToIntValue(value);
            case "float":
                return Math.Round(ToDoubleValue(value), 4);
            case "bool":
                return ToBoolValue(value);
            default:
                throw new InvalidOperationException($"unsupported job resource type: {definition.ResourceType}");
        }
    }

    private static object BuildConsumedJobResourceValue(
        JobResourceDefinition definition,
        object before,
        object after)
    {
        switch (definition.ResourceType)
        {
            case "int":
                return Math.Max(0, ToIntValue(before) - ToIntValue(after));
            case "float":
                return Math.Round(Math.Max(0.0, ToDoubleValue(before) - ToDoubleValue(after)), 4);
            case "bool":
                return ToBoolValue(before) && !ToBoolValue(after);
            default:
                throw new InvalidOperationException($"unsupported job resource type: {definition.ResourceType}");
        }
    }

    private void RebuildResourceKeyViews()
    {
        _snapshotResourceKeys = _jobResourceDefinitions
            .Where(pair => pair.Value.VectorGroup != "internal")
            .Select(pair => pair.Key)
            .ToList();

        _vectorGroupResourceKeys = _jobResourceDefinitions
            .Where(pair => pair.Value.VectorGroup != "internal")
            .OrderBy(pair => pair.Key, StringComparer.Ordinal)
            .GroupBy(pair => pair.Value.VectorGroup)
            .ToDictionary(
                group => group.Key,
                group => (IReadOnlyList<string>)group.Select(pair => pair.Key).ToList(),
                StringComparer.Ordinal);
    }

    private static int ToIntValue(object? value)
    {
        if (value is null)
        {
            return 0;
        }
        return value switch
        {
            bool b => b ? 1 : 0,
            int i => i,
            long l => (int)l,
            double d => (int)d,
            float f => (int)f,
            _ => Convert.ToInt32(value),
        };
    }

    private static double ToDoubleValue(object? value)
    {
        if (value is null)
        {
            return 0.0;
        }
        return value switch
        {
            bool b => b ? 1.0 : 0.0,
            double d => d,
            float f => f,
            int i => i,
            long l => l,
            _ => Convert.ToDouble(value),
        };
    }

    private static bool ToBoolValue(object? value) => value switch
    {
        null => false,
        bool b => b,
        int i => i != 0,
        long l => l != 0,
        double d => d != 0.0,
        float f => f != 0.0f,
        _ => true,
    };
}

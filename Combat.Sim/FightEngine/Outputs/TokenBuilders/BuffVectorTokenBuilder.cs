// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Definitions;
using Combat.Sim.System;

using Combat.Sim.Config;

namespace Combat.Sim.Outputs.TokenBuilders;

/// <summary>
/// Buff 向量 token 装配器（对照 token_builders/buff_vector_token_builder.py）。
/// 按注册的系统/职业状态顺序导出 active / remaining_seconds / remaining_gcds / stacks，
/// 并在末尾追加 buff 分组的职业资源。
/// </summary>
public sealed class BuffVectorTokenBuilder
{
    private readonly string[] _statusFields;
    private readonly string[] _systemStatusKeys;
    private readonly string[] _jobStatusKeys;
    private readonly string[] _buffResourceKeys;
    private readonly string[] _featureKeys;
    private readonly string[] _historyFeatureKeys;

    public BuffVectorTokenBuilder(SystemStateMachine systemMachine)
    {
        // Buff 状态字段模板（config/schema.yaml 单一事实来源，缺失时显式报错），构造时缓存一次
        _statusFields = SchemaConfigLoader.RequireStateVectorFields("buff_state").ToArray();
        _systemStatusKeys = systemMachine.SystemStatusDefinitions.Keys
            .OrderBy(key => key, StringComparer.Ordinal).ToArray();
        _jobStatusKeys = systemMachine.JobStatusDefinitions.Keys
            .OrderBy(key => key, StringComparer.Ordinal).ToArray();
        _buffResourceKeys = systemMachine.JobResourceVectorKeys("buff").ToArray();
        _featureKeys = BuildFeatureKeys();
        _historyFeatureKeys = _featureKeys
            .Select(key => $"before.{key}")
            .Concat(_featureKeys.Select(key => $"after.{key}"))
            .ToArray();
    }

    public IReadOnlyList<string> FeatureKeys => _featureKeys;

    public IReadOnlyList<string> HistoryFeatureKeys => _historyFeatureKeys;

    public Dictionary<string, object?> BuildCurrentToken(StateContext stateContext) =>
        new()
        {
            ["feature_keys"] = _featureKeys.ToList(),
            ["vector"] = BuildVector(stateContext).ToArray(),
        };

    public IReadOnlyList<double> BuildHistoryToken(StateContext before, StateContext after) =>
        BuildVector(before).Concat(BuildVector(after)).ToList();

    /// <summary>
    /// 单个状态上下文的向量装配；当前态与历史态共用。
    /// 状态字段按 schema 逐项取值，未知字段显式报错（schema 增删字段不再静默漂移）。
    /// </summary>
    private IReadOnlyList<double> BuildVector(StateContext stateContext)
    {
        var activeBuffs = stateContext.Buffs.ToDictionary(buff => (buff.Source, buff.StatusKey));
        var vector = new List<double>();
        foreach (var (source, statusKeys) in new[]
                 {
                     ("system", _systemStatusKeys),
                     ("job", _jobStatusKeys),
                 })
        {
            foreach (var statusKey in statusKeys)
            {
                activeBuffs.TryGetValue((source, statusKey), out var buff);
                foreach (var field in _statusFields)
                {
                    vector.Add(ValueFor(buff, field));
                }
            }
        }

        foreach (var resourceKey in _buffResourceKeys)
        {
            vector.Add(ValueUtils.ToFloat(stateContext.Resources[resourceKey]));
        }

        return vector;
    }

    private static double ValueFor(BuffContextEntry? buff, string field) => field switch
    {
        "active" => buff is null ? 0.0 : 1.0,
        "remaining_seconds" => buff?.RemainingSeconds ?? 0.0,
        "remaining_gcds" => buff?.RemainingGcds ?? 0.0,
        "stacks" => buff?.Stacks ?? 0,
        _ => throw new InvalidOperationException($"schema.yaml 中未知的 Buff 状态字段: {field}"),
    };

    private string[] BuildFeatureKeys()
    {
        var keys = new List<string>();
        foreach (var (source, statusKeys) in new[]
                 {
                     ("system", _systemStatusKeys),
                     ("job", _jobStatusKeys),
                 })
        {
            foreach (var statusKey in statusKeys)
            {
                foreach (var field in _statusFields)
                {
                    keys.Add($"{source}.{statusKey}.{field}");
                }
            }
        }

        foreach (var resourceKey in _buffResourceKeys)
        {
            keys.Add($"resource.{resourceKey}");
        }

        return keys.ToArray();
    }
}

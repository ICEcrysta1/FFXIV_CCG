// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.System;

using Combat.Sim.Config;

namespace Combat.Sim.Outputs.TokenBuilders;

/// <summary>
/// 目标 Buff 向量 token 装配器（对照 token_builders/target_buff_vector_token_builder.py）。
/// 统一导出目标侧 DoT 的 active / remaining_seconds / remaining_gcds / stacks
/// 与累计直伤、累计 DoT、当前直伤、当前 GCD 内 DoT 结算威力。
/// </summary>
public sealed class TargetBuffVectorTokenBuilder
{
    private readonly string[] _dotFields;
    private readonly string[] _fixedFields;
    private readonly string[] _registeredDotKeys;
    private readonly string[] _featureKeys;
    private readonly string[] _historyFeatureKeys;

    public TargetBuffVectorTokenBuilder(SystemStateMachine systemMachine)
    {
        // 目标 DoT 字段模板与固定威力字段（config/schema.yaml 单一事实来源，缺失时显式报错），构造时缓存一次
        _dotFields = SchemaConfigLoader.RequireStateVectorFields("target_buff_dot_state").ToArray();
        _fixedFields = SchemaConfigLoader.RequireStateVectorFields("target_buff_fixed").ToArray();
        _registeredDotKeys = systemMachine.RegisteredTargetDotKeys.ToArray();
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
    /// DoT 字段与固定威力字段按 schema 逐项取值，未知字段显式报错
    /// （schema 增删字段不再静默漂移）。
    /// </summary>
    private IReadOnlyList<double> BuildVector(StateContext stateContext)
    {
        var activeDots = stateContext.Dots.ToDictionary(dot => dot.DotKey);
        var vector = new List<double>();
        foreach (var dotKey in _registeredDotKeys)
        {
            activeDots.TryGetValue(dotKey, out var dot);
            foreach (var field in _dotFields)
            {
                vector.Add(ValueForDot(dot, field));
            }
        }

        foreach (var field in _fixedFields)
        {
            vector.Add(ValueForTarget(stateContext.Target, field));
        }
        return vector;
    }

    private static double ValueForDot(DotContextEntry? dot, string field) => field switch
    {
        "active" => dot is null ? 0.0 : 1.0,
        "remaining_seconds" => dot?.RemainingSeconds ?? 0.0,
        "remaining_gcds" => dot?.RemainingGcds ?? 0.0,
        // DoT 无层数概念，存在即 1.0（与 Python 侧一致）
        "stacks" => dot is null ? 0.0 : 1.0,
        _ => throw new InvalidOperationException($"schema.yaml 中未知的目标 DoT 字段: {field}"),
    };

    private static double ValueForTarget(TargetContext target, string field) => field switch
    {
        "target.cumulative_dot_potency" => target.CumulativeDotPotency,
        "target.cumulative_potency" => target.CumulativePotency,
        "target.current_potency" => target.CurrentPotency,
        "target.current_gcd_dot_potency" => target.CurrentGcdDotPotency,
        _ => throw new InvalidOperationException($"schema.yaml 中未知的目标威力字段: {field}"),
    };

    private string[] BuildFeatureKeys()
    {
        var keys = new List<string>();
        foreach (var dotKey in _registeredDotKeys)
        {
            foreach (var field in _dotFields)
            {
                keys.Add($"target.{dotKey}.{field}");
            }
        }

        foreach (var field in _fixedFields)
        {
            keys.Add(field);
        }
        return keys.ToArray();
    }
}

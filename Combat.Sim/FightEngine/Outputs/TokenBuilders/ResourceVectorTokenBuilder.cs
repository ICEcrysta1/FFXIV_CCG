// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.System;

namespace Combat.Sim.Outputs.TokenBuilders;

/// <summary>
/// 量谱向量 token 装配器（对照 token_builders/resource_vector_token_builder.py）。
/// 历史与最新状态 token 都拼接上一动作后与当前请求快照；资源消耗由技能 token 携带。
/// </summary>
public sealed class ResourceVectorTokenBuilder
{
    private readonly string[] _resourceKeys;
    private readonly string[] _featureKeys;
    private readonly string[] _historyFeatureKeys;

    public ResourceVectorTokenBuilder(SystemStateMachine systemMachine)
    {
        _resourceKeys = systemMachine.JobResourceVectorKeys("resource").ToArray();
        _featureKeys = BuildFeatureKeys();
        _historyFeatureKeys = _featureKeys
            .Select(key => $"previous_action_after.{key}")
            .Concat(_featureKeys.Select(key => $"request_state.{key}"))
            .ToArray();
    }

    public IReadOnlyList<string> FeatureKeys => _featureKeys;

    public IReadOnlyList<string> HistoryFeatureKeys => _historyFeatureKeys;

    public IReadOnlyList<double> BuildHistoryToken(
        IReadOnlyDictionary<string, object> before,
        IReadOnlyDictionary<string, object> after) =>
        BuildVector(before).Concat(BuildVector(after)).ToList();

    /// <summary>单个量谱快照的向量装配；两段共用。</summary>
    private IReadOnlyList<double> BuildVector(IReadOnlyDictionary<string, object> resources) =>
        _resourceKeys.Select(key => ValueUtils.ToFloat(resources[key])).ToList();

    private string[] BuildFeatureKeys() => _resourceKeys.ToArray();
}

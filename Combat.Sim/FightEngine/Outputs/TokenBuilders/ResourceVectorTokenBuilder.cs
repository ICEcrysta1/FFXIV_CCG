// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.System;

namespace Combat.Sim.Outputs.TokenBuilders;

/// <summary>
/// 量谱向量 token 装配器（对照 token_builders/resource_vector_token_builder.py）。
/// 历史 token 是 before / after / consumed 三段拼接。
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
            .Select(key => $"before.{key}")
            .Concat(_featureKeys.Select(key => $"after.{key}"))
            .Concat(_featureKeys.Select(key => $"consumed.{key}"))
            .ToArray();
    }

    public IReadOnlyList<string> FeatureKeys => _featureKeys;

    public IReadOnlyList<string> HistoryFeatureKeys => _historyFeatureKeys;

    public Dictionary<string, object?> BuildCurrentToken(IReadOnlyDictionary<string, object> resources) =>
        new()
        {
            ["feature_keys"] = _featureKeys.ToList(),
            ["vector"] = BuildVector(resources).ToArray(),
        };

    public IReadOnlyList<double> BuildHistoryToken(
        IReadOnlyDictionary<string, object> before,
        IReadOnlyDictionary<string, object> after,
        IReadOnlyDictionary<string, object> consumed) =>
        BuildVector(before).Concat(BuildVector(after)).Concat(BuildVector(consumed)).ToList();

    /// <summary>单个量谱快照的向量装配；before / after / consumed 共用。</summary>
    private IReadOnlyList<double> BuildVector(IReadOnlyDictionary<string, object> resources) =>
        _resourceKeys.Select(key => ValueUtils.ToFloat(resources[key])).ToList();

    private string[] BuildFeatureKeys() => _resourceKeys.ToArray();
}

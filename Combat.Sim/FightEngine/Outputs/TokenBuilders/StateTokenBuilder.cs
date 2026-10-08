// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Config;

namespace Combat.Sim.Outputs.TokenBuilders;

/// <summary>
/// 统一状态 token 装配器（对照 token_builders/state_token_builder.py）。
/// 按父级分组契约装配状态 token：状态历史与当前状态共用同一个
/// <see cref="Build"/>，真实状态字段始终为确定的数值。
/// </summary>
public sealed class StateTokenBuilder
{
    private readonly PlayerVectorTokenBuilder _playerVectorTokenBuilder;
    private readonly BuffVectorTokenBuilder _buffVectorTokenBuilder;
    private readonly TargetBuffVectorTokenBuilder _targetBuffVectorTokenBuilder;
    private readonly ResourceVectorTokenBuilder _resourceVectorTokenBuilder;

    public StateTokenBuilder(
        PlayerVectorTokenBuilder playerVectorTokenBuilder,
        BuffVectorTokenBuilder buffVectorTokenBuilder,
        TargetBuffVectorTokenBuilder targetBuffVectorTokenBuilder,
        ResourceVectorTokenBuilder resourceVectorTokenBuilder)
    {
        _playerVectorTokenBuilder = playerVectorTokenBuilder;
        _buffVectorTokenBuilder = buffVectorTokenBuilder;
        _targetBuffVectorTokenBuilder = targetBuffVectorTokenBuilder;
        _resourceVectorTokenBuilder = resourceVectorTokenBuilder;
    }

    public Dictionary<string, object?> BuildMetadata(IReadOnlyList<string> actionKeys) =>
        OutputContextSchema.StateVectorGroupSchemas.ToDictionary(group => group.FeatureKeysField,
            group => (object?)FeatureKeys(group.GroupKey, actionKeys).ToList());

    private IReadOnlyList<string> FeatureKeys(string group, IReadOnlyList<string> actionKeys) => group switch
    {
        "player_state" => _playerVectorTokenBuilder.HistoryFeatureKeys,
        "buff_state" => _buffVectorTokenBuilder.HistoryFeatureKeys,
        "target_buff_state" => _targetBuffVectorTokenBuilder.HistoryFeatureKeys,
        "resource_state" => _resourceVectorTokenBuilder.HistoryFeatureKeys,
        "skill_availability" => SchemaConfigLoader.Instance.StateSnapshots
            .SelectMany(snapshot => actionKeys.Select(key => $"{snapshot}.{key}")).ToArray(),
        _ => throw new InvalidOperationException($"unsupported state group: {group}"),
    };

    /// <summary>当前和历史只读取已冻结的值；策略动作须在到达此处前完成组合。</summary>
    public Dictionary<string, double[]> Build(ModelStateSnapshot snapshot, IReadOnlyList<string> actionKeys)
    {
        var previous = snapshot.PreviousActionAfter.State;
        var request = snapshot.RequestState.State;
        return OutputContextSchema.StateVectorGroupSchemas.ToDictionary(group => group.GroupKey, group => group.GroupKey switch
        {
            "player_state" => _playerVectorTokenBuilder.BuildHistoryToken(previous.Player, request.Player).ToArray(),
            "buff_state" => _buffVectorTokenBuilder.BuildHistoryToken(previous, request).ToArray(),
            "target_buff_state" => _targetBuffVectorTokenBuilder.BuildHistoryToken(previous, request).ToArray(),
            "resource_state" => _resourceVectorTokenBuilder.BuildHistoryToken(previous.Resources, request.Resources).ToArray(),
            "skill_availability" => Availability(snapshot, actionKeys),
            _ => throw new InvalidOperationException($"unsupported state group: {group.GroupKey}"),
        });
    }

    private static double[] Availability(ModelStateSnapshot snapshot, IReadOnlyList<string> actionKeys)
    {
        var frames = new[] { snapshot.PreviousActionAfter, snapshot.RequestState };
        if (actionKeys.Count != actionKeys.Distinct(StringComparer.Ordinal).Count()
            || frames.Any(frame => frame.SkillAvailability.Count != actionKeys.Count
                || actionKeys.Any(key => !frame.SkillAvailability.ContainsKey(key))))
            throw new InvalidOperationException("frozen skill availability must match the complete action layout");
        return frames.SelectMany(frame => actionKeys.Select(key => frame.SkillAvailability[key] ? 1.0 : 0.0)).ToArray();
    }

}

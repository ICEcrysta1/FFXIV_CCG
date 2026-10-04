// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Outputs.TokenBuilders;

/// <summary>
/// 统一状态 token 装配器（对照 token_builders/state_token_builder.py）。
/// 只负责把四组向量拼成一种状态 token：状态历史与当前状态共用同一个
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

    public IReadOnlyList<string> PlayerHistoryFeatureKeys =>
        _playerVectorTokenBuilder.HistoryFeatureKeys;

    public IReadOnlyList<string> BuffHistoryFeatureKeys =>
        _buffVectorTokenBuilder.HistoryFeatureKeys;

    public IReadOnlyList<string> ResourceHistoryFeatureKeys =>
        _resourceVectorTokenBuilder.HistoryFeatureKeys;

    public IReadOnlyList<string> TargetBuffHistoryFeatureKeys =>
        _targetBuffVectorTokenBuilder.HistoryFeatureKeys;

    /// <summary>状态历史与当前状态共用：上一动作后、当前请求两段四组向量。</summary>
    public Dictionary<string, double[]> Build(
        StateContext previousActionAfter,
        StateContext requestState) =>
        new()
        {
            ["player_state"] = _playerVectorTokenBuilder.BuildHistoryToken(
                previousActionAfter.Player, requestState.Player).ToArray(),
            ["buff_state"] = _buffVectorTokenBuilder.BuildHistoryToken(
                previousActionAfter, requestState).ToArray(),
            ["target_buff_state"] = _targetBuffVectorTokenBuilder.BuildHistoryToken(
                previousActionAfter, requestState).ToArray(),
            ["resource_state"] = _resourceVectorTokenBuilder.BuildHistoryToken(
                previousActionAfter.Resources, requestState.Resources).ToArray(),
        };

}

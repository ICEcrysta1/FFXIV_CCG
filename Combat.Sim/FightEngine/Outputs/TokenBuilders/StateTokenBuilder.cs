// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Outputs.TokenBuilders;

/// <summary>
/// 统一状态 token 装配器（对照 token_builders/state_token_builder.py）。
/// 只负责把四组向量拼成一种状态 token：状态历史与候选状态共用同一个
/// <see cref="Build"/>；非法候选额外走 <see cref="BuildIllegalCandidateToken"/>，
/// 保留真实 before 并把 after/consumed 统一写成 null。
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

    /// <summary>状态历史与合法候选共用：before/after 四组向量（对照 build）。</summary>
    public Dictionary<string, double[]> Build(
        StateContext beforeStateContext,
        StateContext afterStateContext,
        IReadOnlyDictionary<string, object> consumedResources) =>
        new()
        {
            ["player_state"] = _playerVectorTokenBuilder.BuildHistoryToken(
                beforeStateContext.Player, afterStateContext.Player).ToArray(),
            ["buff_state"] = _buffVectorTokenBuilder.BuildHistoryToken(
                beforeStateContext, afterStateContext).ToArray(),
            ["target_buff_state"] = _targetBuffVectorTokenBuilder.BuildHistoryToken(
                beforeStateContext, afterStateContext).ToArray(),
            ["resource_state"] = _resourceVectorTokenBuilder.BuildHistoryToken(
                beforeStateContext.Resources, afterStateContext.Resources, consumedResources).ToArray(),
        };

    /// <summary>非法候选：真实 before + null 填充的 after/consumed 段（对照 build_illegal_candidate_token）。</summary>
    public Dictionary<string, double?[]> BuildIllegalCandidateToken(StateContext beforeStateContext)
    {
        var beforePlayer = VectorOf(_playerVectorTokenBuilder.BuildCurrentToken(beforeStateContext.Player));
        var beforeBuff = VectorOf(_buffVectorTokenBuilder.BuildCurrentToken(beforeStateContext));
        var beforeTargetBuff = VectorOf(_targetBuffVectorTokenBuilder.BuildCurrentToken(beforeStateContext));
        var beforeResource = VectorOf(_resourceVectorTokenBuilder.BuildCurrentToken(beforeStateContext.Resources));

        return new Dictionary<string, double?[]>
        {
            ["player_state"] = beforePlayer.Cast<double?>().Concat(NullVector(beforePlayer.Length)).ToArray(),
            ["buff_state"] = beforeBuff.Cast<double?>().Concat(NullVector(beforeBuff.Length)).ToArray(),
            ["target_buff_state"] = beforeTargetBuff.Cast<double?>().Concat(NullVector(beforeTargetBuff.Length)).ToArray(),
            ["resource_state"] = beforeResource
                .Cast<double?>()
                .Concat(NullVector(beforeResource.Length))
                .Concat(NullVector(beforeResource.Length))
                .ToArray(),
        };
    }

    private static double[] VectorOf(Dictionary<string, object?> token)
    {
        if (token["vector"] is not double[] vector)
        {
            throw new InvalidOperationException("state token vector must be a double array");
        }

        return vector;
    }

    private static double?[] NullVector(int size) => new double?[size];
}

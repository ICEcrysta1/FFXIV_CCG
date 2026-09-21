// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Outputs.TokenBuilders;

namespace Combat.Sim.Outputs.ContextBuilders;

/// <summary>
/// 候选状态上下文装配器（对照 candidate_context_builders/candidate_state_context_builder.py）。
/// 与状态历史共用同一个 <see cref="StateTokenBuilder.Build"/>；
/// 非法候选保留真实 before 并把 after/consumed 统一写成 null。
/// </summary>
public sealed class CandidateStateContextBuilder
{
    private readonly StateTokenBuilder _stateTokenBuilder;

    public CandidateStateContextBuilder(StateTokenBuilder stateTokenBuilder)
    {
        _stateTokenBuilder = stateTokenBuilder;
    }

    public Dictionary<string, object?> Build(IReadOnlyList<CandidateContextEntry> entries) =>
        new()
        {
            ["player_state_feature_keys"] = _stateTokenBuilder.PlayerHistoryFeatureKeys.ToList(),
            ["buff_state_feature_keys"] = _stateTokenBuilder.BuffHistoryFeatureKeys.ToList(),
            ["target_buff_state_feature_keys"] = _stateTokenBuilder.TargetBuffHistoryFeatureKeys.ToList(),
            ["resource_state_feature_keys"] = _stateTokenBuilder.ResourceHistoryFeatureKeys.ToList(),
            ["tokens"] = entries
                .Select(entry => entry.Preview.IsLegal
                    ? (object)_stateTokenBuilder.Build(
                        entry.BeforeStateContext,
                        entry.AfterStateContext!,
                        entry.JobResourcesConsumed)
                    : _stateTokenBuilder.BuildIllegalCandidateToken(entry.BeforeStateContext))
                .ToList(),
        };
}

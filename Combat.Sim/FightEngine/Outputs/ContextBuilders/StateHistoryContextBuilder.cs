// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Outputs.TokenBuilders;

namespace Combat.Sim.Outputs.ContextBuilders;

/// <summary>
/// 状态历史上下文装配器（对照 history_context_builders/state_history_context_builder.py）。
/// 与候选状态共用同一个 <see cref="StateTokenBuilder.Build"/>；
/// 增量缓存按条目引用身份复用固定 token，避免滑动窗口逐决策全量重建。
/// </summary>
public sealed class StateHistoryContextBuilder
{
    private readonly StateTokenBuilder _stateTokenBuilder;
    private readonly int? _historyLimit;
    private readonly List<ActionHistoryEntry> _cachedEntries = new();
    private readonly List<Dictionary<string, double[]>> _cachedTokens = new();

    public StateHistoryContextBuilder(StateTokenBuilder stateTokenBuilder, int? historyLimit = null)
    {
        _stateTokenBuilder = stateTokenBuilder;
        _historyLimit = historyLimit;
    }

    public Dictionary<string, object?> Build(CombatState state)
    {
        var history = Limit(state.History);
        var tokenByEntry = new Dictionary<ActionHistoryEntry, Dictionary<string, double[]>>(
            ReferenceEqualityComparer.Instance);
        for (var i = 0; i < _cachedEntries.Count; i++)
        {
            tokenByEntry[_cachedEntries[i]] = _cachedTokens[i];
        }

        var tokens = new List<Dictionary<string, double[]>>();
        foreach (var entry in history)
        {
            if (!tokenByEntry.TryGetValue(entry, out var token))
            {
                token = _stateTokenBuilder.Build(
                    entry.StateBefore,
                    entry.StateAfter,
                    entry.JobResourcesConsumed);
            }

            tokens.Add(token);
        }

        _cachedEntries.Clear();
        _cachedEntries.AddRange(history);
        _cachedTokens.Clear();
        _cachedTokens.AddRange(tokens);
        return new Dictionary<string, object?>
        {
            ["player_state_feature_keys"] = _stateTokenBuilder.PlayerHistoryFeatureKeys.ToList(),
            ["buff_state_feature_keys"] = _stateTokenBuilder.BuffHistoryFeatureKeys.ToList(),
            ["target_buff_state_feature_keys"] = _stateTokenBuilder.TargetBuffHistoryFeatureKeys.ToList(),
            ["resource_state_feature_keys"] = _stateTokenBuilder.ResourceHistoryFeatureKeys.ToList(),
            ["tokens"] = tokens,
        };
    }

    private List<ActionHistoryEntry> Limit(List<ActionHistoryEntry> history)
    {
        if (_historyLimit is null)
        {
            return history;
        }

        if (_historyLimit == 0)
        {
            return new List<ActionHistoryEntry>();
        }

        return history.Skip(Math.Max(0, history.Count - _historyLimit.Value)).ToList();
    }
}

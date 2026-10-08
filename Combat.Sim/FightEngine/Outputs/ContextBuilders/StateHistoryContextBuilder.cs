// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Outputs.TokenBuilders;

namespace Combat.Sim.Outputs.ContextBuilders;

/// <summary>按历史身份与动作布局缓存冻结 token；分组元数据由父级统一生成。</summary>
public sealed class StateHistoryContextBuilder
{
    private readonly StateTokenBuilder _stateTokenBuilder;
    private Dictionary<object, Dictionary<string, double[]>> _tokens = new(ReferenceEqualityComparer.Instance);
    private string[] _actionKeys = Array.Empty<string>();

    public StateHistoryContextBuilder(StateTokenBuilder stateTokenBuilder) => _stateTokenBuilder = stateTokenBuilder;

    internal List<Dictionary<string, double[]>> Build(IReadOnlyList<ModelHistoryRow> rows, IReadOnlyList<string> actionKeys)
    {
        if (!_actionKeys.SequenceEqual(actionKeys))
        {
            _tokens.Clear();
            _actionKeys = actionKeys.ToArray();
        }
        var retained = new Dictionary<object, Dictionary<string, double[]>>(ReferenceEqualityComparer.Instance);
        var result = new List<Dictionary<string, double[]>>(rows.Count);
        foreach (var row in rows)
        {
            if (!_tokens.TryGetValue(row.Identity, out var token))
                token = _stateTokenBuilder.Build(row.ModelState, actionKeys);
            retained.Add(row.Identity, token);
            result.Add(token);
        }
        _tokens = retained;
        return result;
    }
}

// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Outputs.TokenBuilders;

namespace Combat.Sim.Outputs.ContextBuilders;

/// <summary>
/// 技能历史上下文装配器（对照 history_context_builders/skill_history_context_builder.py）。
/// 与候选技能共用同一个 <see cref="SkillTokenBuilder.Build"/>，只是字段来源是历史条目；
/// 增量缓存按条目引用身份复用固定 token，避免滑动窗口逐决策全量重建。
/// </summary>
public sealed class SkillHistoryContextBuilder
{
    private readonly int? _historyLimit;
    private readonly List<ActionHistoryEntry> _cachedEntries = new();
    private readonly List<Dictionary<string, object?>> _cachedTokens = new();

    public SkillHistoryContextBuilder(int? historyLimit = null)
    {
        _historyLimit = historyLimit;
    }

    public List<Dictionary<string, object?>> Build(CombatState state)
    {
        var history = Limit(state.History);
        var tokenByEntry = new Dictionary<ActionHistoryEntry, Dictionary<string, object?>>(
            ReferenceEqualityComparer.Instance);
        for (var i = 0; i < _cachedEntries.Count; i++)
        {
            tokenByEntry[_cachedEntries[i]] = _cachedTokens[i];
        }

        var tokens = new List<Dictionary<string, object?>>();
        foreach (var entry in history)
        {
            if (!tokenByEntry.TryGetValue(entry, out var token))
            {
                token = BuildFromEntry(entry);
            }

            tokens.Add(token);
        }

        _cachedEntries.Clear();
        _cachedEntries.AddRange(history);
        _cachedTokens.Clear();
        _cachedTokens.AddRange(tokens);
        return tokens;
    }

    private Dictionary<string, object?> BuildFromEntry(ActionHistoryEntry entry) =>
        SkillTokenBuilder.Build(
            skillId: entry.SkillId,
            skillKey: entry.SkillKey,
            skillName: entry.SkillName,
            potency: entry.Potency,
            value: entry.Value,
            kind: entry.SkillKind,
            actualMpCost: Math.Max(0, entry.MpBefore - entry.MpAfter),
            castTimeSeconds: entry.CastTimeSeconds,
            gcdWindowSeconds: entry.GcdWindowSeconds,
            isLegal: entry.IsLegal,
            invalidReason: entry.InvalidReason,
            nextCooldownSeconds: entry.NextCooldownSeconds,
            availableCharges: entry.AvailableCharges,
            maxCharges: entry.MaxCharges,
            jobResourcesConsumed: entry.JobResourcesConsumed,
            timeSeconds: entry.TimeSeconds,
            gcdIndex: entry.GcdIndex);

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

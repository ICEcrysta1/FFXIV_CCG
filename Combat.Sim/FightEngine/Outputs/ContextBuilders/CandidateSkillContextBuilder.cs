// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Definitions;
using Combat.Sim.Outputs.TokenBuilders;

namespace Combat.Sim.Outputs.ContextBuilders;

/// <summary>
/// 候选技能上下文装配器（对照 candidate_context_builders/candidate_skill_context_builder.py）。
/// 与技能历史共用同一个 <see cref="SkillTokenBuilder.Build"/>，只是字段来源是候选预演条目。
/// </summary>
public sealed class CandidateSkillContextBuilder
{
    public List<Dictionary<string, object?>> Build(IReadOnlyList<CandidateContextEntry> entries) =>
        entries.Select(entry => BuildFromEntry(entry)).ToList();

    private static Dictionary<string, object?> BuildFromEntry(CandidateContextEntry entry)
    {
        var preview = entry.Preview;
        return SkillTokenBuilder.Build(
            skillId: preview.Skill.GameId,
            skillKey: preview.Skill.Key,
            skillName: preview.Skill.Name,
            potency: preview.Potency,
            value: preview.Value,
            kind: preview.Skill.Kind == ActionKind.Gcd ? "gcd" : "ogcd",
            actualMpCost: preview.ActualMpCost,
            castTimeSeconds: preview.ActualCastSeconds,
            gcdWindowSeconds: preview.GcdWindowSeconds,
            isLegal: preview.IsLegal,
            invalidReason: preview.InvalidReason,
            nextCooldownSeconds: preview.NextCooldownSeconds,
            availableCharges: preview.AvailableCharges,
            maxCharges: preview.MaxCharges,
            jobResourcesConsumed: entry.JobResourcesConsumed,
            timeSeconds: preview.PreviousState.Time,
            gcdIndex: preview.GcdIndex);
    }
}

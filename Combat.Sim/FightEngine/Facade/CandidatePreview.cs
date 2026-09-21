// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;

namespace Combat.Sim.Facade;

/// <summary>
/// 单个候选动作的预演结果（对照 candidate_preview 的条目字典）。
/// 非法候选保留真实 <see cref="PreviousState"/>，<see cref="NextState"/> 为预演前状态，
/// <see cref="CandidateAfterState"/> 为 null。
/// </summary>
public sealed record CandidatePreview(
    SkillDefinition Skill,
    bool IsLegal,
    string InvalidReason,
    double Potency,
    double Value,
    double NextCooldownSeconds,
    int AvailableCharges,
    int MaxCharges,
    double ActualCastSeconds,
    double GcdWindowSeconds,
    double GcdUnitSeconds,
    int ActualMpCost,
    int GcdIndex,
    CombatState PreviousState,
    CombatState NextState,
    CombatState? CandidateAfterState);

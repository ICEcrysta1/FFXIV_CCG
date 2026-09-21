// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Policy;

/// <summary>不进入 SkillBook、冷却或游戏动作历史的模型控制动作。</summary>
public sealed record PolicyActionDefinition(
    string Key,
    int RawId,
    string Name,
    string CandidateKind,
    string Behavior,
    double Value,
    IReadOnlyList<string> Tags);

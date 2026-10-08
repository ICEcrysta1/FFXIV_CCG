// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;

namespace Combat.Sim.Facade;

/// <summary>从同一状态和队列上下文批量提取接受能力，不推进或预演动作。</summary>
internal sealed class SkillAvailabilityBuilder
{
    private readonly CombatStateMachine _machine;
    internal IReadOnlyList<SkillDefinition> Skills { get; }
    internal IReadOnlyList<string> ActionKeys { get; }

    internal SkillAvailabilityBuilder(CombatStateMachine machine)
    {
        _machine = machine;
        Skills = Array.AsReadOnly(machine.SkillBook.EnabledSkills()
            .OrderBy(skill => skill.Key, StringComparer.Ordinal).ToArray());
        ActionKeys = Array.AsReadOnly(Skills.Select(skill => skill.Key).ToArray());
    }

    internal IReadOnlyDictionary<string, bool> Build(CombatState state, bool queueOccupied) =>
        Skills.ToDictionary(skill => skill.Key,
            skill => _machine.EvaluateActionSubmission(state, skill, queueOccupied).Accepted,
            StringComparer.Ordinal);
}

// Copyright (C) 2026 ICE_crystal
// Copyright (C) 2026 SpikeHS (original Machinist implementation)
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;

namespace Combat.Sim.Jobs.Machinist;

public sealed partial class MachinistJobStateMachine
{
    private static ValidationResult ValidateComboOne(CombatState state, SkillDefinition skill) => new(true);

    private static ValidationResult ValidateComboTwo(CombatState state, SkillDefinition skill) => new(true);

    private static ValidationResult ValidateComboThree(CombatState state, SkillDefinition skill) => new(true);

    private static ValidationResult ValidateTool(CombatState state, SkillDefinition skill) => new(true);

    private static ValidationResult ValidateAirAnchor(CombatState state, SkillDefinition skill) => new(true);

    private static ValidationResult ValidateChainSaw(CombatState state, SkillDefinition skill) => new(true);

    private ValidationResult ValidateExcavator(CombatState state, SkillDefinition skill) =>
        state.HasStatus("excavator_ready")
            ? new ValidationResult(true)
            : new ValidationResult(false, "requires_excavator_ready");

    private ValidationResult ValidateBlazingShot(CombatState state, SkillDefinition skill) =>
        state.HasStatus("overheated")
            ? new ValidationResult(true)
            : new ValidationResult(false, "requires_overheated");

    private ValidationResult ValidateFullMetalField(CombatState state, SkillDefinition skill) =>
        state.HasStatus("full_metal_machinist")
            ? new ValidationResult(true)
            : new ValidationResult(false, "requires_full_metal_machinist");

    private ValidationResult ValidateHypercharge(CombatState state, SkillDefinition skill)
    {
        if (state.HasStatus("overheated"))
        {
            return new ValidationResult(false, "already_overheated");
        }

        return state.HasStatus("free_hypercharge") || IntResource(state, "heat") >= 50
            ? new ValidationResult(true)
            : new ValidationResult(false, "not_enough_heat");
    }

    private ValidationResult ValidateWildfire(CombatState state, SkillDefinition skill) =>
        BoolResource(state, "wildfire_active")
            ? new ValidationResult(false, "wildfire_already_active")
            : new ValidationResult(true);

    private ValidationResult ValidateDetonator(CombatState state, SkillDefinition skill) =>
        BoolResource(state, "wildfire_active")
            ? new ValidationResult(true)
            : new ValidationResult(false, "requires_wildfire");

    private static ValidationResult ValidateBarrelStabilizer(CombatState state, SkillDefinition skill) => new(true);

    private static ValidationResult ValidateDirectAbility(CombatState state, SkillDefinition skill) => new(true);

    private static ValidationResult ValidateReassemble(CombatState state, SkillDefinition skill) =>
        state.HasStatus("reassemble")
            ? new ValidationResult(false, "status_already_active")
            : new ValidationResult(true);
}

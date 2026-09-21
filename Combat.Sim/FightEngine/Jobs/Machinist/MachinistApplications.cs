// Copyright (C) 2026 ICE_crystal
// Copyright (C) 2026 SpikeHS (original Machinist implementation)
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.Models.Timeline;
using Combat.Sim.System;

namespace Combat.Sim.Jobs.Machinist;

public sealed partial class MachinistJobStateMachine
{
    private void ApplyComboOne(CombatState nextState)
    {
        GainHeat(nextState, 5);
        SetCombo(nextState, MachinistConstants.ComboSplit);
    }

    private void ApplyComboTwo(CombatState previousState, CombatState nextState)
    {
        if (ComboMatches(previousState, MachinistConstants.ComboSplit))
        {
            GainHeat(nextState, 5);
            SetCombo(nextState, MachinistConstants.ComboSlug);
            return;
        }

        ClearCombo(nextState);
    }

    private void ApplyComboThree(CombatState previousState, CombatState nextState)
    {
        if (ComboMatches(previousState, MachinistConstants.ComboSlug))
        {
            GainHeat(nextState, 5);
            GainBattery(nextState, 10);
        }

        ClearCombo(nextState);
    }

    private void ApplyAirAnchor(CombatState nextState) => GainBattery(nextState, 20);

    private void ApplyChainSaw(CombatState nextState)
    {
        GainBattery(nextState, 20);
        System.GrantRegisteredStatus(nextState, "excavator_ready");
    }

    private void ApplyExcavator(CombatState nextState)
    {
        GainBattery(nextState, 20);
        System.ClearRegisteredStatus(nextState, "excavator_ready");
    }

    private void ApplyBlazingShot(CombatState nextState)
    {
        System.ConsumeRegisteredStatusStack(nextState, "overheated");
        System.ApplyCooldownReduction(nextState, RequireSkill("double_check"), 15.0);
        System.ApplyCooldownReduction(nextState, RequireSkill("checkmate"), 15.0);
    }

    private void ApplyFullMetalField(CombatState nextState) =>
        System.ClearRegisteredStatus(nextState, "full_metal_machinist");

    private void ApplyHypercharge(CombatState nextState)
    {
        if (nextState.HasStatus("free_hypercharge"))
        {
            System.ClearRegisteredStatus(nextState, "free_hypercharge");
        }
        else
        {
            SetResource(nextState, "heat", IntResource(nextState, "heat") - 50);
        }

        System.GrantRegisteredStatus(nextState, "overheated", stacks: 5);
    }

    private void ApplyWildfire(CombatState nextState)
    {
        SetResource(nextState, "wildfire_active", true);
        SetTimelineDeadline(
            nextState,
            WildfireTimelineKey,
            nextState.Time + Job.TimingValue("wildfire_duration"));
        SetResource(nextState, "wildfire_hits", 0);
    }

    private void ApplyDetonator(CombatState previousState, CombatState nextState)
    {
        var basePotency = WildfireBasePotency(previousState);
        var resolvedPotency = System.ResolveAppliedPotency(previousState, basePotency);
        FinishWildfire(nextState, resolvedPotency, lostPotency: 0.0);
    }

    private void ApplyBarrelStabilizer(CombatState nextState)
    {
        System.GrantRegisteredStatus(nextState, "free_hypercharge");
        System.GrantRegisteredStatus(nextState, "full_metal_machinist");
    }

    private void ApplyReassemble(CombatState nextState) =>
        System.GrantRegisteredStatus(nextState, "reassemble");

    private SkillDefinition RequireSkill(string key) =>
        Job.Skills.FirstOrDefault(skill => skill.Key == key)
        ?? throw new KeyNotFoundException($"machinist skill {key} is missing from job config");

    private void RecordWildfireWeaponskill(CombatState state)
    {
        if (!BoolResource(state, "wildfire_active"))
        {
            return;
        }

        var hits = Math.Min(
            ResourceMax("wildfire_hits"),
            IntResource(state, "wildfire_hits") + 1);
        SetResource(state, "wildfire_hits", hits);
    }

    private void ConsumeReassemble(CombatState state, SkillDefinition skill)
    {
        if (state.HasStatus("reassemble") &&
            skill.Potency > 0 &&
            !skill.Tags.Contains("reassemble_immune"))
        {
            System.ClearRegisteredStatus(state, "reassemble");
        }
    }

    private void RecordOverheatedExpiry(CombatState state)
    {
        if (!state.Statuses.TryGetValue("overheated", out var overheated))
        {
            return;
        }

        SetResource(
            state,
            "wasted_overheated_stacks",
            IntResource(state, "wasted_overheated_stacks") + Math.Max(0, overheated.Stacks));
        // 状态移除由系统层在同一次到期结算内完成，这里只负责读走残留层数。
    }

    private void HandleWildfireExpiry(
        TimelineEvent item,
        CombatState state,
        Func<double, bool>? targetableAt)
    {
        if (!BoolResource(state, "wildfire_active"))
        {
            return;
        }

        var basePotency = WildfireBasePotency(state);
        var potential = System.ResolveAppliedPotencyAtOffset(state, basePotency, 0.0);
        var isTargetable = targetableAt?.Invoke(item.Timestamp) ?? state.BossTargetable;
        FinishWildfire(
            state,
            isTargetable ? System.RecordDelayedPotency(state, potential) : 0.0,
            isTargetable ? 0.0 : potential);
    }

    private void FinishWildfire(CombatState state, double resolvedPotency, double lostPotency)
    {
        var hits = IntResource(state, "wildfire_hits");
        SetResource(
            state,
            "wildfire_resolved_potency",
            FloatResource(state, "wildfire_resolved_potency") + Math.Max(0.0, resolvedPotency));
        SetResource(
            state,
            "wildfire_lost_potency",
            FloatResource(state, "wildfire_lost_potency") + Math.Max(0.0, lostPotency));
        SetResource(
            state,
            "wildfire_missed_weaponskills",
            IntResource(state, "wildfire_missed_weaponskills") +
            Math.Max(0, ResourceMax("wildfire_hits") - hits));
        SetResource(state, "wildfire_active", false);
        SetTimelineDeadline(state, WildfireTimelineKey, null);
        SetResource(state, "wildfire_hits", 0);
    }

    private void GainHeat(CombatState state, int amount)
    {
        var current = IntResource(state, "heat");
        var raw = current + Math.Max(0, amount);
        var max = ResourceMax("heat");
        if (raw > max)
        {
            SetResource(
                state,
                "wasted_heat",
                IntResource(state, "wasted_heat") + raw - max);
        }

        SetResource(state, "heat", Math.Min(max, raw));
    }

    private void GainBattery(CombatState state, int amount)
    {
        var current = IntResource(state, "battery");
        var raw = current + Math.Max(0, amount);
        var max = ResourceMax("battery");
        if (raw > max)
        {
            SetResource(
                state,
                "wasted_battery",
                IntResource(state, "wasted_battery") + raw - max);
        }

        SetResource(state, "battery", Math.Min(max, raw));
    }

    private void SetCombo(CombatState state, int stage)
    {
        SetResource(state, "combo_stage", stage);
        SetTimelineDeadline(state, ComboTimelineKey, state.Time + Job.TimingValue("combo_duration"));
    }

    private void ClearCombo(CombatState state)
    {
        SetResource(state, "combo_stage", MachinistConstants.ComboNone);
        SetTimelineDeadline(state, ComboTimelineKey, null);
    }
}

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
        State.Buffs.GrantRegisteredStatus(nextState, "excavator_ready");
    }

    private void ApplyExcavator(CombatState nextState)
    {
        GainBattery(nextState, 20);
        State.Buffs.ClearRegisteredStatus(nextState, "excavator_ready");
    }

    private void ApplyBlazingShot(CombatState nextState)
    {
        State.Buffs.ConsumeRegisteredStatusStack(nextState, "overheated");
        System.ApplyCooldownReduction(nextState, RequireSkill("double_check"), 15.0);
        System.ApplyCooldownReduction(nextState, RequireSkill("checkmate"), 15.0);
    }

    private void ApplyFullMetalField(CombatState nextState) =>
        State.Buffs.ClearRegisteredStatus(nextState, "full_metal_machinist");

    private void ApplyHypercharge(CombatState nextState)
    {
        if (nextState.Job.HasStatus("free_hypercharge"))
        {
            State.Buffs.ClearRegisteredStatus(nextState, "free_hypercharge");
        }
        else
        {
            SetResource(nextState, "heat", IntResource(nextState, "heat") - 50);
        }

        State.Buffs.GrantRegisteredStatus(nextState, "overheated", stacks: 5);
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
        State.Buffs.GrantRegisteredStatus(nextState, "free_hypercharge");
        State.Buffs.GrantRegisteredStatus(nextState, "full_metal_machinist");
    }

    private void ApplyReassemble(CombatState nextState) =>
        State.Buffs.GrantRegisteredStatus(nextState, "reassemble");

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
        if (state.Job.HasStatus("reassemble") &&
            skill.Potency > 0 &&
            !skill.Tags.Contains("reassemble_immune"))
        {
            State.Buffs.ClearRegisteredStatus(state, "reassemble");
        }
    }

    private void RecordOverheatedExpiry(CombatState state)
    {
        if (!state.Job.Statuses.TryGetValue("overheated", out var overheated))
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

    private void GainHeat(CombatState state, int amount) =>
        GainResource(state, "heat", "wasted_heat", amount);

    private void GainBattery(CombatState state, int amount) =>
        GainResource(state, "battery", "wasted_battery", amount);

    /// <summary>机工量谱增长与溢出统计使用相同结算顺序。</summary>
    private void GainResource(CombatState state, string key, string wastedKey, int amount)
    {
        var current = IntResource(state, key);
        var raw = current + Math.Max(0, amount);
        var max = ResourceMax(key);
        if (raw > max)
        {
            SetResource(
                state,
                wastedKey,
                IntResource(state, wastedKey) + raw - max);
        }

        SetResource(state, key, Math.Min(max, raw));
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

// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;

namespace Combat.Sim.Facade;

/// <summary>
/// 候选动作预演。每个合法候选都从完整 SimulationSnapshot 分支并真实提交一次动作，
/// 因而冷却、状态、职业事件和待结算事实与主执行路径完全相同。
/// </summary>
internal static class CandidatePreviewBuilder
{
    internal static IReadOnlyList<CandidatePreview> BuildEntries(
        JobSimulator simulator,
        CombatStateMachine machine)
    {
        var candidates = new List<CandidatePreview>();
        foreach (var skill in machine.SkillBook.EnabledSkills())
        {
            candidates.Add(BuildEntry(simulator.Fork(), machine, skill));
        }
        return candidates;
    }

    private static CandidatePreview BuildEntry(
        JobSimulator preview,
        CombatStateMachine machine,
        SkillDefinition skill)
    {
        var previousState = preview.GetState();
        var submission = preview.SubmitAction(previousState.Time, skill.Key);
        var validation = new ValidationResult(submission.Accepted, submission.Reason);
        var snapshot = machine.BuildSkillSnapshot(previousState, skill, validation);
        var timing = machine.BuildActionTimingPlan(previousState, skill);
        var nextState = previousState;
        CombatState? candidateAfterState = null;

        if (submission.Accepted)
        {
            var acceptedAt = submission.AcceptedTimestamp ?? previousState.Time;
            var effectAt = submission.EffectTimestamp ?? acceptedAt;
            preview.AdvanceTo(effectAt);
            nextState = preview.GetState();
            var observationDelay = skill.Kind == ActionKind.Gcd
                ? timing.NextGcdWindowSeconds
                : timing.ActualOccupancySeconds;
            preview.AdvanceTo(Math.Max(effectAt, acceptedAt + observationDelay));
            candidateAfterState = preview.GetState();
        }

        return new CandidatePreview(
            skill,
            validation.Ok,
            validation.Reason,
            snapshot.Potency,
            snapshot.Value,
            snapshot.NextCooldownSeconds,
            snapshot.AvailableCharges,
            snapshot.MaxCharges,
            timing.ActualCastSeconds,
            timing.GcdWindowSeconds,
            timing.GcdUnitSeconds,
            machine.EstimateActualMpCost(previousState, skill),
            nextState.GcdIndex,
            previousState,
            nextState,
            candidateAfterState);
    }
}

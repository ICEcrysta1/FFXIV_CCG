// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;

namespace Combat.Sim.Facade;

/// <summary>
/// 候选动作预演。合法候选保留完整战斗状态和事件分支并真实提交一次动作，
/// 冷却、状态、职业事件和待结算事实与主执行路径相同，不携带无关的历史前缀。
/// </summary>
internal static class CandidatePreviewBuilder
{
    internal static IReadOnlyList<CandidatePreview> BuildEntries(
        JobSimulator simulator,
        CombatStateMachine machine)
    {
        var candidates = new List<CandidatePreview>();
        var previousState = simulator.GetStateWithoutHistory();
        // 同刻到期事件只在准备分支结算一次，非法候选不再复制整条时间线。
        var prepared = simulator.ForkForPreview();
        var submissionState = prepared.AdvanceTo(previousState.Time);
        var queueOccupied = prepared.HasQueuedAction();
        foreach (var skill in machine.SkillBook.EnabledSkills())
        {
            var submission = machine.EvaluateActionSubmission(submissionState, skill, queueOccupied);
            candidates.Add(BuildEntry(prepared, machine, skill, previousState,
                new ValidationResult(submission.Accepted, submission.Reason)));
        }
        return candidates;
    }

    private static CandidatePreview BuildEntry(
        JobSimulator prepared,
        CombatStateMachine machine,
        SkillDefinition skill,
        CombatState previousState,
        ValidationResult validation)
    {
        var snapshot = machine.BuildSkillSnapshot(previousState, skill, validation);
        var timing = machine.BuildActionTimingPlan(previousState, skill);
        var nextState = previousState;
        CombatState? candidateAfterState = null;

        if (validation.Ok)
        {
            var preview = prepared.ForkForPreview();
            var submission = preview.SubmitAction(previousState.Time, skill.Key);
            var acceptedAt = submission.AcceptedTimestamp ?? previousState.Time;
            var effectAt = submission.EffectTimestamp ?? acceptedAt;
            nextState = preview.AdvanceTo(effectAt);
            var observationDelay = skill.Kind == ActionKind.Gcd
                ? timing.NextGcdWindowSeconds
                : timing.ActualOccupancySeconds;
            var observeAt = Math.Max(effectAt, acceptedAt + observationDelay);
            candidateAfterState = observeAt == effectAt ? nextState : preview.AdvanceTo(observeAt);
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

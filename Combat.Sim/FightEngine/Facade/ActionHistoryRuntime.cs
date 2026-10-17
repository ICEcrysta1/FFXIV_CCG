// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.Outputs;
using Combat.Sim.Jobs;

namespace Combat.Sim.Facade;

/// <summary>
/// 门面动作历史记录：统一记录动作元信息并挂载职业量谱前后快照，控制长度上限
/// （对照 history.ActionHistoryRuntime）。
/// </summary>
public sealed class ActionHistoryRuntime
{
    private readonly Func<CombatState, CombatState, (
        IReadOnlyDictionary<string, object> Before,
        IReadOnlyDictionary<string, object> After,
        IReadOnlyDictionary<string, object> Consumed)> _buildTransition;
    private readonly HistoryRetention _retention;

    public ActionHistoryRuntime(
        JobStateRuntime jobState,
        int? maxHistory = null)
    {
        ArgumentNullException.ThrowIfNull(jobState);
        if (maxHistory is < 0)
        {
            throw new InvalidOperationException($"max_history must be >= 0, got {maxHistory}");
        }
        _buildTransition = jobState.BuildResourceTransition;
        _retention = new HistoryRetention(maxHistory);
    }

    public void RecordActionHistory(
        CombatState nextState,
        CombatState previousState,
        SkillDefinition skill,
        double potency,
        double value,
        double castTimeSeconds,
        double castTimeGcds,
        double gcdWindowSeconds,
        double gcdWindowGcds,
        bool isLegal,
        string invalidReason,
        double nextCooldownSeconds,
        int availableCharges,
        int maxCharges,
        StateContext stateBefore,
        StateContext stateAfter,
        double? recordedTimeSeconds = null,
        double? requestTimestamp = null,
        double? castCompletedTimestamp = null,
        double? effectTimestamp = null,
        Guid? actionInstanceId = null,
        ModelStateSnapshot? modelState = null)
    {
        var (before, after, consumed) = _buildTransition(previousState, nextState);
        nextState.History.Add(new ActionHistoryEntry(
            SkillKey: skill.Key,
            SkillId: skill.GameId,
            SkillName: skill.Name,
            SkillKind: skill.Kind == ActionKind.Gcd ? "gcd" : "ogcd",
            Potency: potency,
            Value: value,
            MpBefore: previousState.Mp,
            MpAfter: nextState.Mp,
            CastTimeSeconds: castTimeSeconds,
            CastTimeGcds: castTimeGcds,
            GcdWindowSeconds: gcdWindowSeconds,
            GcdWindowGcds: gcdWindowGcds,
            IsLegal: isLegal,
            InvalidReason: invalidReason,
            NextCooldownSeconds: nextCooldownSeconds,
            AvailableCharges: availableCharges,
            MaxCharges: maxCharges,
            JobResourcesBefore: before,
            JobResourcesAfter: after,
            JobResourcesConsumed: consumed,
            TimeSeconds: recordedTimeSeconds ?? previousState.Time,
            GcdIndex: nextState.GcdIndex,
            StateBefore: stateBefore,
            StateAfter: stateAfter,
            RequestTimestamp: requestTimestamp,
            CastCompletedTimestamp: castCompletedTimestamp,
            EffectTimestamp: effectTimestamp,
            ActionInstanceId: actionInstanceId,
            ModelState: modelState,
            HistorySequence: nextState.ReserveHistorySequence()));

        _retention.Trim(nextState.History);
    }
}

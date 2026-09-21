// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.Outputs;
using Combat.Sim.System.Registries;

namespace Combat.Sim.System;

/// <summary>
/// 系统层动作历史记录：统一记录动作元信息并挂载职业量谱前后快照，控制长度上限
/// （对照 history.SystemHistoryRuntime）。
/// </summary>
public sealed class SystemHistoryRuntime
{
    private readonly Func<CombatState, CombatState, (
        IReadOnlyDictionary<string, object> Before,
        IReadOnlyDictionary<string, object> After,
        IReadOnlyDictionary<string, object> Consumed)> _buildTransition;
    private readonly int? _maxHistory;

    public SystemHistoryRuntime(
        ResourceStateRegistry resourceState,
        int? maxHistory = null,
        Func<CombatState, CombatState, (
            IReadOnlyDictionary<string, object> Before,
            IReadOnlyDictionary<string, object> After,
            IReadOnlyDictionary<string, object> Consumed)>? buildTransition = null)
    {
        ArgumentNullException.ThrowIfNull(resourceState);
        if (maxHistory is < 0)
        {
            throw new InvalidOperationException($"max_history must be >= 0, got {maxHistory}");
        }
        _buildTransition = buildTransition ?? resourceState.BuildJobResourceTransition;
        _maxHistory = maxHistory;
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
        Guid? actionInstanceId = null)
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
            ActionInstanceId: actionInstanceId));

        if (_maxHistory == 0)
        {
            nextState.History.Clear();
        }
        else if (_maxHistory is { } maxHistory && nextState.History.Count > maxHistory)
        {
            nextState.History.RemoveRange(0, nextState.History.Count - maxHistory);
        }
    }
}

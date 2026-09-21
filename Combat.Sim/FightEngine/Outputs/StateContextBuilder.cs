// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Common;
using Combat.Sim.Config;
using Combat.Sim.Jobs;
using Combat.Sim.Models.Combat;
using Combat.Sim.System;

namespace Combat.Sim.Outputs;

/// <summary>
/// 状态上下文原料装配器（对照 output_context_builder.build_state_context）。
/// 输出历史条目与候选条目的 before/after 状态上下文共用。
/// </summary>
public sealed class StateContextBuilder
{
    private readonly SystemStateMachine _systemMachine;
    private readonly IJobStateMachine _jobMachine;
    private readonly ProjectConfig _project;

    public StateContextBuilder(
        ProjectConfig project,
        SystemStateMachine systemMachine,
        IJobStateMachine jobMachine)
    {
        _project = project;
        _systemMachine = systemMachine;
        _jobMachine = jobMachine;
    }

    public StateContext Build(CombatState state)
    {
        // 状态的 remaining 属性从绝对截止时间派生，输出层不另行维护计时器。
        var currentGcd = _jobMachine.CurrentGcdDuration(state);
        return new StateContext(
            JobTag: _systemMachine.RegisteredJobTag ??
                    throw new InvalidOperationException("job state is not registered"),
            JobName: _project.Job.Name,
            Player: BuildPlayerContext(state, currentGcd),
            Target: BuildTargetContext(state),
            Resources: _systemMachine.BuildJobResourceSnapshot(state),
            Buffs: BuildBuffContext(state, currentGcd),
            Dots: BuildDotContext(state, currentGcd));
    }

    private PlayerContext BuildPlayerContext(CombatState state, double currentGcd)
    {
        var ogcdWindowSeconds = Math.Max(
            0.0,
            state.WeaveWindowRemaining);
        var nextUntargetableSeconds = state.NextDowntimeEta;
        return new PlayerContext(
            TimeSeconds: state.Time,
            Mp: state.Mp,
            MaxMp: state.MaxMp,
            MpRatio: SafeRatio(state.Mp, state.MaxMp),
            GcdIndex: state.GcdIndex,
            CurrentGcdSeconds: currentGcd,
            FightRemainingSeconds: state.FightRemaining,
            GcdRemainingSeconds: state.GcdRemaining,
            CastRemainingSeconds: state.CastRemaining,
            GcdRemainingGcds: GcdUnits.ToGcdUnits(state.GcdRemaining, currentGcd),
            WeaveWindowSeconds: state.WeaveWindowRemaining,
            WeaveWindowGcds: GcdUnits.ToGcdUnits(state.WeaveWindowRemaining, currentGcd),
            OgcdWindowSeconds: ogcdWindowSeconds,
            OgcdWindowGcds: GcdUnits.ToGcdUnits(ogcdWindowSeconds, currentGcd),
            BossTargetable: state.BossTargetable,
            NextUntargetableInSeconds: nextUntargetableSeconds,
            NextUntargetableInGcds: GcdUnits.ToOptionalGcdUnits(nextUntargetableSeconds, currentGcd),
            DowntimeRemainingSeconds: state.DowntimeRemaining,
            DowntimeRemainingGcds: GcdUnits.ToGcdUnits(state.DowntimeRemaining, currentGcd),
            IsMoving: state.IsMoving,
            OgcdsWeaved: state.OgcdsWeaved,
            MaxOgcdPerWindow: state.MaxOgcdPerWindow);
    }

    private List<BuffContextEntry> BuildBuffContext(CombatState state, double currentGcd)
    {
        var buffs = new List<BuffContextEntry>();
        foreach (var (source, definitions) in new (string Source, IReadOnlyDictionary<string, Models.Definitions.StatusDefinition> Definitions)[]
                 {
                     ("system", _systemMachine.SystemStatusDefinitions),
                     ("job", _systemMachine.JobStatusDefinitions),
                 })
        {
            foreach (var (statusKey, definition) in definitions)
            {
                if (!state.Statuses.TryGetValue(statusKey, out var status) ||
                    status.Remaining <= 0 || status.Stacks <= 0)
                {
                    continue;
                }

                buffs.Add(new BuffContextEntry(
                    Source: source,
                    StatusKey: statusKey,
                    StatusId: definition.GameId,
                    RemainingSeconds: Math.Round(status.Remaining, 4),
                    RemainingGcds: Math.Round(GcdUnits.ToGcdUnits(status.Remaining, currentGcd), 4),
                    Stacks: status.Stacks));
            }
        }

        return buffs;
    }

    private List<DotContextEntry> BuildDotContext(CombatState state, double currentGcd)
    {
        var dots = new List<DotContextEntry>();
        foreach (var dotKey in state.Dots.Keys.OrderBy(key => key, StringComparer.Ordinal))
        {
            var dotState = state.Dots[dotKey];
            if (dotState.Remaining <= 0)
            {
                continue;
            }

            dots.Add(new DotContextEntry(
                DotKey: dotKey,
                RemainingSeconds: Math.Round(dotState.Remaining, 4),
                RemainingGcds: Math.Round(GcdUnits.ToGcdUnits(dotState.Remaining, currentGcd), 4),
                TickIntervalSeconds: Math.Round(dotState.TickInterval, 4),
                PotencyPerTick: dotState.PotencyPerTick));
        }

        return dots;
    }

    private static TargetContext BuildTargetContext(CombatState state) =>
        new(
            CumulativePotency: Math.Round(state.CumulativePotency, 4),
            CumulativeDotPotency: Math.Round(state.CumulativeDotPotency, 4),
            CurrentPotency: Math.Round(state.CurrentPotency, 4),
            CurrentGcdDotPotency: Math.Round(state.CurrentGcdDotPotency, 4));

    private static double SafeRatio(double value, double total) =>
        total <= 0 ? 0.0 : value / total;
}

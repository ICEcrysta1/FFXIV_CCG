// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Timeline;

namespace Combat.Sim.Facade;

public enum RandomSequenceStepKind
{
    SubmitAction,
    AdvanceTo,
}

/// <summary>固定随机策略的一次绝对时间操作。</summary>
public sealed record RandomSequenceStep(
    RandomSequenceStepKind Kind,
    int StepIndex,
    IReadOnlyList<string> LegalKeys,
    int? ChosenIndex,
    string? Action,
    double Timestamp,
    ActionSubmissionResult? Result,
    CombatState StateAfter);

/// <summary>
/// 固定 seed 随机策略回放。策略显式选择下一内部事件时间并调用 AdvanceTo，
/// FightEngine 不替策略猜测 GCD 或动画锁推进量。
/// </summary>
public static class RandomSequenceReplay
{
    public const long DefaultSeed = 20260814;
    public const int DefaultMaxSteps = 100;

    private static ulong NextU64(ref ulong state)
    {
        state += 0x9E3779B97F4A7C15UL;
        var z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    public static IReadOnlyList<RandomSequenceStep> Run(
        CombatStateMachine machine,
        long seed,
        int maxSteps)
    {
        var simulator = new JobSimulator(machine);
        var rngState = (ulong)seed;
        var steps = new List<RandomSequenceStep>();

        for (var stepIndex = 1; stepIndex <= maxSteps; stepIndex++)
        {
            var legalKeys = simulator.AvailableActionKeysAt(simulator.Time)
                .OrderBy(key => key, StringComparer.Ordinal)
                .ToList();
            if (legalKeys.Count == 0)
            {
                var nextTimestamp = simulator.GetNextScheduledEventTime();
                if (nextTimestamp is null || nextTimestamp <= simulator.Time)
                {
                    throw new InvalidOperationException(
                        $"决策卡死：第 {stepIndex} 步无合法动作且没有未来事件");
                }

                var state = simulator.AdvanceTo(nextTimestamp.Value);
                steps.Add(new RandomSequenceStep(
                    RandomSequenceStepKind.AdvanceTo,
                    stepIndex,
                    Array.Empty<string>(),
                    null,
                    null,
                    nextTimestamp.Value,
                    null,
                    state));
                continue;
            }

            var chosenIndex = (int)(NextU64(ref rngState) % (ulong)legalKeys.Count);
            var action = legalKeys[chosenIndex];
            var result = simulator.SubmitAction(simulator.Time, action);
            if (!result.Accepted)
            {
                throw new InvalidOperationException(
                    $"合法候选 {action} 在提交时被拒绝: {result.Reason}");
            }
            steps.Add(new RandomSequenceStep(
                RandomSequenceStepKind.SubmitAction,
                stepIndex,
                legalKeys,
                chosenIndex,
                action,
                simulator.Time,
                result,
                simulator.GetState()));
        }

        return steps;
    }
}

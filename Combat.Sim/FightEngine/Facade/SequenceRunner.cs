// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Timeline;

namespace Combat.Sim.Facade;

/// <summary>按调用方提供的绝对时间请求序列执行，不根据校验 reason 猜测推进时间。</summary>
public static class SequenceRunner
{
    public static Dictionary<string, object?> RunActionSequence(
        CombatStateMachine machine,
        IReadOnlyList<ActionRequest> sequence)
    {
        var simulator = new JobSimulator(machine);
        var appliedActions = new List<Dictionary<string, object?>>(sequence.Count);
        double? finalEffectTimestamp = null;

        for (var index = 0; index < sequence.Count; index++)
        {
            var request = sequence[index];
            var result = simulator.SubmitAction(request);
            if (!result.Accepted)
            {
                throw new InvalidOperationException(
                    $"{request.SkillKey} at {request.Timestamp:R} was rejected: {result.Reason}");
            }

            finalEffectTimestamp = Math.Max(
                finalEffectTimestamp ?? double.NegativeInfinity,
                result.EffectTimestamp ?? result.AcceptedTimestamp ?? request.Timestamp);
            appliedActions.Add(new Dictionary<string, object?>
            {
                ["index"] = index + 1,
                ["action"] = request.SkillKey,
                ["request_time_seconds"] = Math.Round(request.Timestamp, 4),
                ["accepted_time_seconds"] = Math.Round(result.AcceptedTimestamp!.Value, 4),
                ["effect_time_seconds"] = Math.Round(result.EffectTimestamp!.Value, 4),
                ["queued"] = result.Queued,
                ["action_instance_id"] = result.ActionInstanceId!.Value.ToString("D"),
            });
        }

        if (finalEffectTimestamp is { } timestamp && timestamp > simulator.Time)
        {
            simulator.AdvanceTo(timestamp);
        }

        return new Dictionary<string, object?>
        {
            ["requested_sequence"] = sequence.Select(item => new Dictionary<string, object?>
            {
                ["timestamp"] = item.Timestamp,
                ["action"] = item.SkillKey,
            }).ToList(),
            ["applied_actions"] = appliedActions,
            ["final_output"] = sequence.Count == 0 ? null : simulator.FormatVectorState(),
        };
    }
}

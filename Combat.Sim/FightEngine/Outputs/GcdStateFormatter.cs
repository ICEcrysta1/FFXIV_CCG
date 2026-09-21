// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Common;
using Combat.Sim.Models.Definitions;

namespace Combat.Sim.Outputs;

/// <summary>
/// GCD 制状态输出格式（对照 outputs/gcd.py）。
/// 把内部状态转成"以 GCD 为主"的外部可消费字段；
/// 量谱里 display_unit 为 seconds 的资源额外输出 *_gcd 换算。
/// </summary>
public sealed class GcdStateFormatter
{
    private readonly IReadOnlyDictionary<string, JobResourceDefinition> _jobResourceDefinitions;

    public GcdStateFormatter(IReadOnlyDictionary<string, JobResourceDefinition> jobResourceDefinitions)
    {
        _jobResourceDefinitions = jobResourceDefinitions;
    }

    public Dictionary<string, object?> Format(StateContext stateContext)
    {
        var player = stateContext.Player;
        var currentGcd = player.CurrentGcdSeconds;
        var payload = new Dictionary<string, object?>
        {
            ["mode"] = "gcd",
            ["gcd_index"] = player.GcdIndex,
            ["current_gcd_seconds"] = currentGcd,
            ["fight_remaining_gcd"] = GcdUnits.ToGcdUnits(player.FightRemainingSeconds, currentGcd),
            ["gcd_remaining_gcd"] = GcdUnits.ToGcdUnits(player.GcdRemainingSeconds, currentGcd),
            ["weave_window_gcd"] = GcdUnits.ToGcdUnits(player.WeaveWindowSeconds, currentGcd),
            ["next_downtime_eta_gcd"] = GcdUnits.ToOptionalGcdUnits(player.NextUntargetableInSeconds, currentGcd),
            ["downtime_remaining_gcd"] = GcdUnits.ToGcdUnits(player.DowntimeRemainingSeconds, currentGcd),
            ["boss_targetable"] = player.BossTargetable,
            ["job_tag"] = stateContext.JobTag,
            ["job_name"] = stateContext.JobName,
            ["mp"] = player.Mp,
            ["max_mp"] = player.MaxMp,
            ["mp_ratio"] = player.MpRatio,
        };

        var normalizedJobPayload = stateContext.Resources.ToDictionary(
            entry => entry.Key,
            entry => (object?)entry.Value);
        foreach (var (key, definition) in _jobResourceDefinitions)
        {
            if (definition.DisplayUnit != "seconds" || !normalizedJobPayload.ContainsKey(key))
            {
                continue;
            }

            var timerSeconds = Convert.ToDouble(normalizedJobPayload[key]);
            normalizedJobPayload.Remove(key);
            normalizedJobPayload[$"{key}_gcd"] = GcdUnits.ToGcdUnits(timerSeconds, currentGcd);
        }

        foreach (var (key, value) in normalizedJobPayload)
        {
            payload[key] = value;
        }

        return payload;
    }
}

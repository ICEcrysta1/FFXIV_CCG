// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Common;

namespace Combat.Sim.Outputs;

/// <summary>秒制状态输出格式（对照 outputs/seconds.py）。</summary>
public sealed class SecondsStateFormatter
{
    public Dictionary<string, object?> Format(StateContext stateContext)
    {
        var player = stateContext.Player;
        var payload = new Dictionary<string, object?>
        {
            ["mode"] = "seconds",
            ["time_seconds"] = player.TimeSeconds,
            ["gcd_index"] = player.GcdIndex,
            ["current_gcd_seconds"] = player.CurrentGcdSeconds,
            ["fight_remaining_seconds"] = player.FightRemainingSeconds,
            ["gcd_remaining_seconds"] = player.GcdRemainingSeconds,
            ["cast_remaining_seconds"] = player.CastRemainingSeconds,
            ["weave_window_seconds"] = player.WeaveWindowSeconds,
            ["next_downtime_eta_seconds"] = player.NextUntargetableInSeconds,
            ["downtime_remaining_seconds"] = player.DowntimeRemainingSeconds,
            ["boss_targetable"] = player.BossTargetable,
            ["job_tag"] = stateContext.JobTag,
            ["job_name"] = stateContext.JobName,
            ["mp"] = player.Mp,
            ["max_mp"] = player.MaxMp,
            ["mp_ratio"] = player.MpRatio,
        };
        foreach (var (key, value) in stateContext.Resources)
        {
            payload[key] = value;
        }

        return payload;
    }
}

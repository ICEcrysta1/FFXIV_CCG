// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Outputs.TokenBuilders;

/// <summary>统一装配独立死亡标记；首次可见的样本步号由回放和转换层补充。</summary>
public static class DeathTokenBuilder
{
    internal const string Token = "<dead/>";

    public static Dictionary<string, object?> Build(double timestamp, long afterHistoryLength) => new()
    {
        ["token"] = Token,
        ["timestamp"] = timestamp,
        ["after_history_length"] = afterHistoryLength,
    };
}

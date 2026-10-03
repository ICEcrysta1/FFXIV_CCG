// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Outputs;

/// <summary>输出记录的保留策略；null 保留完整记录，0 不保留，不影响战斗时长或累计威力。</summary>
public sealed class HistoryRetention
{
    public int? Limit { get; }

    public HistoryRetention(int? limit)
    {
        if (limit is < 0) throw new ArgumentOutOfRangeException(nameof(limit));
        Limit = limit;
    }

    internal void Trim<T>(List<T> entries)
    {
        if (Limit is { } limit && entries.Count > limit)
            entries.RemoveRange(0, entries.Count - limit);
    }
}

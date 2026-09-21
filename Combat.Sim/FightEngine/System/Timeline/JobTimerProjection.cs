// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;

namespace Combat.Sim.System.Timeline;

/// <summary>
/// 职业计时器的输出投影契约。内部只保存绝对截止时刻，输出层按职业量谱 schema
/// 显示本周期累计秒数或剩余秒数。
/// </summary>
internal sealed record JobTimerProjection(
    string OutputKey,
    string DeadlineKey,
    bool IsAccumulated,
    double IntervalSeconds)
{
    public double Project(CombatState state, double? deadline)
    {
        if (deadline is not double dueAt)
        {
            return 0.0;
        }

        var untilDeadline = dueAt - state.Time;
        return IsAccumulated
            ? Math.Max(0.0, IntervalSeconds - untilDeadline)
            : Math.Max(0.0, untilDeadline);
    }
}

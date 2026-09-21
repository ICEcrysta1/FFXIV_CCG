// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Config;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Timeline;

namespace Combat.Sim.System;

/// <summary>自然回蓝仅声明周期并结算单次 tick，中心负责续排。</summary>
public sealed class MpRecoveryRuntime
{
    private readonly SystemMpRecoveryConfig _config;
    public MpRecoveryRuntime(SystemMpRecoveryConfig config)
    {
        if (!double.IsFinite(config.TickIntervalSeconds) || config.TickIntervalSeconds <= 0)
            throw new ArgumentOutOfRangeException(nameof(config));
        _config = config;
    }
    public TimelineEvent DescribeEvent(CombatState state) => new(
        Math.Max(state.Time, state.NaturalMpLastTickAt + _config.TickIntervalSeconds),
        TimelineEventPriority.PeriodicSettlement, TimelineEventKind.MpTick);

    public TimelineMutation HandleTick(TimelineEvent item, CombatState state,
        Func<CombatState, int, int> resolveTickAmount) => new(ApplyState: target =>
    {
        // 回调读取 tick 当刻快照（职业 MP 修正已注册在 JobTimelineRegistry 上），不反推推进窗口。
        var amount = Math.Max(0, resolveTickAmount(state, _config.InCombatAmount));
        target.Mp = Math.Min(target.MaxMp, target.Mp + amount);
        target.NaturalMpLastTickAt = item.Timestamp;
    });
}

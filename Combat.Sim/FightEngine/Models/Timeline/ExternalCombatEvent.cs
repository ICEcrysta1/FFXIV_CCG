// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Timeline;

/// <summary>状态机接受的外部战斗事实键；宿主协议负责把字符串映射到这些稳定键。</summary>
public static class ExternalCombatEventKinds
{
    public const string BossTargetableChanged = "boss_targetable_changed";
    public const string MovementChanged = "movement_changed";
    public const string RaidBuffWindowChanged = "raid_buff_window_changed";
    public const string TargetCountChanged = "target_count_changed";
    public const string PlayerDeath = "player_death";
    public const string PlayerRevived = "player_revived";
    /// <summary>治疗 LB3 复活：系统恢复满 MP，不接受独立的 MP 覆盖。</summary>
    public const string PlayerRevivedLb3 = "player_revived_lb3";
}

/// <summary>带绝对时间戳的战斗外部事实。</summary>
public sealed record ExternalCombatEvent(
    double Timestamp,
    string Kind,
    bool? Value = null,
    int? TargetCount = null,
    double? RemainingSeconds = null,
    int? Mp = null);

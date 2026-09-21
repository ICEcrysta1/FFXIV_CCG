// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Timeline;

/// <summary>
/// 动作被接受后使用的稳定时序计划。ActualCastSeconds 控制完整读条锁，
/// EffectDelaySeconds 控制服务器提前结算，两者不得由调用方重新推算。
/// </summary>
internal sealed record ActionTimingPlan(
    double GcdUnitSeconds,
    double ActualCastSeconds,
    double EffectDelaySeconds,
    double EffectiveGcdSeconds,
    double ActualOccupancySeconds,
    string ActualOccupancySource,
    double GcdWindowSeconds,
    double NextGcdWindowSeconds);

// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Timeline;

/// <summary>外部战斗事实提交结果；死亡重置成功时返回独立标记及已落实历史游标，不作为技能或模型向量。</summary>
public sealed record ExternalEventResult(
    bool Accepted,
    string Reason,
    double Timestamp,
    string? Token = null,
    long? HistoryCursor = null);

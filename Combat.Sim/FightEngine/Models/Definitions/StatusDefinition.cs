// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Definitions;

/// <summary>状态（Buff）静态定义（对照 models.StatusDefinition）。</summary>
public sealed record StatusDefinition(
    string Key,
    int GameId,
    double Duration,
    int MaxStacks = 1);

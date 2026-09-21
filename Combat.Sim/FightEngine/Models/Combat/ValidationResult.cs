// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Combat;

/// <summary>动作合法性检查结果（对照 models.ValidationResult）。</summary>
public sealed record ValidationResult(bool Ok, string Reason = "");

// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Definitions;

/// <summary>
/// 职业专属资源定义（对照 models.JobResourceDefinition）。
/// <c>DefaultValue</c> 运行时类型为 int / double / bool。
/// </summary>
public sealed record JobResourceDefinition(
    string Key,
    string ResourceType,
    object DefaultValue,
    string VectorGroup = "resource",
    string DisplayUnit = "count",
    double? MaxValue = null);

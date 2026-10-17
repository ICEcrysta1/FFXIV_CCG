// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Definitions;

namespace Combat.Sim.Models.Definitions;

/// <summary>职业自身的资源与状态注册定义（对照 models.JobStateRegistration）。</summary>
public sealed record JobStateRegistration(
    string JobTag,
    IReadOnlyDictionary<string, JobResourceDefinition> Resources,
    IReadOnlyDictionary<string, StatusDefinition> Statuses);

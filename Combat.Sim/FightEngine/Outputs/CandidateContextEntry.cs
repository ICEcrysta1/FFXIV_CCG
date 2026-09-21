// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Facade;
using Combat.Sim.Models.Combat;

namespace Combat.Sim.Outputs;

/// <summary>
/// 单个候选动作的装配条目（对照 output_context_builder._build_candidate_context_entries
/// 给预览条目注入的 job_resources_consumed / before / after 状态上下文）。
/// </summary>
public sealed record CandidateContextEntry(
    CandidatePreview Preview,
    IReadOnlyDictionary<string, object> JobResourcesConsumed,
    StateContext BeforeStateContext,
    StateContext? AfterStateContext);

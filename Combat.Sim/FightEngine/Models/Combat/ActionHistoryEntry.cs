// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Outputs;

namespace Combat.Sim.Models.Combat;

/// <summary>
/// 单次动作的历史快照（对照 models.ActionHistoryEntry）。
/// 只保存系统层通用的历史信息，供输出层按职业需要展示。
/// <c>StateBefore</c> / <c>StateAfter</c> 是输出层的状态上下文产物（对照 state_before/state_after）。
/// </summary>
public sealed record ActionHistoryEntry(
    string SkillKey,
    int SkillId,
    string SkillName,
    string SkillKind,
    double Potency,
    double Value,
    int MpBefore,
    int MpAfter,
    double CastTimeSeconds,
    double CastTimeGcds,
    double GcdWindowSeconds,
    double GcdWindowGcds,
    bool IsLegal,
    string InvalidReason,
    double NextCooldownSeconds,
    int AvailableCharges,
    int MaxCharges,
    IReadOnlyDictionary<string, object> JobResourcesBefore,
    IReadOnlyDictionary<string, object> JobResourcesAfter,
    IReadOnlyDictionary<string, object> JobResourcesConsumed,
    double TimeSeconds,
    int GcdIndex,
    StateContext StateBefore,
    StateContext StateAfter,
    double? RequestTimestamp = null,
    double? CastCompletedTimestamp = null,
    double? EffectTimestamp = null,
    Guid? ActionInstanceId = null);

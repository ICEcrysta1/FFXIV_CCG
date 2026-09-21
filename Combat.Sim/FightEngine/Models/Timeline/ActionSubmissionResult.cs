// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Timeline;

/// <summary>绝对时间动作提交结果。</summary>
public sealed record ActionSubmissionResult(
    bool Accepted,
    bool Queued,
    string Reason,
    Guid? ActionInstanceId,
    double RequestTimestamp,
    double? AcceptedTimestamp,
    double? EffectTimestamp,
    double? NextScheduledEventTime);

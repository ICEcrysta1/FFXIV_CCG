// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Timeline;

/// <summary>含可变引用的事件载荷必须实现深拷贝，保证 snapshot/fork 相互隔离。</summary>
internal interface ITimelineEventPayload
{
    object DeepClone();
}

// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using System.Collections.ObjectModel;

namespace Combat.Sim.Outputs;

/// <summary>原状态和同刻接受能力作为整体冻结，供历史、窗口和分支安全共享。</summary>
public sealed record ModelStateFrame(StateContext State, IReadOnlyDictionary<string, bool> SkillAvailability)
{
    public static ModelStateFrame Freeze(StateContext context, IReadOnlyDictionary<string, bool> availability) =>
        new(context with
        {
            Resources = new ReadOnlyDictionary<string, object>(context.Resources.ToDictionary(item => item.Key, item => item.Value)),
            Buffs = Array.AsReadOnly(context.Buffs.ToArray()),
            Dots = Array.AsReadOnly(context.Dots.ToArray()),
        }, new ReadOnlyDictionary<string, bool>(availability.ToDictionary(item => item.Key, item => item.Value, StringComparer.Ordinal)));
}

/// <summary>请求时冻结的两段模型状态；真实执行统计仍由动作历史独立保存。</summary>
public sealed record ModelStateSnapshot(ModelStateFrame PreviousActionAfter, ModelStateFrame RequestState)
{
    public static ModelStateSnapshot Capture(ModelStateFrame? previousActionAfter, ModelStateFrame requestState) =>
        new(previousActionAfter ?? requestState, requestState);
}

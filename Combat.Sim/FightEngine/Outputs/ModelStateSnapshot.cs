// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using System.Collections.ObjectModel;

namespace Combat.Sim.Outputs;

/// <summary>请求时冻结的模型状态；真实动作执行前后统计仍由动作历史独立保存。</summary>
public sealed record ModelStateSnapshot(StateContext PreviousActionAfter, StateContext RequestState)
{
    public static ModelStateSnapshot Capture(StateContext? previousActionAfter, StateContext requestState)
    {
        var request = Freeze(requestState);
        return new(previousActionAfter is null ? request : Freeze(previousActionAfter), request);
    }

    // 容器只读化，保证历史、窗口裁剪和 fork 都不能事后改写请求时的输入。
    internal static StateContext Freeze(StateContext context) => context with
    {
        Resources = new ReadOnlyDictionary<string, object>(context.Resources.ToDictionary(item => item.Key, item => item.Value)),
        Buffs = Array.AsReadOnly(context.Buffs.ToArray()),
        Dots = Array.AsReadOnly(context.Dots.ToArray()),
    };
}

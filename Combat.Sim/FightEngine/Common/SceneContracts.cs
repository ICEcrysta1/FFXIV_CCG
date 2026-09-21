// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Config;

namespace Combat.Sim.Common;

/// <summary>
/// 跨状态机、输出和训练层共享的稳定契约常量（对照 contracts.py）。
/// 定义由 <c>config/schema.yaml</c> 提供（C# 与 Python 单一事实来源），
/// 使用前必须先调用 <see cref="SchemaConfigLoader.Load"/>。
/// </summary>
public static class SceneContracts
{
    /// <summary>Boss 可选中窗口的 scene context key。</summary>
    public static string TargetableWindowContextKey => SceneContextKey("targetable");

    /// <summary>强制移动窗口的 scene context key。</summary>
    public static string ForcedMovementContextKey => SceneContextKey("forced_movement");

    /// <summary>团辅窗口的 scene context key。</summary>
    public static string RaidBuffWindowContextKey => SceneContextKey("raid_buff");

    /// <summary>多目标窗口的 scene context key。</summary>
    public static string TargetCountWindowContextKey => SceneContextKey("target_count");

    /// <summary>scene 时间字段使用的绝对时间模式标识。</summary>
    public static string SceneContextAbsoluteMode => SchemaConfigLoader.Instance.SceneContextAbsoluteMode;

    /// <summary>统一滑步窗口秒数（读条结束前可移动的容差）。</summary>
    public static double SlidecastWindowSeconds => SchemaConfigLoader.Instance.SlidecastWindowSeconds;

    /// <summary>Python scene 查询与 Sidecar 延迟伤害边界共用的浮点容差。</summary>
    public static double SceneEpsilon => SchemaConfigLoader.Instance.SceneEpsilon;

    private static string SceneContextKey(string semanticKey) =>
        SchemaConfigLoader.Instance.SceneContextKeys.TryGetValue(semanticKey, out var key)
            ? key
            : throw new InvalidOperationException($"missing scene context key: {semanticKey}");
}

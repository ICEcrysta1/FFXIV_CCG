// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Jobs.Black.Mage;

/// <summary>
/// 黑魔职业专属常量（对照 black_mage.py 顶部常量表）。
/// 元素极性倍率按层数下标访问：0 无极性、1~3 对应层数。
/// </summary>
public static class BlackMageConstants
{
    /// <summary>AF 下火属性技能威力倍率（下标 = AF 层数）。</summary>
    public static readonly double[] AstralFireFirePotencyMultipliers = { 1.0, 1.4, 1.6, 1.8 };

    /// <summary>AF 下冰属性技能威力倍率（下标 = AF 层数）。</summary>
    public static readonly double[] AstralFireIcePotencyMultipliers = { 1.0, 0.9, 0.8, 0.7 };

    /// <summary>UI 下火属性技能威力倍率（下标 = UI 层数）。</summary>
    public static readonly double[] UmbralIceFirePotencyMultipliers = { 1.0, 0.9, 0.8, 0.7 };

    /// <summary>UI 下冰属性技能威力倍率（下标 = UI 层数）。</summary>
    public static readonly double[] UmbralIceIcePotencyMultipliers = { 1.0, 1.0, 1.0, 1.0 };

    /// <summary>AF 下火属性技能 MP 消耗倍率。</summary>
    public const double AstralFireMpCostMultiplier = 2.0;

    /// <summary>灵极魂在 AF 下对火属性技能的 MP 折扣倍率。</summary>
    public const double UmbralHeartFireMpDiscountMultiplier = 0.5;

    /// <summary>悖论在 AF 下的基础 MP 消耗（后续再按元素倍率折算）。</summary>
    public const int ParadoxFireBaseMpCost = 800;

    /// <summary>醒梦生效时每个 MP tick 额外恢复量。</summary>
    public const int LucidDreamingTickMp = 550;

    /// <summary>每层 UI 在冰属性动作命中时恢复的 MP。</summary>
    public const int UmbralIceMpPerStack = 2500;
}

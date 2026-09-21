// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;

namespace Combat.Sim.Jobs.Black.Mage;

public sealed partial class BlackMageJobStateMachine
{
    /// <summary>技能元素的极性解析结果（对照 _resolve_elemental_aspect 的 "fire"/"ice"/None）。</summary>
    private enum ElementalAspect
    {
        None,
        Fire,
        Ice,
    }

    /// <summary>
    /// 按当前元素极性解析技能实际 MP 消耗（对照 _resolve_elemental_mp_cost）。
    /// base_cost 必须是已折算前的整数基础消耗。
    /// </summary>
    private int ResolveElementalMpCost(
        CombatState state,
        SkillDefinition skill,
        int baseCost,
        bool allowUmbralHeartDiscount = false)
    {
        var resolvedBaseCost = Math.Max(0, baseCost);
        if (resolvedBaseCost <= 0)
        {
            return 0;
        }

        switch (ResolveElementalAspect(state, skill))
        {
            case ElementalAspect.Fire:
                if (InUmbralIce(state))
                {
                    return 0;
                }

                var resolvedCost = (double)resolvedBaseCost;
                if (InAstralFire(state))
                {
                    resolvedCost *= BlackMageConstants.AstralFireMpCostMultiplier;
                    if (allowUmbralHeartDiscount && IntResource(state, "umbral_hearts") > 0)
                    {
                        resolvedCost *= BlackMageConstants.UmbralHeartFireMpDiscountMultiplier;
                    }
                }

                return ToInt32Truncate(resolvedCost);

            case ElementalAspect.Ice:
                // 冰属性技能在 AF/UI 下免费（UI 下命中后由冰系命中回蓝兜底）。
                if (InAstralFire(state) || InUmbralIce(state))
                {
                    return 0;
                }

                return resolvedBaseCost;

            default:
                return resolvedBaseCost;
        }
    }

    /// <summary>按当前元素极性解析技能实际威力倍率（对照 _resolve_elemental_potency_multiplier）。</summary>
    private double ResolveElementalPotencyMultiplier(CombatState state, SkillDefinition skill)
    {
        var astralFire = IntResource(state, "astral_fire");
        var umbralIce = IntResource(state, "umbral_ice");

        switch (ResolveElementalAspect(state, skill))
        {
            case ElementalAspect.Fire:
                if (astralFire > 0)
                {
                    return BlackMageConstants.AstralFireFirePotencyMultipliers[astralFire];
                }

                if (umbralIce > 0)
                {
                    return BlackMageConstants.UmbralIceFirePotencyMultipliers[umbralIce];
                }

                return 1.0;

            case ElementalAspect.Ice:
                if (astralFire > 0)
                {
                    return BlackMageConstants.AstralFireIcePotencyMultipliers[astralFire];
                }

                if (umbralIce > 0)
                {
                    return BlackMageConstants.UmbralIceIcePotencyMultipliers[umbralIce];
                }

                return 1.0;

            default:
                return 1.0;
        }
    }

    /// <summary>
    /// 解析技能在当前状态下的元素极性（对照 _resolve_elemental_aspect）。
    /// 悖论按当前极性归属；其他技能只看自身 tags 的火/冰标记。
    /// </summary>
    private ElementalAspect ResolveElementalAspect(CombatState state, SkillDefinition skill)
    {
        if (skill.Key == "paradox")
        {
            if (InAstralFire(state))
            {
                return ElementalAspect.Fire;
            }

            if (InUmbralIce(state))
            {
                return ElementalAspect.Ice;
            }

            return ElementalAspect.None;
        }

        var isFire = skill.Tags.Contains("fire");
        var isIce = skill.Tags.Contains("ice");
        if (isFire && !isIce)
        {
            return ElementalAspect.Fire;
        }

        if (isIce && !isFire)
        {
            return ElementalAspect.Ice;
        }

        return ElementalAspect.None;
    }
}

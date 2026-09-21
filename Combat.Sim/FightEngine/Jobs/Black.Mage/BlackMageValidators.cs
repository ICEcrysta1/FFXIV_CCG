// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;

namespace Combat.Sim.Jobs.Black.Mage;

public sealed partial class BlackMageJobStateMachine
{
    /// <summary>爆炎：火苗触发时免费，否则按元素折算耗蓝（对照 _validate_fire_iii）。</summary>
    private ValidationResult ValidateFireIii(CombatState state, SkillDefinition skill)
    {
        if (state.Mp < ActualMpCost(state, skill))
        {
            return new ValidationResult(false, "not_enough_mp");
        }

        return new ValidationResult(true);
    }

    /// <summary>炽炎：需要 AF，耗蓝按元素折算并享受灵极魂折扣（对照 _validate_fire_iv）。</summary>
    private ValidationResult ValidateFireIv(CombatState state, SkillDefinition skill)
    {
        if (!InAstralFire(state))
        {
            return new ValidationResult(false, "requires_af");
        }

        if (state.Mp < ActualMpCost(state, skill))
        {
            return new ValidationResult(false, "not_enough_mp");
        }

        return new ValidationResult(true);
    }

    /// <summary>高烈炎：只有耗蓝校验（对照 _validate_high_fire_ii）。</summary>
    private ValidationResult ValidateHighFireIi(CombatState state, SkillDefinition skill)
    {
        if (state.Mp < ActualMpCost(state, skill))
        {
            return new ValidationResult(false, "not_enough_mp");
        }

        return new ValidationResult(true);
    }

    /// <summary>绝望：需要 AF，且 MP 不得低于技能定义的消耗下限（对照 _validate_despair）。</summary>
    private ValidationResult ValidateDespair(CombatState state, SkillDefinition skill)
    {
        if (!InAstralFire(state))
        {
            return new ValidationResult(false, "requires_af");
        }

        if (state.Mp < skill.MpCostFloor)
        {
            return new ValidationResult(false, "not_enough_mp");
        }

        return new ValidationResult(true);
    }

    /// <summary>核爆：需要 AF，且 MP 不得低于技能定义的消耗下限（对照 _validate_flare）。</summary>
    private ValidationResult ValidateFlare(CombatState state, SkillDefinition skill)
    {
        if (!InAstralFire(state))
        {
            return new ValidationResult(false, "requires_af");
        }

        if (state.Mp < skill.MpCostFloor)
        {
            return new ValidationResult(false, "not_enough_mp");
        }

        return new ValidationResult(true);
    }

    /// <summary>耀星：需要攒满天语（对照 _validate_flare_star）。</summary>
    private ValidationResult ValidateFlareStar(CombatState state, SkillDefinition _skill)
    {
        if (IntResource(state, "astral_soul") < ResourceMax("astral_soul"))
        {
            return new ValidationResult(false, "insufficient_astral_soul");
        }

        return new ValidationResult(true);
    }

    /// <summary>冰封：只有耗蓝校验（对照 _validate_blizzard_iii）。</summary>
    private ValidationResult ValidateBlizzardIii(CombatState state, SkillDefinition skill)
    {
        if (state.Mp < ActualMpCost(state, skill))
        {
            return new ValidationResult(false, "not_enough_mp");
        }

        return new ValidationResult(true);
    }

    /// <summary>冰澈：需要 UI，命中后回满灵极魂与 MP（对照 _validate_blizzard_iv）。</summary>
    private ValidationResult ValidateBlizzardIv(CombatState state, SkillDefinition skill)
    {
        if (!InUmbralIce(state))
        {
            return new ValidationResult(false, "requires_ui");
        }

        if (state.Mp < ActualMpCost(state, skill))
        {
            return new ValidationResult(false, "not_enough_mp");
        }

        return new ValidationResult(true);
    }

    /// <summary>玄冰：需要 UI，命中后回满灵极魂与 MP（对照 _validate_freeze）。</summary>
    private ValidationResult ValidateFreeze(CombatState state, SkillDefinition skill)
    {
        if (!InUmbralIce(state))
        {
            return new ValidationResult(false, "requires_ui");
        }

        if (state.Mp < ActualMpCost(state, skill))
        {
            return new ValidationResult(false, "not_enough_mp");
        }

        return new ValidationResult(true);
    }

    /// <summary>高冰冻：只有耗蓝校验（对照 _validate_high_blizzard_ii）。</summary>
    private ValidationResult ValidateHighBlizzardIi(CombatState state, SkillDefinition skill)
    {
        if (state.Mp < ActualMpCost(state, skill))
        {
            return new ValidationResult(false, "not_enough_mp");
        }

        return new ValidationResult(true);
    }

    /// <summary>悖论：需要悖论就绪且 MP 充足（对照 _validate_paradox）。</summary>
    private ValidationResult ValidateParadox(CombatState state, SkillDefinition skill)
    {
        if (!BoolResource(state, "paradox_ready"))
        {
            return new ValidationResult(false, "requires_paradox");
        }

        if (state.Mp < ActualMpCost(state, skill))
        {
            return new ValidationResult(false, "not_enough_mp");
        }

        return new ValidationResult(true);
    }

    /// <summary>通晓消耗技能：需要至少 1 层通晓（对照 _validate_polyglot_spender）。</summary>
    private ValidationResult ValidatePolyglotSpender(CombatState state, SkillDefinition _skill)
    {
        if (IntResource(state, "polyglot") < 1)
        {
            return new ValidationResult(false, "requires_polyglot");
        }

        return new ValidationResult(true);
    }

    /// <summary>雷云 DoT 技能：需要雷云就绪（对照 _validate_thundercloud_dot）。</summary>
    private ValidationResult ValidateThundercloudDot(CombatState state, SkillDefinition _skill)
    {
        if (!BoolResource(state, "thundercloud_ready"))
        {
            return new ValidationResult(false, "requires_thundercloud");
        }

        return new ValidationResult(true);
    }

    /// <summary>星灵移位：需要当前处于任意元素态（对照 _validate_transpose）。</summary>
    private ValidationResult ValidateTranspose(CombatState state, SkillDefinition _skill)
    {
        if (!EnochianActive(state))
        {
            return new ValidationResult(false, "requires_elemental_state");
        }

        return new ValidationResult(true);
    }

    /// <summary>灵极魂：需要 UI 且 Boss 不可选中（对照 _validate_umbral_soul）。</summary>
    private ValidationResult ValidateUmbralSoul(CombatState state, SkillDefinition _skill)
    {
        if (!InUmbralIce(state))
        {
            return new ValidationResult(false, "requires_ui");
        }

        if (state.BossTargetable)
        {
            return new ValidationResult(false, "downtime_only");
        }

        return new ValidationResult(true);
    }

    /// <summary>魔泉：需要 AF（对照 _validate_manafont）。</summary>
    private ValidationResult ValidateManafont(CombatState state, SkillDefinition _skill)
    {
        if (!InAstralFire(state))
        {
            return new ValidationResult(false, "requires_af");
        }

        return new ValidationResult(true);
    }

    /// <summary>详述：需要当前处于任意元素态（对照 _validate_amplifier）。</summary>
    private ValidationResult ValidateAmplifier(CombatState state, SkillDefinition _skill)
    {
        if (!EnochianActive(state))
        {
            return new ValidationResult(false, "requires_elemental_state");
        }

        return new ValidationResult(true);
    }
}

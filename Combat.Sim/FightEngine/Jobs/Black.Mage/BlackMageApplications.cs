// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;

namespace Combat.Sim.Jobs.Black.Mage;

public sealed partial class BlackMageJobStateMachine
{
    /// <summary>炽炎：耗蓝并消耗 1 层灵极魂，+1 天语（对照 _apply_fire_iv）。</summary>
    private void ApplyFireIv(CombatState previousState, CombatState nextState, SkillDefinition skill)
    {
        SpendMp(nextState, ActualMpCost(previousState, skill));
        if (IntResource(nextState, "umbral_hearts") > 0)
        {
            SetResource(nextState, "umbral_hearts", IntResource(nextState, "umbral_hearts") - 1);
        }

        SetResource(
            nextState,
            "astral_soul",
            Math.Min(ResourceMax("astral_soul"), IntResource(nextState, "astral_soul") + 1));
    }

    /// <summary>绝望：清空 MP 并补满 AF（对照 _apply_despair）。</summary>
    private void ApplyDespair(CombatState nextState)
    {
        SpendMp(nextState, nextState.Mp);
        SetAstralFire(nextState, ResourceMax("astral_fire"));
    }

    /// <summary>核爆：消耗全部 MP（灵极魂时减半）、补满 AF、清空灵极魂并 +3 天语（对照 _apply_flare）。</summary>
    private void ApplyFlare(CombatState nextState)
    {
        SpendMp(nextState, FlareCost(nextState));
        SetAstralFire(nextState, ResourceMax("astral_fire"));
        if (IntResource(nextState, "umbral_hearts") > 0)
        {
            SetResource(nextState, "umbral_hearts", 0);
        }

        SetResource(
            nextState,
            "astral_soul",
            Math.Min(ResourceMax("astral_soul"), IntResource(nextState, "astral_soul") + 3));
    }

    /// <summary>耀星：清空全部天语（对照 _apply_flare_star）。</summary>
    private void ApplyFlareStar(CombatState nextState) =>
        SetResource(nextState, "astral_soul", 0);

    /// <summary>冰澈：耗蓝、补满灵极魂并按当前 UI 命中回蓝（对照 _apply_blizzard_iv）。</summary>
    private void ApplyBlizzardIv(CombatState previousState, CombatState nextState, SkillDefinition skill)
    {
        SpendMp(nextState, ActualMpCost(previousState, skill));
        SetResource(nextState, "umbral_hearts", ResourceMax("umbral_hearts"));
        RecoverMpForUmbralIce(nextState);
    }

    /// <summary>玄冰：耗蓝、补满灵极魂并按当前 UI 命中回蓝（对照 _apply_freeze）。</summary>
    private void ApplyFreeze(CombatState previousState, CombatState nextState, SkillDefinition skill)
    {
        SpendMp(nextState, ActualMpCost(previousState, skill));
        SetResource(nextState, "umbral_hearts", ResourceMax("umbral_hearts"));
        RecoverMpForUmbralIce(nextState);
    }

    /// <summary>悖论：AF 下耗蓝并授予火苗，随后消费悖论就绪（对照 _apply_paradox）。</summary>
    private void ApplyParadox(CombatState previousState, CombatState nextState, SkillDefinition skill)
    {
        if (InAstralFire(previousState))
        {
            SpendMp(nextState, ActualMpCost(previousState, skill));
            SetResource(nextState, "firestarter_ready", true);
        }

        SetResource(nextState, "paradox_ready", false);
    }

    /// <summary>通晓消耗技能：扣 1 层通晓（对照 _apply_polyglot_spender）。</summary>
    private void ApplyPolyglotSpender(CombatState nextState) =>
        SetResource(nextState, "polyglot", Math.Max(0, IntResource(nextState, "polyglot") - 1));

    /// <summary>雷云 DoT 技能：消费雷云并按动作前状态快照 DoT 威力挂载（对照 _apply_thundercloud_dot）。</summary>
    private void ApplyThundercloudDot(CombatState previousState, CombatState nextState, SkillDefinition skill)
    {
        SetResource(nextState, "thundercloud_ready", false);
        var snapshotDotPotency = System.ResolveAppliedPotency(
            previousState,
            ResolvePotency(previousState, skill, skill.DotPotency),
            skill);
        System.GrantRegisteredTargetDot(
            nextState,
            skill.DotKey ?? skill.Key,
            skill.DotDuration,
            snapshotDotPotency);
    }

    /// <summary>星灵移位：翻转极性，满层另一侧时授予悖论，并授予雷云（对照 _apply_transpose）。</summary>
    private void ApplyTranspose(CombatState previousState, CombatState nextState)
    {
        if (InAstralFire(previousState))
        {
            SetUmbralIce(nextState, 1);
            if (IntResource(previousState, "astral_fire") == ResourceMax("astral_fire"))
            {
                SetResource(nextState, "paradox_ready", true);
            }
        }
        else if (InUmbralIce(previousState))
        {
            SetAstralFire(nextState, 1);
            if (IntResource(previousState, "umbral_ice") == ResourceMax("umbral_ice") &&
                IntResource(previousState, "umbral_hearts") >= ResourceMax("umbral_hearts"))
            {
                SetResource(nextState, "paradox_ready", true);
            }
        }
        else
        {
            // 当前校验已经禁止无元素态使用星灵移位；这里保留兜底分支，
            // 防止未来放宽校验或回放脏数据时出现未定义状态推进。
            SetAstralFire(nextState, 1);
        }

        SetResource(nextState, "thundercloud_ready", true);
    }

    /// <summary>灵极魂：+1 UI（无 UI 时进 1 层）、+1 灵极魂并按当前 UI 命中回蓝（对照 _apply_umbral_soul）。</summary>
    private void ApplyUmbralSoul(CombatState nextState)
    {
        var currentUi = IntResource(nextState, "umbral_ice");
        var uiNew = currentUi > 0
            ? Math.Min(ResourceMax("umbral_ice"), currentUi + 1)
            : 1;
        SetUmbralIce(nextState, uiNew);
        SetResource(
            nextState,
            "umbral_hearts",
            Math.Min(ResourceMax("umbral_hearts"), IntResource(nextState, "umbral_hearts") + 1));
        RecoverMpForUmbralIce(nextState);
    }

    /// <summary>魔泉：回满 MP 与 AF/灵极魂，授予悖论与雷云（对照 _apply_manafont）。</summary>
    private void ApplyManafont(CombatState nextState)
    {
        nextState.Mp = nextState.MaxMp;
        SetAstralFire(nextState, ResourceMax("astral_fire"));
        SetResource(nextState, "umbral_hearts", ResourceMax("umbral_hearts"));
        SetResource(nextState, "paradox_ready", true);
        SetResource(nextState, "thundercloud_ready", true);
    }

    /// <summary>详述：+1 通晓（对照 _apply_amplifier）。</summary>
    private void ApplyAmplifier(CombatState nextState) => GainPolyglot(nextState);

    /// <summary>火属性 stance 过渡：耗蓝、补满 AF、授予雷云，满 UI+灵极魂时授予悖论（对照 _apply_fire_stance_transition）。</summary>
    private void ApplyFireStanceTransition(CombatState previousState, CombatState nextState, SkillDefinition skill)
    {
        SpendMp(nextState, ActualMpCost(previousState, skill));
        SetAstralFire(nextState, ResourceMax("astral_fire"));
        SetResource(nextState, "thundercloud_ready", true);
        if (IntResource(previousState, "umbral_ice") == ResourceMax("umbral_ice") &&
            IntResource(previousState, "umbral_hearts") >= ResourceMax("umbral_hearts"))
        {
            SetResource(nextState, "paradox_ready", true);
        }
    }

    /// <summary>
    /// 冰属性 stance 过渡：耗蓝、补满 UI、授予雷云，满 AF 时授予悖论，
    /// 并按动作前 UI 层数回蓝（对照 _apply_ice_stance_transition）。
    /// </summary>
    private void ApplyIceStanceTransition(CombatState previousState, CombatState nextState, SkillDefinition skill)
    {
        SpendMp(nextState, ActualMpCost(previousState, skill));
        SetUmbralIce(nextState, ResourceMax("umbral_ice"));
        SetResource(nextState, "thundercloud_ready", true);
        if (IntResource(previousState, "astral_fire") == ResourceMax("astral_fire"))
        {
            SetResource(nextState, "paradox_ready", true);
        }

        var previousUiStacks = IntResource(previousState, "umbral_ice");
        var mpRecovery = previousUiStacks >= ResourceMax("umbral_ice")
            ? nextState.MaxMp
            : BlackMageConstants.UmbralIceMpPerStack * Math.Max(1, previousUiStacks);
        RecoverMp(nextState, mpRecovery);
    }

    /// <summary>写入 AF 层数并清空 UI（对照 _set_astral_fire）。</summary>
    private void SetAstralFire(CombatState state, int stacks)
    {
        var wasElemental = EnochianActive(state);
        SetResource(
            state,
            "astral_fire",
            Math.Max(0, Math.Min(ResourceMax("astral_fire"), stacks)));
        SetResource(state, "umbral_ice", 0);
        StartPolyglotCycleIfEntering(state, wasElemental);
    }

    /// <summary>写入 UI 层数并清空 AF 与天语（对照 _set_umbral_ice）。</summary>
    private void SetUmbralIce(CombatState state, int stacks)
    {
        var wasElemental = EnochianActive(state);
        SetResource(
            state,
            "umbral_ice",
            Math.Max(0, Math.Min(ResourceMax("umbral_ice"), stacks)));
        SetResource(state, "astral_fire", 0);
        SetResource(state, "astral_soul", 0);
        StartPolyglotCycleIfEntering(state, wasElemental);
    }

    /// <summary>
    /// 元素态从无到有时启动通晓周期：下一次结算定在"进入时刻 + 周期"。
    /// 已经在元素态内的重复写入（例如刷新 AF 层数）不会推迟已经排定的结算。
    /// </summary>
    private void StartPolyglotCycleIfEntering(CombatState state, bool wasElemental)
    {
        if (wasElemental)
        {
            return;
        }

        SetTimelineDeadline(
            state,
            PolyglotTimelineKey,
            state.Time + Job.TimingValue("polyglot_interval"));
    }

    /// <summary>核爆的实际消耗：有灵极魂时减半（对照 _flare_cost）。</summary>
    private int FlareCost(CombatState state) =>
        IntResource(state, "umbral_hearts") > 0 ? state.Mp / 2 : state.Mp;

    private void SpendMp(CombatState state, int amount) => state.Mp = Math.Max(0, state.Mp - amount);

    private void RecoverMp(CombatState state, int amount) => state.Mp = Math.Min(state.MaxMp, state.Mp + amount);

    /// <summary>按当前 UI 层数命中回蓝：满层一次回满，否则每层 2500（对照 _recover_mp_for_umbral_ice）。</summary>
    private void RecoverMpForUmbralIce(CombatState state)
    {
        var uiStacks = IntResource(state, "umbral_ice");
        if (uiStacks <= 0)
        {
            return;
        }

        RecoverMp(
            state,
            uiStacks >= ResourceMax("umbral_ice")
                ? 10000
                : BlackMageConstants.UmbralIceMpPerStack * uiStacks);
    }

    /// <summary>累计通晓，不超过上限（对照 _gain_polyglot）。</summary>
    private void GainPolyglot(CombatState state, int stacks = 1) =>
        SetResource(
            state,
            "polyglot",
            Math.Min(ResourceMax("polyglot"), IntResource(state, "polyglot") + Math.Max(0, stacks)));
}

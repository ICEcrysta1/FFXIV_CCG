// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Config;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.System;
using Combat.Sim.System.Timeline;

namespace Combat.Sim.Jobs;

/// <summary>
/// 职业子状态机契约。
/// 系统层与门面只通过该接口驱动职业，不感知具体职业规则。
/// Python 侧用 getattr 判断的可选方法在 C# 侧以默认接口实现表达默认行为。
/// </summary>
public interface IJobStateMachine
{
    /// <summary>
    /// 职业标签（静态成员，注册与路由用，对照职业类的静态 tag）。
    /// 注册表通过静态成员取标签，注册路径不构造实例；
    /// 实现类必须提供静态 Tag，缺失时注册报错。
    /// </summary>
    static virtual string Tag => throw new NotSupportedException(
        "job state machine must define a static Tag");

    /// <summary>门面装配期注入项目配置与系统层引用。</summary>
    void Bind(ProjectConfig project, SystemStateMachine system) { }

    /// <summary>职业默认战斗剩余时间（对照 job.default_fight_remaining）。</summary>
    double DefaultFightRemaining { get; }

    /// <summary>职业向系统层注册的资源与状态定义（对照 build_state_registration）。</summary>
    JobStateRegistration BuildStateRegistration();

    /// <summary>当前状态下的有效基础 GCD 秒数（对照 current_gcd_duration）。</summary>
    double CurrentGcdDuration(CombatState state);

    /// <summary>判断技能在当前状态下是否等效瞬发（对照 is_instant）。</summary>
    bool IsInstant(CombatState state, SkillDefinition skill);

    /// <summary>职业层是否识别某个行为名（对照 supports_behavior）。</summary>
    bool SupportsBehavior(string behavior);

    /// <summary>职业层合法性校验；系统层全部校验通过后调用（对照 validate_action）。</summary>
    ValidationResult ValidateAction(CombatState state, SkillDefinition skill);

    /// <summary>应用职业行为效果（对照 apply_action）。</summary>
    void ApplyAction(CombatState previousState, CombatState nextState, SkillDefinition skill);

    /// <summary>消耗职业时序资源（对照 consume_timing_resources，仅 GCD 调用）。</summary>
    void ConsumeTimingResources(CombatState state, SkillDefinition skill);

    /// <summary>向公共时间线注册职业周期资源、倒计时和 MP 修正规则。</summary>
    void RegisterTimeline(JobTimelineRegistry timeline) { }

    /// <summary>职业可选的读条倍率；默认 1.0（对照可选 cast_time_multiplier）。</summary>
    double CastTimeMultiplier(CombatState state, SkillDefinition skill) => 1.0;

    /// <summary>职业修正后的实际耗蓝；默认 full 语义按当前 MP，否则取技能定义（对照可选 actual_mp_cost）。</summary>
    int ActualMpCost(CombatState state, SkillDefinition skill) =>
        skill.MpCostIsFull ? Math.Max(0, state.Mp) : Math.Max(0, skill.MpCost);

    /// <summary>职业修正后的实际威力；默认返回基础威力（对照可选 resolve_potency）。</summary>
    double ResolvePotency(CombatState state, SkillDefinition skill, double basePotency) =>
        Math.Max(0.0, basePotency);

    /// <summary>按职业状态解析训练辅助 value；默认使用技能 YAML value，不模拟随机暴击/直击。</summary>
    double ResolveActionValue(CombatState state, SkillDefinition skill, double baseValue) =>
        Math.Max(0.0, baseValue);
}

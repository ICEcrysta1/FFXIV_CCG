// Copyright (C) 2026 ICE_crystal
// Copyright (C) 2026 SpikeHS (original Machinist implementation)
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Config;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.Models.Timeline;
using Combat.Sim.System;
using Combat.Sim.System.Timeline;

namespace Combat.Sim.Jobs.Machinist;

/// <summary>
/// 机工职业子状态机。
/// 只处理机工资源、连击、过热/整备状态与野火结算；公共 GCD、冷却、
/// 共享 Buff 和目标威力记录仍由系统层负责。
/// </summary>
public sealed partial class MachinistJobStateMachine : JobResourceAccessors, IJobStateMachine
{
    /// <summary>职业标签（对照 machinist.py 的 tag = "machinist"）。</summary>
    public static string Tag => "machinist";

    /// <summary>连击倒计时的注册 key，同时是它在 JobTimelineDeadlines 里的绝对到期键。</summary>
    private const string ComboTimelineKey = "machinist.combo";

    /// <summary>野火倒计时的注册 key，同时是它在 JobTimelineDeadlines 里的绝对到期键。</summary>
    private const string WildfireTimelineKey = "machinist.wildfire";

    private sealed record BehaviorHandler(
        Func<CombatState, SkillDefinition, ValidationResult>? Validate = null,
        Action<CombatState, CombatState, SkillDefinition>? Apply = null);

    private readonly Dictionary<string, BehaviorHandler> _behaviors;

    /// <summary>行为名、校验与效果在职业内只登记一次；空委托表示无需额外规则。</summary>
    public MachinistJobStateMachine()
    {
        _behaviors = new(StringComparer.Ordinal)
        {
            ["mch_combo_one"] = new(Apply: (_, next, _) => ApplyComboOne(next)),
            ["mch_combo_two"] = new(Apply: (previous, next, _) => ApplyComboTwo(previous, next)),
            ["mch_combo_three"] = new(Apply: (previous, next, _) => ApplyComboThree(previous, next)),
            ["mch_tool"] = new(),
            ["mch_air_anchor"] = new(Apply: (_, next, _) => ApplyAirAnchor(next)),
            ["mch_chain_saw"] = new(Apply: (_, next, _) => ApplyChainSaw(next)),
            ["mch_excavator"] = new(ValidateExcavator, (_, next, _) => ApplyExcavator(next)),
            ["mch_blazing_shot"] = new(ValidateBlazingShot, (_, next, _) => ApplyBlazingShot(next)),
            ["mch_full_metal_field"] = new(ValidateFullMetalField, (_, next, _) => ApplyFullMetalField(next)),
            ["mch_hypercharge"] = new(ValidateHypercharge, (_, next, _) => ApplyHypercharge(next)),
            ["mch_wildfire"] = new(ValidateWildfire, (_, next, _) => ApplyWildfire(next)),
            ["mch_detonator"] = new(ValidateDetonator, (previous, next, _) => ApplyDetonator(previous, next)),
            ["mch_barrel_stabilizer"] = new(Apply: (_, next, _) => ApplyBarrelStabilizer(next)),
            ["mch_reassemble"] = new(ValidateReassemble, (_, next, _) => ApplyReassemble(next)),
            ["mch_direct_ability"] = new(),
        };
    }

    /// <summary>机工配置契约由机工模块持有，公共加载器只调用注册的职业校验。</summary>
    public static void ValidateConfig(JobConfig job)
    {
        if (!job.Timing.TryGetValue("wildfire_duration", out var wildfireDuration) ||
            !job.ResourceLimits.TryGetValue("wildfire_remaining", out var wildfireRemainingMax))
        {
            return;
        }

        if (Math.Abs(wildfireDuration - wildfireRemainingMax) > 1e-9)
        {
            throw new InvalidOperationException(
                "job config wildfire_duration and resources.wildfire_remaining.max_value " +
                $"must match, got {wildfireDuration} and {wildfireRemainingMax}");
        }
    }

    private ProjectConfig? _project;
    private JobConfig? _job;
    private ProjectConfig Project => _project ?? throw new InvalidOperationException(
        "machinist job state machine is not bound; Bind(project, system) must be called before use");

    private JobConfig Job => _job ?? throw new InvalidOperationException(
        "machinist job state machine is not bound; Bind(project, system) must be called before use");

    public void Bind(ProjectConfig project, SystemStateMachine system)
    {
        _project = project;
        _job = project.Job;
        BindSystem(system);
    }

    public double DefaultFightRemaining => Job.DefaultFightRemaining;

    /// <summary>向系统层注册机工资源与职业状态。</summary>
    public JobStateRegistration BuildStateRegistration() =>
        new(
            Tag,
            new Dictionary<string, JobResourceDefinition>(StringComparer.Ordinal)
            {
                ["heat"] = new("heat", "int", 0),
                ["battery"] = new("battery", "int", 0),
                ["combo_stage"] = new("combo_stage", "int", MachinistConstants.ComboNone),
                ["combo_remaining"] = new(
                    "combo_remaining", "float", 0.0, DisplayUnit: "seconds"),
                ["wasted_heat"] = new("wasted_heat", "int", 0),
                ["wasted_battery"] = new("wasted_battery", "int", 0),
                ["wasted_overheated_stacks"] = new("wasted_overheated_stacks", "int", 0),
                ["wildfire_active"] = new("wildfire_active", "bool", false),
                ["wildfire_remaining"] = new(
                    "wildfire_remaining", "float", 0.0, DisplayUnit: "seconds"),
                ["wildfire_hits"] = new("wildfire_hits", "int", 0),
                ["wildfire_missed_weaponskills"] = new("wildfire_missed_weaponskills", "int", 0),
                ["wildfire_resolved_potency"] = new(
                    "wildfire_resolved_potency", "float", 0.0, DisplayUnit: "potency"),
                ["wildfire_lost_potency"] = new(
                    "wildfire_lost_potency", "float", 0.0, DisplayUnit: "potency"),
            },
            new Dictionary<string, StatusDefinition>(Job.Statuses, StringComparer.Ordinal));

    public bool SupportsBehavior(string behavior) => behavior is not null && _behaviors.ContainsKey(behavior);

    /// <summary>校验机工职业专属动作合法性。</summary>
    public ValidationResult ValidateAction(CombatState state, SkillDefinition skill) =>
        skill.Behavior is { } behavior && _behaviors.TryGetValue(behavior, out var handler) &&
        handler.Validate is { } validate
            ? validate(state, skill)
            : new ValidationResult(true);

    /// <summary>应用机工职业专属动作效果，并处理野火命中与整备消费。</summary>
    public void ApplyAction(CombatState previousState, CombatState nextState, SkillDefinition skill)
    {
        if (skill.Behavior is not { } behavior || !_behaviors.TryGetValue(behavior, out var handler))
        {
            throw new ArgumentOutOfRangeException(
                nameof(skill.Behavior), skill.Behavior,
                $"machinist does not support behavior: {skill.Behavior}");
        }

        handler.Apply?.Invoke(previousState, nextState, skill);

        if (skill.Kind == ActionKind.Gcd && skill.Tags.Contains("weaponskill"))
        {
            RecordWildfireWeaponskill(nextState);
            ConsumeReassemble(nextState, skill);
        }
    }

    public double CurrentGcdDuration(CombatState state) => state.BaseGcd ?? Project.System.BaseGcd;

    public bool IsInstant(CombatState state, SkillDefinition skill) =>
        skill.Kind == ActionKind.Ogcd || skill.CastTime <= 0.0;

    public double CastTimeMultiplier(CombatState state, SkillDefinition skill) => 1.0;

    public int ActualMpCost(CombatState state, SkillDefinition skill) => 0;

    public void ConsumeTimingResources(CombatState state, SkillDefinition skill) { }

    /// <summary>解析连击、起爆与过热状态下的实际基础威力。</summary>
    public double ResolvePotency(CombatState state, SkillDefinition skill, double basePotency)
    {
        var potency = Math.Max(0.0, basePotency);
        if (skill.Key == "heated_slug_shot" && ComboMatches(state, MachinistConstants.ComboSplit))
        {
            potency = 320.0;
        }
        else if (skill.Key == "heated_clean_shot" && ComboMatches(state, MachinistConstants.ComboSlug))
        {
            potency = 420.0;
        }
        else if (skill.Key == "detonator" && BoolResource(state, "wildfire_active"))
        {
            potency = WildfireBasePotency(state);
        }

        if (state.HasStatus("overheated") && skill.Tags.Contains("single_target_weaponskill"))
        {
            potency += MachinistConstants.OverheatedSingleTargetPotencyBonus;
        }

        return potency;
    }

    /// <summary>整备使下一个符合条件的武器技能使用双保证对应的训练 value。</summary>
    public double ResolveActionValue(CombatState state, SkillDefinition skill, double baseValue)
    {
        var value = Math.Max(0.0, baseValue);
        var reassembleApplies =
            state.HasStatus("reassemble") &&
            skill.Kind == ActionKind.Gcd &&
            skill.Potency > 0 &&
            skill.Tags.Contains("weaponskill") &&
            !skill.Tags.Contains("reassemble_immune");
        return reassembleApplies ? Math.Max(2.0, value) : value;
    }

    public void RegisterTimeline(JobTimelineRegistry timeline)
    {
        timeline.RegisterCountdown(
            ComboTimelineKey,
            "combo_remaining",
            state => TimelineDeadline(state, ComboTimelineKey),
            (_, state, _) => new TimelineMutation(ApplyState: ClearCombo));
        // 状态到期钩子归 StatusTimelineRuntime 所有（它才是状态到期的拥有者），
        // 时间资源仍注册在 timeline 上。
        System.StatusTimeline.RegisterExpiryHook("overheated", RecordOverheatedExpiry);
        timeline.RegisterCountdown(
            WildfireTimelineKey,
            "wildfire_remaining",
            state => BoolResource(state, "wildfire_active")
                ? TimelineDeadline(state, WildfireTimelineKey)
                : null,
            (item, state, targetableAt) =>
                new TimelineMutation(ApplyState: target =>
                    HandleWildfireExpiry(item, target, targetableAt)));
    }

    private bool ComboMatches(CombatState state, int stage) =>
        IntResource(state, "combo_stage") == stage &&
        TimelineDeadline(state, ComboTimelineKey) is double deadline &&
        deadline > state.Time + SystemConstants.ZeroEpsilon;

    private double WildfireBasePotency(CombatState state) =>
        IntResource(state, "wildfire_hits") * Job.TimingValue("wildfire_potency_per_hit");
}

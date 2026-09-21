// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Config;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.Models.Timeline;
using Combat.Sim.System;
using Combat.Sim.System.Timeline;

namespace Combat.Sim.Jobs.Black.Mage;

/// <summary>
/// 黑魔职业子状态机。
/// 只处理黑魔自己的职业规则：AF/UI、通晓、悖论、雷云、火苗等资源与专属时间钩子；
/// 公共时间轴、公共冷却和空输出由系统状态机负责。
/// 注册走 JobMachineRegistry，标签读取静态 Tag，注册路径不构造实例；
/// 实例由门面装配期经 Bind 注入项目配置与系统层引用。
/// </summary>
public sealed partial class BlackMageJobStateMachine : JobResourceAccessors, IJobStateMachine
{
    /// <summary>职业标签（对照 black_mage.py 的 tag = "black_mage"）。</summary>
    public static string Tag => "black_mage";

    /// <summary>通晓周期的注册 key，同时是它在 JobTimelineDeadlines 里的绝对截止时刻键。</summary>
    private const string PolyglotTimelineKey = "black_mage.polyglot";

    private ProjectConfig? _project;
    private JobConfig? _job;
    /// <summary>黑魔负责的行为集合（对照 _behaviors 字典的键）。</summary>
    private static readonly HashSet<string> Behaviors = new(StringComparer.Ordinal)
    {
        "fire_iii",
        "high_fire_ii",
        "fire_iv",
        "despair",
        "flare",
        "flare_star",
        "blizzard_iii",
        "blizzard_iv",
        "freeze",
        "high_blizzard_ii",
        "paradox",
        "polyglot_spender",
        "thundercloud_dot",
        "transpose",
        "umbral_soul",
        "manafont",
        "amplifier",
    };

    private JobConfig Job => _job ?? throw new InvalidOperationException(
        "black mage job state machine is not bound; Bind(project, system) must be called before use");

    private ProjectConfig Project => _project ?? throw new InvalidOperationException(
        "black mage job state machine is not bound; Bind(project, system) must be called before use");

    /// <summary>门面装配期注入项目配置与系统层引用（对照构造参数 project_config / system_machine）。</summary>
    public void Bind(ProjectConfig project, SystemStateMachine system)
    {
        _project = project;
        _job = project.Job;
        BindSystem(system);
    }

    /// <summary>职业默认战斗剩余时间（对照 job.default_fight_remaining）。</summary>
    public double DefaultFightRemaining => Job.DefaultFightRemaining;

    /// <summary>向系统层注册黑魔专属资源与职业 Buff（对照 build_state_registration）。</summary>
    public JobStateRegistration BuildStateRegistration() =>
        new(
            Tag,
            new Dictionary<string, JobResourceDefinition>(StringComparer.Ordinal)
            {
                ["astral_fire"] = new("astral_fire", "int", 0),
                ["umbral_ice"] = new("umbral_ice", "int", 0),
                ["umbral_hearts"] = new("umbral_hearts", "int", 0),
                ["astral_soul"] = new("astral_soul", "int", 0),
                ["polyglot"] = new("polyglot", "int", 0),
                ["polyglot_timer"] = new("polyglot_timer", "float", 0.0, DisplayUnit: "seconds"),
                ["paradox_ready"] = new("paradox_ready", "bool", false),
                ["thundercloud_ready"] = new("thundercloud_ready", "bool", false, VectorGroup: "buff"),
                ["firestarter_ready"] = new("firestarter_ready", "bool", false, VectorGroup: "buff"),
            },
            new Dictionary<string, StatusDefinition>(Job.Statuses, StringComparer.Ordinal));

    /// <summary>返回黑魔职业机是否负责这个行为名（对照 supports_behavior）。</summary>
    public bool SupportsBehavior(string behavior) => Behaviors.Contains(behavior);

    /// <summary>
    /// 校验黑魔职业专属动作合法性（对照 validate_action）。
    /// 系统层行为也会路由到这里，不是职业负责的行为直接放行，由上游继续走系统层校验；
    /// 行为已注册但没有对应校验规则时返回 unknown_behavior。
    /// </summary>
    public ValidationResult ValidateAction(CombatState state, SkillDefinition skill)
    {
        if (!SupportsBehavior(skill.Behavior))
        {
            return new ValidationResult(true);
        }

        return skill.Behavior switch
        {
            "fire_iii" => ValidateFireIii(state, skill),
            "high_fire_ii" => ValidateHighFireIi(state, skill),
            "fire_iv" => ValidateFireIv(state, skill),
            "despair" => ValidateDespair(state, skill),
            "flare" => ValidateFlare(state, skill),
            "flare_star" => ValidateFlareStar(state, skill),
            "blizzard_iii" => ValidateBlizzardIii(state, skill),
            "blizzard_iv" => ValidateBlizzardIv(state, skill),
            "freeze" => ValidateFreeze(state, skill),
            "high_blizzard_ii" => ValidateHighBlizzardIi(state, skill),
            "paradox" => ValidateParadox(state, skill),
            "polyglot_spender" => ValidatePolyglotSpender(state, skill),
            "thundercloud_dot" => ValidateThundercloudDot(state, skill),
            "transpose" => ValidateTranspose(state, skill),
            "umbral_soul" => ValidateUmbralSoul(state, skill),
            "manafont" => ValidateManafont(state, skill),
            "amplifier" => ValidateAmplifier(state, skill),
            _ => new ValidationResult(false, $"unknown_behavior:{skill.Behavior}"),
        };
    }

    /// <summary>应用黑魔职业专属动作效果（对照 apply_action）。</summary>
    public void ApplyAction(CombatState previousState, CombatState nextState, SkillDefinition skill)
    {
        switch (skill.Behavior)
        {
            case "fire_iii":
            case "high_fire_ii":
                ApplyFireStanceTransition(previousState, nextState, skill);
                break;
            case "fire_iv":
                ApplyFireIv(previousState, nextState, skill);
                break;
            case "despair":
                ApplyDespair(nextState);
                break;
            case "flare":
                ApplyFlare(nextState);
                break;
            case "flare_star":
                ApplyFlareStar(nextState);
                break;
            case "blizzard_iii":
            case "high_blizzard_ii":
                ApplyIceStanceTransition(previousState, nextState, skill);
                break;
            case "blizzard_iv":
                ApplyBlizzardIv(previousState, nextState, skill);
                break;
            case "freeze":
                ApplyFreeze(previousState, nextState, skill);
                break;
            case "paradox":
                ApplyParadox(previousState, nextState, skill);
                break;
            case "polyglot_spender":
                ApplyPolyglotSpender(nextState);
                break;
            case "thundercloud_dot":
                ApplyThundercloudDot(previousState, nextState, skill);
                break;
            case "transpose":
                ApplyTranspose(previousState, nextState);
                break;
            case "umbral_soul":
                ApplyUmbralSoul(nextState);
                break;
            case "manafont":
                ApplyManafont(nextState);
                break;
            case "amplifier":
                ApplyAmplifier(nextState);
                break;
            default:
                throw new ArgumentOutOfRangeException(
                    nameof(skill.Behavior), skill.Behavior,
                    $"black mage does not support behavior: {skill.Behavior}");
        }
    }

    public void RegisterTimeline(JobTimelineRegistry timeline)
    {
        timeline.RegisterPeriodicResource(
            PolyglotTimelineKey,
            Job.TimingValue("polyglot_interval"),
            "polyglot_timer",
            // 只在元素态内计时；元素态结束后不再描述下一次结算。
            state => EnochianActive(state) ? TimelineDeadline(state, PolyglotTimelineKey) : null,
            (_, state) => new TimelineMutation(ApplyState: target =>
            {
                // 结算一层通晓，并把下一次结算推进一个完整周期。
                GainPolyglot(target);
                SetTimelineDeadline(
                    target,
                    PolyglotTimelineKey,
                    target.Time + Job.TimingValue("polyglot_interval"));
            }));

        timeline.RegisterMpTickModifier((state, naturalAmount) =>
        {
            if (InAstralFire(state))
            {
                return 0;
            }

            var totalAmount = Math.Max(0, naturalAmount);
            if (state.Statuses.TryGetValue("lucid_dreaming", out var lucidStatus) &&
                lucidStatus.Remaining > SystemConstants.ZeroEpsilon)
            {
                totalAmount += BlackMageConstants.LucidDreamingTickMp;
            }

            return totalAmount;
        });
    }

    /// <summary>返回当前全局视角下的有效 GCD 秒数（对照 current_gcd_duration）。</summary>
    public double CurrentGcdDuration(CombatState state)
    {
        var gcd = Project.System.BaseGcd;
        if (state.HasStatus("ley_lines"))
        {
            gcd *= Job.TimingValue("ley_lines_haste_multiplier");
        }

        return gcd;
    }

    /// <summary>判断黑魔技能在当前状态下是否等效瞬发（对照 is_instant）。</summary>
    public bool IsInstant(CombatState state, SkillDefinition skill)
    {
        if (skill.Kind == ActionKind.Ogcd || skill.CastTime <= 0)
        {
            return true;
        }

        if (skill.Key == "fire_iii" && BoolResource(state, "firestarter_ready"))
        {
            return true;
        }

        if (state.HasStatus("swiftcast") || state.HasStatus("triplecast"))
        {
            return true;
        }

        return false;
    }

    /// <summary>返回火冰极性对硬读条施加的职业专属倍率（对照 cast_time_multiplier）。</summary>
    public double CastTimeMultiplier(CombatState state, SkillDefinition skill)
    {
        if (skill.Kind != ActionKind.Gcd || skill.CastTime <= 0)
        {
            return 1.0;
        }

        if (IntResource(state, "astral_fire") >= ResourceMax("astral_fire") && skill.Tags.Contains("ice"))
        {
            return 0.5;
        }

        if (IntResource(state, "umbral_ice") >= ResourceMax("umbral_ice") && skill.Tags.Contains("fire"))
        {
            return 0.5;
        }

        return 1.0;
    }

    /// <summary>解析黑魔技能的实际 MP 消耗（对照 actual_mp_cost）。</summary>
    public int ActualMpCost(CombatState state, SkillDefinition skill)
    {
        if (skill.Key == "fire_iii" && BoolResource(state, "firestarter_ready"))
        {
            return 0;
        }

        if (skill.Key == "despair")
        {
            return state.Mp;
        }

        if (skill.Key == "flare")
        {
            return FlareCost(state);
        }

        if (skill.Key == "paradox")
        {
            var baseCost = InAstralFire(state) ? BlackMageConstants.ParadoxFireBaseMpCost : 0;
            return ResolveElementalMpCost(state, skill, baseCost);
        }

        if (skill.MpCostIsFull)
        {
            return 0;
        }

        return ResolveElementalMpCost(
            state,
            skill,
            skill.MpCost,
            allowUmbralHeartDiscount: skill.Key == "fire_iv");
    }

    /// <summary>解析黑魔技能的实际威力（对照 resolve_potency，悖论不做极性倍率）。</summary>
    public double ResolvePotency(CombatState state, SkillDefinition skill, double basePotency)
    {
        var resolvedBasePotency = Math.Max(0.0, basePotency);
        if (resolvedBasePotency <= 0.0)
        {
            return 0.0;
        }

        if (skill.Key == "paradox")
        {
            return resolvedBasePotency;
        }

        return resolvedBasePotency * ResolveElementalPotencyMultiplier(state, skill);
    }

    /// <summary>在公共时序推进后，消耗会影响读条的黑魔资源（对照 consume_timing_resources）。</summary>
    public void ConsumeTimingResources(CombatState state, SkillDefinition skill)
    {
        if (skill.Kind != ActionKind.Gcd || skill.CastTime <= 0)
        {
            return;
        }

        // 火苗提供的免费瞬发 F3 不应消耗任何瞬发层。
        if (skill.Key == "fire_iii" && BoolResource(state, "firestarter_ready"))
        {
            SetResource(state, "firestarter_ready", false);
            return;
        }

        // 两种通用瞬发同时存在时，游戏优先消耗即刻咏唱，再消耗三连咏唱。
        if (state.HasStatus("swiftcast"))
        {
            System.ClearRegisteredStatus(state, "swiftcast");
        }
        else if (state.HasStatus("triplecast"))
        {
            System.ConsumeRegisteredStatusStack(state, "triplecast");
        }
    }

    private bool InAstralFire(CombatState state) => IntResource(state, "astral_fire") > 0;

    private bool InUmbralIce(CombatState state) => IntResource(state, "umbral_ice") > 0;

    private bool EnochianActive(CombatState state) => InAstralFire(state) || InUmbralIce(state);
}

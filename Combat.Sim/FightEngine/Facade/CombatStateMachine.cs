// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Common;
using Combat.Sim.Config;
using Combat.Sim.Jobs;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.Models.Timeline;
using Combat.Sim.Outputs;
using Combat.Sim.Skills;
using Combat.Sim.State;
using Combat.Sim.System;
using Combat.Sim.System.Timeline;

namespace Combat.Sim.Facade;

/// <summary>
/// 统一战斗状态机门面（对照 state_machine.CombatStateMachine）。
/// 只负责职业路由与系统/职业拼装，不承载具体职业规则；
/// 动作时序、合法性校验、效果与威力记录委托给系统层与职业层。
/// </summary>
public sealed class CombatStateMachine
{
    private readonly double _skillTableBaseGcd;
    private readonly string? _projectRoot;
    private readonly StateContextBuilder _stateContextBuilder;

    internal int? MaxHistory { get; }

    internal ProjectConfig Project { get; }
    public string JobTag { get; }
    internal SystemStateMachine SystemMachine { get; }
    internal IJobStateMachine JobMachine { get; }
    internal ActionHistoryRuntime History { get; }
    internal SkillBook SkillBook { get; }

    // 规则可被多个队列并发读取；带增量缓存的输出路由器必须由各模拟游标单独持有。
    internal StateOutputRouter CreateOutputRouter() =>
        new(Project, SystemMachine, JobMachine, precisionConfigRoot: _projectRoot);

    /// <summary>
    /// 直接以配置对象构造门面。
    /// 前置条件：<paramref name="projectRoot"/> 供 tensor 输出懒加载精度配置；
    /// 懒加载 config/precision.yaml 使用；省略时共享 schema 按仓库根自动加载
    /// （要求可执行文件位于仓库目录树内，游戏插件等独立嵌入场景必须显式传 projectRoot），
    /// canonical / 快照输出不受影响，首次 tensor 输出会抛 InvalidOperationException
    /// （Python 端恒有 PROJECT_ROOT 可用）。
    /// </summary>
    public CombatStateMachine(
        ProjectConfig projectConfig,
        string jobTag,
        int? maxHistory = null,
        string? projectRoot = null)
    {
        // 共享 schema（config/schema.yaml）是输出层契约的单一事实来源：
        // 缺 projectRoot 时按仓库根兜底加载，保证任意构造路径下输出层立即可用
        SchemaConfigLoader.Load(projectRoot ?? RepoRootLocator.Find());
        Project = projectConfig;
        JobTag = jobTag;
        _projectRoot = projectRoot;
        MaxHistory = new HistoryRetention(maxHistory).Limit;
        SystemMachine = new SystemStateMachine(
            projectConfig.EngineTiming,
            projectConfig.System.Potency,
            projectConfig.System.MpRecovery);
        SystemMachine.RegisterSystemStatuses(projectConfig.System.Statuses);
        JobMachine = BuildJobMachine(projectConfig, jobTag, SystemMachine);
        JobMachine.State.Register(
            JobMachine.BuildStateRegistration(),
            resourceLimits: projectConfig.Job.ResourceLimits);
        JobMachine.RegisterTimeline(JobMachine.State.Timeline);
        History = new ActionHistoryRuntime(JobMachine.State);
        SkillBook = SkillBook.FromProjectConfig(projectConfig);
        SystemMachine.RegisterJobTargetDots(SkillBook.EnabledSkills());
        ValidateSkillContracts();
        _stateContextBuilder = new StateContextBuilder(projectConfig, SystemMachine, JobMachine);
        _skillTableBaseGcd = projectConfig.System.SkillTableBaseGcd;
    }

    /// <summary>
    /// 从默认配置创建状态机（对照 from_default_config）。
    /// C# 侧不读取 .env，职业标签必须由调用方显式传入。
    /// </summary>
    public static CombatStateMachine FromDefaultConfig(
        string projectRoot,
        string jobTag,
        double? actualBaseGcd = null,
        int? maxHistory = null)
    {
        SchemaConfigLoader.Load(projectRoot);
        var projectConfig = ProjectConfigLoader.LoadProjectConfig(projectRoot, jobTag);
        if (actualBaseGcd is { } baseGcd)
        {
            projectConfig = projectConfig with { System = projectConfig.System with { BaseGcd = baseGcd } };
        }
        return new CombatStateMachine(projectConfig, jobTag, maxHistory, projectRoot);
    }

    /// <summary>创建当前职业的初始战斗状态（对照 initial_state）。</summary>
    internal CombatState InitialState(
        double? fightRemaining = null, int? maxMp = null, double startTime = 0)
    {
        var state = SystemMachine.InitialState(
            fightRemaining ?? JobMachine.DefaultFightRemaining,
            maxMp ?? Project.EngineResources.MaxMp, startTime);
        JobMachine.State.Initialize(state);
        return state;
    }

    /// <summary>门面组合两层规则；两层都不拥有独立时钟或推进循环。</summary>
    internal CombatTimelineRuntime CreateTimeline(CombatState state, Func<double, bool>? targetableAt = null) =>
        CreateTimeline(SystemMachine, JobMachine.State, state, key => ResolveSkill(key).Charges, targetableAt);

    internal static CombatTimelineRuntime CreateTimeline(SystemStateMachine system, JobStateRuntime job,
        CombatState state, Func<string, int> getMaxCharges, Func<double, bool>? targetableAt = null)
    {
        var timeline = new CombatTimelineRuntime(state);
        system.RegisterTimelineHandlers(timeline, getMaxCharges, job.Timeline.ResolveMpTick);
        timeline.RegisterHandler(TimelineEventKind.StatusExpired, (item, current) => item.Scope switch
        {
            TimelineEventScope.System => system.StatusTimeline.HandleExpired(item, current),
            TimelineEventScope.Job => job.StatusTimeline.HandleExpired(item, current),
            _ => throw new ArgumentOutOfRangeException(nameof(item), "unsupported status expiration scope"),
        });
        timeline.RegisterHandler(TimelineEventKind.JobPeriodicTick, job.Timeline.HandlePeriodic);
        timeline.RegisterHandler(TimelineEventKind.JobTimerExpired,
            (item, current) => job.Timeline.HandleCountdown(item, current, targetableAt));
        timeline.ConfigureResources(current => system.PlayerState.DescribeEvents(current)
            .Concat(system.StatusTimeline.DescribeEvents(current))
            .Concat(job.StatusTimeline.DescribeEvents(current))
            .Concat(system.DotTimeline.DescribeEvents(current))
            .Concat(system.CooldownRuntime.DescribeEvents(current))
            .Concat(job.Timeline.DescribeEvents(current))
            .Append(system.MpRecovery.DescribeEvent(current)).ToArray());
        return timeline;
    }

    private BuffStateRegistry StatusOwner(string key)
    {
        if (SystemMachine.SystemStatusDefinitions.ContainsKey(key)) return SystemMachine.BuffState;
        if (JobMachine.State.Buffs.Definitions.ContainsKey(key)) return JobMachine.State.Buffs;
        throw new KeyNotFoundException($"unregistered status: {key}");
    }

    private ValidationResult ValidateSharedBehavior(CombatState state, SkillDefinition skill)
    {
        if (skill.AppliesStatuses.Count == 0 || skill.Behavior != "grant_status")
            return new ValidationResult(true);
        foreach (var key in skill.AppliesStatuses)
        {
            var owner = StatusOwner(key);
            var validation = owner.ValidateRegisteredStatus(state, key);
            if (!validation.Ok) return validation;
        }
        return new ValidationResult(true);
    }

    private bool ApplySharedBehavior(CombatState state, SkillDefinition skill)
    {
        if (skill.AppliesStatuses.Count == 0) return false;
        foreach (var key in skill.AppliesStatuses) StatusOwner(key).GrantRegisteredStatus(state, key);
        return true;
    }

    /// <summary>按 key 或游戏内 id 解析技能定义（对照 resolve_skill）。</summary>
    internal SkillDefinition ResolveSkill(string key) => SkillBook.Get(key);

    internal SkillDefinition ResolveSkill(int gameId) => SkillBook.Get(gameId);

    /// <summary>校验一次动作在当前状态下是否合法（对照 validate_action）。</summary>
    internal ValidationResult ValidateAction(CombatState state, string skillRef) =>
        ValidateAction(state, ResolveSkill(skillRef));

    internal ValidationResult ValidateAction(CombatState state, SkillDefinition skill)
        => ValidateActionCore(state, skill, includeTemporalLocks: true);

    private ValidationResult ValidateActionCore(
        CombatState state,
        SkillDefinition skill,
        bool includeTemporalLocks,
        double? actualCastSecondsOverride = null)
    {
        if (!skill.Enabled)
        {
            return new ValidationResult(false, "disabled");
        }
        if (!SupportsSkillBehavior(skill))
        {
            return new ValidationResult(false, "unsupported_behavior");
        }

        var isInstant = actualCastSecondsOverride is { } observedCast
            ? observedCast <= SystemConstants.ZeroEpsilon
            : JobMachine.IsInstant(state, skill);
        var hasAvailableCharge = SystemMachine.HasAvailableCharge(state, skill);
        var commonValidation = SystemMachine.ValidateCommonAction(
            state,
            skill,
            isInstant: isInstant,
            hasAvailableCharge: hasAvailableCharge,
            includeTemporalLocks: includeTemporalLocks);
        if (!commonValidation.Ok)
        {
            return commonValidation;
        }

        var sharedValidation = ValidateSharedBehavior(state, skill);
        if (!sharedValidation.Ok)
        {
            return sharedValidation;
        }

        return JobMachine.ValidateAction(state, skill);
    }

    /// <summary>
    /// 统一判断动作应立即接受、进入容量一队列或拒绝。进入队列前仍会执行全部非时间锁校验，
    /// 因而 MP、通晓、目标和职业资源不足的动作不会等待资源生成。
    /// </summary>
    internal ActionSubmissionPlan EvaluateActionSubmission(
        CombatState state,
        SkillDefinition skill,
        bool queueOccupied,
        double? actualCastSecondsOverride = null)
    {
        var validation = ValidateActionCore(
            state,
            skill,
            includeTemporalLocks: true,
            actualCastSecondsOverride);
        if (validation.Ok)
        {
            return new(true, false, "", state.Time);
        }

        if (validation.Reason is not ("cast_locked" or "gcd_locked" or "cooldown_locked"))
        {
            return new(false, false, validation.Reason, state.Time);
        }

        var nonTemporalValidation = ValidateActionCore(
            state,
            skill,
            includeTemporalLocks: false,
            actualCastSecondsOverride);
        if (!nonTemporalValidation.Ok)
        {
            return new(false, false, nonTemporalValidation.Reason, state.Time);
        }

        var readyAt = state.Time;
        if (state.CastRemaining > SystemConstants.ZeroEpsilon)
        {
            readyAt = Math.Max(readyAt, state.CastEndsAt);
        }
        if (skill.Kind == ActionKind.Gcd && state.GcdRemaining > SystemConstants.ZeroEpsilon)
        {
            readyAt = Math.Max(readyAt, state.GcdReadyAt);
        }
        if (!SystemMachine.HasAvailableCharge(state, skill))
        {
            var cooldownReadyAt = SystemMachine.NextChargeReadyAt(state, skill);
            if (cooldownReadyAt is null)
            {
                return new(false, false, "cooldown_locked", state.Time);
            }
            readyAt = Math.Max(readyAt, cooldownReadyAt.Value);
        }

        // 已有排队动作只占用“需要等待后再接受”的队列槽；当前请求若在此刻
        // 已经合法（典型是 GCD 队列期间仍可立即施放的 oGCD），不应被误判为队列占用。
        if (queueOccupied)
        {
            return new(false, false, "action_queue_occupied", state.Time);
        }

        var queueWindow = Math.Max(0.0, Project.EngineTiming.ActionQueueWindowSeconds);
        if (readyAt - state.Time > queueWindow + SystemConstants.ZeroEpsilon)
        {
            return new(false, false, validation.Reason, readyAt);
        }

        return new(true, true, "", readyAt);
    }

    /// <summary>应用动作占用、冷却和职业时序资源消耗（对照 _apply_action_timing）。</summary>
    internal void ApplyActionTiming(
        CombatState state,
        SkillDefinition skill,
        double effectiveGcd,
        double actualCastSeconds,
        double? gcdStartTimestamp = null)
    {
        SystemMachine.ApplyActionTiming(state, skill, effectiveGcd, actualCastSeconds, gcdStartTimestamp);
        SystemMachine.ConsumeCooldown(state, skill);
        if (skill.Kind == ActionKind.Gcd)
        {
            JobMachine.ConsumeTimingResources(state, skill);
        }
    }

    /// <summary>应用系统技能、共享行为、职业行为和目标威力（对照 _apply_action_effect）。</summary>
    internal void ApplyActionEffect(CombatState sourceState, CombatState state, SkillDefinition skill)
    {
        var handledShared = ApplySharedBehavior(state, skill);
        if (!handledShared && JobMachine.SupportsBehavior(skill.Behavior))
        {
            JobMachine.ApplyAction(sourceState, state, skill);
        }
        SystemMachine.RecordTargetPotency(
            state,
            sourceState,
            skill,
            ResolveJobAdjustedPotency(sourceState, skill, basePotency: skill.Potency));
    }

    /// <summary>按当前状态的共享增伤倍率解析职业修正后的实际威力（对照 _resolve_job_adjusted_potency）。</summary>
    internal double ResolveJobAdjustedPotency(CombatState state, SkillDefinition skill, double basePotency) =>
        JobMachine.ResolvePotency(state, skill, basePotency);

    internal double ResolveCastTimeMultiplier(CombatState state, SkillDefinition skill) =>
        JobMachine.CastTimeMultiplier(state, skill);

    private static IJobStateMachine BuildJobMachine(ProjectConfig projectConfig, string jobTag, SystemStateMachine systemMachine)
    {
        var machine = JobMachineRegistry.Get(jobTag);
        if (machine is null)
        {
            var supported = string.Join(", ", JobMachineRegistry.RegisteredTags());
            throw new InvalidOperationException($"unsupported job tag: {jobTag}; supported={supported}");
        }
        if (projectConfig.Job.Key != jobTag)
        {
            throw new InvalidOperationException(
                $"loaded config job does not match requested tag: config={projectConfig.Job.Key}, requested={jobTag}");
        }
        machine.Bind(projectConfig, systemMachine);
        return machine;
    }

    private void ValidateSkillContracts()
    {
        foreach (var key in SkillBook.Keys())
        {
            var definition = SkillBook.Get(key);
            if (SupportsSkillBehavior(definition))
            {
                continue;
            }
            throw new InvalidOperationException(
                $"unsupported behavior for skill {definition.Key}: {definition.Behavior}");
        }
    }

    private bool SupportsSkillBehavior(SkillDefinition skill) =>
        SystemMachine.SupportsBehavior(skill.Behavior) || JobMachine.SupportsBehavior(skill.Behavior);

    /// <summary>
    /// 构造一次绝对时间动作生命周期所需的纯时序计划。
    /// 该方法只读取状态，不推进时间、不写入动作效果，供动作时序校验和时间线执行复用。
    /// </summary>
    internal ActionTimingPlan BuildActionTimingPlan(
        CombatState state,
        SkillDefinition skill,
        double? actualCastSecondsOverride = null)
    {
        var gcdUnitSeconds = JobMachine.CurrentGcdDuration(state);
        var isInstant = JobMachine.IsInstant(state, skill);
        var actualCastSeconds = actualCastSecondsOverride is { } observedCast
            ? Math.Max(0.0, observedCast)
            : BuildActualCastSeconds(
                skill,
                isInstant: isInstant,
                gcdUnitSeconds: gcdUnitSeconds,
                castTimeMultiplier: ResolveCastTimeMultiplier(state, skill));
        var effectDelaySeconds = actualCastSeconds > CombatTimelineRuntime.TimeEpsilon
            ? Math.Max(0.0, actualCastSeconds - SceneContracts.SlidecastWindowSeconds)
            : 0.0;
        var effectiveGcd = BuildEffectiveGcdSeconds(skill, gcdUnitSeconds);
        var (actualOccupancySeconds, actualOccupancySource, gcdWindowSeconds) = BuildActionMetrics(
            skill,
            actualCastSeconds,
            effectiveGcd);
        return new ActionTimingPlan(
            gcdUnitSeconds,
            actualCastSeconds,
            effectDelaySeconds,
            effectiveGcd,
            actualOccupancySeconds,
            actualOccupancySource,
            gcdWindowSeconds,
            BuildNextGcdWindowSeconds(
                skill,
                state,
                actualOccupancySeconds,
                gcdWindowSeconds));
    }

    /// <summary>
    /// 在 ActionEffect 事件中记录真实游戏动作历史。
    /// 历史主时间以 effect time 为准，同时保留请求、完整读条结束和动作实例 id。
    /// </summary>
    internal void RecordTimelineActionHistory(
        CombatState requestState,
        CombatState effectState,
        SkillDefinition skill,
        ActionTimingPlan timing,
        Guid actionInstanceId,
        double requestTimestamp,
        double castCompletedTimestamp,
        double effectTimestamp,
        ModelStateSnapshot modelState)
    {
        var stateBefore = _stateContextBuilder.Build(requestState);
        var stateAfter = _stateContextBuilder.Build(effectState);
        // 威力和训练价值沿用请求快照，冷却反映动作消耗后的效果快照。
        var (nextCooldownSeconds, availableCharges, maxCharges) =
            SystemMachine.BuildCooldownSnapshot(effectState, skill);
        History.RecordActionHistory(
            effectState,
            requestState,
            skill,
            potency: ResolveJobAdjustedPotency(requestState, skill, skill.Potency),
            value: JobMachine.ResolveActionValue(requestState, skill, skill.Value),
            castTimeSeconds: timing.ActualCastSeconds,
            castTimeGcds: GcdUnits.ToGcdUnits(timing.ActualCastSeconds, timing.GcdUnitSeconds),
            gcdWindowSeconds: timing.GcdWindowSeconds,
            gcdWindowGcds: GcdUnits.ToGcdUnits(timing.GcdWindowSeconds, timing.GcdUnitSeconds),
            isLegal: true,
            invalidReason: "",
            nextCooldownSeconds: nextCooldownSeconds,
            availableCharges: availableCharges,
            maxCharges: maxCharges,
            stateBefore,
            stateAfter,
            recordedTimeSeconds: effectState.Time,
            requestTimestamp: requestTimestamp,
            castCompletedTimestamp: castCompletedTimestamp,
            effectTimestamp: effectTimestamp,
            actionInstanceId: actionInstanceId,
            modelState: modelState);
    }

    /// <summary>动作占用元信息（对照 _build_action_metrics 返回值）。</summary>
    internal (double Seconds, string Source, double GcdWindow) BuildActionMetrics(
        SkillDefinition skill,
        double actualCastSeconds,
        double effectiveGcd)
    {
        if (skill.Kind == ActionKind.Gcd)
        {
            return actualCastSeconds <= 0
                ? (0.0, "instant", effectiveGcd)
                : (actualCastSeconds, "cast_time", effectiveGcd);
        }
        return (0.0, "instant", 0.0);
    }

    internal double BuildActualCastSeconds(
        SkillDefinition skill,
        bool isInstant,
        double gcdUnitSeconds,
        double castTimeMultiplier = 1.0)
    {
        if (isInstant || skill.CastTime <= 0)
        {
            return 0.0;
        }
        return ScaleNominalGcdSeconds(skill.CastTime, gcdUnitSeconds) * Math.Max(0.0, castTimeMultiplier);
    }

    internal double BuildEffectiveGcdSeconds(SkillDefinition skill, double gcdUnitSeconds) =>
        skill.Kind != ActionKind.Gcd
            ? skill.RecastTime
            : ScaleNominalGcdSeconds(skill.RecastTime, gcdUnitSeconds);

    internal double BuildNextGcdWindowSeconds(
        SkillDefinition skill,
        CombatState state,
        double actualOccupancySeconds,
        double gcdWindowSeconds)
    {
        if (skill.Kind == ActionKind.Gcd)
        {
            return Math.Max(actualOccupancySeconds, gcdWindowSeconds);
        }
        return Math.Max(0.0, state.GcdRemaining);
    }

    internal sealed record ActionSubmissionPlan(
        bool Accepted,
        bool Queued,
        string Reason,
        double AcceptedTimestamp);

    private double ScaleNominalGcdSeconds(double nominalSeconds, double gcdUnitSeconds)
    {
        if (_skillTableBaseGcd <= 0)
        {
            throw new InvalidOperationException("skill_table_base_gcd must be > 0");
        }
        return nominalSeconds * gcdUnitSeconds / _skillTableBaseGcd;
    }
}

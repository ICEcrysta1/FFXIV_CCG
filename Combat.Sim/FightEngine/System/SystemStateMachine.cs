// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Timeline;
using Combat.Sim.Config;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.Outputs;
using Combat.Sim.System.Registries;
using Combat.Sim.System.Timeline;

namespace Combat.Sim.System;

/// <summary>
/// 系统状态机总调度门面：组装全部系统层子模块，统一协调公共时间轴、
/// Buff 注册中心、职业量谱注册中心与历史记录（对照 machine.SystemStateMachine）。
/// </summary>
public sealed class SystemStateMachine
{
    private readonly EngineTimingConfig _engineTiming;
    private readonly SystemPotencyConfig _potencyConfig;

    public BuffStateRegistry BuffState { get; }
    public ResourceStateRegistry ResourceState { get; }
    public StatusTimelineRuntime StatusTimeline { get; }
    public DotTimelineRuntime DotTimeline { get; }
    public MpRecoveryRuntime MpRecovery { get; }
    public CooldownRuntime CooldownRuntime { get; }
    public JobTimelineRegistry JobTimeline { get; }
    public PlayerStateRuntime PlayerState { get; }
    public SystemHistoryRuntime History { get; }

    /// <summary>引擎时序配置，定义动作请求队列窗口。</summary>
    public EngineTimingConfig EngineTiming => _engineTiming;

    public SystemStateMachine(
        EngineTimingConfig engineTiming,
        SystemPotencyConfig potencyConfig,
        SystemMpRecoveryConfig mpRecoveryConfig,
        int? maxHistory = null)
    {
        _engineTiming = engineTiming;
        _potencyConfig = potencyConfig;
        BuffState = new BuffStateRegistry();
        ResourceState = new ResourceStateRegistry();
        StatusTimeline = new StatusTimelineRuntime();
        DotTimeline = new DotTimelineRuntime();
        MpRecovery = new MpRecoveryRuntime(mpRecoveryConfig);
        CooldownRuntime = new CooldownRuntime();
        JobTimeline = new JobTimelineRegistry();
        PlayerState = new PlayerStateRuntime();
        History = new SystemHistoryRuntime(
            ResourceState,
            buildTransition: BuildJobResourceTransition,
            maxHistory: maxHistory);
    }

    public IReadOnlyDictionary<string, StatusDefinition> SystemStatusDefinitions => BuffState.SystemStatusDefinitions;
    public IReadOnlyDictionary<string, StatusDefinition> JobStatusDefinitions => BuffState.JobStatusDefinitions;
    public IReadOnlyDictionary<string, JobResourceDefinition> JobResourceDefinitions => ResourceState.JobResourceDefinitions;
    public string? RegisteredJobTag => ResourceState.RegisteredJobTag;
    public IReadOnlyList<string> RegisteredTargetDotKeys => DotTimeline.RegisteredDotKeys;
    public IReadOnlyList<string> JobResourceSnapshotKeys => ResourceState.SnapshotResourceKeys;

    public IReadOnlyList<string> JobResourceVectorKeys(string vectorGroup) =>
        ResourceState.VectorResourceKeys(vectorGroup);

    public double? JobResourceMax(string key) => ResourceState.ResourceMaxValue(key);

    public void RegisterSystemStatuses(IReadOnlyDictionary<string, StatusDefinition> statuses) =>
        BuffState.RegisterSystemStatuses(statuses);

    public void RegisterJobState(JobStateRegistration registration, IReadOnlyDictionary<string, double>? resourceLimits = null)
    {
        BuffState.RegisterJobStatuses(registration.JobTag, registration.Statuses);
        ResourceState.RegisterJobResources(registration.JobTag, registration.Resources, resourceLimits);
    }

    public void RegisterJobTargetDots(IEnumerable<SkillDefinition> skills) =>
        DotTimeline.RegisterDots(skills);

    /// <summary>创建系统层默认初始状态。</summary>
    public CombatState InitialState(double fightRemaining, int maxMp, double startTime = 0)
    {
        var state = new CombatState
        {
            FightRemaining = fightRemaining,
            Mp = maxMp,
            MaxMp = maxMp,
            JobResources = ResourceState.BuildInitialJobResources(),
        };
        state.SetTimelineTime(startTime);
        state.GcdReadyAt = startTime;
        state.CastEndsAt = startTime;
        state.WeaveEndsAt = startTime;
        state.NaturalMpLastTickAt = startTime;
        state.DowntimeEndsAt = startTime;
        state.FightRemaining = fightRemaining;
        return state;
    }

    /// <summary>系统层是否识别某个共享行为名。</summary>
    public bool SupportsBehavior(string behavior) => BuffState.SupportsBehavior(behavior);

    public ValidationResult ValidateCommonAction(
        CombatState state,
        SkillDefinition skill,
        bool isInstant,
        bool hasAvailableCharge,
        bool includeTemporalLocks = true) =>
        PlayerState.ValidateCommonAction(
            state,
            skill,
            isInstant,
            hasAvailableCharge,
            includeTemporalLocks);

    public ValidationResult ValidateSharedBehavior(CombatState state, SkillDefinition skill) =>
        BuffState.ValidateSharedBehavior(state, skill);

    public void ApplyActionTiming(
        CombatState state,
        SkillDefinition skill,
        double effectiveGcd,
        double actualCastSeconds,
        double? gcdStartTimestamp = null) =>
        PlayerState.ApplyActionTiming(state, skill, effectiveGcd, actualCastSeconds, gcdStartTimestamp);

    public bool ApplySharedBehavior(CombatState state, SkillDefinition skill) =>
        BuffState.ApplySharedBehavior(state, skill);

    /// <summary>装配公共领域处理器，时间线独占推进与事件续排。</summary>
    public CombatTimelineRuntime CreateTimeline(CombatState state,
        Func<string, int> getMaxCharges,
        Func<double, bool>? targetableAt = null)
    {
        var timeline = new CombatTimelineRuntime(state);
        timeline.RegisterHandler(TimelineEventKind.GcdReady, PlayerState.HandleGcdReady);
        timeline.RegisterHandler(TimelineEventKind.SceneChanged, HandleSceneChanged);
        timeline.RegisterHandler(TimelineEventKind.StatusExpired, StatusTimeline.HandleExpired);
        timeline.RegisterHandler(TimelineEventKind.DotTick, DotTimeline.HandleTick);
        timeline.RegisterHandler(TimelineEventKind.CooldownChargeReady,
            (item, current) => CooldownRuntime.HandleReady(item, current, getMaxCharges));
        timeline.RegisterHandler(TimelineEventKind.MpTick,
            (item, current) => MpRecovery.HandleTick(
                item,
                current,
                (tickState, amount) => JobTimeline.ResolveMpTick(tickState, amount)));
        timeline.RegisterHandler(TimelineEventKind.JobPeriodicTick, JobTimeline.HandlePeriodic);
        timeline.RegisterHandler(
            TimelineEventKind.JobTimerExpired,
            (item, current) => JobTimeline.HandleCountdown(item, current, targetableAt));
        timeline.ConfigureResources(current => PlayerState.DescribeEvents(current)
            .Concat(StatusTimeline.DescribeEvents(current))
            .Concat(DotTimeline.DescribeEvents(current))
            .Concat(CooldownRuntime.DescribeEvents(current))
            .Concat(JobTimeline.DescribeEvents(current))
            .Append(MpRecovery.DescribeEvent(current)).ToArray());
        return timeline;
    }

    private TimelineMutation HandleSceneChanged(TimelineEvent item, CombatState state)
    {
        if (item.Payload is not ExternalCombatEvent external)
        {
            return PlayerState.HandleSceneChanged(item, state);
        }

        return new TimelineMutation(ApplyState: target =>
        {
            switch (external.Kind)
            {
                case ExternalCombatEventKinds.BossTargetableChanged when external.Value is bool targetable:
                    target.BossTargetable = targetable;
                    break;
                case ExternalCombatEventKinds.MovementChanged when external.Value is bool moving:
                    target.IsMoving = moving;
                    break;
                case ExternalCombatEventKinds.RaidBuffWindowChanged when external.Value is bool active:
                    if (active)
                    {
                        BuffState.GrantRegisteredStatus(
                            target,
                            "raid_buff_window",
                            external.RemainingSeconds);
                    }
                    else
                    {
                        BuffState.ClearRegisteredStatus(target, "raid_buff_window");
                    }
                    break;
                case ExternalCombatEventKinds.TargetCountChanged when external.TargetCount is int targetCount:
                    target.TargetCount = targetCount;
                    break;
                default:
                    throw new ArgumentException($"unsupported external scene event: {external.Kind}");
            }
        });
    }

    internal void ValidateExternalEvent(ExternalCombatEvent external)
    {
        var hasRaidBuffWindow = BuffState.SystemStatusDefinitions.ContainsKey("raid_buff_window")
            || BuffState.JobStatusDefinitions.ContainsKey("raid_buff_window");
        var supported = external.Kind switch
        {
            ExternalCombatEventKinds.BossTargetableChanged =>
                external.Value is bool
                && external.TargetCount is null
                && external.RemainingSeconds is null,
            ExternalCombatEventKinds.MovementChanged =>
                external.Value is bool
                && external.TargetCount is null
                && external.RemainingSeconds is null,
            ExternalCombatEventKinds.RaidBuffWindowChanged =>
                hasRaidBuffWindow
                && external.Value is bool active
                && external.TargetCount is null
                && (active
                    ? external.RemainingSeconds is null or > 0
                    : external.RemainingSeconds is null),
            ExternalCombatEventKinds.TargetCountChanged =>
                external.Value is null
                && external.TargetCount is >= 0
                && external.RemainingSeconds is null,
            _ => false,
        };
        if (!supported)
        {
            throw new ArgumentException($"unsupported external scene event: {external.Kind}", nameof(external));
        }
    }

    public bool HasAvailableCharge(CombatState state, SkillDefinition skill) =>
        CooldownRuntime.HasAvailableCharge(state, skill);

    public double? NextChargeReadyAt(CombatState state, SkillDefinition skill) =>
        CooldownRuntime.NextChargeReadyAt(state, skill);

    public (double NextCooldownSeconds, int AvailableCharges, int MaxCharges) BuildCooldownSnapshot(
        CombatState state,
        SkillDefinition skill) =>
        CooldownRuntime.BuildCooldownSnapshot(state, skill);

    public void ConsumeCooldown(CombatState state, SkillDefinition skill) =>
        CooldownRuntime.ConsumeCooldown(state, skill);

    internal void ApplyCooldownReduction(CombatState state, SkillDefinition skill, double seconds) =>
        CooldownRuntime.ApplyReduction(state, skill, seconds);

    public void RecordActionHistory(
        CombatState nextState,
        CombatState previousState,
        SkillDefinition skill,
        double potency,
        double value,
        double castTimeSeconds,
        double castTimeGcds,
        double gcdWindowSeconds,
        double gcdWindowGcds,
        bool isLegal,
        string invalidReason,
        double nextCooldownSeconds,
        int availableCharges,
        int maxCharges,
        StateContext stateBefore,
        StateContext stateAfter,
        double? recordedTimeSeconds = null,
        double? requestTimestamp = null,
        double? castCompletedTimestamp = null,
        double? effectTimestamp = null,
        Guid? actionInstanceId = null) =>
        History.RecordActionHistory(
            nextState,
            previousState,
            skill,
            potency,
            value,
            castTimeSeconds,
            castTimeGcds,
            gcdWindowSeconds,
            gcdWindowGcds,
            isLegal,
            invalidReason,
            nextCooldownSeconds,
            availableCharges,
            maxCharges,
            stateBefore,
            stateAfter,
            recordedTimeSeconds,
            requestTimestamp,
            castCompletedTimestamp,
            effectTimestamp,
            actionInstanceId);

    public StatusDefinition StatusDefinitionOf(string key) => BuffState.StatusDefinitionOf(key);

    internal object GetJobResource(CombatState state, string key) => ResourceState.GetJobResource(state, key);

    internal void SetJobResource(CombatState state, string key, object value) =>
        ResourceState.SetJobResource(state, key, value);

    public Dictionary<string, object> BuildJobResourceSnapshot(CombatState state)
    {
        var snapshot = ResourceState.BuildJobResourceSnapshot(state);
        // 职业时间资源的真相是绝对截止时刻，这里投影回输出契约期望的相对秒数视图。
        // 截止时刻取自注册项声明的活跃条件，保证"不描述到期"与"视图显示 0"一致。
        foreach (var (projection, deadline) in JobTimeline.ActiveTimerDeadlines(state))
        {
            if (!snapshot.ContainsKey(projection.OutputKey))
            {
                continue;
            }

            snapshot[projection.OutputKey] = ResourceState.NormalizeResourceValue(
                projection.OutputKey,
                projection.Project(state, deadline));
        }

        return snapshot;
    }

    /// <summary>导出一次动作前后的量谱快照与消耗量谱。</summary>
    public (IReadOnlyDictionary<string, object> Before, IReadOnlyDictionary<string, object> After,
        IReadOnlyDictionary<string, object> Consumed) BuildJobResourceTransition(
        CombatState previousState,
        CombatState nextState)
    {
        var before = BuildJobResourceSnapshot(previousState);
        var after = BuildJobResourceSnapshot(nextState);
        return ResourceState.BuildJobResourceTransition(before, after);
    }

    /// <summary>按当前状态的共享增伤倍率解析实际威力。</summary>
    public double ResolveAppliedPotency(CombatState state, double basePotency, SkillDefinition? skill = null)
    {
        var resolvedBasePotency = Math.Max(0.0, basePotency);
        if (resolvedBasePotency <= 0.0)
        {
            return 0.0;
        }
        double multiplier = 1.0;
        if (skill is not null)
        {
            multiplier *= ResolveAoeTargetMultiplier(skill, state.TargetCount);
        }
        if (state.HasStatus("burst_potion"))
        {
            multiplier *= _potencyConfig.BurstPotionMultiplier;
        }
        if (state.HasStatus("raid_buff_window"))
        {
            multiplier *= _potencyConfig.RaidBuffWindowMultiplier;
        }
        return resolvedBasePotency * multiplier;
    }

    /// <summary>按技能自身的额外目标衰减计算总目标威力倍率。</summary>
    public static double ResolveAoeTargetMultiplier(SkillDefinition skill, int targetCount)
    {
        if (skill.MaxTargets <= 1 || targetCount <= 1)
        {
            return 1.0;
        }
        var effectiveTargetCount = Math.Min(targetCount, skill.MaxTargets);
        var secondaryMultiplier = Math.Max(0.0, 1.0 - skill.AoeSecondaryReduction);
        return 1.0 + secondaryMultiplier * (effectiveTargetCount - 1);
    }

    /// <summary>按推进窗口内的事件时刻解析共享增伤倍率（延迟威力用）。</summary>
    public double ResolveAppliedPotencyAtOffset(CombatState state, double basePotency, double offsetSeconds)
    {
        var resolvedBasePotency = Math.Max(0.0, basePotency);
        if (resolvedBasePotency <= 0.0)
        {
            return 0.0;
        }
        var offset = Math.Max(0.0, offsetSeconds);
        double multiplier = 1.0;
        if (state.Statuses.TryGetValue("burst_potion", out var burstPotion) &&
            burstPotion.Remaining - offset > 0.0)
        {
            multiplier *= _potencyConfig.BurstPotionMultiplier;
        }
        if (state.Statuses.TryGetValue("raid_buff_window", out var raidBuff) &&
            raidBuff.Remaining - offset > 0.0)
        {
            multiplier *= _potencyConfig.RaidBuffWindowMultiplier;
        }
        return resolvedBasePotency * multiplier;
    }

    /// <summary>记录已经在实际伤害时刻解析完成的延迟威力，返回解析后的威力。</summary>
    public double RecordDelayedPotency(CombatState state, double resolvedPotency)
    {
        var resolved = Math.Max(0.0, resolvedPotency);
        state.CumulativePotency += resolved;
        return resolved;
    }

    public void RecordTargetPotency(CombatState state, CombatState sourceState, SkillDefinition skill, double directPotency)
    {
        var directPotencyValue = ResolveAppliedPotency(sourceState, directPotency, skill);
        state.CurrentPotency = directPotencyValue;
        state.CurrentGcdDotPotency = 0.0;
        state.CumulativePotency += directPotencyValue;
    }

    public void GrantRegisteredStatus(CombatState state, string key, double? remaining = null, int? stacks = null) =>
        BuffState.GrantRegisteredStatus(state, key, remaining, stacks);

    public void ClearRegisteredStatus(CombatState state, string key) =>
        BuffState.ClearRegisteredStatus(state, key);

    public void ConsumeRegisteredStatusStack(CombatState state, string key) =>
        BuffState.ConsumeRegisteredStatusStack(state, key);

    public void GrantRegisteredTargetDot(
        CombatState state,
        string key,
        double remaining,
        double potencyPerTick,
        double tickInterval = 3.0) =>
        DotTimeline.GrantRegisteredDot(state, key, remaining, potencyPerTick, tickInterval);
}

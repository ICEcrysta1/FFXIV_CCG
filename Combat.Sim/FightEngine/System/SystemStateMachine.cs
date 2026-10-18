// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Timeline;
using Combat.Sim.Config;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.Outputs;
using Combat.Sim.State;
using Combat.Sim.System.Timeline;

namespace Combat.Sim.System;

/// <summary>
/// 系统状态机总调度门面：组装全部系统层子模块，统一协调公共时间轴、
/// 公共 Buff、MP、冷却与目标 DoT；职业状态由职业模块持有。
/// </summary>
public sealed class SystemStateMachine
{
    private readonly EngineTimingConfig _engineTiming;
    private readonly SystemPotencyConfig _potencyConfig;
    private readonly SystemMpRecoveryConfig _mpRecoveryConfig;

    public BuffStateRegistry BuffState { get; }
    public StatusTimelineRuntime StatusTimeline { get; }
    public DotTimelineRuntime DotTimeline { get; }
    public MpRecoveryRuntime MpRecovery { get; }
    public CooldownRuntime CooldownRuntime { get; }
    public PlayerStateRuntime PlayerState { get; }

    /// <summary>引擎时序配置，定义动作请求队列窗口。</summary>
    public EngineTimingConfig EngineTiming => _engineTiming;

    public SystemStateMachine(
        EngineTimingConfig engineTiming,
        SystemPotencyConfig potencyConfig,
        SystemMpRecoveryConfig mpRecoveryConfig)
    {
        _engineTiming = engineTiming;
        _potencyConfig = potencyConfig;
        _mpRecoveryConfig = mpRecoveryConfig;
        BuffState = new BuffStateRegistry();
        StatusTimeline = new StatusTimelineRuntime();
        DotTimeline = new DotTimelineRuntime();
        MpRecovery = new MpRecoveryRuntime(mpRecoveryConfig);
        CooldownRuntime = new CooldownRuntime();
        PlayerState = new PlayerStateRuntime();
    }

    public IReadOnlyDictionary<string, StatusDefinition> SystemStatusDefinitions => BuffState.Definitions;
    public IReadOnlyList<string> RegisteredTargetDotKeys => DotTimeline.RegisteredDotKeys;

    public void RegisterSystemStatuses(IReadOnlyDictionary<string, StatusDefinition> statuses) =>
        BuffState.RegisterStatuses(statuses);

    public void RegisterJobTargetDots(IEnumerable<SkillDefinition> skills) =>
        DotTimeline.RegisterDots(skills);

    /// <summary>创建系统层默认初始状态。</summary>
    internal CombatState InitialState(double fightRemaining, int maxMp, double startTime = 0)
    {
        var state = new CombatState
        {
            FightRemaining = fightRemaining,
            Mp = maxMp,
            MaxMp = maxMp,
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

    public void ApplyActionTiming(
        CombatState state,
        SkillDefinition skill,
        double effectiveGcd,
        double actualCastSeconds,
        double? gcdStartTimestamp = null) =>
        PlayerState.ApplyActionTiming(state, skill, effectiveGcd, actualCastSeconds, gcdStartTimestamp);

    /// <summary>死亡标记只重置系统拥有的 MP 与公共 Buff，保留 DoT 和冷却。</summary>
    internal void ApplyDeath(CombatState state)
    {
        state.IsDead = true;
        state.Mp = 0;
        state.Statuses.Clear();
    }

    /// <summary>注册公共规则；职业 MP 修正只影响恢复量，MP 写入仍由系统执行。</summary>
    internal void RegisterTimelineHandlers(CombatTimelineRuntime timeline,
        Func<string, int> getMaxCharges, Func<CombatState, int, int>? mpTickModifier = null)
    {
        timeline.RegisterHandler(TimelineEventKind.GcdReady, PlayerState.HandleGcdReady);
        timeline.RegisterHandler(TimelineEventKind.SceneChanged, HandleSceneChanged);
        timeline.RegisterHandler(TimelineEventKind.DotTick, DotTimeline.HandleTick);
        timeline.RegisterHandler(TimelineEventKind.CooldownChargeReady,
            (item, current) => CooldownRuntime.HandleReady(item, current, getMaxCharges));
        timeline.RegisterHandler(TimelineEventKind.MpTick,
            (item, current) => MpRecovery.HandleTick(item, current, mpTickModifier ?? ((_, amount) => amount)));
        timeline.RegisterHandler(TimelineEventKind.PlayerRevived, (item, _) => new TimelineMutation(ApplyState: target =>
        {
            var external = (ExternalCombatEvent)item.Payload!;
            target.IsDead = false;
            target.Mp = external.Kind == ExternalCombatEventKinds.PlayerRevivedLb3
                ? target.MaxMp
                : Math.Min(target.MaxMp, external.Mp ?? (int)(target.MaxMp * _mpRecoveryConfig.RevivalRatio));
        }));
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
        var hasRaidBuffWindow = BuffState.Definitions.ContainsKey("raid_buff_window");
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
            ExternalCombatEventKinds.PlayerDeath =>
                external.Value is null
                && external.TargetCount is null
                && external.RemainingSeconds is null,
            ExternalCombatEventKinds.PlayerRevived =>
                external.Value is null
                && external.TargetCount is null
                && external.RemainingSeconds is null
                && external.Mp is null or >= 0,
            ExternalCombatEventKinds.PlayerRevivedLb3 =>
                external.Value is null
                && external.TargetCount is null
                && external.RemainingSeconds is null
                && external.Mp is null,
            _ => false,
        };
        if (!supported || (external.Mp is not null && external.Kind != ExternalCombatEventKinds.PlayerRevived))
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

    public void GrantRegisteredTargetDot(
        CombatState state,
        string key,
        double remaining,
        double potencyPerTick,
        double tickInterval = 3.0) =>
        DotTimeline.GrantRegisteredDot(state, key, remaining, potencyPerTick, tickInterval);
}

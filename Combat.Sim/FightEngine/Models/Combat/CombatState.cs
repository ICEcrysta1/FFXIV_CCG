// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Outputs;

namespace Combat.Sim.Models.Combat;

/// <summary>
/// 单个时刻的战斗快照（对照 models.CombatState）。
/// 可变容器类型；Clone 显式复制可变字段，避免整树深拷贝的开销。
/// </summary>
public sealed class CombatState
{
    private double _time;
    public double Time => _time;

    internal void SetTimelineTime(double time)
    {
        BindResourceTimes();
        _time = time;
        BindResourceTimes();
    }
    internal void BindResourceTimes()
    {
        foreach (var item in Statuses.Values) item.BindTime(Time);
        Job.BindTime(Time);
        foreach (var item in Dots.Values) item.BindTime(Time);
        foreach (var item in Cooldowns.Values) item.BindTime(Time);
    }
    public int GcdIndex { get; set; } = 0;
    /// <summary>队列自己的基础 GCD；未指定时使用共享职业配置的默认值。</summary>
    public double? BaseGcd { get; internal set; }
    public double FightEndsAt { get; internal set; } = 600.0;
    public double FightRemaining { get => Math.Max(0, FightEndsAt - Time); internal set => FightEndsAt = Time + value; }
    public double? NextDowntimeStartsAt { get; internal set; }
    public double? NextDowntimeEta { get => NextDowntimeStartsAt is double t ? Math.Max(0, t - Time) : null; internal set => NextDowntimeStartsAt = value is double t ? Time + t : null; }
    public double DowntimeEndsAt { get; internal set; }
    private double _downtimeDuration;
    public double DowntimeRemaining
    {
        get => BossTargetable ? _downtimeDuration : Math.Max(0, DowntimeEndsAt - Time);
        internal set { _downtimeDuration = value; DowntimeEndsAt = Time + value; }
    }
    public int Mp { get; set; } = 10000;
    public int MaxMp { get; set; } = 10000;
    // 死亡期间只暂停自然回蓝；冷却、充能和目标 DoT 继续沿统一时间线结算。
    internal bool IsDead { get; set; }
    public double NaturalMpLastTickAt { get; internal set; }
    public double NaturalMpTickProgress { get => Math.Max(0, Time - NaturalMpLastTickAt); internal set => NaturalMpLastTickAt = Time - value; }
    public double CastEndsAt { get; internal set; } = 0.0;
    public double CastRemaining { get => Math.Max(0, CastEndsAt - Time); internal set => CastEndsAt = Time + value; }
    public double GcdReadyAt { get; internal set; } = 0.0;
    public double GcdRemaining { get => Math.Max(0, GcdReadyAt - Time); internal set => GcdReadyAt = Time + value; }
    public double WeaveEndsAt { get; internal set; } = 0.0;
    public double WeaveWindowRemaining { get => Math.Max(0, WeaveEndsAt - Time); internal set => WeaveEndsAt = Time + value; }
    public int OgcdsWeaved { get; set; } = 0;
    public int MaxOgcdPerWindow { get; set; } = 3;
    public bool IsMoving { get; set; } = false;
    private bool _bossTargetable = true;
    public bool BossTargetable
    {
        get => _bossTargetable;
        set
        {
            if (_bossTargetable && !value) DowntimeEndsAt = Time + _downtimeDuration;
            _bossTargetable = value;
        }
    }
    public int TargetCount { get; set; } = 1;
    public double CumulativePotency { get; set; } = 0.0;
    public double CumulativeDotPotency { get; set; } = 0.0;
    public double CurrentPotency { get; set; } = 0.0;
    public double CurrentGcdDotPotency { get; set; } = 0.0;
    public JobState Job { get; set; } = new();
    public Dictionary<string, CooldownState> Cooldowns { get; set; } = new();
    public Dictionary<string, StatusState> Statuses { get; set; } = new();
    public Dictionary<string, DotState> Dots { get; set; } = new();
    public List<ActionHistoryEntry> History { get; set; } = new();
    // 与历史保留窗口独立；只跟踪最近一次已接受的真实或 policy 请求。
    internal Guid? LastDecisionId { get; set; }
    internal ModelStateFrame? LastDecisionAfter { get; set; }
    // 最近一次已落实动作的基准，供死亡取消尚未完成的请求后恢复。
    internal ModelStateFrame? LastCompletedDecisionAfter { get; set; }
    // 真实技能生效与 policy 记录共用递增序号，独立于历史裁剪和输出时间舍入。
    internal long LastHistorySequence { get; set; }
    internal long ReserveHistorySequence() => LastHistorySequence = checked(LastHistorySequence + 1);

    /// <summary>
    /// 显式复制战斗状态：标量直接值拷贝，容器与元素复制；
    /// history 默认复制列表本身但复用条目对象。
    /// <paramref name="copyHistory"/> 为 false 只适用于"明确不会改写 history 列表"的只读预演场景。
    /// </summary>
    public CombatState Clone(bool copyHistory = true)
    {
        BindResourceTimes();
        var clone = new CombatState
        {
            GcdIndex = GcdIndex,
            BaseGcd = BaseGcd,
            FightEndsAt = FightEndsAt,
            NextDowntimeStartsAt = NextDowntimeStartsAt,
            DowntimeEndsAt = DowntimeEndsAt,
            _downtimeDuration = _downtimeDuration,
            Mp = Mp,
            MaxMp = MaxMp,
            IsDead = IsDead,
            NaturalMpLastTickAt = NaturalMpLastTickAt,
            CastEndsAt = CastEndsAt,
            GcdReadyAt = GcdReadyAt,
            WeaveEndsAt = WeaveEndsAt,
            OgcdsWeaved = OgcdsWeaved,
            MaxOgcdPerWindow = MaxOgcdPerWindow,
            IsMoving = IsMoving,
            _bossTargetable = _bossTargetable,
            TargetCount = TargetCount,
            CumulativePotency = CumulativePotency,
            CumulativeDotPotency = CumulativeDotPotency,
            CurrentPotency = CurrentPotency,
            CurrentGcdDotPotency = CurrentGcdDotPotency,
            Job = Job.Clone(),
            Cooldowns = Cooldowns.ToDictionary(pair => pair.Key, pair => pair.Value.Clone()),
            Statuses = Statuses.ToDictionary(pair => pair.Key, pair => pair.Value.Clone()),
            Dots = Dots.ToDictionary(pair => pair.Key, pair => pair.Value.Clone()),
            History = copyHistory ? new List<ActionHistoryEntry>(History) : History,
            LastDecisionId = LastDecisionId,
            LastDecisionAfter = LastDecisionAfter,
            LastCompletedDecisionAfter = LastCompletedDecisionAfter,
            LastHistorySequence = LastHistorySequence,
        };
        clone.SetTimelineTime(Time);
        return clone;
    }

    /// <summary>事件载荷和策略记录只需要当时的状态，不能递归携带历史列表。</summary>
    internal CombatState CloneWithoutHistory()
    {
        var clone = Clone(copyHistory: false);
        clone.History = new();
        return clone;
    }

    public bool HasStatus(string key) =>
        Statuses.TryGetValue(key, out var status) && status.Remaining > 0 && status.Stacks > 0;

    public int StatusStacks(string key)
    {
        if (!Statuses.TryGetValue(key, out var status) || status.Remaining <= 0)
        {
            return 0;
        }
        return status.Stacks;
    }

    public double DotRemaining(string key)
    {
        if (!Dots.TryGetValue(key, out var dot))
        {
            return 0.0;
        }
        return Math.Max(dot.Remaining, 0.0);
    }
}

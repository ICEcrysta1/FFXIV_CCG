// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Timeline;

namespace Combat.Sim.System.Timeline;

/// <summary>
/// 职业时间资源注册表。职业只声明周期与倒计时的**绝对截止时刻**，时间推进与事件排空由时间线内核负责；
/// 注册表不提供按秒推进的入口，也不保存任何剩余量。
/// </summary>
public sealed class JobTimelineRegistry
{
    private sealed record PeriodicRegistration(
        string Key,
        string OutputTimerKey,
        double IntervalSeconds,
        Func<CombatState, double?> Deadline,
        Func<TimelineEvent, CombatState, TimelineMutation> HandleTick);

    private sealed record CountdownRegistration(
        string Key,
        string OutputTimerKey,
        Func<CombatState, double?> Deadline,
        Func<TimelineEvent, CombatState, Func<double, bool>?, TimelineMutation> HandleExpiry);

    private readonly Dictionary<string, PeriodicRegistration> _periodic = new(StringComparer.Ordinal);
    private readonly Dictionary<string, CountdownRegistration> _countdowns = new(StringComparer.Ordinal);
    private Func<CombatState, int, int>? _mpTickModifier;
    /// <summary>
    /// 按注册项自己声明的活跃条件给出每个对外计时器视图的截止时刻，用于对外投影：
    /// 与 <see cref="DescribeEvents"/> 走同一份声明，因此"不描述"与"视图显示 0"始终一致。
    /// </summary>
    internal IReadOnlyList<(JobTimerProjection Projection, double? Deadline)> ActiveTimerDeadlines(
        CombatState state) =>
        _periodic.Values
            .OrderBy(item => item.Key, StringComparer.Ordinal)
            .Select(item => (
                new JobTimerProjection(item.OutputTimerKey, item.Key, IsAccumulated: true, item.IntervalSeconds),
                item.Deadline(state)))
            .Concat(_countdowns.Values
                .OrderBy(item => item.Key, StringComparer.Ordinal)
                .Select(item => (
                    new JobTimerProjection(item.OutputTimerKey, item.Key, IsAccumulated: false, IntervalSeconds: 0.0),
                    item.Deadline(state))))
            .ToArray();

    /// <summary>
    /// 注册职业周期资源。周期由 <paramref name="intervalSeconds"/> 声明，
    /// 下一次结算时刻由 <paramref name="deadline"/> 从绝对时基读出；返回 null 表示当前不在计时。
    /// </summary>
    public void RegisterPeriodicResource(
        string key,
        double intervalSeconds,
        string outputTimerKey,
        Func<CombatState, double?> deadline,
        Func<TimelineEvent, CombatState, TimelineMutation> handleTick)
    {
        ValidateKey(key);
        ValidateOutputTimerKey(outputTimerKey);
        if (!double.IsFinite(intervalSeconds) || intervalSeconds <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(intervalSeconds));
        ArgumentNullException.ThrowIfNull(deadline);
        ArgumentNullException.ThrowIfNull(handleTick);
        if (IsRegisteredKey(key) || !_periodic.TryAdd(
                key, new(key, outputTimerKey, intervalSeconds, deadline, handleTick)))
        {
            throw new InvalidOperationException($"job periodic resource already registered: {key}");
        }
    }

    /// <summary>
    /// 注册职业倒计时。到期时刻由 <paramref name="deadline"/> 从绝对时基读出；
    /// 返回 null 表示当前没有在计时的倒计时。
    /// </summary>
    public void RegisterCountdown(
        string key,
        string outputTimerKey,
        Func<CombatState, double?> deadline,
        Func<TimelineEvent, CombatState, Func<double, bool>?, TimelineMutation> handleExpiry)
    {
        ValidateKey(key);
        ValidateOutputTimerKey(outputTimerKey);
        ArgumentNullException.ThrowIfNull(deadline);
        ArgumentNullException.ThrowIfNull(handleExpiry);
        if (IsRegisteredKey(key) || !_countdowns.TryAdd(key, new(key, outputTimerKey, deadline, handleExpiry)))
        {
            throw new InvalidOperationException($"job countdown already registered: {key}");
        }
    }

    public void RegisterMpTickModifier(Func<CombatState, int, int> modifier)
    {
        ArgumentNullException.ThrowIfNull(modifier);
        if (_mpTickModifier is not null)
            throw new InvalidOperationException("job MP tick modifier already registered");
        _mpTickModifier = modifier;
    }

    internal int ResolveMpTick(CombatState state, int naturalAmount) =>
        Math.Max(0, _mpTickModifier?.Invoke(state, naturalAmount) ?? naturalAmount);

    /// <summary>
    /// 只声明仍然在计时的资源截止时刻：注册项不保存剩余量，也不随推进重算。
    /// 因此同一状态在同一时刻重复描述的结果相同，不依赖推进被切成几段。
    /// </summary>
    internal IEnumerable<TimelineEvent> DescribeEvents(CombatState state)
    {
        foreach (var registration in _periodic.Values.OrderBy(item => item.Key, StringComparer.Ordinal))
        {
            if (registration.Deadline(state) is not double deadline)
                continue;

            yield return new(
                Math.Max(state.Time, deadline),
                TimelineEventPriority.PeriodicSettlement,
                TimelineEventKind.JobPeriodicTick,
                registration.Key);
        }

        foreach (var registration in _countdowns.Values.OrderBy(item => item.Key, StringComparer.Ordinal))
        {
            if (registration.Deadline(state) is not double deadline)
                continue;

            yield return new(
                Math.Max(state.Time, deadline),
                TimelineEventPriority.ExpirationAndCooldown,
                TimelineEventKind.JobTimerExpired,
                registration.Key);
        }
    }

    internal TimelineMutation HandlePeriodic(TimelineEvent item, CombatState state) =>
        _periodic.TryGetValue(item.OwnerKey ?? string.Empty, out var registration)
            ? registration.HandleTick(item, state)
            : TimelineMutation.Empty;

    internal TimelineMutation HandleCountdown(
        TimelineEvent item,
        CombatState state,
        Func<double, bool>? targetableAt = null)
    {
        if (!_countdowns.TryGetValue(item.OwnerKey ?? string.Empty, out var registration))
            return TimelineMutation.Empty;

        return registration.HandleExpiry(item, state, targetableAt);
    }

    private static void ValidateKey(string key)
    {
        if (string.IsNullOrWhiteSpace(key))
            throw new ArgumentException("job timeline key must not be empty", nameof(key));
    }

    private static void ValidateOutputTimerKey(string outputTimerKey)
    {
        if (string.IsNullOrWhiteSpace(outputTimerKey))
            throw new ArgumentException("output timer key must not be empty", nameof(outputTimerKey));
    }

    private bool IsRegisteredKey(string key) => _periodic.ContainsKey(key) || _countdowns.ContainsKey(key);
}

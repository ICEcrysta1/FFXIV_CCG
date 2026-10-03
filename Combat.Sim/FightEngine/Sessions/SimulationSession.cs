// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Facade;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Policy;
using Combat.Sim.Models.Timeline;
using Combat.Sim.Policy;

namespace Combat.Sim.Sessions;

/// <summary>在同一队列锁内生成的请求结果和时间游标，避免调用方分次读取发生竞态。</summary>
public sealed record SessionResult<T>(double Timestamp, double? NextScheduledEventTime, T Value);

/// <summary>只读容量诊断；计数不需要复制历史或事件载荷。</summary>
public sealed record SessionStatistics(
    double Timestamp, int ActionHistoryCount, int PolicyHistoryCount,
    int PendingEventCount, int QueueEntryCount, int PendingSettlementCount);

/// <summary>
/// 一个可增量推进的独立队列。调用按队列串行，不同队列可由不同宿主线程并行执行。
/// 同队列的事件时间顺序由调用方保证；关闭后旧句柄永久失效，不能误操作新队列。
/// </summary>
public sealed class SimulationSession : IDisposable
{
    private readonly object _gate = new();
    private SimulationEngine? _owner;
    private JobSimulator? _simulator;
    private PolicySession? _policy;

    internal SimulationSession(SimulationEngine owner, long id, JobSimulator simulator, PolicySession policy)
    {
        _owner = owner;
        Id = id;
        _simulator = simulator;
        _policy = policy;
    }

    public long Id { get; }
    public bool IsClosed { get { lock (_gate) return _simulator is null; } }

    public SessionResult<bool> Reset(
        int? historyLimit, double? actualBaseGcd = null,
        double initialTimestamp = 0, double? fightRemaining = null)
    {
        lock (_gate)
        {
            RequireSimulator();
            // 全部构造成功才替换；错误参数不能销毁仍可用的旧上下文。
            var (simulator, policy) = _owner!.BuildRuntime(
                historyLimit, actualBaseGcd, initialTimestamp, fightRemaining);
            _simulator = simulator;
            _policy = policy;
            return Result(simulator, true);
        }
    }

    public SessionResult<bool> AdvanceTo(double timestamp) => Execute(simulator =>
    {
        simulator.AdvanceClockTo(timestamp);
        return true;
    });

    public SessionResult<ActionSubmissionResult> SubmitAction(
        double timestamp, string skillKey, double? actualCastSeconds = null) =>
        Execute(simulator =>
        {
            if (actualCastSeconds is { } duration && (!double.IsFinite(duration) || duration < 0))
                throw new ArgumentOutOfRangeException(nameof(actualCastSeconds));
            return simulator.SubmitAction(timestamp, skillKey, actualCastSeconds);
        });

    public SessionResult<ValidationResult> ValidateActionAt(double timestamp, string skillKey) =>
        Execute(simulator => simulator.ValidateActionAt(timestamp, skillKey));

    public SessionResult<ExternalEventResult> ApplyExternalEvent(ExternalCombatEvent externalEvent) =>
        Execute(simulator => simulator.ApplyExternalEvent(externalEvent));

    public SessionResult<PolicyDecision> RecordPolicyAction(
        double timestamp, string actionKey, double nextObservationTimestamp) =>
        Execute(simulator => _policy!.Record(simulator, timestamp, actionKey, nextObservationTimestamp));

    public SessionResult<Dictionary<string, object?>> ObserveAt(
        double timestamp, string format = "seconds", double? nextObservationTimestamp = null) =>
        Execute(simulator =>
        {
            if (format is not ("seconds" or "gcd" or "vector"))
                throw new ArgumentException($"unsupported observe format: {format}", nameof(format));
            if (format == "vector" && (nextObservationTimestamp is not { } next
                || !double.IsFinite(next) || next < timestamp))
                throw new ArgumentOutOfRangeException(nameof(nextObservationTimestamp));
            simulator.AdvanceClockTo(timestamp);
            return format == "vector"
                ? _policy!.BuildVectorContext(simulator, nextObservationTimestamp!.Value)
                : simulator.FormatState(format);
        });

    public SessionStatistics GetStatistics()
    {
        lock (_gate)
        {
            var simulator = RequireSimulator();
            var counts = simulator.GetStatistics();
            return new(simulator.Time, counts.History, _policy!.HistoryCount,
                counts.PendingEvents, counts.QueueEntries, counts.PendingSettlements);
        }
    }

    internal SessionResult<T> Execute<T>(Func<JobSimulator, T> action)
    {
        lock (_gate)
        {
            var simulator = RequireSimulator();
            return Result(simulator, action(simulator));
        }
    }

    private static SessionResult<T> Result<T>(JobSimulator simulator, T value) =>
        new(simulator.Time, simulator.GetNextScheduledEventTime(), value);

    private JobSimulator RequireSimulator() =>
        _simulator ?? throw new ObjectDisposedException(nameof(SimulationSession));

    internal void CloseCore()
    {
        lock (_gate)
        {
            _simulator = null;
            _policy = null;
            _owner = null;
        }
    }

    // 不持有队列锁再取注册锁，关闭、重置和并发请求使用一致的锁顺序。
    public void Dispose() => Volatile.Read(ref _owner)?.Release(this);
}

// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Facade;
using Combat.Sim.Outputs;
using Combat.Sim.Policy;

namespace Combat.Sim.Sessions;

/// <summary>
/// 多队列宿主：每个引擎只装载一份职业规则和配置，队列各自持有战斗、事件和输出上下文。
/// 注册锁只保护生命周期；执行请求使用各队列自己的锁，不在全局锁内运行模拟。
/// </summary>
public sealed class SimulationEngine : IDisposable
{
    private readonly object _gate = new();
    private readonly CombatStateMachine _rules;
    private readonly PolicyActionRegistry _policyRegistry;
    private readonly Dictionary<long, SimulationSession> _sessions = new();
    private long _nextId;
    private bool _disposed;

    public SimulationEngine(string projectRoot, string jobTag, int capacity = 16)
    {
        if (capacity < 1) throw new ArgumentOutOfRangeException(nameof(capacity));
        Capacity = capacity;
        _rules = CombatStateMachine.FromDefaultConfig(projectRoot, jobTag);
        _policyRegistry = PolicyActionRegistry.Load(projectRoot, _rules.SkillBook);
    }

    public int Capacity { get; }

    /// <summary>嵌入宿主可注入已装配的规则与策略索引，多个队列共用同一份配置。</summary>
    public SimulationEngine(CombatStateMachine rules, PolicyActionRegistry policyRegistry, int capacity = 16)
    {
        ArgumentNullException.ThrowIfNull(rules);
        ArgumentNullException.ThrowIfNull(policyRegistry);
        if (capacity < 1) throw new ArgumentOutOfRangeException(nameof(capacity));
        Capacity = capacity;
        _rules = rules;
        _policyRegistry = policyRegistry;
    }
    public string JobTag => _rules.JobTag;
    public int ActiveCount { get { lock (_gate) return _sessions.Count; } }

    /// <summary>
    /// 显式选择输出历史保留量；null 表示完整记录，与队列执行时长无关。
    /// 满载立即拒绝，由宿主控制在途任务数量，不在引擎内积压无界请求。
    /// </summary>
    public SimulationSession CreateSession(
        int? historyLimit,
        double? actualBaseGcd = null,
        double initialTimestamp = 0,
        double? fightRemaining = null)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_sessions.Count >= Capacity)
                throw new InvalidOperationException($"simulation engine capacity reached: {Capacity}");
            var (simulator, policy) = BuildRuntime(historyLimit, actualBaseGcd, initialTimestamp, fightRemaining);
            var session = new SimulationSession(this, checked(++_nextId), simulator, policy);
            _sessions.Add(session.Id, session);
            return session;
        }
    }

    internal (JobSimulator, PolicySession) BuildRuntime(
        int? historyLimit, double? actualBaseGcd, double initialTimestamp, double? fightRemaining)
    {
        if (!double.IsFinite(initialTimestamp))
            throw new ArgumentOutOfRangeException(nameof(initialTimestamp));
        if (actualBaseGcd is { } gcd && (!double.IsFinite(gcd) || gcd <= 0))
            throw new ArgumentOutOfRangeException(nameof(actualBaseGcd));
        if (fightRemaining is { } duration && (!double.IsFinite(duration) || duration < 0
            || !double.IsFinite(initialTimestamp + duration)))
            throw new ArgumentOutOfRangeException(nameof(fightRemaining));
        var retention = new HistoryRetention(historyLimit);
        var state = _rules.InitialState(fightRemaining, startTime: initialTimestamp);
        state.BaseGcd = actualBaseGcd;
        return (new JobSimulator(_rules, state, retention), new PolicySession(_policyRegistry, retention));
    }

    internal void Release(SimulationSession session)
    {
        lock (_gate)
        {
            if (_sessions.Remove(session.Id)) session.CloseCore();
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var session in _sessions.Values) session.CloseCore();
            _sessions.Clear();
        }
    }
}

// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Facade;
using Combat.Sim.Models.Policy;
using Combat.Sim.Outputs;

namespace Combat.Sim.Policy;

/// <summary>纯策略决策历史；记录 wait 不会向 FightEngine 提交动作。</summary>
public sealed class PolicyDecisionHistory
{
    private readonly PolicyActionRegistry _registry;
    private readonly List<PolicyDecision> _entries = new();

    public PolicyDecisionHistory(PolicyActionRegistry registry, HistoryRetention? retention = null)
    {
        _registry = registry;
        Retention = retention ?? new HistoryRetention(null);
    }

    internal HistoryRetention Retention { get; }
    internal int Count => _entries.Count;
    // 仅供同队列输出层只读访问；公开快照仍防御性复制。
    internal IReadOnlyList<PolicyDecision> ReadEntries => _entries;

    public IReadOnlyList<PolicyDecision> Entries => _entries.Select(item => item.DeepClone()).ToArray();

    public PolicyDecision Record(
        JobSimulator simulator,
        double timestamp,
        string actionKey,
        double nextObservationTimestamp)
    {
        var action = _registry.Get(actionKey);
        if (Math.Abs(simulator.Time - timestamp) > System.Timeline.CombatTimelineRuntime.TimeEpsilon)
        {
            throw new InvalidOperationException(
                $"policy decision timestamp must equal current observation time: {timestamp:R} != {simulator.Time:R}");
        }
        var before = simulator.GetStateWithoutHistory();
        if (nextObservationTimestamp < timestamp)
            throw new ArgumentOutOfRangeException(nameof(nextObservationTimestamp));
        var (modelState, historySequence) = simulator.RecordModelDecision(Guid.NewGuid(), before, completed: true);
        // wait 无即时游戏效果，其动作后状态为落实决策时的真实状态；后续变化由真实推进产生。
        var after = before.CloneWithoutHistory();
        var decision = new PolicyDecision(action, timestamp, before.GcdIndex,
            before, after, modelState, historySequence);
        _entries.Add(decision);
        Retention.Trim(_entries);
        return decision.DeepClone();
    }

    public IReadOnlyList<PolicyDecision> CreateSnapshot() => Entries;

    public void RestoreSnapshot(IEnumerable<PolicyDecision> snapshot)
    {
        var restored = snapshot.Select(item => item.DeepClone()).ToList();
        Retention.Trim(restored);
        _entries.Clear();
        _entries.AddRange(restored);
    }

    public PolicyDecisionHistory Fork()
    {
        var fork = new PolicyDecisionHistory(_registry, Retention);
        fork.RestoreSnapshot(_entries);
        return fork;
    }
}

// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Facade;
using Combat.Sim.Models.Policy;

namespace Combat.Sim.Policy;

/// <summary>纯策略决策历史；记录 wait 不会向 FightEngine 提交动作。</summary>
public sealed class PolicyDecisionHistory
{
    private readonly PolicyActionRegistry _registry;
    private readonly List<PolicyDecision> _entries = new();

    public PolicyDecisionHistory(PolicyActionRegistry registry)
    {
        _registry = registry;
    }

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
        var before = simulator.GetState();
        if (nextObservationTimestamp < timestamp)
            throw new ArgumentOutOfRangeException(nameof(nextObservationTimestamp));
        var branch = simulator.Fork();
        var after = branch.ObserveAt(nextObservationTimestamp);
        var decision = new PolicyDecision(action, timestamp, before.GcdIndex, before, after);
        _entries.Add(decision);
        return decision.DeepClone();
    }

    public IReadOnlyList<PolicyDecision> CreateSnapshot() => Entries;

    public void RestoreSnapshot(IEnumerable<PolicyDecision> snapshot)
    {
        _entries.Clear();
        _entries.AddRange(snapshot.Select(item => item.DeepClone()));
    }

    public PolicyDecisionHistory Fork()
    {
        var fork = new PolicyDecisionHistory(_registry);
        fork.RestoreSnapshot(_entries);
        return fork;
    }
}

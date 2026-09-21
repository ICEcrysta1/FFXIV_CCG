// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Facade;
using Combat.Sim.Models.Policy;

namespace Combat.Sim.Policy;

/// <summary>持有独立策略历史，并为宿主输出包含策略候选的完整模型上下文。</summary>
public sealed class PolicySession
{
    private readonly PolicyDecisionHistory _history;
    private readonly PolicyContextBuilder _contextBuilder;

    private PolicySession(PolicyActionRegistry registry)
    {
        _history = new PolicyDecisionHistory(registry);
        _contextBuilder = new PolicyContextBuilder(registry);
    }

    public static PolicySession Create(string projectRoot, JobSimulator simulator)
    {
        ArgumentNullException.ThrowIfNull(simulator);
        var registry = PolicyActionRegistry.Load(projectRoot, simulator.Rules.SkillBook);
        return new PolicySession(registry);
    }

    public Dictionary<string, object?> BuildVectorContext(
        JobSimulator simulator,
        double nextObservationTimestamp) =>
        _contextBuilder.BuildVectorContext(simulator, _history, nextObservationTimestamp);

    public PolicyDecision Record(
        JobSimulator simulator,
        double timestamp,
        string actionKey,
        double nextObservationTimestamp) =>
        _history.Record(simulator, timestamp, actionKey, nextObservationTimestamp);

    public IReadOnlyList<PolicyDecision> CreateSnapshot() => _history.CreateSnapshot();

    public void RestoreSnapshot(IEnumerable<PolicyDecision> snapshot) =>
        _history.RestoreSnapshot(snapshot);
}

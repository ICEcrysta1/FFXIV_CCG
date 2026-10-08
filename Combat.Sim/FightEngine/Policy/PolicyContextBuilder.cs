// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using System.Collections.ObjectModel;
using Combat.Sim.Facade;
using Combat.Sim.Models.Policy;
using Combat.Sim.Outputs;
using Combat.Sim.Outputs.TokenBuilders;

namespace Combat.Sim.Policy;

/// <summary>在向量化前组合动作布局与冻结历史；公共输出只由 OutputContextBuilder 装配。</summary>
public sealed class PolicyContextBuilder
{
    private readonly PolicyActionRegistry _registry;
    private Dictionary<object, ModelHistoryRow> _historyRows = new(ReferenceEqualityComparer.Instance);
    private string[] _realActionKeys = Array.Empty<string>();
    private string[] _actionKeys = Array.Empty<string>();

    public PolicyContextBuilder(PolicyActionRegistry registry) => _registry = registry;

    public Dictionary<string, object?> BuildVectorContext(
        JobSimulator simulator, PolicyDecisionHistory history, double nextObservationTimestamp)
    {
        var state = simulator.GetState();
        if (!double.IsFinite(nextObservationTimestamp) || nextObservationTimestamp < state.Time)
            throw new ArgumentOutOfRangeException(nameof(nextObservationTimestamp));
        EnsureActionLayout(simulator.ActionKeys);
        var router = simulator.OutputRouter;
        var realValues = simulator.BuildActionValues(state);
        var values = simulator.ActionKeys.Select((key, index) => (key, value: realValues[index]))
            .Concat(_registry.Actions.Select(action => (key: action.Key, value: action.Value)))
            .ToDictionary(item => item.key, item => item.value, StringComparer.Ordinal);
        var current = Compose(simulator.CaptureCurrentModelState(state));
        var retained = new Dictionary<object, ModelHistoryRow>(ReferenceEqualityComparer.Instance);
        var merged = new List<ModelHistoryRow>();
        foreach (var row in router.CollectHistory(state))
        {
            if (!_historyRows.TryGetValue(row.Identity, out var composed))
                composed = row with { ModelState = Compose(row.ModelState) };
            retained.Add(row.Identity, composed);
            merged.Add(composed);
        }
        foreach (var decision in history.ReadEntries)
        {
            if (!_historyRows.TryGetValue(decision, out var row))
                row = new ModelHistoryRow(decision, decision.HistorySequence, Compose(decision.ModelState),
                    BuildSkillToken(decision.Action, router.BuildNoopResourceTransition(decision.StateBefore)),
                    new Dictionary<string, double>
                    {
                        ["cumulative_potency"] = decision.StateAfter.CumulativePotency,
                        ["cumulative_dot_potency"] = decision.StateAfter.CumulativeDotPotency,
                    });
            retained.Add(decision, row);
            merged.Add(row);
        }
        _historyRows = retained;
        merged.Sort((left, right) => left.Sequence.CompareTo(right.Sequence));
        for (var index = 0; index < merged.Count; index++)
            if (merged[index].Sequence <= 0 || (index > 0 && merged[index - 1].Sequence == merged[index].Sequence))
                throw new InvalidOperationException("history sequence must be positive and unique");
        history.Retention.Trim(merged);
        return router.FormatVectors(state, _actionKeys, _actionKeys.Select(key => values[key]).ToArray(), current, merged);
    }

    private void EnsureActionLayout(IReadOnlyList<string> realActionKeys)
    {
        if (_realActionKeys.SequenceEqual(realActionKeys) && _actionKeys.Length > 0) return;
        _realActionKeys = realActionKeys.ToArray();
        _actionKeys = realActionKeys.Concat(_registry.Actions.Select(action => action.Key))
            .OrderBy(key => key, StringComparer.Ordinal).ToArray();
        if (_actionKeys.Distinct(StringComparer.Ordinal).Count() != _actionKeys.Length)
            throw new InvalidOperationException("policy and real action keys must be unique");
        _historyRows.Clear();
    }

    private ModelStateSnapshot Compose(ModelStateSnapshot snapshot) =>
        new(Compose(snapshot.PreviousActionAfter), Compose(snapshot.RequestState));

    private ModelStateFrame Compose(ModelStateFrame frame)
    {
        // 只组合新字典，原状态与原冻结表保持不可变；当前 policy 动作原始可用性恒为真。
        var availability = frame.SkillAvailability.ToDictionary(item => item.Key, item => item.Value, StringComparer.Ordinal);
        foreach (var action in _registry.Actions) availability.Add(action.Key, true);
        return new ModelStateFrame(frame.State, new ReadOnlyDictionary<string, bool>(availability));
    }

    private static Dictionary<string, object?> BuildSkillToken(
        PolicyActionDefinition action, IReadOnlyDictionary<string, object> consumed) =>
        SkillTokenBuilder.Build(
            skillId: action.RawId, skillKey: action.Key, skillName: action.Name,
            potency: 0, value: action.Value, kind: action.Kind, actualMpCost: 0,
            castTimeSeconds: 0, gcdWindowSeconds: 0, isLegal: true, invalidReason: "",
            nextCooldownSeconds: 0, availableCharges: 1, maxCharges: 1, jobResourcesConsumed: consumed);
}

// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Facade;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Policy;
using Combat.Sim.Outputs;
using Combat.Sim.Outputs.TokenBuilders;

namespace Combat.Sim.Policy;

/// <summary>
/// 在真实 FightEngine 上下文之上合并 policy 动作词表和 policy 历史。
/// policy 动作只产生模型 token，不进入 SkillBook、SubmitAction 或游戏动作历史。
/// </summary>
public sealed class PolicyContextBuilder
{
    private readonly PolicyActionRegistry _registry;
    private Dictionary<PolicyDecision, (Dictionary<string, object?> Skill, Dictionary<string, double[]> State)>
        _historyTokens = new(ReferenceEqualityComparer.Instance);

    public PolicyContextBuilder(PolicyActionRegistry registry)
    {
        _registry = registry;
    }

    public Dictionary<string, object?> BuildVectorContext(
        JobSimulator simulator,
        PolicyDecisionHistory history,
        double nextObservationTimestamp)
    {
        var state = simulator.GetState();
        if (nextObservationTimestamp < state.Time)
            throw new ArgumentOutOfRangeException(nameof(nextObservationTimestamp));

        var output = simulator.FormatVectorState(state);
        var router = simulator.OutputRouter;
        var keys = (List<string>)output[OutputContextSchema.ActionKeysKey]!;
        var legalMask = (List<bool>)output[OutputContextSchema.ActionLegalMaskKey]!;
        var values = (List<double>)output[OutputContextSchema.ActionValuesKey]!;
        var actions = keys.Select((key, index) => (Key: key, Legal: legalMask[index], Value: values[index]))
            .Concat(_registry.Actions.Select(action => (Key: action.Key, Legal: true, Value: action.Value)))
            .OrderBy(action => action.Key, StringComparer.Ordinal).ToArray();
        output[OutputContextSchema.ActionKeysKey] = actions.Select(action => action.Key).ToList();
        output[OutputContextSchema.ActionLegalMaskKey] = actions.Select(action => action.Legal).ToList();
        output[OutputContextSchema.ActionValuesKey] = actions.Select(action => action.Value).ToList();

        MergePolicyHistory(output, history, router, state.History);
        return output;
    }

    private void MergePolicyHistory(
        Dictionary<string, object?> output,
        PolicyDecisionHistory history,
        StateOutputRouter router,
        IReadOnlyList<ActionHistoryEntry> realHistory)
    {
        var skillHistory = (List<Dictionary<string, object?>>)
            output[OutputContextSchema.SkillHistoryContextKey]!;
        var stateHistoryContext = (Dictionary<string, object?>)
            output[OutputContextSchema.StateHistoryContextKey]!;
        var stateHistory = (List<Dictionary<string, double[]>>)stateHistoryContext["tokens"]!;
        var executionMetrics = (List<Dictionary<string, double>>)stateHistoryContext["execution_metrics"]!;

        var entries = realHistory.TakeLast(skillHistory.Count).ToArray();
        if (entries.Length != skillHistory.Count || stateHistory.Count != skillHistory.Count
            || executionMetrics.Count != skillHistory.Count)
            throw new InvalidOperationException("real history and output tokens must be aligned");

        var merged = new List<(long Sequence, Dictionary<string, object?> Skill,
            Dictionary<string, double[]> State, Dictionary<string, double> Metrics)>();
        for (var index = 0; index < skillHistory.Count; index++)
        {
            merged.Add((entries[index].HistorySequence, skillHistory[index], stateHistory[index], executionMetrics[index]));
        }

        var retainedTokens = new Dictionary<PolicyDecision,
            (Dictionary<string, object?> Skill, Dictionary<string, double[]> State)>(ReferenceEqualityComparer.Instance);
        foreach (var decision in history.ReadEntries)
        {
            if (!_historyTokens.TryGetValue(decision, out var token))
            {
                var consumed = router.BuildNoopResourceTransition(decision.StateBefore);
                token = (BuildSkillToken(decision.Action, consumed),
                    router.BuildModelStateToken(decision.ModelState));
            }
            retainedTokens.Add(decision, token);
            merged.Add((
                decision.HistorySequence,
                token.Skill,
                token.State,
                new Dictionary<string, double>
                {
                    ["cumulative_potency"] = decision.StateAfter.CumulativePotency,
                    ["cumulative_dot_potency"] = decision.StateAfter.CumulativeDotPotency,
                }));
        }
        // 只保留当前窗口中的条目，重置、恢复及滑动淘汰都不累积旧快照。
        _historyTokens = retainedTokens;

        // 同一单调战斗游标上的真实效果与等待共用实际写入顺序，展示用时间不参与排序。
        merged.Sort((left, right) => left.Sequence.CompareTo(right.Sequence));
        for (var index = 0; index < merged.Count; index++)
        {
            if (merged[index].Sequence <= 0 || (index > 0 && merged[index - 1].Sequence == merged[index].Sequence))
                throw new InvalidOperationException("history sequence must be positive and unique");
        }
        history.Retention.Trim(merged);
        skillHistory.Clear();
        skillHistory.AddRange(merged.Select(item => item.Skill));
        stateHistory.Clear();
        stateHistory.AddRange(merged.Select(item => item.State));
        executionMetrics.Clear();
        executionMetrics.AddRange(merged.Select(item => item.Metrics));
    }

    private static Dictionary<string, object?> BuildSkillToken(
        PolicyActionDefinition action,
        IReadOnlyDictionary<string, object> consumed) =>
        SkillTokenBuilder.Build(
            skillId: action.RawId,
            skillKey: action.Key,
            skillName: action.Name,
            potency: 0,
            value: action.Value,
            kind: action.Kind,
            actualMpCost: 0,
            castTimeSeconds: 0,
            gcdWindowSeconds: 0,
            isLegal: true,
            invalidReason: "",
            nextCooldownSeconds: 0,
            availableCharges: 1,
            maxCharges: 1,
            jobResourcesConsumed: consumed);
}

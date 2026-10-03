// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Facade;
using Combat.Sim.Models.Policy;
using Combat.Sim.Outputs;
using Combat.Sim.Outputs.TokenBuilders;

namespace Combat.Sim.Policy;

/// <summary>
/// 在真实 FightEngine 上下文之上合并 policy 候选和 policy 历史。
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
        var state = simulator.GetStateWithoutHistory();
        if (nextObservationTimestamp < state.Time)
            throw new ArgumentOutOfRangeException(nameof(nextObservationTimestamp));

        var output = simulator.FormatVectorState();
        var branch = simulator.ForkForPreview();
        var after = branch.ObserveAt(nextObservationTimestamp);
        var router = simulator.OutputRouter;
        var consumed = router.BuildNoopResourceTransition(state);

        var candidateSkills = (List<Dictionary<string, object?>>)
            output[OutputContextSchema.CandidateSkillContextKey]!;
        var candidateStateContext = (Dictionary<string, object?>)
            output[OutputContextSchema.CandidateStateContextKey]!;
        var candidateStates = (List<object>)candidateStateContext["tokens"]!;
        for (var index = _registry.Actions.Count - 1; index >= 0; index--)
        {
            var action = _registry.Actions[index];
            candidateSkills.Insert(0, BuildSkillToken(
                action,
                state.Time,
                consumed));
            candidateStates.Insert(0, router.BuildStateTransitionToken(state, after));
        }

        MergePolicyHistory(output, history, router);
        return output;
    }

    private void MergePolicyHistory(
        Dictionary<string, object?> output,
        PolicyDecisionHistory history,
        StateOutputRouter router)
    {
        var skillHistory = (List<Dictionary<string, object?>>)
            output[OutputContextSchema.SkillHistoryContextKey]!;
        var stateHistoryContext = (Dictionary<string, object?>)
            output[OutputContextSchema.StateHistoryContextKey]!;
        var stateHistory = (List<Dictionary<string, double[]>>)stateHistoryContext["tokens"]!;

        var merged = new List<(double Time, int Order, Dictionary<string, object?> Skill,
            Dictionary<string, double[]> State)>();
        for (var index = 0; index < skillHistory.Count; index++)
        {
            var time = Convert.ToDouble(skillHistory[index]["time_seconds"]);
            merged.Add((time, index, skillHistory[index], stateHistory[index]));
        }

        var order = skillHistory.Count;
        var retainedTokens = new Dictionary<PolicyDecision,
            (Dictionary<string, object?> Skill, Dictionary<string, double[]> State)>(ReferenceEqualityComparer.Instance);
        foreach (var decision in history.ReadEntries)
        {
            if (!_historyTokens.TryGetValue(decision, out var token))
            {
                var consumed = router.BuildNoopResourceTransition(decision.StateBefore);
                token = (BuildSkillToken(decision.Action, decision.Timestamp, consumed),
                    router.BuildStateTransitionToken(decision.StateBefore, decision.StateAfter));
            }
            retainedTokens.Add(decision, token);
            merged.Add((
                decision.Timestamp,
                order++,
                token.Skill,
                token.State));
        }
        // 只保留当前窗口中的条目，重置、恢复及滑动淘汰都不累积旧快照。
        _historyTokens = retainedTokens;

        merged.Sort((left, right) =>
        {
            var byTime = left.Time.CompareTo(right.Time);
            return byTime != 0 ? byTime : left.Order.CompareTo(right.Order);
        });
        history.Retention.Trim(merged);
        skillHistory.Clear();
        skillHistory.AddRange(merged.Select(item => item.Skill));
        stateHistory.Clear();
        stateHistory.AddRange(merged.Select(item => item.State));
    }

    private static Dictionary<string, object?> BuildSkillToken(
        PolicyActionDefinition action,
        double timestamp,
        IReadOnlyDictionary<string, object> consumed) =>
        SkillTokenBuilder.Build(
            skillId: action.RawId,
            skillKey: action.Key,
            skillName: action.Name,
            potency: 0,
            value: action.Value,
            kind: action.CandidateKind,
            actualMpCost: 0,
            castTimeSeconds: 0,
            gcdWindowSeconds: 0,
            isLegal: true,
            invalidReason: "",
            nextCooldownSeconds: 0,
            availableCharges: 1,
            maxCharges: 1,
            jobResourcesConsumed: consumed,
            timeSeconds: timestamp);
}

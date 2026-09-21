// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Config;
using Combat.Sim.Facade;
using Combat.Sim.Jobs;
using Combat.Sim.Models.Combat;
using Combat.Sim.Outputs.ContextBuilders;
using Combat.Sim.Outputs.TokenBuilders;
using Combat.Sim.System;

namespace Combat.Sim.Outputs;

/// <summary>
/// 统一输出上下文构建模块（对照 output_context_builder.py）：
/// 只负责定义"输出包含什么内容"并调度各类 builder，翻译交给 formatter。
/// </summary>
public sealed class OutputContextBuilder
{
    private readonly SystemStateMachine _systemMachine;
    private readonly StateContextBuilder _stateContextBuilder;
    private readonly BuffVectorTokenBuilder _buffVectorTokenBuilder;
    private readonly TargetBuffVectorTokenBuilder _targetBuffVectorTokenBuilder;
    private readonly ResourceVectorTokenBuilder _resourceVectorTokenBuilder;
    private readonly StateTokenBuilder _stateTokenBuilder;
    private readonly SkillHistoryContextBuilder _skillHistoryContextBuilder;
    private readonly StateHistoryContextBuilder _stateHistoryContextBuilder;
    private readonly CandidateSkillContextBuilder _candidateSkillContextBuilder;
    private readonly CandidateStateContextBuilder _candidateStateContextBuilder;

    public OutputContextBuilder(
        ProjectConfig project,
        SystemStateMachine systemMachine,
        IJobStateMachine jobMachine,
        int? historyLimit = null)
    {
        _systemMachine = systemMachine;
        _stateContextBuilder = new StateContextBuilder(project, systemMachine, jobMachine);
        var playerVectorTokenBuilder = new PlayerVectorTokenBuilder();
        _buffVectorTokenBuilder = new BuffVectorTokenBuilder(systemMachine);
        _targetBuffVectorTokenBuilder = new TargetBuffVectorTokenBuilder(systemMachine);
        _resourceVectorTokenBuilder = new ResourceVectorTokenBuilder(systemMachine);
        _stateTokenBuilder = new StateTokenBuilder(
            playerVectorTokenBuilder,
            _buffVectorTokenBuilder,
            _targetBuffVectorTokenBuilder,
            _resourceVectorTokenBuilder);
        _skillHistoryContextBuilder = new SkillHistoryContextBuilder(historyLimit);
        _stateHistoryContextBuilder = new StateHistoryContextBuilder(_stateTokenBuilder, historyLimit);
        _candidateSkillContextBuilder = new CandidateSkillContextBuilder();
        _candidateStateContextBuilder = new CandidateStateContextBuilder(_stateTokenBuilder);
    }

    /// <summary>装配单个状态的状态上下文原料（对照 build_state_context）。</summary>
    public StateContext BuildStateContext(CombatState state) => _stateContextBuilder.Build(state);

    internal IReadOnlyDictionary<string, object> BuildNoopResourceTransition(CombatState state)
    {
        var (_, _, consumed) = _systemMachine.BuildJobResourceTransition(state, state);
        return consumed;
    }

    internal Dictionary<string, double[]> BuildStateTransitionToken(
        CombatState before,
        CombatState after,
        IReadOnlyDictionary<string, object> consumed) =>
        _stateTokenBuilder.Build(BuildStateContext(before), BuildStateContext(after), consumed);

    /// <summary>装配 canonical 顶层上下文（对照 build_context）。</summary>
    public Dictionary<string, object?> BuildContext(
        CombatState state,
        IReadOnlyList<CandidatePreview>? candidatePreviews = null)
    {
        var candidateContextEntries = BuildCandidateContextEntries(candidatePreviews ?? Array.Empty<CandidatePreview>());
        return new Dictionary<string, object?>
        {
            ["job_tag"] = _systemMachine.RegisteredJobTag ??
                          throw new InvalidOperationException("job state is not registered"),
            ["schema_version"] = OutputContextSchema.CanonicalContextSchemaVersion,
            ["scene_context"] = SceneContextSchema.BuildEmptySceneContext(),
            ["skill_history_context"] = _skillHistoryContextBuilder.Build(state),
            ["state_history_context"] = _stateHistoryContextBuilder.Build(state),
            ["candidate_skill_context"] = _candidateSkillContextBuilder.Build(candidateContextEntries),
            ["candidate_state_context"] = _candidateStateContextBuilder.Build(candidateContextEntries),
        };
    }

    /// <summary>
    /// 给候选预演条目注入消耗量谱与 before/after 状态上下文（对照 _build_candidate_context_entries）。
    /// before/after 状态上下文按状态引用身份缓存，避免同一状态被多个候选重复装配。
    /// </summary>
    private List<CandidateContextEntry> BuildCandidateContextEntries(
        IReadOnlyList<CandidatePreview> candidatePreviews)
    {
        var beforeStateContextCache = new Dictionary<CombatState, StateContext>(ReferenceEqualityComparer.Instance);
        var afterStateContextCache = new Dictionary<CombatState, StateContext>(ReferenceEqualityComparer.Instance);
        var noopTransitionCache = new Dictionary<CombatState, IReadOnlyDictionary<string, object>>(
            ReferenceEqualityComparer.Instance);

        var entries = new List<CandidateContextEntry>();
        foreach (var preview in candidatePreviews)
        {
            var previousState = preview.PreviousState;
            if (!beforeStateContextCache.TryGetValue(previousState, out var beforeStateContext))
            {
                beforeStateContext = BuildStateContext(previousState);
                beforeStateContextCache[previousState] = beforeStateContext;
            }

            StateContext? afterStateContext = null;
            IReadOnlyDictionary<string, object> jobResourcesConsumed;
            if (preview.IsLegal)
            {
                var (_, _, consumed) = _systemMachine.BuildJobResourceTransition(
                    previousState,
                    preview.NextState);
                jobResourcesConsumed = consumed;
                var candidateAfterState = preview.CandidateAfterState ??
                                          throw new InvalidOperationException(
                                              "legal candidate must have candidate_after_state");
                if (!afterStateContextCache.TryGetValue(candidateAfterState, out var cachedAfterStateContext))
                {
                    afterStateContext = BuildStateContext(candidateAfterState);
                    afterStateContextCache[candidateAfterState] = afterStateContext;
                }
                else
                {
                    afterStateContext = cachedAfterStateContext;
                }
            }
            else
            {
                // 非法候选没有 after 状态：量谱消耗用"自身对自身"的 noop 过渡，after 上下文保持 null。
                if (!noopTransitionCache.TryGetValue(previousState, out var cachedConsumedResources))
                {
                    var (_, _, consumed) = _systemMachine.BuildJobResourceTransition(
                        previousState,
                        previousState);
                    jobResourcesConsumed = consumed;
                    noopTransitionCache[previousState] = jobResourcesConsumed;
                }
                else
                {
                    jobResourcesConsumed = cachedConsumedResources;
                }
            }

            entries.Add(new CandidateContextEntry(
                Preview: preview,
                JobResourcesConsumed: jobResourcesConsumed,
                BeforeStateContext: beforeStateContext,
                AfterStateContext: afterStateContext));
        }

        return entries;
    }
}

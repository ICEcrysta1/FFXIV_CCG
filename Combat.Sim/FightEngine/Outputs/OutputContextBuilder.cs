// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Config;
using Combat.Sim.Jobs;
using Combat.Sim.Models.Combat;
using Combat.Sim.Outputs.ContextBuilders;
using Combat.Sim.Outputs.TokenBuilders;
using Combat.Sim.System;

namespace Combat.Sim.Outputs;

/// <summary>只在输出装配内使用的已冻结历史行，保持技能、状态与执行统计对齐。</summary>
internal sealed record ModelHistoryRow(object Identity, long Sequence, ModelStateSnapshot ModelState,
    Dictionary<string, object?> Skill, Dictionary<string, double> ExecutionMetrics);

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
        _stateHistoryContextBuilder = new StateHistoryContextBuilder(_stateTokenBuilder);
    }

    /// <summary>装配单个状态的状态上下文原料（对照 build_state_context）。</summary>
    public StateContext BuildStateContext(CombatState state) => _stateContextBuilder.Build(state);

    internal IReadOnlyDictionary<string, object> BuildNoopResourceTransition(CombatState state)
    {
        var (_, _, consumed) = _systemMachine.BuildJobResourceTransition(state, state);
        return consumed;
    }

    internal IReadOnlyList<ModelHistoryRow> CollectHistory(CombatState state)
    {
        var skills = _skillHistoryContextBuilder.Build(state);
        var entries = state.History.TakeLast(skills.Count).ToArray();
        return entries.Select((entry, index) => new ModelHistoryRow(entry, entry.HistorySequence,
            entry.ModelState ?? throw new InvalidOperationException("history is missing its frozen model state"),
            skills[index], new Dictionary<string, double>
            {
                ["cumulative_potency"] = entry.StateAfter.Target.CumulativePotency,
                ["cumulative_dot_potency"] = entry.StateAfter.Target.CumulativeDotPotency,
            })).ToArray();
    }

    /// <summary>装配 canonical 顶层上下文（对照 build_context）。</summary>
    internal Dictionary<string, object?> BuildContext(
        CombatState state,
        IReadOnlyList<string> actionKeys,
        IReadOnlyList<double> actionValues,
        ModelStateSnapshot current,
        IReadOnlyList<ModelHistoryRow>? history = null)
    {
        if (actionKeys.Count != actionValues.Count)
            throw new ArgumentException("action output arrays must have matching lengths");
        history ??= CollectHistory(state);
        var historyContext = _stateTokenBuilder.BuildMetadata(actionKeys);
        historyContext["tokens"] = _stateHistoryContextBuilder.Build(history, actionKeys);
        historyContext["execution_metrics"] = history.Select(row => row.ExecutionMetrics).ToList();
        var currentContext = _stateTokenBuilder.BuildMetadata(actionKeys);
        currentContext["tokens"] = new List<Dictionary<string, double[]>> { _stateTokenBuilder.Build(current, actionKeys) };
        return new Dictionary<string, object?>
        {
            ["job_tag"] = _systemMachine.RegisteredJobTag ??
                          throw new InvalidOperationException("job state is not registered"),
            ["schema_version"] = OutputContextSchema.CanonicalContextSchemaVersion,
            // 与真实效果、已完成 policy 等待一一对应的累计行数；不受历史裁剪影响。
            ["history_cursor"] = state.LastHistorySequence,
            ["scene_context"] = SceneContextSchema.BuildEmptySceneContext(),
            ["skill_history_context"] = history.Select(row => row.Skill).ToList(),
            ["state_history_context"] = historyContext,
            ["current_state_context"] = currentContext,
            ["action_keys"] = actionKeys.ToList(),
            ["action_legal_mask"] = actionKeys.Select(key => current.RequestState.SkillAvailability[key]).ToList(),
            ["action_values"] = actionValues.ToList(),
        };
    }

}

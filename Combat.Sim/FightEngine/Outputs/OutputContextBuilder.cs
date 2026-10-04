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
        _stateHistoryContextBuilder = new StateHistoryContextBuilder(_stateTokenBuilder, historyLimit);
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
        CombatState after) =>
        _stateTokenBuilder.Build(BuildStateContext(before), BuildStateContext(after));

    /// <summary>装配 canonical 顶层上下文（对照 build_context）。</summary>
    public Dictionary<string, object?> BuildContext(
        CombatState state,
        IReadOnlyList<string> actionKeys,
        IReadOnlyList<bool> actionLegalMask,
        IReadOnlyList<double> actionValues)
    {
        if (actionKeys.Count != actionLegalMask.Count || actionKeys.Count != actionValues.Count)
            throw new ArgumentException("action output arrays must have matching lengths");
        var current = BuildStateContext(state);
        return new Dictionary<string, object?>
        {
            ["job_tag"] = _systemMachine.RegisteredJobTag ??
                          throw new InvalidOperationException("job state is not registered"),
            ["schema_version"] = OutputContextSchema.CanonicalContextSchemaVersion,
            ["scene_context"] = SceneContextSchema.BuildEmptySceneContext(),
            ["skill_history_context"] = _skillHistoryContextBuilder.Build(state),
            ["state_history_context"] = _stateHistoryContextBuilder.Build(state),
            ["current_state_context"] = new Dictionary<string, object?>
            {
                ["player_state_feature_keys"] = _stateTokenBuilder.PlayerHistoryFeatureKeys.ToList(),
                ["buff_state_feature_keys"] = _stateTokenBuilder.BuffHistoryFeatureKeys.ToList(),
                ["target_buff_state_feature_keys"] = _stateTokenBuilder.TargetBuffHistoryFeatureKeys.ToList(),
                ["resource_state_feature_keys"] = _stateTokenBuilder.ResourceHistoryFeatureKeys.ToList(),
                // 阶段 1 沿用状态 token 字段，两段均为请求时的真实当前快照。
                ["tokens"] = new List<Dictionary<string, double[]>> { _stateTokenBuilder.Build(current, current) },
            },
            ["action_keys"] = actionKeys.ToList(),
            ["action_legal_mask"] = actionLegalMask.ToList(),
            ["action_values"] = actionValues.ToList(),
        };
    }

}

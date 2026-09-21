// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Config;
using Combat.Sim.Facade;
using Combat.Sim.Jobs;
using Combat.Sim.Models.Combat;
using Combat.Sim.System;

namespace Combat.Sim.Outputs;

/// <summary>
/// 统一输出路由器（对照 outputs/router.py）：
/// 1. 状态快照输出，秒制与 GCD 制两种格式；
/// 2. canonical / tensor 两类上下文输出。
/// </summary>
public sealed class StateOutputRouter
{
    private readonly SystemStateMachine _systemMachine;
    private readonly IJobStateMachine _jobMachine;
    private readonly OutputContextBuilder _contextBuilder;
    private readonly string? _precisionConfigRoot;
    private readonly SecondsStateFormatter _secondsFormatter;
    private readonly GcdStateFormatter _gcdFormatter;
    private ModelTensorFormatter? _modelTensorFormatter;

    public StateOutputRouter(
        ProjectConfig project,
        SystemStateMachine systemMachine,
        IJobStateMachine jobMachine,
        string? precisionConfigRoot = null,
        int? historyLimit = null)
    {
        _systemMachine = systemMachine;
        _jobMachine = jobMachine;
        _precisionConfigRoot = precisionConfigRoot;
        _contextBuilder = new OutputContextBuilder(project, systemMachine, jobMachine, historyLimit);
        _secondsFormatter = new SecondsStateFormatter();
        _gcdFormatter = new GcdStateFormatter(systemMachine.JobResourceDefinitions);
    }

    /// <summary>输出单个状态快照（对照 format；mode 为 seconds / gcd）。</summary>
    public Dictionary<string, object?> Format(CombatState state, string mode)
    {
        var stateContext = _contextBuilder.BuildStateContext(state);
        return mode switch
        {
            "seconds" => _secondsFormatter.Format(stateContext),
            "gcd" => _gcdFormatter.Format(stateContext),
            _ => throw new ArgumentException(
                $"unsupported output mode: {mode}; supported=gcd, seconds", nameof(mode)),
        };
    }

    /// <summary>输出单个状态快照的 canonical 模型结果（对照 format_vectors）。</summary>
    public Dictionary<string, object?> FormatVectors(
        CombatState state,
        IReadOnlyList<CandidatePreview> candidatePreviews) =>
        ModelVectorFormatter.Format(
            _contextBuilder.BuildContext(state, candidatePreviews));

    /// <summary>输出单个状态快照的 tensor 友好模型结果（对照 format_tensors，懒加载精度配置）。</summary>
    public object? FormatTensors(
        CombatState state,
        IReadOnlyList<CandidatePreview> candidatePreviews)
    {
        _modelTensorFormatter ??= LoadTensorFormatter();
        return _modelTensorFormatter.Format(
            _contextBuilder.BuildContext(state, candidatePreviews));
    }

    /// <summary>输出状态上下文原料（对照 build_state_context）。</summary>
    public StateContext BuildStateContext(CombatState state) =>
        _contextBuilder.BuildStateContext(state);

    internal IReadOnlyDictionary<string, object> BuildNoopResourceTransition(CombatState state) =>
        _contextBuilder.BuildNoopResourceTransition(state);

    internal Dictionary<string, double[]> BuildStateTransitionToken(
        CombatState before,
        CombatState after,
        IReadOnlyDictionary<string, object> consumed) =>
        _contextBuilder.BuildStateTransitionToken(before, after, consumed);

    private ModelTensorFormatter LoadTensorFormatter()
    {
        if (_precisionConfigRoot is null)
        {
            throw new InvalidOperationException(
                "tensor output requires a precision config root; construct the machine with a project root");
        }

        var precision = PrecisionConfigLoader.Load(_precisionConfigRoot);
        return new ModelTensorFormatter(precision.IntDtype, precision.FloatDtype);
    }
}

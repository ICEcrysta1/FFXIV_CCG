// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Timeline;
using Combat.Sim.Outputs;
using Combat.Sim.System.Timeline;

namespace Combat.Sim.Facade;

/// <summary>
/// 面向宿主的绝对时间模拟器。它持有唯一的战斗游标和事件队列；动作、外部事实、
/// 观测、快照与分支都必须经过此对象，不能绕过时间线直接改写状态。
/// </summary>
public sealed class JobSimulator
{
    private readonly CombatStateMachine _machine;
    private readonly CombatTimelineRuntime _timeline;
    private readonly HistoryRetention _historyRetention;
    private StateOutputRouter? _outputRouter;
    internal StateOutputRouter OutputRouter => _outputRouter ??= _machine.CreateOutputRouter();
    internal HistoryRetention HistoryRetention => _historyRetention;

    public JobSimulator(
        CombatStateMachine machine,
        double initialTimestamp = 0,
        double? fightRemaining = null)
        : this(machine, machine.InitialState(fightRemaining, startTime: initialTimestamp))
    {
    }

    internal JobSimulator(CombatStateMachine machine, CombatState initialState, HistoryRetention? historyRetention = null)
    {
        _machine = machine;
        _historyRetention = historyRetention ?? new HistoryRetention(machine.MaxHistory);
        _timeline = BuildTimeline(initialState);
    }

    private JobSimulator(CombatStateMachine machine, CombatTimelineRuntime timeline, HistoryRetention historyRetention)
    {
        _machine = machine;
        _timeline = timeline;
        _historyRetention = historyRetention;
    }

    public string JobTag => _machine.JobTag;
    public double Time => _timeline.CurrentTime;
    public CombatState GetState() => _timeline.GetState();
    internal CombatState GetStateWithoutHistory() => _timeline.GetStateWithoutHistory();
    internal void AdvanceClockTo(double timestamp) => _timeline.AdvanceClockTo(timestamp);

    /// <summary>在绝对请求时刻提交真实游戏动作。</summary>
    public ActionSubmissionResult SubmitAction(double timestamp, string skillKey) =>
        SubmitAction(new ActionRequest(timestamp, skillKey));

    public ActionSubmissionResult SubmitAction(
        double timestamp,
        string skillKey,
        double? actualCastSeconds) =>
        SubmitAction(new ActionRequest(timestamp, skillKey), actualCastSeconds);

    public ActionSubmissionResult SubmitAction(ActionRequest request)
        => SubmitAction(request, actualCastSeconds: null);

    private ActionSubmissionResult SubmitAction(
        ActionRequest request,
        double? actualCastSeconds)
    {
        ArgumentNullException.ThrowIfNull(request);
        var skill = _machine.ResolveSkill(request.SkillKey);
        _timeline.AdvanceClockTo(request.Timestamp);

        var requestState = _timeline.GetStateWithoutHistory();
        var submission = _machine.EvaluateActionSubmission(
            requestState,
            skill,
            queueOccupied: HasQueuedAction(),
            actualCastSecondsOverride: actualCastSeconds);
        if (!submission.Accepted)
        {
            return new(
                Accepted: false,
                Queued: false,
                Reason: submission.Reason,
                ActionInstanceId: null,
                RequestTimestamp: request.Timestamp,
                AcceptedTimestamp: null,
                EffectTimestamp: null,
                NextScheduledEventTime: _timeline.GetNextScheduledEventTime());
        }

        var actionId = Guid.NewGuid();
        var timing = _machine.BuildActionTimingPlan(
            requestState,
            skill,
            actualCastSeconds);
        var modelState = RecordModelDecision(actionId, requestState, completed: false).Snapshot;
        var payload = new ActionLifecyclePayload(
            actionId,
            request,
            skill,
            requestState,
            timing,
            submission.AcceptedTimestamp,
            modelState);
        _timeline.Schedule(new TimelineEvent(
            submission.AcceptedTimestamp,
            TimelineEventPriority.ActionAccepted,
            TimelineEventKind.ActionAccepted,
            skill.Key,
            actionId,
            payload));

        if (!submission.Queued)
        {
            // 同刻命令在已有到期事实结算后接受；事件处理器仍是唯一动作状态写入点。
            _timeline.AdvanceClockTo(request.Timestamp);
        }

        return new(
            Accepted: true,
            Queued: submission.Queued,
            Reason: "",
            ActionInstanceId: actionId,
            RequestTimestamp: request.Timestamp,
            AcceptedTimestamp: submission.AcceptedTimestamp,
            EffectTimestamp: submission.AcceptedTimestamp + timing.EffectDelaySeconds,
            NextScheduledEventTime: _timeline.GetNextScheduledEventTime());
    }

    /// <summary>把逻辑时钟推进到绝对时刻并排空所有到期事件。</summary>
    public CombatState AdvanceTo(double timestamp)
    {
        return _timeline.AdvanceTo(timestamp);
    }

    public CombatState ObserveAt(double timestamp) => AdvanceTo(timestamp);

    public ValidationResult ValidateActionAt(double timestamp, string skillKey)
    {
        var skill = _machine.ResolveSkill(skillKey);
        _timeline.AdvanceClockTo(timestamp);
        if (HasQueuedAction())
        {
            return new ValidationResult(false, "action_queue_occupied");
        }
        return _machine.ValidateAction(_timeline.GetStateWithoutHistory(), skill);
    }

    public IReadOnlyList<string> AvailableActionKeysAt(double timestamp)
    {
        _timeline.AdvanceClockTo(timestamp);
        if (HasQueuedAction())
        {
            return Array.Empty<string>();
        }
        return _machine.AvailableActionKeys(_timeline.GetStateWithoutHistory());
    }

    public double? GetNextScheduledEventTime() => _timeline.GetNextScheduledEventTime();

    /// <summary>提交带绝对时间戳的外部战斗事实。</summary>
    public ExternalEventResult ApplyExternalEvent(ExternalCombatEvent externalEvent)
    {
        ArgumentNullException.ThrowIfNull(externalEvent);
        _machine.SystemMachine.ValidateExternalEvent(externalEvent);
        _timeline.Schedule(new TimelineEvent(
            externalEvent.Timestamp,
            TimelineEventPriority.ExternalScene,
            TimelineEventKind.SceneChanged,
            externalEvent.Kind,
            Payload: externalEvent));
        _timeline.AdvanceClockTo(externalEvent.Timestamp);
        return new(true, "", externalEvent.Timestamp);
    }

    public SimulationSnapshot CreateSnapshot() => _timeline.CreateSnapshot();

    public void RestoreSnapshot(SimulationSnapshot snapshot) => _timeline.RestoreSnapshot(snapshot.DeepClone());

    public JobSimulator Fork() => new(_machine, _timeline.Fork(), _historyRetention);
    // 策略历史快照不携带历史前缀，待结算事件与队列载荷仍完整隔离。
    internal JobSimulator ForkWithoutHistory() => new(_machine, _timeline.Fork(includeHistory: false), _historyRetention);

    public Dictionary<string, object?> FormatState(string mode = "seconds") =>
        OutputRouter.Format(GetState(), mode);

    public Dictionary<string, object?> FormatVectorState() => FormatVectorState(GetState());

    internal Dictionary<string, object?> FormatVectorState(CombatState state)
    {
        var actions = BuildActionOutput(state);
        return OutputRouter.FormatVectors(state, actions.Keys, actions.LegalMask, actions.Values);
    }

    public object? FormatTensorState()
    {
        var state = GetState();
        var actions = BuildActionOutput(state);
        return OutputRouter.FormatTensors(state, actions.Keys, actions.LegalMask, actions.Values);
    }

    private (List<string> Keys, List<bool> LegalMask, List<double> Values) BuildActionOutput(CombatState state)
    {
        var skills = _machine.SkillBook.EnabledSkills().OrderBy(skill => skill.Key, StringComparer.Ordinal).ToArray();
        var queueOccupied = HasQueuedAction();
        // 与真实 SubmitAction 使用同一接受规则；仅判断请求，不推进技能生效或复制未来状态。
        return (skills.Select(skill => skill.Key).ToList(),
            skills.Select(skill => _machine.EvaluateActionSubmission(state, skill, queueOccupied).Accepted).ToList(),
            skills.Select(skill => _machine.JobMachine.ResolveActionValue(state, skill, skill.Value)).ToList());
    }

    internal CombatStateMachine Rules => _machine;

    /// <summary>真实动作和 policy 动作共用请求关联与状态冻结；不改变战斗时钟。</summary>
    internal (ModelStateSnapshot Snapshot, long HistorySequence) RecordModelDecision(
        Guid decisionId, CombatState requestState, bool completed)
    {
        var modelState = ModelStateSnapshot.Capture(
            requestState.LastDecisionAfter, OutputRouter.BuildStateContext(requestState));
        long historySequence = 0;
        _timeline.ApplyMutation(new TimelineMutation(ApplyState: state =>
        {
            state.LastDecisionId = decisionId;
            state.LastDecisionAfter = completed ? modelState.RequestState : null;
            // 等待在决策落实时进入历史；真实技能在后续生效记录时领取序号。
            if (completed) historySequence = state.ReserveHistorySequence();
        }));
        return (modelState, historySequence);
    }

    internal (int History, int PendingEvents, int QueueEntries, int PendingSettlements) GetStatistics() =>
        (_timeline.HistoryCount, _timeline.PendingEventCount,
            _timeline.QueueEntryCount, _timeline.PendingSettlementCount);

    internal bool HasQueuedAction() => _timeline.HasQueuedAction();

    private CombatTimelineRuntime BuildTimeline(CombatState state)
    {
        var timeline = _machine.SystemMachine.CreateTimeline(
            state,
            key => _machine.ResolveSkill(key).Charges);
        timeline.RegisterHandler(TimelineEventKind.ActionAccepted, HandleActionAccepted);
        timeline.RegisterHandler(TimelineEventKind.CastCompleted, HandleCastCompleted);
        timeline.RegisterHandler(TimelineEventKind.ActionEffect, HandleActionEffect);
        return timeline;
    }

    private TimelineMutation HandleActionAccepted(TimelineEvent item, CombatState state)
    {
        var payload = (ActionLifecyclePayload)item.Payload!;
        var events = new List<TimelineEvent>
        {
            new(
                item.Timestamp + payload.Timing.EffectDelaySeconds,
                TimelineEventPriority.ConfirmedActionEffect,
                TimelineEventKind.ActionEffect,
                payload.Skill.Key,
                payload.ActionInstanceId,
                payload),
        };
        if (payload.Timing.ActualCastSeconds > payload.Timing.EffectDelaySeconds + CombatTimelineRuntime.TimeEpsilon)
        {
            events.Add(new TimelineEvent(
                item.Timestamp + payload.Timing.ActualCastSeconds,
                TimelineEventPriority.DecisionBoundary,
                TimelineEventKind.CastCompleted,
                payload.Skill.Key,
                payload.ActionInstanceId,
                payload));
        }
        return new TimelineMutation(
            ApplyState: target => _machine.ApplyActionTiming(
                target,
                payload.Skill,
                payload.Timing.EffectiveGcdSeconds,
                payload.Timing.ActualCastSeconds,
                payload.Request.Timestamp),
            EventsToSchedule: events);
    }

    private static TimelineMutation HandleCastCompleted(TimelineEvent item, CombatState state)
    {
        // 读条剩余量由绝对截止时刻派生；该事件只提供完整读条结束的稳定调度边界。
        return TimelineMutation.Empty;
    }

    private TimelineMutation HandleActionEffect(TimelineEvent item, CombatState state)
    {
        var payload = (ActionLifecyclePayload)item.Payload!;
        return new TimelineMutation(ApplyState: target =>
        {
            _machine.ApplyActionEffect(payload.RequestState, target, payload.Skill);
            _machine.RecordTimelineActionHistory(
                payload.RequestState,
                target,
                payload.Skill,
                payload.Timing,
                payload.ActionInstanceId,
                payload.Request.Timestamp,
                payload.AcceptedTimestamp + payload.Timing.ActualCastSeconds,
                item.Timestamp,
                payload.ModelState);
            _historyRetention.Trim(target.History);
        });
    }
}

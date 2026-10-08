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
    private readonly SkillAvailabilityBuilder _availability;
    private static readonly IReadOnlySet<TimelineEventKind> InstanceHandlers = new HashSet<TimelineEventKind>
    {
        TimelineEventKind.ActionAccepted, TimelineEventKind.CastCompleted, TimelineEventKind.ActionEffect,
    };
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
        _availability = new SkillAvailabilityBuilder(machine);
        _historyRetention = historyRetention ?? new HistoryRetention(machine.MaxHistory);
        _timeline = BuildTimeline(initialState);
    }

    private JobSimulator(CombatStateMachine machine, CombatTimelineRuntime timeline,
        HistoryRetention historyRetention, SkillAvailabilityBuilder availability)
    {
        _machine = machine;
        _availability = availability;
        _timeline = timeline;
        _historyRetention = historyRetention;
        BindTimelineHandlers(timeline);
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
        var submission = _machine.EvaluateActionSubmission(_timeline.GetStateWithoutHistory(), skill, HasQueuedAction());
        return new ValidationResult(submission.Accepted, submission.Reason);
    }

    public IReadOnlyList<string> AvailableActionKeysAt(double timestamp)
    {
        _timeline.AdvanceClockTo(timestamp);
        var availability = _availability.Build(_timeline.GetStateWithoutHistory(), HasQueuedAction());
        return _availability.ActionKeys.Where(key => availability[key]).ToArray();
    }

    public double? GetNextScheduledEventTime() => _timeline.GetNextScheduledEventTime();

    /// <summary>同刻事实全部校验、全部入队后再推进，效果不能读取半更新的场景。</summary>
    public ExternalEventResult ApplyExternalEvents(IReadOnlyList<ExternalCombatEvent> externalEvents)
    {
        ArgumentNullException.ThrowIfNull(externalEvents);
        if (externalEvents.Count == 0) throw new ArgumentException("external event batch must not be empty", nameof(externalEvents));
        var batch = externalEvents.ToArray();
        ArgumentNullException.ThrowIfNull(batch[0]);
        var timestamp = batch[0].Timestamp;
        var kinds = new HashSet<string>(StringComparer.Ordinal);
        foreach (var item in batch)
        {
            ArgumentNullException.ThrowIfNull(item);
            if (!double.IsFinite(item.Timestamp) || item.Timestamp < 0 || item.Timestamp < Time
                || item.Timestamp != timestamp)
                throw new ArgumentException("external event batch requires one finite, non-past timestamp", nameof(externalEvents));
            if (!kinds.Add(item.Kind))
                throw new ArgumentException($"conflicting external facts: {item.Kind}", nameof(externalEvents));
            if (item.RemainingSeconds is { } remaining && !double.IsFinite(remaining))
                throw new ArgumentException("external event duration must be finite", nameof(externalEvents));
            _machine.SystemMachine.ValidateExternalEvent(item);
        }
        // 保留调用方的稳定顺序，同刻全部 ExternalScene 均先于动作效果。
        foreach (var item in batch)
            _timeline.Schedule(new TimelineEvent(timestamp, TimelineEventPriority.ExternalScene,
                TimelineEventKind.SceneChanged, item.Kind, Payload: item));
        _timeline.AdvanceClockTo(timestamp);
        return new(true, "", timestamp);
    }

    public SimulationSnapshot CreateSnapshot() => _timeline.CreateSnapshot();

    public void RestoreSnapshot(SimulationSnapshot snapshot) => _timeline.RestoreSnapshot(snapshot.DeepClone());

    public JobSimulator Fork() => ForkCore(includeHistory: true);
    // 策略历史快照不携带历史前缀，待结算事件与队列载荷仍完整隔离。
    internal JobSimulator ForkWithoutHistory() => ForkCore(includeHistory: false);

    private JobSimulator ForkCore(bool includeHistory) =>
        new(_machine, _timeline.Fork(includeHistory, InstanceHandlers), _historyRetention, _availability);

    public Dictionary<string, object?> FormatState(string mode = "seconds") =>
        OutputRouter.Format(GetState(), mode);

    public Dictionary<string, object?> FormatVectorState() => FormatVectorState(GetState());

    internal Dictionary<string, object?> FormatVectorState(CombatState state)
    {
        var snapshot = CaptureCurrentModelState(state);
        return OutputRouter.FormatVectors(state, _availability.ActionKeys, BuildActionValues(state), snapshot);
    }

    public object? FormatTensorState()
    {
        var state = GetState();
        var snapshot = CaptureCurrentModelState(state);
        return OutputRouter.FormatTensors(state, _availability.ActionKeys, BuildActionValues(state), snapshot);
    }

    internal IReadOnlyList<string> ActionKeys => _availability.ActionKeys;
    internal IReadOnlyList<double> BuildActionValues(CombatState state) =>
        _availability.Skills.Select(skill => _machine.JobMachine.ResolveActionValue(state, skill, skill.Value)).ToArray();

    internal ModelStateFrame CaptureModelStateFrame(CombatState state) =>
        ModelStateFrame.Freeze(OutputRouter.BuildStateContext(state), _availability.Build(state, HasQueuedAction()));

    internal ModelStateSnapshot CaptureCurrentModelState(CombatState state) =>
        ModelStateSnapshot.Capture(state.LastDecisionAfter, CaptureModelStateFrame(state));

    private void CompleteModelDecision(Guid decisionId, CombatState state)
    {
        // 仅新请求自己的效果可更新基准；捕获 mutation 已变更状态，不能重新读取旧 timeline 状态。
        if (state.LastDecisionId == decisionId)
            state.LastDecisionAfter = CaptureModelStateFrame(state);
    }

    internal CombatStateMachine Rules => _machine;

    /// <summary>真实动作和 policy 动作共用请求关联与状态冻结；不改变战斗时钟。</summary>
    internal (ModelStateSnapshot Snapshot, long HistorySequence) RecordModelDecision(
        Guid decisionId, CombatState requestState, bool completed)
    {
        var modelState = CaptureCurrentModelState(requestState);
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
        BindTimelineHandlers(timeline);
        return timeline;
    }

    private void BindTimelineHandlers(CombatTimelineRuntime timeline)
    {
        timeline.RegisterHandler(TimelineEventKind.ActionAccepted, HandleActionAccepted);
        timeline.RegisterHandler(TimelineEventKind.CastCompleted, HandleCastCompleted);
        timeline.RegisterHandler(TimelineEventKind.ActionEffect, HandleActionEffect);
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
            CompleteModelDecision(payload.ActionInstanceId, target);
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

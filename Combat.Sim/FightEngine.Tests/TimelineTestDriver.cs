using Combat.Sim.Common;
using Combat.Sim.Facade;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Timeline;

namespace Combat.Sim.Facade;

/// <summary>
/// 测试专用的绝对时间驱动器。所有执行都通过 JobSimulator 的正式协议完成。
/// </summary>
internal static class TimelineTestDriver
{
    internal sealed record ActionResult(
        CombatState PreviousState,
        CombatState NextState,
        ValidationResult Validation,
        double GcdUnitSeconds,
        double ActualOccupancySeconds,
        double ActualOccupancyGcd,
        string ActualOccupancySource,
        double GcdWindowSeconds,
        double GcdWindowGcd,
        double NextGcdWindowSeconds,
        double NextGcdWindowGcd);

    public static ActionResult Execute(CombatStateMachine machine, CombatState state, string skillKey)
    {
        var previous = state.Clone();
        var skill = machine.ResolveSkill(skillKey);
        var timing = machine.BuildActionTimingPlan(previous, skill);
        var simulator = new JobSimulator(machine, previous);
        var submission = simulator.SubmitAction(previous.Time, skillKey);
        if (!submission.Accepted)
        {
            throw new InvalidOperationException($"{skillKey}: {submission.Reason}");
        }

        simulator.AdvanceTo(submission.EffectTimestamp ?? previous.Time);
        var next = simulator.GetState();
        var remainingWindow = Math.Max(
            0,
            previous.Time + timing.NextGcdWindowSeconds - next.Time);
        return new ActionResult(
            previous,
            next,
            new ValidationResult(true),
            timing.GcdUnitSeconds,
            timing.ActualOccupancySeconds,
            GcdUnits.ToGcdUnits(timing.ActualOccupancySeconds, timing.GcdUnitSeconds),
            timing.ActualOccupancySource,
            timing.GcdWindowSeconds,
            GcdUnits.ToGcdUnits(timing.GcdWindowSeconds, timing.GcdUnitSeconds),
            remainingWindow,
            GcdUnits.ToGcdUnits(remainingWindow, timing.GcdUnitSeconds));
    }

    public static CombatState AdvanceBy(
        CombatStateMachine machine,
        CombatState state,
        double seconds,
        Func<double, bool>? targetableAt = null)
    {
        var timeline = machine.SystemMachine.CreateTimeline(
            state,
            key => machine.ResolveSkill(key).Charges,
            targetableAt);
        return timeline.AdvanceTo(state.Time + seconds);
    }

    public static IReadOnlyList<CandidatePreview> BuildCandidatePreviews(
        CombatStateMachine machine,
        CombatState state) => new JobSimulator(machine, state).BuildCandidatePreviews();

    public static Dictionary<string, object?> FormatVectorState(
        CombatStateMachine machine,
        CombatState state) => new JobSimulator(machine, state).FormatVectorState();

    public static Dictionary<string, object?> FormatState(
        CombatStateMachine machine,
        CombatState state,
        string mode = "seconds") => new JobSimulator(machine, state).FormatState(mode);

    public static object? FormatTensorState(
        CombatStateMachine machine,
        CombatState state) => new JobSimulator(machine, state).FormatTensorState();

    public static Dictionary<string, object?> FormatResult(
        CombatStateMachine machine,
        ActionResult result) => FormatVectorState(machine, result.NextState);

    public static ReplayStateCache CreateReplayCache(
        CombatStateMachine machine,
        CombatState? initialState = null) =>
        new(new JobSimulator(machine, initialState ?? machine.InitialState()));
}

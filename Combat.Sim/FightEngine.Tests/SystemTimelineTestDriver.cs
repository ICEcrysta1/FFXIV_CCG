using Combat.Sim.Models.Combat;

namespace Combat.Sim.System;

internal static class SystemTimelineTestDriver
{
    public static CombatState AdvanceBy(
        SystemStateMachine machine,
        CombatState state,
        double seconds,
        Func<string, int>? getMaxCharges = null)
    {
        var timeline = machine.CreateTimeline(state, getMaxCharges ?? (_ => 1));
        return timeline.AdvanceTo(state.Time + seconds);
    }
}

// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Jobs.Timeline;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.System.Registries;
using Combat.Sim.System.Timeline;

namespace Combat.Sim.Jobs;

/// <summary>职业自己的状态规则与定义；运行中的可变值只保存在各队列的 JobState 中。</summary>
public sealed class JobStateRuntime
{
    public ResourceStateRegistry Resources { get; } = new();
    public BuffStateRegistry Buffs { get; } = new(state => state.Job.Statuses);
    public StatusTimelineRuntime StatusTimeline { get; } = new(state => state.Job.Statuses);
    public JobTimelineRegistry Timeline { get; } = new();
    public string JobTag => Resources.RegisteredJobTag ??
        throw new InvalidOperationException("job state is not registered");

    public void Register(JobStateRegistration registration, IReadOnlyDictionary<string, double>? resourceLimits = null)
    {
        Buffs.RegisterStatuses(registration.Statuses);
        Resources.RegisterJobResources(registration.JobTag, registration.Resources, resourceLimits);
    }

    public void Initialize(CombatState state) => state.Job.Resources = Resources.BuildInitialJobResources();

    public Dictionary<string, object> BuildResourceSnapshot(CombatState state)
    {
        var snapshot = Resources.BuildJobResourceSnapshot(state);
        foreach (var (projection, deadline) in Timeline.ActiveTimerDeadlines(state))
        {
            if (snapshot.ContainsKey(projection.OutputKey))
                snapshot[projection.OutputKey] = Resources.NormalizeResourceValue(
                    projection.OutputKey, projection.Project(state, deadline));
        }
        return snapshot;
    }

    public (IReadOnlyDictionary<string, object> Before, IReadOnlyDictionary<string, object> After,
        IReadOnlyDictionary<string, object> Consumed) BuildResourceTransition(
        CombatState previousState, CombatState nextState) =>
        Resources.BuildJobResourceTransition(BuildResourceSnapshot(previousState), BuildResourceSnapshot(nextState));
}

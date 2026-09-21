// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Timeline;

namespace Combat.Sim.Facade;

/// <summary>保存完整模拟快照的增量回放缓存。</summary>
public sealed class ReplayStateCache
{
    private readonly JobSimulator _simulator;
    private readonly List<SimulationSnapshot> _snapshots = new();

    public ReplayStateCache(JobSimulator simulator)
    {
        ArgumentNullException.ThrowIfNull(simulator);
        _simulator = simulator.Fork();
    }

    public CombatState CurrentState => _simulator.GetState();

    public SimulationSnapshot PushSnapshot()
    {
        var snapshot = _simulator.CreateSnapshot().DeepClone();
        _snapshots.Add(snapshot);
        return snapshot.DeepClone();
    }

    public SimulationSnapshot SnapshotAt(int index) => _snapshots[index].DeepClone();

    public CombatState AdvanceTo(double timestamp, bool snapshot = false)
    {
        var state = _simulator.AdvanceTo(timestamp);
        if (snapshot)
        {
            PushSnapshot();
        }
        return state;
    }

    public ActionSubmissionResult SubmitAction(ActionRequest request, bool snapshot = true)
    {
        var result = _simulator.SubmitAction(request);
        if (snapshot)
        {
            PushSnapshot();
        }
        return result;
    }

    public void RestoreSnapshot(SimulationSnapshot snapshot) => _simulator.RestoreSnapshot(snapshot);
}

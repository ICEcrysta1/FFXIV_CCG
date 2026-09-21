// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Combat;

/// <summary>冷却桶保存绝对回充时刻，剩余时间按当前观测时刻派生。</summary>
public sealed class CooldownState
{
    internal double ObservedAt;
    private bool _bound;
    public int AvailableCharges { get; set; }
    public List<double> RechargeReadyAt { get; internal set; } = new();
    public IReadOnlyList<double> RechargeTimers => RechargeReadyAt.Select(t => Math.Max(0, t - ObservedAt)).ToArray();
    public CooldownState(int availableCharges, List<double>? rechargeTimers = null)
    {
        AvailableCharges = availableCharges;
        RechargeReadyAt = rechargeTimers ?? new();
    }
    internal void BindTime(double time)
    {
        if (!_bound)
        {
            RechargeReadyAt = RechargeReadyAt.Select(t => t + time).ToList();
            _bound = true;
        }
        ObservedAt = time;
    }
    public CooldownState Clone()
    {
        var clone = (CooldownState)MemberwiseClone();
        clone.RechargeReadyAt = new(RechargeReadyAt);
        return clone;
    }
}

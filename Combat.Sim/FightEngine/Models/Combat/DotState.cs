// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Combat;

/// <summary>DoT 保存绝对过期和下次结算时刻，剩余时间按当前观测时刻派生。</summary>
public sealed class DotState
{
    internal double ObservedAt;
    private bool _bound;
    public double ExpiresAt { get; internal set; }
    public double Remaining { get => Math.Max(0, ExpiresAt - ObservedAt); set => ExpiresAt = ObservedAt + value; }
    public double PotencyPerTick { get; set; }
    public double TickInterval { get; set; }
    public double NextTickAt { get; internal set; }
    public double NextTickInSeconds { get => Math.Max(0, NextTickAt - ObservedAt); set => NextTickAt = ObservedAt + value; }
    public DotState(double remaining, double potencyPerTick, double tickInterval = 3.0, double nextTickInSeconds = 3.0)
    {
        Remaining = remaining;
        PotencyPerTick = potencyPerTick;
        TickInterval = tickInterval;
        NextTickInSeconds = nextTickInSeconds;
    }
    internal void BindTime(double time)
    {
        if (!_bound)
        {
            ExpiresAt += time;
            NextTickAt += time;
            _bound = true;
        }
        ObservedAt = time;
    }
    public DotState Clone() => (DotState)MemberwiseClone();
}

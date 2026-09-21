// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Combat;

/// <summary>状态保存绝对过期时刻，剩余时间按当前观测时刻派生。</summary>
public sealed class StatusState
{
    internal double ObservedAt;
    private bool _bound;
    public double ExpiresAt { get; internal set; }
    public double Remaining { get => Math.Max(0, ExpiresAt - ObservedAt); set => ExpiresAt = ObservedAt + value; }
    public int Stacks { get; set; }
    public StatusState(double remaining, int stacks = 1)
    {
        Remaining = remaining;
        Stacks = stacks;
    }
    internal void BindTime(double time)
    {
        if (!_bound)
        {
            ExpiresAt += time;
            _bound = true;
        }
        ObservedAt = time;
    }
    public StatusState Clone() => (StatusState)MemberwiseClone();
}

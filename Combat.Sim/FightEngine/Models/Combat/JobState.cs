// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Combat;

/// <summary>单个队列的职业状态；公共玩家状态、冷却和目标 DoT 不属于此容器。</summary>
public sealed class JobState
{
    public Dictionary<string, object> Resources { get; set; } = new();
    public Dictionary<string, StatusState> Statuses { get; set; } = new();
    // 绝对截止时刻是职业计时器的权威值，量谱里的剩余时间只作输出投影。
    public Dictionary<string, double> TimelineDeadlines { get; set; } = new();

    internal void BindTime(double time)
    {
        foreach (var status in Statuses.Values) status.BindTime(time);
    }

    public JobState Clone() => new()
    {
        Resources = new(Resources),
        Statuses = Statuses.ToDictionary(pair => pair.Key, pair => pair.Value.Clone()),
        TimelineDeadlines = new(TimelineDeadlines),
    };

    public object? GetResource(string key, object? fallback = null) =>
        Resources.TryGetValue(key, out var value) ? value : fallback;

    public void SetResource(string key, object value) => Resources[key] = value;

    public bool HasStatus(string key) =>
        Statuses.TryGetValue(key, out var status) && status.Remaining > 0 && status.Stacks > 0;

    public int StatusStacks(string key) =>
        Statuses.TryGetValue(key, out var status) && status.Remaining > 0 ? status.Stacks : 0;
}

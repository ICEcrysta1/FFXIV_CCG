// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using static Combat.Sim.System.SystemConstants;

namespace Combat.Sim.System.Registries;

/// <summary>
/// 系统层 Buff 注册中心：统一托管系统 Buff 与职业 Buff 定义，
/// 并处理共享 grant_status 行为的校验与应用（对照 buff_state.BuffStateRegistry）。
/// </summary>
public sealed class BuffStateRegistry
{
    public Dictionary<string, StatusDefinition> SystemStatusDefinitions { get; private set; } = new();
    public Dictionary<string, StatusDefinition> JobStatusDefinitions { get; private set; } = new();
    public string? RegisteredJobTag { get; private set; }

    public void RegisterSystemStatuses(IReadOnlyDictionary<string, StatusDefinition> statuses)
    {
        SystemStatusDefinitions = new Dictionary<string, StatusDefinition>(statuses, StringComparer.Ordinal);
    }

    public void RegisterJobStatuses(string jobTag, IReadOnlyDictionary<string, StatusDefinition> statuses)
    {
        RegisteredJobTag = jobTag;
        JobStatusDefinitions = new Dictionary<string, StatusDefinition>(statuses, StringComparer.Ordinal);
    }

    /// <summary>系统层是否识别这个共享 Buff 行为名。</summary>
    public bool SupportsBehavior(string behavior) => behavior == "grant_status";

    public ValidationResult ValidateSharedBehavior(CombatState state, SkillDefinition skill)
    {
        if (skill.AppliesStatuses.Count == 0 || skill.Behavior != "grant_status")
        {
            return new ValidationResult(true);
        }
        foreach (var statusKey in skill.AppliesStatuses)
        {
            var definition = StatusDefinitionOf(statusKey);
            if (!state.Statuses.TryGetValue(statusKey, out var current))
            {
                continue;
            }
            if (current.Remaining > 0 && current.Stacks >= definition.MaxStacks)
            {
                return new ValidationResult(false, "status_already_active");
            }
        }
        return new ValidationResult(true);
    }

    /// <summary>应用系统层共享 Buff 行为（grant_status），返回是否处理了。</summary>
    public bool ApplySharedBehavior(CombatState state, SkillDefinition skill)
    {
        if (skill.AppliesStatuses.Count == 0)
        {
            return false;
        }
        foreach (var statusKey in skill.AppliesStatuses)
        {
            GrantRegisteredStatus(state, statusKey);
        }
        return true;
    }

    public StatusDefinition StatusDefinitionOf(string key)
    {
        if (SystemStatusDefinitions.TryGetValue(key, out var systemDefinition))
        {
            return systemDefinition;
        }
        if (JobStatusDefinitions.TryGetValue(key, out var jobDefinition))
        {
            return jobDefinition;
        }
        throw new KeyNotFoundException($"unregistered status: {key}");
    }

    /// <summary>
    /// 授予已注册状态：默认使用配置的持续时间和最大层数；
    /// 外部环境输入（如实时剩余时间）也可以直接写入。
    /// </summary>
    public void GrantRegisteredStatus(CombatState state, string key, double? remaining = null, int? stacks = null)
    {
        GrantStatus(state, StatusDefinitionOf(key), key, remaining, stacks);
    }

    public void ClearRegisteredStatus(CombatState state, string key) => state.Statuses.Remove(key);

    public void ConsumeRegisteredStatusStack(CombatState state, string key)
    {
        if (!state.Statuses.TryGetValue(key, out var status))
        {
            return;
        }
        status.Stacks -= 1;
        if (status.Stacks <= 0)
        {
            ClearRegisteredStatus(state, key);
        }
    }

    private static void GrantStatus(
        CombatState state,
        StatusDefinition definition,
        string key,
        double? remaining,
        int? stacks)
    {
        var remainingSeconds = remaining ?? definition.Duration;
        remainingSeconds = Math.Max(0.0, remainingSeconds);
        var stackCount = stacks ?? definition.MaxStacks;
        stackCount = Math.Max(0, Math.Min(definition.MaxStacks, stackCount));
        if (remainingSeconds <= 0.0 || stackCount <= 0)
        {
            state.Statuses.Remove(key);
            return;
        }
        state.Statuses[key] = new StatusState(remainingSeconds, stackCount);
        state.Statuses[key].BindTime(state.Time);
    }
}

// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;

namespace Combat.Sim.State;

/// <summary>
/// 单一所有者的 Buff 注册中心：只操作构造时指定的状态容器，
/// 负责单个状态的层数校验、授予、清除和消耗；技能级路由由门面处理。
/// </summary>
public sealed class BuffStateRegistry
{
    private readonly Func<CombatState, Dictionary<string, StatusState>> _statuses;

    public IReadOnlyDictionary<string, StatusDefinition> Definitions { get; private set; } =
        new Dictionary<string, StatusDefinition>(StringComparer.Ordinal);

    public BuffStateRegistry(Func<CombatState, Dictionary<string, StatusState>>? statuses = null)
    {
        _statuses = statuses ?? (state => state.Statuses);
    }

    public void RegisterStatuses(IReadOnlyDictionary<string, StatusDefinition> statuses)
    {
        Definitions = new Dictionary<string, StatusDefinition>(statuses, StringComparer.Ordinal);
    }

    /// <summary>是否识别这个共享 Buff 行为名。</summary>
    public bool SupportsBehavior(string behavior) => behavior == "grant_status";

    /// <summary>只在所属容器内检查状态，门面只负责按定义路由。</summary>
    public ValidationResult ValidateRegisteredStatus(CombatState state, string key)
    {
        var definition = StatusDefinitionOf(key);
        return _statuses(state).TryGetValue(key, out var current) &&
               current.Remaining > 0 && current.Stacks >= definition.MaxStacks
            ? new ValidationResult(false, "status_already_active")
            : new ValidationResult(true);
    }

    public StatusDefinition StatusDefinitionOf(string key)
    {
        if (Definitions.TryGetValue(key, out var definition))
        {
            return definition;
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

    public void ClearRegisteredStatus(CombatState state, string key) => _statuses(state).Remove(key);

    public void ConsumeRegisteredStatusStack(CombatState state, string key)
    {
        if (!_statuses(state).TryGetValue(key, out var status))
        {
            return;
        }
        status.Stacks -= 1;
        if (status.Stacks <= 0)
        {
            ClearRegisteredStatus(state, key);
        }
    }

    private void GrantStatus(
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
            _statuses(state).Remove(key);
            return;
        }
        var status = new StatusState(remainingSeconds, stackCount);
        status.BindTime(state.Time);
        _statuses(state)[key] = status;
    }
}

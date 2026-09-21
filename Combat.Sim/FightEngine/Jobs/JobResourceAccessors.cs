// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Combat;
using Combat.Sim.System;

namespace Combat.Sim.Jobs;

/// <summary>
/// 职业状态机共享的量谱访问辅助方法。
/// 职业层只负责具体规则，不重复维护系统量谱的读写和类型转换语义。
/// </summary>
public abstract class JobResourceAccessors
{
    private SystemStateMachine? _system;

    /// <summary>获取已由 Bind 注入的系统层；未绑定时保持 fail-fast 语义。</summary>
    protected SystemStateMachine System => _system ?? throw new InvalidOperationException(
        "job state machine is not bound; Bind(project, system) must be called before use");

    /// <summary>由具体职业的 Bind 转交系统层引用。</summary>
    protected void BindSystem(SystemStateMachine system) => _system = system;

    protected object Resource(CombatState state, string key) => System.GetJobResource(state, key);

    protected int ResourceMax(string key)
    {
        var maxValue = System.JobResourceMax(key);
        if (maxValue is null)
        {
            throw new KeyNotFoundException($"job resource {key} is missing max_value");
        }

        return ToInt32Truncate(maxValue.Value);
    }

    protected int IntResource(CombatState state, string key) => ToInt32Truncate(Resource(state, key));

    protected double FloatResource(CombatState state, string key) => Convert.ToDouble(Resource(state, key));

    protected bool BoolResource(CombatState state, string key) => Convert.ToBoolean(Resource(state, key));

    protected void SetResource(CombatState state, string key, object value) =>
        System.SetJobResource(state, key, value);

    /// <summary>读取职业时间资源的绝对截止时刻；null 表示当前不在计时。</summary>
    protected static double? TimelineDeadline(CombatState state, string key) =>
        state.JobTimelineDeadlines.TryGetValue(key, out var deadline) ? deadline : null;

    /// <summary>写入职业时间资源的绝对截止时刻；传 null 表示停止计时。</summary>
    protected static void SetTimelineDeadline(CombatState state, string key, double? deadline)
    {
        if (deadline is double value)
        {
            state.JobTimelineDeadlines[key] = value;
            return;
        }

        state.JobTimelineDeadlines.Remove(key);
    }

    /// <summary>对照 Python int() 的截断语义，避免 Convert.ToInt32 的银行家舍入。</summary>
    protected static int ToInt32Truncate(object value) => value switch
    {
        int intValue => intValue,
        double doubleValue => (int)doubleValue,
        bool boolValue => boolValue ? 1 : 0,
        _ => throw new InvalidCastException($"cannot truncate {value.GetType().Name} to int"),
    };
}

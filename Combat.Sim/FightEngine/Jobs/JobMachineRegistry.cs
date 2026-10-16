// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using System.Reflection;
using Combat.Sim.Config;

namespace Combat.Sim.Jobs;

/// <summary>
/// 职业子状态机注册中心。
/// 门面按职业标签从这里取工厂；未注册职业返回 null，由门面统一报错。
/// 内置职业在静态构造期自动发现注册：扫描当前程序集的 IJobStateMachine 实现，
/// 等价于 Python 侧模块导入时的自动发现，注册中心不显式引用任何具体职业。
/// </summary>
public static class JobMachineRegistry
{
    static JobMachineRegistry()
    {
        RegisterAssembly(typeof(JobMachineRegistry).Assembly);
    }

    private sealed record Registration(
        Type? Type,
        Func<IJobStateMachine> Factory,
        Action<JobConfig>? ValidateConfig);

    private static readonly Dictionary<string, Registration> _registrations = new(StringComparer.Ordinal);

    /// <summary>
    /// 自动发现并注册指定程序集内的全部职业状态机。
    /// 宿主加载独立职业 DLL 后可调用此入口，核心不引用具体职业程序集。
    /// 实现类只需提供静态 Tag（注册路径不构造实例，标签直接读取）与无参构造
    /// （配置与系统层引用经 Bind 注入）；缺静态 Tag 的实现直接报错，避免静默漏注册。
    /// 与手动 <see cref="Register{T}"/> 共用同一写路径，幂等语义完全对称：
    /// 自动发现注册后再手动注册同一类型同样幂等返回。
    /// </summary>
    public static void RegisterAssembly(Assembly assembly)
    {
        ArgumentNullException.ThrowIfNull(assembly);
        foreach (var type in assembly.GetTypes())
        {
            if (type.IsAbstract || type.IsInterface || type.ContainsGenericParameters ||
                !typeof(IJobStateMachine).IsAssignableFrom(type))
            {
                continue;
            }

            // 与泛型注册共用静态接口分派，同时支持显式实现的标签与配置校验。
            var register = typeof(JobMachineRegistry)
                .GetMethod(nameof(RegisterDiscoveredType), BindingFlags.NonPublic | BindingFlags.Static)!
                .MakeGenericMethod(type).CreateDelegate<Action>();
            register();
        }
    }

    private static void RegisterDiscoveredType<T>() where T : IJobStateMachine
    {
        var tag = T.Tag;
        if (string.IsNullOrEmpty(tag))
        {
            throw new InvalidOperationException(
                $"job state machine must define a non-empty static Tag: {typeof(T).FullName}");
        }

        RegisterType(tag, typeof(T), T.ValidateConfig);
    }

    /// <summary>
    /// 按标签注册职业类型：同一类型重复注册幂等，注册路径不构造实例
    /// （标签直接来自静态成员 <see cref="IJobStateMachine.Tag"/>，工厂延迟到 <see cref="Get"/>）。
    /// 类型与工厂互斥对称：任一入口先占用标签后，另一入口注册同一标签都抛错
    /// （对照 register_job_state_machine）。
    /// 注意：注册表是进程级全局空间，测试与宿主共享，测试专用职业 tag 需全局唯一；
    /// 注册本身不保证并发安全，应在进程启动期单线程完成注册。
    /// </summary>
    private static void RegisterType(string tag, Type type, Action<JobConfig>? validateConfig)
    {
        if (_registrations.TryGetValue(tag, out var existing))
        {
            if (existing.Type == type)
            {
                return;
            }
            throw new InvalidOperationException($"duplicate job state machine tag: {tag}");
        }
        _registrations.Add(tag, new(type, () => (IJobStateMachine)Activator.CreateInstance(type)!, validateConfig));
    }

    /// <summary>
    /// 注册一个职业子状态机类型；同一类型重复注册幂等，注册路径不构造实例。
    /// </summary>
    public static void Register<T>() where T : IJobStateMachine, new() =>
        RegisterDiscoveredType<T>();

    /// <summary>按标签注册职业工厂；标签已被占用时抛错（工厂无法比较类型，不做幂等）。</summary>
    public static void Register(string tag, Func<IJobStateMachine> factory)
    {
        if (!_registrations.TryAdd(tag, new(null, factory, null)))
        {
            throw new InvalidOperationException($"duplicate job state machine tag: {tag}");
        }
    }

    /// <summary>按职业标签取工厂；未注册返回 null（对照 get_job_state_machine_type）。</summary>
    public static IJobStateMachine? Get(string jobTag) =>
        _registrations.TryGetValue(jobTag, out var registration) ? registration.Factory() : null;

    /// <summary>由职业自己校验专属配置；公共加载器只按标签派发。</summary>
    internal static void ValidateConfig(JobConfig job)
    {
        if (_registrations.TryGetValue(job.Key, out var registration))
            registration.ValidateConfig?.Invoke(job);
    }

    /// <summary>返回当前已注册的职业标签（排序，对照 registered_job_tags）。</summary>
    public static IReadOnlyList<string> RegisteredTags() =>
        _registrations.Keys.OrderBy(k => k, StringComparer.Ordinal).ToList();
}

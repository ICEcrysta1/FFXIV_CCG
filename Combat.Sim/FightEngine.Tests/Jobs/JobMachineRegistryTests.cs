using System.Reflection;
using System.Reflection.Emit;
using Combat.Sim.Config;
using Combat.Sim.Jobs;
using Combat.Sim.Models.Definitions;
using Xunit;

namespace FightEngine.Tests.Jobs;

public class JobMachineRegistryTests
{
    [Fact]
    public void 同一类型重复注册幂等()
    {
        JobMachineRegistry.Register<RegistryJobA>();
        JobMachineRegistry.Register<RegistryJobA>(); // 不抛错
        Assert.Contains("registry_job_a", JobMachineRegistry.RegisteredTags());
    }

    [Fact]
    public void 注册路径不构造实例()
    {
        CountingJob.Constructions = 0;
        JobMachineRegistry.Register<CountingJob>();
        JobMachineRegistry.Register<CountingJob>(); // 幂等路径同样不构造
        Assert.Equal(0, CountingJob.Constructions);

        // 只有 Get 才会走工厂构造
        var machine = JobMachineRegistry.Get("counting_job");
        Assert.NotNull(machine);
        Assert.Equal(1, CountingJob.Constructions);
    }

    [Fact]
    public void 不同类型占用同一标签抛错()
    {
        JobMachineRegistry.Register<RegistryJobA>();
        var ex = Assert.Throws<InvalidOperationException>(() => JobMachineRegistry.Register<RegistryJobB>());
        Assert.Contains("duplicate job state machine tag: registry_job_a", ex.Message);
    }

    [Fact]
    public void 工厂先占标签后类型注册抛错()
    {
        // 工厂先占 tag → 类型注册必须抛错，且不覆盖原工厂（互斥对称的反方向）
        JobMachineRegistry.Register("registry_preempted_job", () => new PreemptedJob());
        var ex = Assert.Throws<InvalidOperationException>(() => JobMachineRegistry.Register<PreemptedJob>());
        Assert.Contains("duplicate job state machine tag: registry_preempted_job", ex.Message);
        Assert.NotNull(JobMachineRegistry.Get("registry_preempted_job"));
    }

    [Fact]
    public void 工厂重复注册抛错()
    {
        JobMachineRegistry.Register("registry_factory_job", () => new RegistryJobA());
        var ex = Assert.Throws<InvalidOperationException>(
            () => JobMachineRegistry.Register("registry_factory_job", () => new RegistryJobA()));
        Assert.Contains("duplicate job state machine tag: registry_factory_job", ex.Message);
    }

    [Fact]
    public void 类型先占标签后工厂注册抛错()
    {
        JobMachineRegistry.Register<RegistryJobA>();
        Assert.Throws<InvalidOperationException>(() =>
            JobMachineRegistry.Register("registry_job_a", () => new RegistryJobB()));
        Assert.IsType<RegistryJobA>(JobMachineRegistry.Get("registry_job_a"));
    }

    [Fact]
    public void 类型注册挂载职业配置校验但不构造实例()
    {
        ValidatedRegistryJob.Constructions = 0;
        JobMachineRegistry.Register<ValidatedRegistryJob>();
        var job = EmptyJob("registry_validated_job");
        var error = Assert.Throws<InvalidOperationException>(() => JobMachineRegistry.ValidateConfig(job));
        Assert.Equal("registry job config is invalid", error.Message);
        Assert.Equal(0, ValidatedRegistryJob.Constructions);
    }

    [Fact]
    public void 外部程序集可重复发现职业及其配置校验()
    {
        // 独立程序集只引用公共契约；注册过程必须保持延迟构造。
        var tag = "external_registry_" + Guid.NewGuid().ToString("N");
        var assembly = AssemblyBuilder.DefineDynamicAssembly(new AssemblyName(tag), AssemblyBuilderAccess.Run);
        var type = assembly.DefineDynamicModule(tag).DefineType(
            "ExternalJob", TypeAttributes.Public, typeof(ValidatedRegistryJob));
        type.AddInterfaceImplementation(typeof(IJobStateMachine));
        var getter = type.DefineMethod("get_Tag",
            MethodAttributes.Public | MethodAttributes.Static | MethodAttributes.SpecialName,
            typeof(string), Type.EmptyTypes);
        var il = getter.GetILGenerator();
        il.Emit(OpCodes.Ldstr, tag);
        il.Emit(OpCodes.Ret);
        type.DefineProperty("Tag", PropertyAttributes.None, typeof(string), null).SetGetMethod(getter);
        // 派生类的新静态属性必须映射到接口，泛型 T.Tag 才会使用外部职业标签。
        type.DefineMethodOverride(getter, typeof(IJobStateMachine).GetProperty(nameof(IJobStateMachine.Tag))!.GetMethod!);

        var constructor = type.DefineConstructor(MethodAttributes.Public, CallingConventions.Standard, Type.EmptyTypes);
        il = constructor.GetILGenerator();
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Call, typeof(ValidatedRegistryJob).GetConstructor(Type.EmptyTypes)!);
        il.Emit(OpCodes.Ldstr, "external job must remain unconstructed during registration");
        il.Emit(OpCodes.Newobj, typeof(InvalidOperationException).GetConstructor(new[] { typeof(string) })!);
        il.Emit(OpCodes.Throw);

        type.CreateType();

        JobMachineRegistry.RegisterAssembly(assembly);
        JobMachineRegistry.RegisterAssembly(assembly);
        Assert.Contains(tag, JobMachineRegistry.RegisteredTags());
        var error = Assert.Throws<InvalidOperationException>(() => JobMachineRegistry.ValidateConfig(EmptyJob(tag)));
        Assert.Equal("registry job config is invalid", error.Message);
        Assert.Throws<TargetInvocationException>(() => JobMachineRegistry.Get(tag));
    }

    [Theory]
    [InlineData(MethodAttributes.Public, true)]
    [InlineData(MethodAttributes.Private, false)]
    [InlineData(MethodAttributes.Family, false)]
    [InlineData(MethodAttributes.Assembly, false)]
    public void 外部程序集拒绝缺少公开无参构造的职业且不登记标签(
        MethodAttributes visibility, bool hasParameters)
    {
        var tag = "external_invalid_constructor_" + Guid.NewGuid().ToString("N");
        var assembly = AssemblyBuilder.DefineDynamicAssembly(new AssemblyName(tag), AssemblyBuilderAccess.Run);
        var type = assembly.DefineDynamicModule(tag).DefineType(
            "ExternalInvalidConstructorJob", TypeAttributes.Public, typeof(ValidatedRegistryJob));
        type.AddInterfaceImplementation(typeof(IJobStateMachine));
        var getter = type.DefineMethod("get_Tag",
            MethodAttributes.Public | MethodAttributes.Static | MethodAttributes.SpecialName,
            typeof(string), Type.EmptyTypes);
        var il = getter.GetILGenerator();
        il.Emit(OpCodes.Ldstr, tag);
        il.Emit(OpCodes.Ret);
        type.DefineMethodOverride(getter, typeof(IJobStateMachine).GetProperty(nameof(IJobStateMachine.Tag))!.GetMethod!);

        var constructor = type.DefineConstructor(visibility, CallingConventions.Standard,
            hasParameters ? new[] { typeof(int) } : Type.EmptyTypes);
        il = constructor.GetILGenerator();
        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Call, typeof(ValidatedRegistryJob).GetConstructor(Type.EmptyTypes)!);
        il.Emit(OpCodes.Ret);
        var jobType = type.CreateType()!;
        ValidatedRegistryJob.Constructions = 0;

        var error = Assert.Throws<InvalidOperationException>(() => JobMachineRegistry.RegisterAssembly(assembly));

        Assert.Contains("public parameterless constructor", error.Message);
        Assert.Contains(jobType.FullName!, error.Message);
        Assert.DoesNotContain(tag, JobMachineRegistry.RegisteredTags());
        Assert.Null(JobMachineRegistry.Get(tag));
        Assert.Equal(0, ValidatedRegistryJob.Constructions);
    }

    [Fact]
    public void 外部程序集接受具有隐式无参构造的值类型职业()
    {
        var tag = "external_value_registry_" + Guid.NewGuid().ToString("N");
        var assembly = AssemblyBuilder.DefineDynamicAssembly(new AssemblyName(tag), AssemblyBuilderAccess.Run);
        var type = assembly.DefineDynamicModule(tag).DefineType(
            "ExternalValueJob", TypeAttributes.Public | TypeAttributes.Sealed | TypeAttributes.SequentialLayout,
            typeof(ValueType));
        type.AddInterfaceImplementation(typeof(IJobStateMachine));
        var getter = type.DefineMethod("get_Tag",
            MethodAttributes.Public | MethodAttributes.Static | MethodAttributes.SpecialName,
            typeof(string), Type.EmptyTypes);
        var il = getter.GetILGenerator();
        il.Emit(OpCodes.Ldstr, tag);
        il.Emit(OpCodes.Ret);
        type.DefineMethodOverride(getter, typeof(IJobStateMachine).GetProperty(nameof(IJobStateMachine.Tag))!.GetMethod!);

        // 工厂只需创建值类型，职业行为不参与本次注册检查。
        foreach (var method in typeof(IJobStateMachine).GetMethods().Where(method => method.IsAbstract))
        {
            var implementation = type.DefineMethod(method.Name, MethodAttributes.Public | MethodAttributes.Virtual,
                method.ReturnType, method.GetParameters().Select(parameter => parameter.ParameterType).ToArray());
            il = implementation.GetILGenerator();
            il.Emit(OpCodes.Newobj, typeof(NotSupportedException).GetConstructor(Type.EmptyTypes)!);
            il.Emit(OpCodes.Throw);
            type.DefineMethodOverride(implementation, method);
        }
        var jobType = type.CreateType()!;
        Assert.Null(jobType.GetConstructor(Type.EmptyTypes));

        JobMachineRegistry.RegisterAssembly(assembly);
        JobMachineRegistry.RegisterAssembly(assembly);

        Assert.Contains(tag, JobMachineRegistry.RegisteredTags());
        Assert.Equal(jobType, JobMachineRegistry.Get(tag)?.GetType());
    }

    [Fact]
    public void 外部程序集识别显式静态接口标签且与泛型注册幂等()
    {
        var assemblyName = "external_explicit_registry_" + Guid.NewGuid().ToString("N");
        var assembly = AssemblyBuilder.DefineDynamicAssembly(
            new AssemblyName(assemblyName), AssemblyBuilderAccess.Run);
        var type = assembly.DefineDynamicModule(assemblyName).DefineType(
            "ExternalExplicitJob", TypeAttributes.Public, typeof(ExplicitTagRegistryJob));
        type.DefineDefaultConstructor(MethodAttributes.Public);
        var jobType = type.CreateType()!;
        ExplicitTagRegistryJob.Constructions = 0;

        // 动态职业继承显式接口映射，没有可供扫描器查找的 public Tag 属性。
        JobMachineRegistry.RegisterAssembly(assembly);
        JobMachineRegistry.RegisterAssembly(assembly);
        typeof(JobMachineRegistry).GetMethods()
            .Single(method => method.Name == nameof(JobMachineRegistry.Register) && method.IsGenericMethodDefinition)
            .MakeGenericMethod(jobType).Invoke(null, null);

        Assert.Contains(ExplicitTagRegistryJob.RegisteredTag, JobMachineRegistry.RegisteredTags());
        var error = Assert.Throws<InvalidOperationException>(() =>
            JobMachineRegistry.ValidateConfig(EmptyJob(ExplicitTagRegistryJob.RegisteredTag)));
        Assert.Equal("explicit registry job config is invalid", error.Message);
        Assert.Equal(0, ExplicitTagRegistryJob.Constructions);

        Assert.Equal(jobType, JobMachineRegistry.Get(ExplicitTagRegistryJob.RegisteredTag)?.GetType());
        Assert.Equal(1, ExplicitTagRegistryJob.Constructions);
    }

    private static JobConfig EmptyJob(string key) => new(
        key, key, new Dictionary<string, double>(), new Dictionary<string, double>(),
        new Dictionary<string, StatusDefinition>(), Array.Empty<SkillDefinition>());

    [Fact]
    public void 按标签查询与标签排序()
    {
        JobMachineRegistry.Register("registry_zzz_job", () => new RegistryJobA());
        JobMachineRegistry.Register("registry_aaa_job", () => new RegistryJobA());
        var tags = JobMachineRegistry.RegisteredTags();
        Assert.Contains("registry_aaa_job", tags);
        Assert.Contains("registry_zzz_job", tags);
        Assert.Equal(tags.OrderBy(t => t, StringComparer.Ordinal), tags);

        Assert.NotNull(JobMachineRegistry.Get("registry_aaa_job"));
        Assert.Null(JobMachineRegistry.Get("not_registered_anywhere"));
    }

    [Fact]
    public void 内置职业自动发现注册()
    {
        // 注册中心不显式引用具体职业类型，内置职业经程序集扫描自动注册；
        // 这里只按标签查询，验证自动发现链路可用（注册路径不构造实例）。
        Assert.Contains("black_mage", JobMachineRegistry.RegisteredTags());
        Assert.NotNull(JobMachineRegistry.Get("black_mage"));
        Assert.Contains("machinist", JobMachineRegistry.RegisteredTags());
        Assert.NotNull(JobMachineRegistry.Get("machinist"));
    }

    [Fact]
    public void 自动发现后再手动注册同一类型幂等()
    {
        // 自动发现与 Register<T> 共用同一写路径：black_mage 已被静态构造期扫描注册，
        // 再走手动类型注册同一类型必须幂等返回（修复前会因 _factories 占用而抛错）。
        JobMachineRegistry.Register<global::Combat.Sim.Jobs.Black.Mage.BlackMageJobStateMachine>();
        Assert.Contains("black_mage", JobMachineRegistry.RegisteredTags());
        Assert.NotNull(JobMachineRegistry.Get("black_mage"));
    }
}

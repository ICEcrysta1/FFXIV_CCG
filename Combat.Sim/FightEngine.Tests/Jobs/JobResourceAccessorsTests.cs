using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;
using Combat.Sim.Jobs;
using Combat.Sim.System;
using FightEngine.Tests.System;

namespace FightEngine.Tests.Jobs;

public class JobResourceAccessorsTests
{
    [Fact]
    public void SharedAccessorsReadWriteAndConvertJobResources()
    {
        var system = new SystemStateMachine(
            SystemTestKit.Timing,
            SystemTestKit.Potency,
            SystemTestKit.MpRecovery);
        system.RegisterJobState(new JobStateRegistration(
            "test_job",
            new Dictionary<string, JobResourceDefinition>
            {
                ["count"] = new("count", "int", 0, MaxValue: 3),
                ["ratio"] = new("ratio", "float", 2.5),
                ["enabled"] = new("enabled", "bool", true),
            },
            new Dictionary<string, StatusDefinition>()));

        var state = system.InitialState(fightRemaining: 600.0, maxMp: 10000);
        var accessors = new TestAccessors();
        accessors.Attach(system);

        Assert.Equal(0, accessors.ReadInt(state, "count"));
        Assert.Equal(2.5, accessors.ReadFloat(state, "ratio"));
        Assert.True(accessors.ReadBool(state, "enabled"));
        Assert.Equal(3, accessors.ReadMax("count"));

        accessors.Write(state, "count", 2);
        Assert.Equal(2, accessors.Read(state, "count"));
        Assert.Equal(1, accessors.Truncate(1.9));
        Assert.Equal(0, accessors.Truncate(false));
    }

    [Fact]
    public void SharedAccessorsFailBeforeSystemBinding()
    {
        var accessors = new TestAccessors();

        var exception = Assert.Throws<InvalidOperationException>(
            () => accessors.Read(new CombatState(), "count"));

        Assert.Contains("Bind(project, system)", exception.Message);
    }

    private sealed class TestAccessors : JobResourceAccessors
    {
        public void Attach(SystemStateMachine system) => BindSystem(system);

        public object Read(CombatState state, string key) => Resource(state, key);

        public int ReadMax(string key) => ResourceMax(key);

        public int ReadInt(CombatState state, string key) => IntResource(state, key);

        public double ReadFloat(CombatState state, string key) => FloatResource(state, key);

        public bool ReadBool(CombatState state, string key) => BoolResource(state, key);

        public void Write(CombatState state, string key, object value) => SetResource(state, key, value);

        public int Truncate(object value) => ToInt32Truncate(value);
    }
}

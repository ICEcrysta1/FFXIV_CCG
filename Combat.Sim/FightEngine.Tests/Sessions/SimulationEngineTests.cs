using System.Text.Json;
using Combat.Sim.Common;
using Combat.Sim.Models.Timeline;
using Combat.Sim.Policy;
using Combat.Sim.Sessions;

namespace FightEngine.Tests.Sessions;

public sealed class SimulationEngineTests
{
    private static string Root => RepoRootLocator.Find();

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void 默认容量为一百且未申请队列时保持空宿主(bool injectRules)
    {
        using var engine = CreateCapacityEngine(injectRules);

        Assert.Equal(100, SimulationEngine.MaxCapacity);
        Assert.Equal(SimulationEngine.MaxCapacity, engine.Capacity);
        Assert.Equal(0, engine.ActiveCount);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void 默认和显式一百容量可独立运行且满载释放后恢复(bool injectRules, bool explicitCapacity)
    {
        using var engine = CreateCapacityEngine(injectRules,
            explicitCapacity ? SimulationEngine.MaxCapacity : null);
        var sessions = Enumerable.Range(0, SimulationEngine.MaxCapacity)
            .Select(index => engine.CreateSession(4, 2.0 + index * 0.001, index * 100, 30)).ToArray();

        Assert.Equal(100, engine.ActiveCount);
        Assert.Equal(100, sessions.Select(session => session.Id).Distinct().Count());
        // 不同队列使用不同起点、基础 GCD 和历史长度，检查共享配置不串改运行状态。
        Parallel.For(0, sessions.Length, index =>
        {
            var session = sessions[index];
            var actions = 1 + index % 3;
            for (var step = 0; step < actions; step++)
            {
                var time = index * 100 + step * 3.0;
                Assert.True(session.SubmitAction(time, "heated_split_shot").Value.Accepted);
                session.AdvanceTo(time + 0.7);
                session.RecordPolicyAction(time + 0.7, "ogcd_wait", time + 3);
            }
            var timestamp = index * 100 + (actions - 1) * 3.0 + 0.7;
            var statistics = session.GetStatistics();
            Assert.Equal(timestamp, statistics.Timestamp);
            Assert.Equal(actions, statistics.ActionHistoryCount);
            Assert.Equal(actions, statistics.PolicyHistoryCount);
            Assert.Equal(2.0 + index * 0.001,
                Convert.ToDouble(session.ObserveAt(timestamp).Value["current_gcd_seconds"]), precision: 9);
        });

        Assert.Throws<InvalidOperationException>(() => engine.CreateSession(4));
        Assert.Equal(100, engine.ActiveCount);
        sessions[0].Dispose();
        Assert.Equal(99, engine.ActiveCount);
        Assert.True(sessions[0].IsClosed);
        Assert.All(sessions.Skip(1), session => Assert.False(session.IsClosed));
        using var replacement = engine.CreateSession(4);
        Assert.Equal(100, engine.ActiveCount);
        Assert.NotEqual(sessions[0].Id, replacement.Id);
        Assert.True(replacement.SubmitAction(0, "heated_split_shot").Value.Accepted);
        replacement.AdvanceTo(0.7);
        Assert.Equal(1, replacement.GetStatistics().ActionHistoryCount);
        Assert.Throws<InvalidOperationException>(() => engine.CreateSession(4));
    }

    [Theory]
    [InlineData(false, 0)]
    [InlineData(false, 101)]
    [InlineData(true, 0)]
    [InlineData(true, 101)]
    public void 两种构造入口均拒绝容量越界(bool injectRules, int capacity)
    {
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => CreateCapacityEngine(injectRules, capacity));
        Assert.Equal("capacity", error.ParamName);
    }

    [Theory]
    [InlineData(-3.5)]
    [InlineData(-3.0)]
    public void 预读负起点允许当前及未来的有限同刻场景批次(double timestamp)
    {
        using var engine = new SimulationEngine(Root, "black_mage", 1);
        using var session = engine.CreateSession(8, initialTimestamp: -3.5);

        var applied = session.ApplyExternalEvents(new[]
        {
            new ExternalCombatEvent(timestamp, ExternalCombatEventKinds.MovementChanged, true),
            new ExternalCombatEvent(timestamp, ExternalCombatEventKinds.TargetCountChanged, TargetCount: 2),
        });

        Assert.True(applied.Value.Accepted);
        Assert.Equal(timestamp, applied.Timestamp);
        Assert.Equal(timestamp, session.GetStatistics().Timestamp);
        Assert.Equal("movement_locked", session.ValidateActionAt(timestamp, "fire_iii").Value.Reason);
        Assert.True(session.ApplyExternalEvents(new[]
        {
            new ExternalCombatEvent(timestamp, ExternalCombatEventKinds.MovementChanged, false),
        }).Value.Accepted);
        Assert.True(session.SubmitAction(timestamp, "fire_iii").Value.Accepted);
    }

    [Theory]
    [InlineData(-4.0, -4.0)]
    [InlineData(-3.5000000001, -3.5000000001)]
    [InlineData(-3.0, double.NaN)]
    [InlineData(-3.0, double.NegativeInfinity)]
    [InlineData(-3.0, double.PositiveInfinity)]
    [InlineData(-3.0, -2.5)]
    public void 预读负起点拒绝过去非有限或不同时刻批次且整批零变更(double firstTimestamp, double secondTimestamp)
    {
        using var engine = new SimulationEngine(Root, "black_mage", 1);
        using var session = engine.CreateSession(8, initialTimestamp: -3.5);
        var before = JsonSerializer.Serialize(session.ObserveAt(-3.5, "vector", -3.5).Value);
        var statistics = session.GetStatistics();

        Assert.Throws<ArgumentException>(() => session.ApplyExternalEvents(new[]
        {
            new ExternalCombatEvent(firstTimestamp, ExternalCombatEventKinds.BossTargetableChanged, false),
            new ExternalCombatEvent(secondTimestamp, ExternalCombatEventKinds.MovementChanged, true),
        }));

        Assert.Equal(statistics, session.GetStatistics());
        Assert.Equal(before, JsonSerializer.Serialize(session.ObserveAt(-3.5, "vector", -3.5).Value));
    }

    [Fact]
    public void 单元素批次通过正式会话结算且非法整批不改变状态()
    {
        using var engine = new SimulationEngine(Root, "black_mage", 1);
        using var batch = engine.CreateSession(8);
        var fact = new ExternalCombatEvent(1, ExternalCombatEventKinds.MovementChanged, true);
        var applied = batch.ApplyExternalEvents(new[] { fact });
        Assert.True(applied.Value.Accepted);
        Assert.Equal(1, applied.Timestamp);
        Assert.Equal("movement_locked", batch.ValidateActionAt(1, "fire_iii").Value.Reason);
        var before = JsonSerializer.Serialize(batch.ObserveAt(1, "vector", 1).Value);
        Assert.Throws<ArgumentException>(() => batch.ApplyExternalEvents(new[]
        {
            new ExternalCombatEvent(2, ExternalCombatEventKinds.MovementChanged, false),
            new ExternalCombatEvent(2, ExternalCombatEventKinds.TargetCountChanged, TargetCount: -1),
        }));
        Assert.Equal(before, JsonSerializer.Serialize(batch.ObserveAt(1, "vector", 1).Value));
        Assert.Equal(1, batch.GetStatistics().Timestamp);
        Assert.True(batch.ApplyExternalEvents(new[]
        {
            new ExternalCombatEvent(1, ExternalCombatEventKinds.MovementChanged, false),
        }).Value.Accepted);
        Assert.True(batch.ValidateActionAt(1, "fire_iii").Value.Ok);
    }

    [Theory]
    [InlineData("black_mage", "blizzard_iii")]
    [InlineData("machinist", "heated_split_shot")]
    public void 十六队列多线程与独立模拟器逐步输出一致(string job, string action)
    {
        using var engine = new SimulationEngine(Root, job, 16);
        var sessions = Enumerable.Range(0, 16)
            .Select(index => engine.CreateSession(3, 2.0 + index * 0.03, index * 100, 180)).ToArray();
        // 基准将基础 GCD 写入配置；共享引擎将基础 GCD 放在各队列状态中。
        var references = Enumerable.Range(0, 16)
            .Select(index => new JobSimulator(
                CombatStateMachine.FromDefaultConfig(Root, job, 2.0 + index * 0.03, 3),
                index * 100, 180)).ToArray();
        var policies = references.Select(simulator => PolicySession.Create(Root, simulator)).ToArray();
        Parallel.For(0, sessions.Length, index =>
        {
            var session = sessions[index];
            var reference = references[index];
            for (var step = 0; step < 12; step++)
            {
                var time = index * 100 + step * 6.0;
                var fact = new ExternalCombatEvent(time, ExternalCombatEventKinds.TargetCountChanged,
                    TargetCount: 1 + index % 3);
                session.ApplyExternalEvents(new ExternalCombatEvent[] { fact });
                reference.ApplyExternalEvents(new ExternalCombatEvent[] { fact });
                if (step == 0)
                {
                    var buff = new ExternalCombatEvent(time, ExternalCombatEventKinds.RaidBuffWindowChanged,
                        Value: true, RemainingSeconds: 20);
                    session.ApplyExternalEvents(new ExternalCombatEvent[] { buff });
                    reference.ApplyExternalEvents(new ExternalCombatEvent[] { buff });
                }
                var actual = session.SubmitAction(time, action).Value;
                var expected = reference.SubmitAction(time, action);
                Assert.True(actual.Accepted);
                Assert.Equal(expected.AcceptedTimestamp, actual.AcceptedTimestamp);
                Assert.Equal(expected.EffectTimestamp, actual.EffectTimestamp);
                session.AdvanceTo(time + 4);
                reference.AdvanceTo(time + 4);
                session.RecordPolicyAction(time + 4, "ogcd_wait", time + 6);
                policies[index].Record(reference, time + 4, "ogcd_wait", time + 6);
                Assert.Equal(JsonSerializer.Serialize(reference.FormatState()),
                    JsonSerializer.Serialize(session.ObserveAt(time + 4).Value));
                var context = session.ObserveAt(time + 4, "vector", time + 6).Value;
                Assert.Equal(JsonSerializer.Serialize(policies[index].BuildVectorContext(reference, time + 6)),
                    JsonSerializer.Serialize(context));
            }
        });
        Assert.Equal(16, engine.ActiveCount);
    }

    [Fact]
    public void 队列独立暂停接收新事件且单队列错误不影响其他队列()
    {
        using var engine = new SimulationEngine(Root, "black_mage", 2);
        using var first = engine.CreateSession(8);
        using var second = engine.CreateSession(8);
        Assert.True(first.SubmitAction(0, "fire_iii").Value.Accepted);
        first.ApplyExternalEvents(new ExternalCombatEvent[] { new(0.1, ExternalCombatEventKinds.MovementChanged, Value: true) });
        second.AdvanceTo(1000);
        Assert.Equal(0.1, first.GetStatistics().Timestamp);
        Assert.Throws<ArgumentOutOfRangeException>(() => first.AdvanceTo(-1));
        Assert.Throws<KeyNotFoundException>(() => first.SubmitAction(0.1, "missing"));
        Assert.Throws<ArgumentOutOfRangeException>(() => first.SubmitAction(10, "blizzard_iii", double.NaN));
        Assert.Equal(0.1, first.GetStatistics().Timestamp);
        Assert.True(second.SubmitAction(1000, "blizzard_iii").Value.Accepted);
        first.AdvanceTo(4);
        Assert.Equal(1, first.GetStatistics().ActionHistoryCount);
        Assert.Equal("movement_locked", first.ValidateActionAt(4, "blizzard_iii").Value.Reason);
        Assert.Equal(1000, second.GetStatistics().Timestamp);
    }

    [Fact]
    public void 满载拒绝且释放后旧句柄不会命中新队列()
    {
        using var engine = new SimulationEngine(Root, "machinist", 1);
        var old = engine.CreateSession(4);
        Assert.Throws<InvalidOperationException>(() => engine.CreateSession(4));
        old.Dispose();
        old.Dispose();
        Assert.Equal(0, engine.ActiveCount);
        using var next = engine.CreateSession(4);
        Assert.NotEqual(old.Id, next.Id);
        Assert.Throws<ObjectDisposedException>(() => old.AdvanceTo(0));
        engine.Dispose();
        Assert.True(next.IsClosed);
        Assert.Equal(0, engine.ActiveCount);
        Assert.Throws<ObjectDisposedException>(() => engine.CreateSession(4));
        Assert.Throws<ObjectDisposedException>(() => next.Reset(4));
    }

    [Fact]
    public void 重置清除全部上下文且失败重置保留原队列()
    {
        using var engine = new SimulationEngine(Root, "black_mage");
        using var session = engine.CreateSession(8);
        session.SubmitAction(0, "fire_iii");
        session.RecordPolicyAction(0, "ogcd_wait", 4);
        session.ApplyExternalEvents(new ExternalCombatEvent[] { new(0, ExternalCombatEventKinds.MovementChanged, Value: true) });
        Assert.Throws<ArgumentOutOfRangeException>(() => session.Reset(8, double.NaN));
        Assert.Equal(1, session.GetStatistics().PolicyHistoryCount);
        session.Reset(8, 2.1, 20, 60);
        Assert.Equal(0, session.GetStatistics().ActionHistoryCount);
        Assert.Equal(0, session.GetStatistics().PolicyHistoryCount);
        var state = session.ObserveAt(24).Value;
        Assert.True(session.ValidateActionAt(24, "blizzard_iii").Value.Ok);
        Assert.Equal(0, Convert.ToInt32(state["gcd_index"]));
        Assert.Equal(2.1, Convert.ToDouble(state["current_gcd_seconds"]));
        Assert.Equal(56, Convert.ToDouble(state["fight_remaining_seconds"]));
    }

    [Fact]
    public void 相同队列多线程请求串行记录且生命周期反复复用不积压()
    {
        using var engine = new SimulationEngine(Root, "machinist", 1);
        for (var iteration = 0; iteration < 20; iteration++)
        {
            using var session = engine.CreateSession(null);
            Parallel.For(0, 64, _ => session.RecordPolicyAction(0, "ogcd_wait", 0));
            Assert.Equal(64, session.GetStatistics().PolicyHistoryCount);
            Assert.Equal(0, session.GetStatistics().Timestamp);
        }
        Assert.Equal(0, engine.ActiveCount);
    }

    [Fact]
    public void 完整记录与有限窗口走同一执行路径且累计指标覆盖整场()
    {
        using var engine = new SimulationEngine(Root, "machinist", 3);
        using var full = engine.CreateSession(null);
        using var window = engine.CreateSession(8);
        using var none = engine.CreateSession(0);
        for (var step = 0; step < 512; step++)
        {
            var time = step * 2.5;
            foreach (var session in new[] { full, window, none })
            {
                Assert.True(session.SubmitAction(time, "heated_split_shot").Value.Accepted);
                session.AdvanceTo(time + 0.7);
                session.RecordPolicyAction(time + 0.7, "ogcd_wait", time + 2.5);
                session.AdvanceTo(time + 2.5);
            }
        }
        var expected = JsonSerializer.Serialize(full.ObserveAt(1280).Value);
        Assert.Equal(expected, JsonSerializer.Serialize(window.ObserveAt(1280).Value));
        Assert.Equal(expected, JsonSerializer.Serialize(none.ObserveAt(1280).Value));
        Assert.Equal(512, full.GetStatistics().ActionHistoryCount);
        Assert.Equal(512, full.GetStatistics().PolicyHistoryCount);
        Assert.Equal(8, window.GetStatistics().ActionHistoryCount);
        Assert.Equal(8, window.GetStatistics().PolicyHistoryCount);
        Assert.Equal(0, none.GetStatistics().ActionHistoryCount);
        Assert.Equal(0, none.GetStatistics().PolicyHistoryCount);
        var fullRows = Rows(full.ObserveAt(1280, "vector", 1280).Value);
        var windowRows = Rows(window.ObserveAt(1280, "vector", 1280).Value);
        Assert.Equal(1024, fullRows.Count);
        Assert.Equal(JsonSerializer.Serialize(fullRows.TakeLast(8)), JsonSerializer.Serialize(windowRows));
        Assert.Empty(Rows(none.ObserveAt(1280, "vector", 1280).Value));
        Assert.Equal(JsonSerializer.Serialize(full.ObserveAt(1280, "vector", 1280).Value["current_state_context"]),
            JsonSerializer.Serialize(none.ObserveAt(1280, "vector", 1280).Value["current_state_context"]));
        Assert.InRange(window.GetStatistics().QueueEntryCount, 0, 80);
    }

    private static SimulationEngine CreateCapacityEngine(bool injectRules, int? capacity = null)
    {
        if (!injectRules)
            return capacity is { } configuredCapacity
                ? new SimulationEngine(Root, "machinist", configuredCapacity)
                : new SimulationEngine(Root, "machinist");
        var rules = CombatStateMachine.FromDefaultConfig(Root, "machinist");
        var policyRegistry = PolicyActionRegistry.Load(Root, rules.SkillBook);
        return capacity is { } injectedCapacity
            ? new SimulationEngine(rules, policyRegistry, injectedCapacity)
            : new SimulationEngine(rules, policyRegistry);
    }

    private static List<Dictionary<string, object?>> Rows(Dictionary<string, object?> context) =>
        Assert.IsType<List<Dictionary<string, object?>>>(context["skill_history_context"]);
}

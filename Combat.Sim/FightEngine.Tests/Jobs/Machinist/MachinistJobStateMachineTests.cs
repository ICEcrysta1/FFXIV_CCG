using Combat.Sim.Facade;
using Combat.Sim.Models.Combat;

namespace FightEngine.Tests.Jobs.Machinist;

/// <summary>
/// 机工职业子状态机测试。
/// 覆盖机工状态机的资源、状态、连击、野火和动态 value 语义。
/// </summary>
public class MachinistJobStateMachineTests
{
    private static readonly string RepoRoot = FindRepoRoot();

    private static CombatStateMachine BuildMachine() =>
        CombatStateMachine.FromDefaultConfig(RepoRoot, "machinist");

    private static int Int(CombatState state, string key) =>
        Assert.IsType<int>(state.GetJobResource(key));

    private static double Float(CombatState state, string key) =>
        Assert.IsType<double>(state.GetJobResource(key));

    private static bool Bool(CombatState state, string key) =>
        Assert.IsType<bool>(state.GetJobResource(key));

    /// <summary>读取职业时间资源的对外视图（即模型输入看到的相对秒数）。</summary>
    private static double TimerView(CombatStateMachine machine, CombatState state, string key) =>
        Convert.ToDouble(machine.SystemMachine.BuildJobResourceSnapshot(state)[key]);

    private static CombatState ReadyAfterGcd(CombatStateMachine machine, CombatState state) =>
        state.GcdRemaining <= 0.0 ? state : TimelineTestDriver.AdvanceBy(machine, state, state.GcdRemaining);

    private static CombatState StartWildfire(CombatStateMachine machine, int hits)
    {
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "wildfire").NextState;
        state.SetJobResource("wildfire_hits", hits);
        return state;
    }

    private static string FindRepoRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "config", "default.yaml")))
            {
                return dir.FullName;
            }
        }

        throw new InvalidOperationException("仓库根未找到（缺少 config/default.yaml）");
    }

    [Fact]
    public void ScopeRegistrationMatchesPythonContract()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();

        Assert.Contains("wildfire", machine.SkillBook.Keys());
        Assert.Contains("detonator", machine.SkillBook.Keys());
        Assert.DoesNotContain("automaton_queen", machine.SkillBook.Keys());
        Assert.Contains("heat", state.JobResources.Keys);
        Assert.Contains("wildfire_hits", state.JobResources.Keys);
        Assert.DoesNotContain("queen_heat", state.JobResources.Keys);

        state = TimelineTestDriver.Execute(machine, state, "reassemble").NextState;
        Assert.True(state.HasStatus("reassemble"));
    }

    [Fact]
    public void ReassembleRaisesNextWeaponSkillValueAndConsumes()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();

        var initialPayload = TimelineTestDriver.FormatVectorState(machine, state);
        var initialCandidates = (List<Dictionary<string, object?>>)initialPayload["candidate_skill_context"];
        var initialSplitShot = initialCandidates.Single(token => (string)token["skill_key"]! == "heated_split_shot");
        Assert.Equal(1.0, (double)initialSplitShot["value"]!, 5);

        state = TimelineTestDriver.Execute(machine, state, "reassemble").NextState;

        var candidatePayload = TimelineTestDriver.FormatVectorState(machine, state);
        var candidates = (List<Dictionary<string, object?>>)candidatePayload["candidate_skill_context"];
        var splitShot = candidates.Single(token => (string)token["skill_key"]! == "heated_split_shot");
        Assert.Equal(2.0, (double)splitShot["value"]!, 5);

        state = TimelineTestDriver.Execute(machine, state, "heated_split_shot").NextState;
        var history = (List<Dictionary<string, object?>>)TimelineTestDriver.FormatVectorState(machine, state)["skill_history_context"];
        Assert.Equal(2.0, (double)history[^1]["value"]!, 5);
        Assert.False(state.HasStatus("reassemble"));
    }

    [Fact]
    public void CombosResolvePotencyAndGenerateHeatAndBattery()
    {
        var machine = BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "heated_split_shot").NextState;

        Assert.Equal(5, Int(state, "heat"));
        Assert.Equal(1, Int(state, "combo_stage"));

        state = ReadyAfterGcd(machine, state);
        state = TimelineTestDriver.Execute(machine, state, "heated_slug_shot").NextState;
        Assert.Equal(10, Int(state, "heat"));
        Assert.Equal(2, Int(state, "combo_stage"));
        Assert.Equal(320.0, state.CurrentPotency);

        state = ReadyAfterGcd(machine, state);
        state = TimelineTestDriver.Execute(machine, state, "heated_clean_shot").NextState;
        Assert.Equal(15, Int(state, "heat"));
        Assert.Equal(10, Int(state, "battery"));
        Assert.Equal(0, Int(state, "combo_stage"));
        Assert.Equal(0.0, Float(state, "combo_remaining"));
        Assert.Equal(420.0, state.CurrentPotency);
    }

    [Fact]
    public void ComboExpiryAndResourceOverflowAreRecorded()
    {
        var machine = BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "heated_split_shot").NextState;
        state = TimelineTestDriver.AdvanceBy(machine, state, 30.0);

        Assert.Equal(0, Int(state, "combo_stage"));
        Assert.Equal(0.0, Float(state, "combo_remaining"));

        state.SetJobResource("heat", 100);
        state = ReadyAfterGcd(machine, state);
        state = TimelineTestDriver.Execute(machine, state, "heated_split_shot").NextState;
        Assert.Equal(100, Int(state, "heat"));
        Assert.Equal(5, Int(state, "wasted_heat"));

        state.SetJobResource("battery", 100);
        state = ReadyAfterGcd(machine, state);
        state = TimelineTestDriver.Execute(machine, state, "air_anchor").NextState;
        Assert.Equal(100, Int(state, "battery"));
        Assert.Equal(20, Int(state, "wasted_battery"));
    }

    [Fact]
    public void JobSpecificValidationAndStatusApplicationsMatchPython()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();

        Assert.Equal("requires_excavator_ready", machine.ValidateAction(state, "excavator").Reason);
        Assert.Equal("requires_overheated", machine.ValidateAction(state, "blazing_shot").Reason);
        Assert.Equal("requires_full_metal_machinist", machine.ValidateAction(state, "full_metal_field").Reason);
        Assert.Equal("not_enough_heat", machine.ValidateAction(state, "hypercharge").Reason);
        Assert.Equal("requires_wildfire", machine.ValidateAction(state, "detonator").Reason);

        state = TimelineTestDriver.Execute(machine, state, "barrel_stabilizer").NextState;
        Assert.True(state.HasStatus("free_hypercharge"));
        Assert.True(state.HasStatus("full_metal_machinist"));

        state = TimelineTestDriver.Execute(machine, state, "full_metal_field").NextState;
        Assert.False(state.HasStatus("full_metal_machinist"));

        state = ReadyAfterGcd(machine, state);
        state.SetJobResource("heat", 50);
        state = TimelineTestDriver.Execute(machine, state, "hypercharge").NextState;
        Assert.Equal(50, Int(state, "heat"));
        Assert.True(state.HasStatus("overheated"));
        Assert.Equal(5, state.StatusStacks("overheated"));

        state = ReadyAfterGcd(machine, state);
        state = TimelineTestDriver.Execute(machine, state, "blazing_shot").NextState;
        Assert.Equal(4, state.StatusStacks("overheated"));
    }

    [Fact]
    public void ChainSawAndExcavatorManageBatteryAndReplacementStatus()
    {
        var machine = BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "chain_saw").NextState;

        Assert.Equal(20, Int(state, "battery"));
        Assert.True(state.HasStatus("excavator_ready"));
        state = ReadyAfterGcd(machine, state);
        Assert.True(machine.ValidateAction(state, "excavator").Ok);

        state = TimelineTestDriver.Execute(machine, state, "excavator").NextState;

        Assert.Equal(40, Int(state, "battery"));
        Assert.False(state.HasStatus("excavator_ready"));
    }

    [Fact]
    public void WildfireCountsWeaponskillsAndCapsAtSix()
    {
        var machine = BuildMachine();
        var state = StartWildfire(machine, hits: 0);

        state = TimelineTestDriver.Execute(machine, state, "double_check").NextState;
        Assert.Equal(0, Int(state, "wildfire_hits"));

        state = TimelineTestDriver.Execute(machine, state, "heated_split_shot").NextState;
        Assert.Equal(1, Int(state, "wildfire_hits"));

        state.SetJobResource("wildfire_hits", 6);
        state = ReadyAfterGcd(machine, state);
        state = TimelineTestDriver.Execute(machine, state, "heated_slug_shot").NextState;
        Assert.Equal(6, Int(state, "wildfire_hits"));
    }

    [Fact]
    public void WildfireExpiryUsesDamageTimeBuffsAndClearsState()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();
        machine.SystemMachine.GrantRegisteredStatus(state, "raid_buff_window", remaining: 1.0);
        state = TimelineTestDriver.Execute(machine, state, "wildfire").NextState;
        state.SetJobResource("wildfire_hits", 2);

        state = TimelineTestDriver.AdvanceBy(machine, state, 10.0);

        Assert.False(state.HasStatus("raid_buff_window"));
        Assert.Equal(480.0, state.CumulativePotency);
        Assert.Equal(480.0, Float(state, "wildfire_resolved_potency"));
        Assert.False(Bool(state, "wildfire_active"));
        Assert.Equal(4, Int(state, "wildfire_missed_weaponskills"));
    }

    [Fact]
    public void WildfireExpiryUsesBuffAddedBeforeDamageTime()
    {
        var machine = BuildMachine();
        var state = StartWildfire(machine, hits: 2);
        state = TimelineTestDriver.AdvanceBy(machine, state, 9.0);
        machine.SystemMachine.GrantRegisteredStatus(state, "raid_buff_window", remaining: 2.0);

        state = TimelineTestDriver.AdvanceBy(machine, state, 1.0);

        var expected = 480.0 * machine.Project.System.Potency.RaidBuffWindowMultiplier;
        Assert.Equal(expected, state.CumulativePotency, 8);
        Assert.Equal(expected, Float(state, "wildfire_resolved_potency"), 8);
    }

    [Fact]
    public void DetonatorResolvesCurrentHitsAndRecordsHistoryPotencyOnce()
    {
        var machine = BuildMachine();
        var state = StartWildfire(machine, hits: 4);
        machine.SystemMachine.GrantRegisteredStatus(state, "raid_buff_window", remaining: 20.0);

        state = TimelineTestDriver.Execute(machine, state, "detonator").NextState;
        var expectedBase = 4 * 240.0;
        var expectedResolved = expectedBase * machine.Project.System.Potency.RaidBuffWindowMultiplier;

        Assert.Equal(expectedResolved, state.CurrentPotency, 8);
        Assert.Equal(expectedResolved, state.CumulativePotency, 8);
        Assert.Equal(expectedResolved, Float(state, "wildfire_resolved_potency"), 8);
        Assert.Equal(2, Int(state, "wildfire_missed_weaponskills"));
        Assert.False(Bool(state, "wildfire_active"));
        Assert.Equal(expectedBase, state.History[^1].Potency);

        state = TimelineTestDriver.AdvanceBy(machine, state, 1.0);
        Assert.Equal("requires_wildfire", machine.ValidateAction(state, "detonator").Reason);
    }

    [Fact]
    public void WildfireExpiryRecordsLostPotencyDuringDowntime()
    {
        var machine = BuildMachine();
        var state = StartWildfire(machine, hits: 2);
        state.NextDowntimeEta = 5.0;
        state.DowntimeRemaining = 20.0;

        state = TimelineTestDriver.AdvanceBy(machine,
            state,
            10.0,
            targetableAt: eventTime => eventTime < 5.0 || eventTime >= 25.0);

        Assert.Equal(0.0, state.CumulativePotency);
        Assert.Equal(0.0, Float(state, "wildfire_resolved_potency"));
        Assert.Equal(480.0, Float(state, "wildfire_lost_potency"));
    }

    [Fact]
    public void ExpiringOverheatedStacksAreRecorded()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("heat", 50);
        state = TimelineTestDriver.Execute(machine, state, "hypercharge").NextState;

        state = TimelineTestDriver.AdvanceBy(machine, state, 10.0);

        Assert.False(state.HasStatus("overheated"));
        Assert.Equal(5, Int(state, "wasted_overheated_stacks"));
    }

    [Theory]
    [InlineData(9.99995)]
    [InlineData(9.99999995)]
    public void 临界同步不会取消过热到期统计且只记录一次(double elapsedBeforeExpiry)
    {
        var machine = BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("heat", 50);
        state = TimelineTestDriver.Execute(machine, state, "hypercharge").NextState;

        state = TimelineTestDriver.AdvanceBy(machine, state, elapsedBeforeExpiry);
        Assert.Equal(0, Int(state, "wasted_overheated_stacks"));

        state = TimelineTestDriver.AdvanceBy(machine, state, 1.0);
        Assert.False(state.HasStatus("overheated"));
        Assert.Equal(5, Int(state, "wasted_overheated_stacks"));

        state = TimelineTestDriver.AdvanceBy(machine, state, 1.0);
        Assert.Equal(5, Int(state, "wasted_overheated_stacks"));
    }

    [Theory]
    [InlineData("status")]
    [InlineData("downtime")]
    [InlineData("mp_tick")]
    public void 野火到期与同刻事件并发时仍会结算(string tieKind)
    {
        var machine = BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "wildfire").NextState;
        var duration = TimerView(machine, state, "wildfire_remaining");

        // 让另一类事件恰好在野火到期的同一时刻派发。这些事件会触发资源重新同步，
        // 而野火的剩余量此时已经被推进清零；到期事件必须仍然派发，
        // 否则 wildfire_active 会永久卡住，detonator 也会一直合法。
        switch (tieKind)
        {
            case "status":
                state.Statuses["probe_buff"] = new StatusState(duration);
                break;
            case "downtime":
                state.NextDowntimeEta = duration;
                state.DowntimeRemaining = 5.0;
                break;
            case "mp_tick":
                state.Mp = 0;
                state.NaturalMpTickProgress = 3.0 - duration;
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(tieKind), tieKind, "unknown tie kind");
        }

        state = TimelineTestDriver.AdvanceBy(machine, state, duration + 0.5);

        Assert.False(Bool(state, "wildfire_active"));
        Assert.Equal(0.0, TimerView(machine, state, "wildfire_remaining"));
    }
}

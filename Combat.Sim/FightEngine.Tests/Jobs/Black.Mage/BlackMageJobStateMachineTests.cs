using System.Globalization;
using System.Text.RegularExpressions;
using Combat.Sim.Facade;
using Combat.Sim.Models.Combat;
using Combat.Sim.Models.Definitions;

namespace FightEngine.Tests.Jobs.Black.Mage;

/// <summary>
/// 黑魔职业子状态机测试。
/// 依赖输出层的断言（ui_mp_tick_progress）与操纵 Python 内部 _behaviors 的用例未迁移；
/// unknown_behavior 分支在 C# 侧不可注入，以"未注册行为直接放行"等价替代。
/// </summary>
public class BlackMageJobStateMachineTests
{
    private static readonly string RepoRoot = FindRepoRoot();

    private static CombatStateMachine BuildMachine() =>
        CombatStateMachine.FromDefaultConfig(RepoRoot, "black_mage");

    /// <summary>推进到 GCD 与读条锁都结束（对照 tests.helpers.ready_after_gcd）。</summary>
    private static CombatState ReadyAfterGcd(CombatStateMachine machine, CombatState state)
    {
        var waitSeconds = Math.Max(state.GcdRemaining, state.CastRemaining);
        if (waitSeconds <= 0)
        {
            return state;
        }

        return TimelineTestDriver.AdvanceBy(machine, state, waitSeconds);
    }

    private static int Int(CombatState state, string key) => Assert.IsType<int>(state.GetJobResource(key));

    private static bool Bool(CombatState state, string key) => Assert.IsType<bool>(state.GetJobResource(key));

    private static double Float(CombatState state, string key) => Assert.IsType<double>(state.GetJobResource(key));

    /// <summary>读取职业时间资源的对外视图（即模型输入看到的相对秒数）。</summary>
    private static double TimerView(CombatStateMachine machine, CombatState state, string key) =>
        Convert.ToDouble(machine.SystemMachine.BuildJobResourceSnapshot(state)[key]);

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
    public void FireThreeEntersAstralFireThree()
    {
        var machine = BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState;

        Assert.Equal(3, Int(state, "astral_fire"));
        Assert.Equal(0, Int(state, "umbral_ice"));
        Assert.Equal(8000, state.Mp);
        Assert.Equal(1, state.GcdIndex);
    }

    [Fact]
    public void HistoryResourceSnapshotsProjectAbsoluteTimers()
    {
        var machine = BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState;
        state = TimelineTestDriver.AdvanceBy(machine, state, state.GcdRemaining);

        var nextState = TimelineTestDriver.Execute(machine, state, "fire_iv").NextState;
        var entry = nextState.History[^1];

        Assert.Equal(0.0, Convert.ToDouble(entry.JobResourcesBefore["polyglot_timer"]), 5);
        Assert.Equal(2.0, Convert.ToDouble(entry.JobResourcesAfter["polyglot_timer"]), 5);
    }

    [Fact]
    public void FireThreeCostsDoubleMpInAstralFire()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("astral_fire", 3);
        state.Mp = 5000;

        var nextState = TimelineTestDriver.Execute(machine, state, "fire_iii").NextState;

        Assert.Equal(1000, nextState.Mp);
    }

    [Fact]
    public void AvailableActionsMatchCurrentState()
    {
        var machine = BuildMachine();
        var actionKeys = machine.AvailableActionKeys(machine.InitialState()).ToHashSet();

        Assert.Contains("fire_iii", actionKeys);
        Assert.Contains("ley_lines", actionKeys);
        Assert.DoesNotContain("fire_iv", actionKeys);
    }

    [Fact]
    public void ElementalSpendersRequireMatchingPolarity()
    {
        var machine = BuildMachine();

        var astralFire = machine.InitialState();
        astralFire.SetJobResource("astral_fire", 3);
        Assert.True(machine.ValidateAction(astralFire, "fire_iv").Ok);
        Assert.Equal("requires_ui", machine.ValidateAction(astralFire, "blizzard_iv").Reason);

        var umbralIce = machine.InitialState();
        umbralIce.SetJobResource("umbral_ice", 3);
        Assert.True(machine.ValidateAction(umbralIce, "blizzard_iv").Ok);
        Assert.Equal("requires_af", machine.ValidateAction(umbralIce, "fire_iv").Reason);
    }

    [Fact]
    public void ElementalCastTimeReductionOnlyAppliesAtThreeStacks()
    {
        var machine = BuildMachine();

        var astralFire = machine.InitialState();
        astralFire.SetJobResource("astral_fire", 3);
        var fireToIce = TimelineTestDriver.Execute(machine, astralFire, "blizzard_iii");
        Assert.Equal(1.75, fireToIce.NextState.History[^1].CastTimeSeconds, 5);

        var umbralIceOne = machine.InitialState();
        umbralIceOne.SetJobResource("umbral_ice", 1);
        var uiOneBlizzard = TimelineTestDriver.Execute(machine, umbralIceOne, "blizzard_iii");
        Assert.Equal(3.5, uiOneBlizzard.NextState.History[^1].CastTimeSeconds, 5);

        var umbralIceThree = machine.InitialState();
        umbralIceThree.SetJobResource("umbral_ice", 3);
        var iceToFire = TimelineTestDriver.Execute(machine, umbralIceThree, "fire_iii");
        Assert.Equal(1.75, iceToFire.NextState.History[^1].CastTimeSeconds, 5);
    }

    [Fact]
    public void FireFourConsumesMpAndGrantsAstralSoul()
    {
        var machine = BuildMachine();
        var state = ReadyAfterGcd(machine, TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState);

        var result = TimelineTestDriver.Execute(machine, state, "fire_iv");
        state = result.NextState;

        Assert.Equal(1, Int(state, "astral_soul"));
        Assert.Equal(6400, state.Mp);
    }

    [Fact]
    public void FireFourRemainsLegalWhenFlareStarIsReady()
    {
        var machine = BuildMachine();
        var state = ReadyAfterGcd(machine, TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState);
        state.SetJobResource("astral_soul", 6);

        var validation = machine.ValidateAction(state, "fire_iv");
        Assert.True(validation.Ok);

        var nextState = TimelineTestDriver.Execute(machine, state, "fire_iv").NextState;
        Assert.Equal(6, Int(nextState, "astral_soul"));
        Assert.Equal(6400, nextState.Mp);
    }

    [Fact]
    public void FlareConsumesAllUmbralHeartsHalvesMpAndGrantsThreeSouls()
    {
        var machine = BuildMachine();
        var state = ReadyAfterGcd(machine, TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState);
        state.Mp = 4000;
        state.SetJobResource("umbral_hearts", 3);
        state.SetJobResource("astral_soul", 2);

        var nextState = TimelineTestDriver.Execute(machine, state, "flare").NextState;

        Assert.Equal(3, Int(nextState, "astral_fire"));
        Assert.Equal(0, Int(nextState, "umbral_ice"));
        Assert.Equal(0, Int(nextState, "umbral_hearts"));
        Assert.Equal(5, Int(nextState, "astral_soul"));
        Assert.Equal(2000, nextState.Mp);
    }

    [Fact]
    public void FlareConsumesAllMpAndCapsSoulsWithoutUmbralHearts()
    {
        var machine = BuildMachine();
        var state = ReadyAfterGcd(machine, TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState);
        state.Mp = 2400;
        state.SetJobResource("astral_soul", 4);

        var nextState = TimelineTestDriver.Execute(machine, state, "flare").NextState;

        Assert.Equal(0, Int(nextState, "umbral_hearts"));
        Assert.Equal(6, Int(nextState, "astral_soul"));
        Assert.Equal(0, nextState.Mp);
    }

    [Fact]
    public void BlizzardCycleRestoresIceResources()
    {
        var machine = BuildMachine();
        var state = ReadyAfterGcd(machine, TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState);
        state.Mp = 0;

        var blizzardIii = TimelineTestDriver.Execute(machine, state, "blizzard_iii");
        Assert.Equal(2500, blizzardIii.NextState.Mp);

        state = ReadyAfterGcd(machine, blizzardIii.NextState);
        state = TimelineTestDriver.Execute(machine, state, "blizzard_iv").NextState;
        Assert.Equal(3, Int(state, "umbral_ice"));
        Assert.Equal(3, Int(state, "umbral_hearts"));
        Assert.Equal(10000, state.Mp);
    }

    [Fact]
    public void BlizzardFourUsesNominalMpInConfigButIsFreeInIce()
    {
        var machine = BuildMachine();
        Assert.Equal(800, machine.SkillBook.Get("blizzard_iv").MpCost);

        var neutralState = machine.InitialState();
        var neutralValidation = machine.ValidateAction(neutralState, "blizzard_iv");
        Assert.False(neutralValidation.Ok);
        Assert.Equal("requires_ui", neutralValidation.Reason);

        var iceState = ReadyAfterGcd(machine, TimelineTestDriver.Execute(machine, neutralState, "blizzard_iii").NextState);
        iceState.Mp = 0;
        Assert.True(machine.ValidateAction(iceState, "blizzard_iv").Ok);

        var nextState = TimelineTestDriver.Execute(machine, iceState, "blizzard_iv").NextState;
        Assert.Equal(10000, nextState.Mp);
    }

    [Fact]
    public void BlizzardThreeRecoveryUsesPreviousUmbralIceStacks()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("umbral_ice", 2);
        state.Mp = 0;

        var nextState = TimelineTestDriver.Execute(machine, state, "blizzard_iii").NextState;

        Assert.Equal(3, Int(nextState, "umbral_ice"));
        // 读条期间 t=3 的自然回蓝先结算，技能 effect 再按原 UI2 增加 5000 MP。
        Assert.Equal(5200, nextState.Mp);
    }

    [Fact]
    public void FreezeRequiresUiAndRestoresMp()
    {
        var machine = BuildMachine();
        var neutralValidation = machine.ValidateAction(machine.InitialState(), "freeze");
        Assert.False(neutralValidation.Ok);
        Assert.Equal("requires_ui", neutralValidation.Reason);

        var state = machine.InitialState();
        state.SetJobResource("umbral_ice", 3);
        state.Mp = 0;

        var nextState = TimelineTestDriver.Execute(machine, state, "freeze").NextState;
        Assert.Equal(3, Int(nextState, "umbral_hearts"));
        Assert.Equal(10000, nextState.Mp);
    }

    [Fact]
    public void ParadoxRequiresProcAndGrantsFirestarterInFire()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();
        Assert.Throws<InvalidOperationException>(() => TimelineTestDriver.Execute(machine, state, "paradox"));

        state.SetJobResource("paradox_ready", true);
        state.SetJobResource("astral_fire", 3);
        state.Mp = 2000;
        var nextState = TimelineTestDriver.Execute(machine, state, "paradox").NextState;
        Assert.False(Bool(nextState, "paradox_ready"));
        Assert.True(Bool(nextState, "firestarter_ready"));
        Assert.Equal(400, nextState.Mp);
    }

    [Fact]
    public void ParadoxInUmbralIceDoesNotRestoreMp()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("astral_fire", 3);
        state.Mp = 0;
        var blizzardIii = TimelineTestDriver.Execute(machine, state, "blizzard_iii");
        state = TimelineTestDriver.AdvanceBy(machine, blizzardIii.NextState, blizzardIii.NextGcdWindowSeconds);
        var mpBeforeParadox = state.Mp;

        var nextState = TimelineTestDriver.Execute(machine, state, "paradox").NextState;

        Assert.Equal(3, Int(state, "umbral_ice"));
        Assert.Equal(mpBeforeParadox, nextState.Mp);
        Assert.False(Bool(nextState, "paradox_ready"));
    }

    [Fact]
    public void HighThunderConsumesThundercloudAndAppliesDot()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("thundercloud_ready", true);
        var nextState = TimelineTestDriver.Execute(machine, state, "high_thunder").NextState;

        Assert.Contains("high_thunder", machine.SystemMachine.RegisteredTargetDotKeys);
        Assert.False(Bool(nextState, "thundercloud_ready"));
        Assert.Equal(30.0, nextState.Dots["high_thunder"].Remaining, 5);
    }

    [Fact]
    public void HighThunderTwoReusesHighThunderDotSlot()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("thundercloud_ready", true);

        var nextState = TimelineTestDriver.Execute(machine, state, "high_thunder_ii").NextState;

        Assert.Contains("high_thunder", machine.SystemMachine.RegisteredTargetDotKeys);
        Assert.False(Bool(nextState, "thundercloud_ready"));
        Assert.Equal(24.0, nextState.Dots["high_thunder"].Remaining, 5);
    }

    [Fact]
    public void XenoglossyConsumesPolyglot()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("polyglot", 2);

        var nextState = TimelineTestDriver.Execute(machine, state, "xenoglossy").NextState;

        Assert.Equal(1, Int(nextState, "polyglot"));
    }

    [Fact]
    public void AmplifierIsLegalButNoopsAtPolyglotCap()
    {
        var machine = BuildMachine();
        var state = ReadyAfterGcd(machine, TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState);
        state.SetJobResource("polyglot", 3);

        Assert.True(machine.ValidateAction(state, "amplifier").Ok);

        var nextState = TimelineTestDriver.Execute(machine, state, "amplifier").NextState;
        Assert.Equal(3, Int(nextState, "polyglot"));
    }

    [Fact]
    public void HighBlizzardTwoEntersUiThreeAndGrantsThundercloud()
    {
        var machine = BuildMachine();
        var nextState = TimelineTestDriver.Execute(machine, machine.InitialState(), "high_blizzard_ii").NextState;

        Assert.Equal(3, Int(nextState, "umbral_ice"));
        Assert.Equal(0, Int(nextState, "astral_fire"));
        Assert.True(Bool(nextState, "thundercloud_ready"));
        Assert.Equal(10000, nextState.Mp);
    }

    [Fact]
    public void HighFireTwoIsFreeInUiAndGrantsParadox()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("umbral_ice", 3);
        state.SetJobResource("umbral_hearts", 3);
        state.Mp = 0;

        var nextState = TimelineTestDriver.Execute(machine, state, "high_fire_ii").NextState;

        Assert.Equal(3, Int(nextState, "astral_fire"));
        Assert.Equal(0, Int(nextState, "umbral_ice"));
        Assert.True(Bool(nextState, "paradox_ready"));
        Assert.True(Bool(nextState, "thundercloud_ready"));
        Assert.Equal(0, nextState.Mp);
    }

    [Fact]
    public void HighFireTwoCostsDoubleMpInAstralFire()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("astral_fire", 3);
        state.Mp = 3000;

        var nextState = TimelineTestDriver.Execute(machine, state, "high_fire_ii").NextState;

        Assert.Equal(0, nextState.Mp);
    }

    [Fact]
    public void TriplecastMakesNextHardcastInstant()
    {
        var machine = BuildMachine();
        var state = ReadyAfterGcd(machine, TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState);
        state = TimelineTestDriver.Execute(machine, state, "triplecast").NextState;
        Assert.Equal(3, state.StatusStacks("triplecast"));

        var nextState = TimelineTestDriver.Execute(machine, state, "fire_iv").NextState;
        Assert.Equal(2, nextState.StatusStacks("triplecast"));
        Assert.Equal(2.5, nextState.WeaveWindowRemaining, 5);
    }

    [Fact]
    public void FirestarterDoesNotConsumeTriplecastForFireThree()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("firestarter_ready", true);
        state = TimelineTestDriver.Execute(machine, state, "triplecast").NextState;

        var nextState = TimelineTestDriver.Execute(machine, state, "fire_iii").NextState;

        Assert.False(Bool(nextState, "firestarter_ready"));
        Assert.Equal(3, nextState.StatusStacks("triplecast"));
        Assert.Equal(10000, nextState.Mp);
    }

    [Fact]
    public void FirestarterDoesNotConsumeSwiftcastForFireThree()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("firestarter_ready", true);
        state = TimelineTestDriver.Execute(machine, state, "swiftcast").NextState;

        var nextState = TimelineTestDriver.Execute(machine, state, "fire_iii").NextState;

        Assert.False(Bool(nextState, "firestarter_ready"));
        Assert.True(nextState.HasStatus("swiftcast"));
        Assert.Equal(10000, nextState.Mp);
    }

    [Fact]
    public void SwiftcastIsConsumedBeforeTriplecast()
    {
        var machine = BuildMachine();
        var state = ReadyAfterGcd(machine, TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState);
        state = TimelineTestDriver.Execute(machine, state, "triplecast").NextState;
        state = TimelineTestDriver.Execute(machine, state, "swiftcast").NextState;

        var nextState = TimelineTestDriver.Execute(machine, state, "fire_iv").NextState;

        Assert.False(nextState.HasStatus("swiftcast"));
        Assert.Equal(3, nextState.StatusStacks("triplecast"));
    }

    [Fact]
    public void RemovedUtilitySkillsAreAbsentFromTheModelSkillBook()
    {
        var machine = BuildMachine();

        Assert.False(machine.SkillBook.Contains("retrace"));
        Assert.False(machine.SkillBook.Contains("manaward"));
        Assert.False(machine.SkillBook.Contains("surecast"));
        Assert.False(machine.JobMachine.SupportsBehavior("ley_lines_utility"));

        var available = machine.AvailableActionKeys(machine.InitialState());
        Assert.DoesNotContain("retrace", available);
        Assert.DoesNotContain("manaward", available);
        Assert.DoesNotContain("surecast", available);
    }

    [Fact]
    public void TransposeRequiresElementalState()
    {
        var machine = BuildMachine();
        var validation = machine.ValidateAction(machine.InitialState(), "transpose");
        Assert.False(validation.Ok);
        Assert.Equal("requires_elemental_state", validation.Reason);

        Assert.Throws<InvalidOperationException>(() => TimelineTestDriver.Execute(machine, machine.InitialState(), "transpose"));

        var state = ReadyAfterGcd(machine, TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState);
        Assert.True(machine.ValidateAction(state, "transpose").Ok);
    }

    [Fact]
    public void TransposeGrantsParadoxOnlyAtRequiredElementalThresholds()
    {
        var machine = BuildMachine();

        CombatState TransposeFrom(int astralFire, int umbralIce, int umbralHearts)
        {
            var state = machine.InitialState();
            state.SetJobResource("astral_fire", astralFire);
            state.SetJobResource("umbral_ice", umbralIce);
            state.SetJobResource("umbral_hearts", umbralHearts);
            return TimelineTestDriver.Execute(machine, state, "transpose").NextState;
        }

        Assert.False(Bool(TransposeFrom(1, 0, 0), "paradox_ready"));
        Assert.True(Bool(TransposeFrom(3, 0, 0), "paradox_ready"));
        Assert.False(Bool(TransposeFrom(0, 1, 3), "paradox_ready"));
        Assert.False(Bool(TransposeFrom(0, 3, 2), "paradox_ready"));
        Assert.True(Bool(TransposeFrom(0, 3, 3), "paradox_ready"));
    }

    [Fact]
    public void DespairFromAstralFireOneRestoresAstralFireThree()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();
        state.SetJobResource("astral_fire", 1);
        state.Mp = 2000;

        var nextState = TimelineTestDriver.Execute(machine, state, "despair").NextState;

        Assert.Equal(3, Int(nextState, "astral_fire"));
        Assert.Equal(0, nextState.Mp);
    }

    [Fact]
    public void PolyglotGrowsOverTimeWhenElementalStateIsActive()
    {
        var machine = BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState;

        var nextState = TimelineTestDriver.AdvanceBy(machine, state, 30.0);

        Assert.Equal(1, Int(nextState, "polyglot"));
    }

    [Fact]
    public void HumanRotationTraceReplaysCleanly()
    {
        var tracePath = Path.Combine(RepoRoot, "old", "docs", "human_rotation.txt");
        var rows = new List<(int StepIndex, double ActionTime, int SkillId)>();
        var pattern = new Regex(@"^\s*(\d+)\s+(\d+\.\d+)s\s+(\d+)\s+", RegexOptions.Compiled);
        foreach (var line in File.ReadLines(tracePath))
        {
            var match = pattern.Match(line);
            if (!match.Success)
            {
                continue;
            }

            rows.Add((
                int.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture),
                double.Parse(match.Groups[2].Value, CultureInfo.InvariantCulture),
                int.Parse(match.Groups[3].Value, CultureInfo.InvariantCulture)));
        }

        Assert.NotEmpty(rows);

        var machine = BuildMachine();
        var state = machine.InitialState();
        double? prevTime = null;
        foreach (var (stepIndex, actionTime, skillId) in rows)
        {
            if (prevTime is { } previous)
            {
                state = TimelineTestDriver.AdvanceBy(machine, state, actionTime - previous);
            }

            var skill = machine.SkillBook.Get(skillId);
            var validation = machine.ValidateAction(state, skill.Key);
            Assert.True(validation.Ok,
                $"human_rotation failed at step={stepIndex}, time={actionTime}, " +
                $"skill={skill.Key}, reason={validation.Reason}");

            state = TimelineTestDriver.Execute(machine, state, skill.Key).NextState;
            prevTime = actionTime;
        }
    }

    [Fact]
    public void UnregisteredBehaviorPassesThrough()
    {
        var machine = BuildMachine();
        var jobMachine = machine.JobMachine;

        // 行为不在黑魔集合中时直接放行，由上游继续走系统层校验（对照 validate_action 的放行分支）。
        var fakeSkill = new SkillDefinition(
            Key: "test",
            GameId: -1,
            Name: "test",
            Kind: ActionKind.Gcd,
            Behavior: "_test_fake");

        var result = jobMachine.ValidateAction(machine.InitialState(), fakeSkill);
        Assert.True(result.Ok);
    }

    [Fact]
    public void AstralFireBlocksSharedNaturalMpRecovery()
    {
        var machine = BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState;
        state.Mp = 6000;

        state = TimelineTestDriver.AdvanceBy(machine, state, 6.0);

        Assert.Equal(6000, state.Mp);
    }

    [Fact]
    public void AstralFireBlocksLucidDreamingMpRecovery()
    {
        var machine = BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState;
        state.Mp = 4000;
        state = TimelineTestDriver.Execute(machine, state, "lucid_dreaming").NextState;

        state = TimelineTestDriver.AdvanceBy(machine, state, 21.0);

        Assert.Equal(4000, state.Mp);
        Assert.False(state.HasStatus("lucid_dreaming"));
    }

    [Fact]
    public void UmbralSoulStacksUiAndHearts()
    {
        var machine = BuildMachine();
        var state = TimelineTestDriver.Execute(machine, machine.InitialState(), "fire_iii").NextState;
        state = TimelineTestDriver.AdvanceBy(machine, state, 2.5);
        var transpose = TimelineTestDriver.Execute(machine, state, "transpose");
        state = TimelineTestDriver.AdvanceBy(machine, transpose.NextState, transpose.NextGcdWindowSeconds);
        state.BossTargetable = false;
        state.DowntimeRemaining = 30.0;
        var firstSoul = TimelineTestDriver.Execute(machine, state, "umbral_soul");
        state = TimelineTestDriver.AdvanceBy(machine, firstSoul.NextState, firstSoul.NextGcdWindowSeconds);
        var secondSoul = TimelineTestDriver.Execute(machine, state, "umbral_soul");
        state = secondSoul.NextState;

        Assert.Equal(3, Int(state, "umbral_ice"));
        Assert.Equal(2, Int(state, "umbral_hearts"));
    }

    [Fact]
    public void PolyglotTimerResetsWithoutElementalState()
    {
        var machine = BuildMachine();
        var state = machine.InitialState();
        // 走对外视图写入：由翻译层换算成绝对截止时刻。
        machine.SystemMachine.SetJobResource(state, "polyglot_timer", 12.0);
        state.SetJobResource("polyglot", 1);

        var nextState = TimelineTestDriver.AdvanceBy(machine, state, 5.0);

        // 无元素态时通晓周期不在计时，对外视图归零，且不产生通晓。
        Assert.Equal(0.0, TimerView(machine, nextState, "polyglot_timer"), 5);
        Assert.Equal(1, Int(nextState, "polyglot"));
    }
}

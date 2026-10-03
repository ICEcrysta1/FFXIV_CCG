using Combat.Sim.Sessions;
using Combat.Sim.Facade;

namespace FightEngine.Tests.Facade;

public sealed class RandomSequenceReplayTests
{
    private static readonly string RepoRoot = FindRepoRoot();

    [Fact]
    public void 固定seed绝对时间随机序列可以逐步复现()
    {
        using var engine = new SimulationEngine(RepoRoot, "black_mage");
        using var firstSession = engine.CreateSession(null);
        using var secondSession = engine.CreateSession(null);
        var first = RandomSequenceReplay.Run(
            firstSession,
            RandomSequenceReplay.DefaultSeed,
            RandomSequenceReplay.DefaultMaxSteps);
        var second = RandomSequenceReplay.Run(
            secondSession,
            RandomSequenceReplay.DefaultSeed,
            RandomSequenceReplay.DefaultMaxSteps);

        Assert.Equal(first.Count, second.Count);
        for (var index = 0; index < first.Count; index++)
        {
            Assert.Equal(first[index].Kind, second[index].Kind);
            Assert.Equal(first[index].Action, second[index].Action);
            Assert.Equal(first[index].Timestamp, second[index].Timestamp, 9);
            Assert.Equal(first[index].StateAfter.Time, second[index].StateAfter.Time, 9);
            Assert.Equal(first[index].StateAfter.Mp, second[index].StateAfter.Mp);
            Assert.Equal(first[index].StateAfter.GcdIndex, second[index].StateAfter.GcdIndex);
            Assert.DoesNotContain("ogcd_wait", first[index].LegalKeys);
        }
    }

    [Fact]
    public void 多队列固定随机流与串行逐步一致()
    {
        using var engine = new SimulationEngine(RepoRoot, "black_mage", 16);
        IReadOnlyList<RandomSequenceStep> expected;
        using (var session = engine.CreateSession(4))
            expected = RandomSequenceReplay.Run(session, 42, 40);
        Parallel.For(0, 16, index =>
        {
            using var session = engine.CreateSession(4);
            var actual = RandomSequenceReplay.Run(session, 42, 40);
            Assert.Equal(expected.Select(s => (s.Kind, s.Action, s.Timestamp, s.StateAfter.Mp)),
                         actual.Select(s => (s.Kind, s.Action, s.Timestamp, s.StateAfter.Mp)));
        });
        Assert.Equal(0, engine.ActiveCount);
    }

    private static string FindRepoRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "config", "default.yaml")))
                return dir.FullName;
        }
        throw new InvalidOperationException("仓库根未找到（缺少 config/default.yaml）");
    }
}

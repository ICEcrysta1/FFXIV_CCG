using Combat.Sim.Facade;
using Combat.Sim.Models.Combat;

namespace FightEngine.Tests.Outputs;

/// <summary>输出层测试共享：仓库根定位、黑魔状态机与向量取值辅助（对照 tests/helpers.py）。</summary>
public static class OutputsTestKit
{
    public static readonly string RepoRoot = FindRepoRoot();

    public static CombatStateMachine BuildMachine() =>
        CombatStateMachine.FromDefaultConfig(RepoRoot, "black_mage");

    /// <summary>推进到 GCD 就绪（对照 ready_after_gcd）。</summary>
    public static CombatState ReadyAfterGcd(CombatStateMachine machine, CombatState state) =>
        state.GcdRemaining <= 0 ? state : TimelineTestDriver.AdvanceBy(machine, state, state.GcdRemaining);

    /// <summary>按 feature key 取状态历史 token 的向量值（对照 history_vector_value）。</summary>
    public static double HistoryVectorValue(
        Dictionary<string, object?> historyState,
        string groupKey,
        string featureKey,
        int tokenIndex = 0)
    {
        var featureKeys = (List<string>)historyState[$"{groupKey}_feature_keys"];
        var tokens = (List<Dictionary<string, double[]>>)historyState["tokens"];
        return tokens[tokenIndex][groupKey][featureKeys.IndexOf(featureKey)];
    }

    /// <summary>按 feature key 取请求时的当前状态向量。</summary>
    public static double CurrentStateVectorValue(
        Dictionary<string, object?> payload,
        string groupKey,
        string featureKey)
    {
        return HistoryVectorValue((Dictionary<string, object?>)payload["current_state_context"]!, groupKey, featureKey);
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
}

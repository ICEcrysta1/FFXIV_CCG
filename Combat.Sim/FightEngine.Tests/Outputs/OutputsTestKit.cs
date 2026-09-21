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

    /// <summary>按技能 key 与 feature key 取候选状态 token 的向量值（对照 candidate_state_vector_value）。</summary>
    public static double? CandidateStateVectorValue(
        Dictionary<string, object?> payload,
        string skillKey,
        string groupKey,
        string featureKey)
    {
        var candidateSkillContext = (List<Dictionary<string, object?>>)payload["candidate_skill_context"];
        var candidateStateContext = (Dictionary<string, object?>)payload["candidate_state_context"];
        var tokens = (List<object>)candidateStateContext["tokens"];
        var index = candidateSkillContext.FindIndex(token => (string)token["skill_key"] == skillKey);
        var vector = VectorOf(tokens[index], groupKey);
        var featureKeys = (List<string>)candidateStateContext[$"{groupKey}_feature_keys"];
        return vector[featureKeys.IndexOf(featureKey)];
    }

    /// <summary>取候选状态 token 某个分组的向量（合法/非法 token 统一转 double? 列表）。</summary>
    public static IReadOnlyList<double?> VectorOf(object token, string groupKey) =>
        token switch
        {
            Dictionary<string, double[]> legal => legal[groupKey].Select(value => (double?)value).ToList(),
            Dictionary<string, double?[]> illegal => illegal[groupKey].ToList(),
            _ => throw new InvalidOperationException($"unsupported token type: {token.GetType().Name}"),
        };

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

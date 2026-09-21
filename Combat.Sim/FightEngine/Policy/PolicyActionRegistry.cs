// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Policy;
using Combat.Sim.Skills;

namespace Combat.Sim.Policy;

/// <summary>独立的 policy 控制动作索引；构造时拒绝与真实技能冲突。</summary>
public sealed class PolicyActionRegistry
{
    private readonly IReadOnlyList<PolicyActionDefinition> _ordered;
    private readonly Dictionary<string, PolicyActionDefinition> _byKey;

    public PolicyActionRegistry(IEnumerable<PolicyActionDefinition> actions, SkillBook skillBook)
    {
        _ordered = actions.ToArray();
        _byKey = new Dictionary<string, PolicyActionDefinition>(StringComparer.Ordinal);
        var rawIds = new HashSet<int>();
        foreach (var action in _ordered)
        {
            if (!_byKey.TryAdd(action.Key, action))
                throw new InvalidOperationException($"duplicate policy action key: {action.Key}");
            if (!rawIds.Add(action.RawId))
                throw new InvalidOperationException($"duplicate policy action raw_id: {action.RawId}");
            if (skillBook.Contains(action.Key) || skillBook.Contains(action.RawId))
                throw new InvalidOperationException($"policy action conflicts with game skill: {action.Key}");
            if (action.CandidateKind is not ("gcd" or "ogcd"))
                throw new InvalidOperationException(
                    $"unsupported policy candidate kind for {action.Key}: {action.CandidateKind}");
        }
    }

    public static PolicyActionRegistry Load(string projectRoot, SkillBook skillBook) =>
        new(PolicyConfigLoader.Load(projectRoot), skillBook);

    public IReadOnlyList<PolicyActionDefinition> Actions => _ordered;
    public PolicyActionDefinition Get(string key) => _byKey[key];
    public bool Contains(string key) => _byKey.ContainsKey(key);
}

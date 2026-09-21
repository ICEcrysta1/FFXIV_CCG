// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Common;
using Combat.Sim.Models.Policy;

namespace Combat.Sim.Policy;

/// <summary>独立读取 policy_actions.yaml，不参与项目技能配置装配。</summary>
public static class PolicyConfigLoader
{
    public static IReadOnlyList<PolicyActionDefinition> Load(string projectRoot)
    {
        var path = Path.Combine(projectRoot, "config", "policy_actions.yaml");
        var root = YamlConfig.LoadYamlMapping(path, "policy action config");
        var raw = YamlValues.RequireKey(root, "policy_actions", path);
        if (raw is not Dictionary<string, object?> actions)
        {
            throw new InvalidOperationException($"{path}: policy_actions must be a mapping");
        }

        return actions.Select(pair => Build(pair.Key, pair.Value, path)).ToArray();
    }

    private static PolicyActionDefinition Build(string key, object? raw, string path)
    {
        if (raw is not Dictionary<string, object?> data)
        {
            throw new InvalidOperationException($"{path}: policy action {key} must be a mapping");
        }

        var tags = YamlValues.Get(data, "tags", new List<object?>()) is List<object?> values
            ? values.Select(YamlValues.ToText).ToArray()
            : throw new InvalidOperationException($"{path}: policy action {key}.tags must be a sequence");
        return new PolicyActionDefinition(
            key,
            YamlValues.ToInt(YamlValues.RequireKey(data, "raw_id", path), path),
            YamlValues.ToText(YamlValues.RequireKey(data, "name", path)),
            YamlValues.ToText(YamlValues.RequireKey(data, "candidate_kind", path)),
            YamlValues.ToText(YamlValues.RequireKey(data, "behavior", path)),
            YamlValues.ToDouble(YamlValues.Get(data, "value", 1.0), path),
            tags);
    }
}

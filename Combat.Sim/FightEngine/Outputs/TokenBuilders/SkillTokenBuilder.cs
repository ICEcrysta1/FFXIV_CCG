// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Outputs.TokenBuilders;

/// <summary>
/// 统一技能 token 装配器（对照 token_builders/skill_token_builder.py）。
/// 技能历史与策略动作历史都通过同一个 build 函数生成固定字段。
/// </summary>
public static class SkillTokenBuilder
{
    public static Dictionary<string, object?> Build(
        int skillId,
        string skillKey,
        string skillName,
        double potency,
        double value,
        string kind,
        int actualMpCost,
        double castTimeSeconds,
        double gcdWindowSeconds,
        bool isLegal,
        string invalidReason,
        double nextCooldownSeconds,
        int availableCharges,
        int maxCharges,
        IReadOnlyDictionary<string, object> jobResourcesConsumed)
    {
        var encodedKind = kind switch
        {
            "gcd" => 1,
            "ogcd" => 0,
            _ => throw new InvalidOperationException($"unsupported skill kind: {kind}"),
        };

        var payload = new Dictionary<string, object?>
        {
            ["skill_id"] = skillId,
            ["skill_key"] = skillKey,
            ["skill_name"] = skillName,
            ["potency"] = potency,
            ["value"] = value,
            // kind 同时作为输出 token 字段和 skill 数值特征；1=gcd，0=ogcd。
            ["kind"] = encodedKind,
            ["actual_mp_cost"] = actualMpCost,
            ["cast_time"] = new Dictionary<string, object?>
            {
                ["seconds"] = Math.Round(castTimeSeconds, 4),
            },
            ["gcd_window"] = new Dictionary<string, object?>
            {
                ["seconds"] = Math.Round(gcdWindowSeconds, 4),
            },
            ["is_legal"] = isLegal,
            ["invalid_reason"] = invalidReason,
            ["next_cooldown_seconds"] = Math.Round(nextCooldownSeconds, 4),
            ["available_charges"] = availableCharges,
            ["max_charges"] = maxCharges,
            ["job_resources_consumed"] = jobResourcesConsumed.ToDictionary(
                entry => entry.Key,
                entry => (object?)entry.Value),
        };
        return payload;
    }
}

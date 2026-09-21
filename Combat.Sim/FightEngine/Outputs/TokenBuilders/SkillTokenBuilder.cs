// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Outputs.TokenBuilders;

/// <summary>
/// 统一技能 token 装配器（对照 token_builders/skill_token_builder.py）。
/// 只提供这一个 build 函数：技能历史条目与候选技能条目都调用它，
/// 只是调用方传入的属性来源不同（历史条目 vs 候选预演条目）。
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
        IReadOnlyDictionary<string, object> jobResourcesConsumed,
        double? timeSeconds = null,
        int? gcdIndex = null)
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
        if (timeSeconds is not null)
        {
            payload["time_seconds"] = Math.Round(timeSeconds.Value, 4);
        }

        if (gcdIndex is not null)
        {
            payload["gcd_index"] = gcdIndex;
        }

        return payload;
    }
}

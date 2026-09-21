// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Config;
using Combat.Sim.Models.Definitions;

namespace Combat.Sim.Skills;

/// <summary>
/// 当前职业的技能查询表（对照 skills.SkillBook）。
/// 只从正式项目配置构建 key / game_id 索引，不接收单个技能列表之外的反推来源。
/// </summary>
public sealed class SkillBook
{
    private readonly IReadOnlyList<SkillDefinition> _ordered;
    private readonly Dictionary<string, SkillDefinition> _byKey;
    private readonly Dictionary<int, SkillDefinition> _byGameId;

    public SkillBook(IEnumerable<SkillDefinition> skills)
    {
        _ordered = skills.ToList();
        _byKey = new Dictionary<string, SkillDefinition>(StringComparer.Ordinal);
        _byGameId = new Dictionary<int, SkillDefinition>();
        foreach (var skill in _ordered)
        {
            if (_byKey.ContainsKey(skill.Key))
            {
                throw new InvalidOperationException($"duplicate skill key: {skill.Key}");
            }
            if (_byGameId.ContainsKey(skill.GameId))
            {
                throw new InvalidOperationException($"duplicate skill game_id: {skill.GameId}");
            }
            _byKey[skill.Key] = skill;
            _byGameId[skill.GameId] = skill;
        }
    }

    /// <summary>按项目配置构建：系统技能 + 职业技能（对照 from_project_config）。</summary>
    public static SkillBook FromProjectConfig(ProjectConfig projectConfig) =>
        new(projectConfig.System.Skills.Concat(projectConfig.Job.Skills));

    /// <summary>按 key 查询技能定义；未命中抛 <see cref="KeyNotFoundException"/>。</summary>
    public SkillDefinition Get(string key) => _byKey[key];

    /// <summary>按游戏内 id 查询技能定义；未命中抛 <see cref="KeyNotFoundException"/>。</summary>
    public SkillDefinition Get(int gameId) => _byGameId[gameId];

    /// <summary>返回全部启用技能，可选按技能类型过滤（对照 enabled_skills）。</summary>
    public IReadOnlyList<SkillDefinition> EnabledSkills(ActionKind? kind = null)
    {
        var skills = _ordered.Where(skill => skill.Enabled);
        if (kind is { } kindValue)
        {
            skills = skills.Where(skill => skill.Kind == kindValue);
        }
        return skills.ToList();
    }

    /// <summary>返回有序技能 key 列表（对照 keys）。</summary>
    public IReadOnlyList<string> Keys() => _ordered.Select(skill => skill.Key).ToList();

    public bool Contains(string key) => _byKey.ContainsKey(key);

    public bool Contains(int gameId) => _byGameId.ContainsKey(gameId);
}

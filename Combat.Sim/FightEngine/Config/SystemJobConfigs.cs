// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Models.Definitions;

namespace Combat.Sim.Config;

/// <summary>职业配置（对照 config.JobConfig）。</summary>
public sealed record JobConfig(
    string Key,
    string Name,
    IReadOnlyDictionary<string, double> Timing,
    IReadOnlyDictionary<string, double> ResourceLimits,
    IReadOnlyDictionary<string, StatusDefinition> Statuses,
    IReadOnlyList<SkillDefinition> Skills)
{
    /// <summary>对应 JobConfig.default_fight_remaining。</summary>
    public double DefaultFightRemaining => Timing["default_fight_remaining"];

    /// <summary>对应 JobConfig.timing_value(key)。</summary>
    public double TimingValue(string key) => Timing[key];
}

/// <summary>系统级共享配置（对照 config.SystemConfig）。</summary>
public sealed record SystemConfig(
    double BaseGcd,
    double SkillTableBaseGcd,
    SystemMpRecoveryConfig MpRecovery,
    SystemPotencyConfig Potency,
    IReadOnlyList<string> RaidBuffWindowMarkerSkills,
    double RaidBuffWindowDuration,
    IReadOnlyDictionary<string, StatusDefinition> Statuses,
    IReadOnlyList<SkillDefinition> Skills);

/// <summary>项目总配置（对照 config.ProjectConfig）。</summary>
public sealed record ProjectConfig(
    string Name,
    string Version,
    RuntimeConfig Runtime,
    EngineTimingConfig EngineTiming,
    EngineResourceConfig EngineResources,
    SystemConfig System,
    JobConfig Job);

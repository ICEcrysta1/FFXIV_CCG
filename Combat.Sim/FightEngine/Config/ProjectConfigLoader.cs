// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Common;
using Combat.Sim.Models.Definitions;

namespace Combat.Sim.Config;

/// <summary>
/// 项目配置加载层。
/// 只负责读取 YAML 配置并装配成配置数据类，不处理战斗逻辑；
/// 职业标签必须由调用方显式传入（C# 侧不读取 .env）。
/// </summary>
public static class ProjectConfigLoader
{
    /// <summary>按项目根目录加载默认配置（<c>&lt;root&gt;/config/default.yaml</c>）。</summary>
    public static ProjectConfig LoadProjectConfig(string projectRoot, string jobTag)
    {
        var configPath = Path.Combine(projectRoot, "config", "default.yaml");
        return LoadProjectConfigFromPath(configPath, jobTag);
    }

    /// <summary>从显式 default.yaml 路径加载；项目根由配置路径推导（config 的上级目录）。</summary>
    public static ProjectConfig LoadProjectConfigFromPath(string configPath, string jobTag)
    {
        var payload = YamlConfig.LoadYamlMapping(configPath);
        var projectRoot = Directory.GetParent(Path.GetDirectoryName(configPath)!)!.FullName;

        var runtimeData = RequireMapping(payload, "runtime", configPath);
        var engineTimingData = RequireMapping(payload, "engine_timing", configPath);
        var engineResourceData = RequireMapping(payload, "engine_resources", configPath);

        var systemConfigPath = YamlValues.ResolveProjectPath(
            YamlValues.RequireKey(runtimeData, "system_config", configPath),
            projectRoot);
        var jobConfigPath = ResolveJobConfigPath(runtimeData, jobTag, projectRoot, configPath);
        var systemPayload = YamlConfig.LoadYamlMapping(systemConfigPath);
        var jobPayload = YamlConfig.LoadYamlMapping(jobConfigPath);

        var runtime = BuildRuntimeConfig(jobTag, systemConfigPath, jobConfigPath);
        var (engineTiming, engineResources) = BuildEngineConfigs(
            engineTimingData,
            engineResourceData,
            configPath);
        var system = BuildSystemConfig(systemPayload, systemConfigPath);
        var job = BuildJobConfig(jobPayload, jobConfigPath);
        ValidateCrossContracts(system, job);

        var projectData = RequireMapping(payload, "project", configPath);
        return new ProjectConfig(
            Name: YamlValues.ToText(YamlValues.RequireKey(projectData, "name", configPath)),
            Version: YamlValues.ToText(YamlValues.RequireKey(projectData, "version", configPath)),
            Runtime: runtime,
            EngineTiming: engineTiming,
            EngineResources: engineResources,
            System: system,
            Job: job);
    }

    private static RuntimeConfig BuildRuntimeConfig(
        string jobTag,
        string systemConfigPath,
        string jobConfigPath)
    {
        return new RuntimeConfig(
            JobTag: jobTag,
            SystemConfigPath: systemConfigPath,
            JobConfigPath: jobConfigPath);
    }

    private static (EngineTimingConfig Timing, EngineResourceConfig Resources) BuildEngineConfigs(
        Dictionary<string, object?> engineTimingData,
        Dictionary<string, object?> engineResourceData,
        string configPath)
    {
        var timing = new EngineTimingConfig(
            ActionQueueWindowSeconds: YamlValues.ToDouble(
                YamlValues.RequireKey(engineTimingData, "action_queue_window_seconds", configPath), configPath));
        var resources = new EngineResourceConfig(
            MaxMp: YamlValues.ToInt(
                YamlValues.RequireKey(engineResourceData, "max_mp", configPath), configPath));
        return (timing, resources);
    }

    private static SystemConfig BuildSystemConfig(
        Dictionary<string, object?> systemPayload,
        string systemConfigPath)
    {
        var systemTiming = RequireMapping(systemPayload, "timing", systemConfigPath);
        var systemPotencyData = GetMappingOrEmpty(systemPayload, "potency", systemConfigPath);
        var raidBuffWindowData = RequireMapping(systemPayload, "raid_buff_window", systemConfigPath);
        var markerSkills = ToStringList(
            YamlValues.RequireKey(raidBuffWindowData, "marker_skills", systemConfigPath));
        if (markerSkills.Count == 0)
        {
            throw new InvalidOperationException("raid_buff_window.marker_skills must not be empty");
        }
        if (markerSkills.Distinct().Count() != markerSkills.Count)
        {
            throw new InvalidOperationException("raid_buff_window.marker_skills must not contain duplicates");
        }
        var raidBuffWindowDuration = YamlValues.ToDouble(
            YamlValues.RequireKey(raidBuffWindowData, "duration_seconds", systemConfigPath), systemConfigPath);
        if (raidBuffWindowDuration <= 0)
        {
            throw new InvalidOperationException("raid_buff_window.duration_seconds must be positive");
        }

        var mpRecoveryData = RequireMapping(systemPayload, "mp_recovery", systemConfigPath);
        return new SystemConfig(
            BaseGcd: YamlValues.ToDouble(
                YamlValues.RequireKey(systemTiming, "base_gcd", systemConfigPath), systemConfigPath),
            SkillTableBaseGcd: YamlValues.ToDouble(
                YamlValues.RequireKey(systemTiming, "skill_table_base_gcd", systemConfigPath), systemConfigPath),
            MpRecovery: new SystemMpRecoveryConfig(
                TickIntervalSeconds: YamlValues.ToDouble(
                    YamlValues.RequireKey(mpRecoveryData, "tick_interval_seconds", systemConfigPath), systemConfigPath),
                InCombatAmount: YamlValues.ToInt(
                    YamlValues.RequireKey(mpRecoveryData, "in_combat_amount", systemConfigPath), systemConfigPath)),
            Potency: new SystemPotencyConfig(
                BurstPotionMultiplier: YamlValues.ToDouble(
                    YamlValues.Get(systemPotencyData, "burst_potion_multiplier", 1.05), systemConfigPath),
                RaidBuffWindowMultiplier: YamlValues.ToDouble(
                    YamlValues.Get(systemPotencyData, "raid_buff_window_multiplier", 1.10), systemConfigPath)),
            RaidBuffWindowMarkerSkills: markerSkills,
            RaidBuffWindowDuration: raidBuffWindowDuration,
            Statuses: BuildStatusDefinitions(
                GetMappingOrEmpty(systemPayload, "statuses", systemConfigPath), systemConfigPath),
            Skills: BuildSkillDefinitions(
                GetMappingOrEmpty(systemPayload, "skills", systemConfigPath), systemConfigPath));
    }

    private static JobConfig BuildJobConfig(
        Dictionary<string, object?> jobPayload,
        string jobConfigPath)
    {
        var jobTiming = RequireMapping(jobPayload, "timing", jobConfigPath);
        var jobData = RequireMapping(jobPayload, "job", jobConfigPath);
        return new JobConfig(
            Key: YamlValues.ToText(YamlValues.RequireKey(jobData, "key", jobConfigPath)),
            Name: YamlValues.ToText(YamlValues.RequireKey(jobData, "name", jobConfigPath)),
            Timing: jobTiming.ToDictionary(
                pair => pair.Key,
                pair => YamlValues.ToDouble(pair.Value, jobConfigPath),
                StringComparer.Ordinal),
            ResourceLimits: BuildResourceLimits(
                YamlValues.Get(jobPayload, "resources", null), jobConfigPath),
            Statuses: BuildStatusDefinitions(
                GetMappingOrEmpty(jobPayload, "statuses", jobConfigPath), jobConfigPath),
            Skills: BuildSkillDefinitions(
                GetMappingOrEmpty(jobPayload, "skills", jobConfigPath), jobConfigPath));
    }

    private static void ValidateCrossContracts(SystemConfig system, JobConfig job)
    {
        ValidateTimingResourceContracts(job);
        ValidateStatusDefinitions(system, job);
        ValidateStatusConflicts(system, job);
        ValidateSkillConflicts(system, job);
        ValidateSkillStatusReferences(system, job);
    }

    /// <summary>按职业标签解析已注册的职业 YAML（对照 _resolve_job_config_path）。</summary>
    private static string ResolveJobConfigPath(
        Dictionary<string, object?> runtimeData,
        string jobTag,
        string projectRoot,
        string configPath)
    {
        var configuredPaths = YamlValues.Get(runtimeData, "job_configs", null);
        if (configuredPaths is not Dictionary<string, object?> paths)
        {
            throw new InvalidOperationException("runtime.job_configs must be a mapping");
        }
        if (!paths.TryGetValue(jobTag, out var configuredPath))
        {
            var supported = string.Join(", ", paths.Keys.OrderBy(k => k, StringComparer.Ordinal));
            throw new InvalidOperationException(
                $"no job config registered for {jobTag}; supported={supported}");
        }
        return YamlValues.ResolveProjectPath(configuredPath, projectRoot);
    }

    /// <summary>构建状态定义（对照 _build_status_definitions）。</summary>
    private static Dictionary<string, StatusDefinition> BuildStatusDefinitions(
        Dictionary<string, object?> rawStatuses,
        string context)
    {
        var definitions = new Dictionary<string, StatusDefinition>(StringComparer.Ordinal);
        foreach (var (key, rawPayload) in rawStatuses)
        {
            var payload = RequireMappingValue(rawPayload, $"status {key}", context);
            definitions[key] = new StatusDefinition(
                Key: key,
                GameId: YamlValues.ToInt(YamlValues.RequireKey(payload, "game_id", context), context),
                Duration: YamlValues.ToDouble(YamlValues.RequireKey(payload, "duration", context), context),
                MaxStacks: YamlValues.ToInt(YamlValues.Get(payload, "max_stacks", 1L), context));
        }
        return definitions;
    }

    /// <summary>读取职业量谱上限（对照 _build_resource_limits）。</summary>
    private static Dictionary<string, double> BuildResourceLimits(object? rawResources, string context)
    {
        if (rawResources is null)
        {
            return new Dictionary<string, double>(StringComparer.Ordinal);
        }
        if (rawResources is not Dictionary<string, object?> resources)
        {
            throw new InvalidOperationException("job resources must be a mapping");
        }
        var limits = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (var (key, rawPayload) in resources)
        {
            var payload = RequireMappingValue(rawPayload, $"job resource {key}", context);
            if (!payload.TryGetValue("max_value", out var rawMaxValue))
            {
                throw new InvalidOperationException($"job resource {key} is missing max_value");
            }
            var maxValue = YamlValues.ToDouble(rawMaxValue, context);
            if (!double.IsFinite(maxValue) || maxValue <= 0.0)
            {
                throw new InvalidOperationException(
                    $"job resource {key}: max_value must be positive and finite, got {maxValue}");
            }
            limits[key] = maxValue;
        }
        return limits;
    }

    /// <summary>校验时序配置与通用资源归一化上限的一致性。</summary>
    private static void ValidateTimingResourceContracts(JobConfig job)
    {
        if (!job.Timing.TryGetValue("wildfire_duration", out var wildfireDuration) ||
            !job.ResourceLimits.TryGetValue("wildfire_remaining", out var wildfireRemainingMax))
        {
            return;
        }

        if (Math.Abs(wildfireDuration - wildfireRemainingMax) > 1e-9)
        {
            throw new InvalidOperationException(
                "job config wildfire_duration and resources.wildfire_remaining.max_value " +
                $"must match, got {wildfireDuration} and {wildfireRemainingMax}");
        }
    }

    /// <summary>构建技能定义（对照 _build_skill_definitions）。</summary>
    private static IReadOnlyList<SkillDefinition> BuildSkillDefinitions(
        Dictionary<string, object?> rawSkills,
        string context)
    {
        var skills = new List<SkillDefinition>();
        foreach (var (key, rawPayload) in rawSkills)
        {
            var payload = RequireMappingValue(rawPayload, $"skill {key}", context);
            var rawMpCost = YamlValues.Get(payload, "mp_cost", 0L);
            if (rawMpCost is string mpCostText && mpCostText != "full")
            {
                throw new InvalidOperationException($"unsupported mp_cost semantic for skill {key}: {mpCostText}");
            }
            var mpCostIsFull = rawMpCost is string;
            var mpCost = mpCostIsFull
                ? 0
                : YamlValues.ToInt(rawMpCost, context);
            var mpCostFloor = mpCostIsFull
                ? YamlValues.ToInt(YamlValues.Get(payload, "mp_cost_floor", 0L), context)
                : YamlValues.ToInt(YamlValues.Get(payload, "mp_cost_floor", mpCost), context);

            var aoeSecondaryReduction = YamlValues.ToDouble(
                YamlValues.Get(payload, "aoe_secondary_reduction", 1.0), context);
            if (aoeSecondaryReduction is < 0.0 or > 1.0)
            {
                throw new InvalidOperationException(
                    $"skill {key}: aoe_secondary_reduction must be between 0 and 1, got {aoeSecondaryReduction}");
            }
            var value = YamlValues.ToDouble(YamlValues.Get(payload, "value", 1.0), context);
            if (!double.IsFinite(value) || value <= 0.0)
            {
                throw new InvalidOperationException(
                    $"skill {key}: value must be a positive finite number, got {value}");
            }

            var potency = YamlValues.ToInt(YamlValues.Get(payload, "potency", 0L), context);
            var dotPotency = YamlValues.ToInt(YamlValues.Get(payload, "dot_potency", 0L), context);
            var rawRequiresTarget = YamlValues.Get(payload, "requires_target", null);
            var requiresTarget = rawRequiresTarget is not null
                ? YamlValues.ToBool(rawRequiresTarget)
                : potency > 0 || dotPotency > 0;

            var rawDotKey = YamlValues.Get(payload, "dot_key", null);
            skills.Add(new SkillDefinition(
                Key: key,
                GameId: YamlValues.ToInt(YamlValues.RequireKey(payload, "game_id", context), context),
                Name: YamlValues.ToText(YamlValues.RequireKey(payload, "name", context)),
                Kind: ParseActionKind(YamlValues.ToText(YamlValues.RequireKey(payload, "kind", context))),
                Behavior: YamlValues.ToText(YamlValues.RequireKey(payload, "behavior", context)),
                Potency: potency,
                Value: value,
                CastTime: YamlValues.ToDouble(YamlValues.Get(payload, "cast_time", 0.0), context),
                RecastTime: YamlValues.ToDouble(YamlValues.Get(payload, "recast_time", 2.5), context),
                MpCostIsFull: mpCostIsFull,
                MpCost: mpCost,
                MpCostFloor: mpCostFloor,
                Cooldown: YamlValues.ToDouble(YamlValues.Get(payload, "cooldown", 0.0), context),
                Charges: YamlValues.ToInt(YamlValues.Get(payload, "charges", 1L), context),
                Enabled: YamlValues.ToBool(YamlValues.Get(payload, "enabled", true)),
                MaxTargets: YamlValues.ToInt(YamlValues.Get(payload, "max_targets", 1L), context),
                AoeSecondaryReduction: aoeSecondaryReduction,
                DotPotency: dotPotency,
                DotDuration: YamlValues.ToDouble(YamlValues.Get(payload, "dot_duration", 0.0), context),
                DotKey: rawDotKey is null ? null : YamlValues.ToText(rawDotKey),
                RequiresTarget: requiresTarget,
                AppliesStatuses: ToStringList(YamlValues.Get(payload, "applies_statuses", null)),
                Tags: ToStringList(YamlValues.Get(payload, "tags", null))));
        }
        return skills;
    }

    /// <summary>校验状态定义的 max_stacks 合法（对照 _validate_status_definitions）。</summary>
    private static void ValidateStatusDefinitions(SystemConfig system, JobConfig job)
    {
        foreach (var (source, statuses) in new[]
                 {
                     ("system", system.Statuses),
                     ("job", job.Statuses),
                 })
        {
            foreach (var (key, definition) in statuses)
            {
                if (definition.MaxStacks < 1)
                {
                    throw new InvalidOperationException(
                        $"{source} status {key}: max_stacks must be >= 1, got {definition.MaxStacks}");
                }
            }
        }
    }

    /// <summary>校验系统/职业状态 key 与 game_id 不冲突（对照 _validate_status_conflicts）。</summary>
    private static void ValidateStatusConflicts(SystemConfig system, JobConfig job)
    {
        var duplicateKeys = system.Statuses.Keys.Intersect(job.Statuses.Keys).OrderBy(k => k).ToList();
        if (duplicateKeys.Count > 0)
        {
            throw new InvalidOperationException(
                $"duplicate status keys across system/job config: {string.Join(", ", duplicateKeys)}");
        }
        var systemGameIds = system.Statuses.ToDictionary(pair => pair.Value.GameId, pair => pair.Key);
        foreach (var (key, definition) in job.Statuses)
        {
            if (systemGameIds.TryGetValue(definition.GameId, out var systemKey))
            {
                throw new InvalidOperationException(
                    "duplicate status game_id across system/job config: " +
                    $"{definition.GameId} ({systemKey}, {key})");
            }
        }
    }

    /// <summary>校验系统/职业技能 key 与 game_id 不冲突（对照 _validate_skill_conflicts）。</summary>
    private static void ValidateSkillConflicts(SystemConfig system, JobConfig job)
    {
        var duplicateKeys = system.Skills.Select(s => s.Key)
            .Intersect(job.Skills.Select(s => s.Key)).OrderBy(k => k).ToList();
        if (duplicateKeys.Count > 0)
        {
            throw new InvalidOperationException(
                $"duplicate skill keys across system/job config: {string.Join(", ", duplicateKeys)}");
        }
        var systemGameIds = system.Skills.ToDictionary(skill => skill.GameId, skill => skill.Key);
        foreach (var skill in job.Skills)
        {
            if (systemGameIds.TryGetValue(skill.GameId, out var systemKey))
            {
                throw new InvalidOperationException(
                    "duplicate skill game_id across system/job config: " +
                    $"{skill.GameId} ({systemKey}, {skill.Key})");
            }
        }
    }

    /// <summary>校验 grant_status 必须有状态、状态引用必须已注册（对照 _validate_skill_status_references）。</summary>
    private static void ValidateSkillStatusReferences(SystemConfig system, JobConfig job)
    {
        var statusKeys = new HashSet<string>(
            system.Statuses.Keys.Concat(job.Statuses.Keys), StringComparer.Ordinal);
        foreach (var (source, skills) in new[]
                 {
                     ("system", system.Skills),
                     ("job", job.Skills),
                 })
        {
            foreach (var skill in skills)
            {
                if (skill.Behavior == "grant_status" && skill.AppliesStatuses.Count == 0)
                {
                    throw new InvalidOperationException(
                        $"{source} skill {skill.Key} uses grant_status but applies_statuses is empty");
                }
                foreach (var statusKey in skill.AppliesStatuses)
                {
                    if (!statusKeys.Contains(statusKey))
                    {
                        throw new InvalidOperationException(
                            $"{source} skill {skill.Key} references unregistered status: {statusKey}");
                    }
                }
            }
        }
    }

    private static Dictionary<string, object?> RequireMapping(
        Dictionary<string, object?> payload,
        string key,
        string context)
    {
        var value = YamlValues.RequireKey(payload, key, context);
        return RequireMappingValue(value, key, context);
    }

    private static Dictionary<string, object?> GetMappingOrEmpty(
        Dictionary<string, object?> payload,
        string key,
        string context)
    {
        var value = YamlValues.Get(payload, key, null);
        if (value is null)
        {
            return new Dictionary<string, object?>(StringComparer.Ordinal);
        }
        return RequireMappingValue(value, key, context);
    }

    private static Dictionary<string, object?> RequireMappingValue(object? value, string key, string context)
    {
        if (value is not Dictionary<string, object?> mapping)
        {
            throw new InvalidOperationException($"{key} must be a mapping: {context}");
        }
        return mapping;
    }

    private static List<string> ToStringList(object? value)
    {
        if (value is null)
        {
            return new List<string>();
        }
        if (value is not List<object?> items)
        {
            throw new InvalidOperationException($"expected a sequence, got {value}");
        }
        return items.Select(YamlValues.ToText).ToList();
    }

    private static ActionKind ParseActionKind(string raw)
    {
        switch (raw)
        {
            case "gcd":
                return ActionKind.Gcd;
            case "ogcd":
                return ActionKind.Ogcd;
            default:
                throw new InvalidOperationException($"unsupported action kind: {raw}");
        }
    }
}

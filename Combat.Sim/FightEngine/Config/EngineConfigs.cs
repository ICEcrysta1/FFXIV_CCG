// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Config;

/// <summary>运行期配置（对照 config.RuntimeConfig）。</summary>
public sealed record RuntimeConfig(
    string JobTag,
    string SystemConfigPath,
    string JobConfigPath);

/// <summary>引擎时序配置（对照 config.EngineTimingConfig）。</summary>
public sealed record EngineTimingConfig(
    double ActionQueueWindowSeconds);

/// <summary>引擎资源配置（对照 config.EngineResourceConfig）。</summary>
public sealed record EngineResourceConfig(int MaxMp);

/// <summary>系统威力倍率配置（对照 config.SystemPotencyConfig）。</summary>
public sealed record SystemPotencyConfig(
    double BurstPotionMultiplier,
    double RaidBuffWindowMultiplier);

/// <summary>系统自然回蓝配置（对照 config.SystemMpRecoveryConfig）。</summary>
public sealed record SystemMpRecoveryConfig(
    double TickIntervalSeconds,
    int InCombatAmount);

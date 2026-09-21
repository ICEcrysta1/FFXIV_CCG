// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Outputs;

/// <summary>状态上下文里的玩家段（对照 output_context_builder 的 player 段）。</summary>
public sealed record PlayerContext(
    double TimeSeconds,
    int Mp,
    int MaxMp,
    double MpRatio,
    int GcdIndex,
    double CurrentGcdSeconds,
    double FightRemainingSeconds,
    double GcdRemainingSeconds,
    double CastRemainingSeconds,
    double GcdRemainingGcds,
    double WeaveWindowSeconds,
    double WeaveWindowGcds,
    double OgcdWindowSeconds,
    double OgcdWindowGcds,
    bool BossTargetable,
    double? NextUntargetableInSeconds,
    double? NextUntargetableInGcds,
    double DowntimeRemainingSeconds,
    double DowntimeRemainingGcds,
    bool IsMoving,
    int OgcdsWeaved,
    int MaxOgcdPerWindow);

/// <summary>状态上下文里的目标段（对照 output_context_builder 的 target 段）。</summary>
public sealed record TargetContext(
    double CumulativePotency,
    double CumulativeDotPotency,
    double CurrentPotency,
    double CurrentGcdDotPotency);

/// <summary>状态上下文里的单个 Buff 条目（对照 output_context_builder 的 buffs 段）。</summary>
public sealed record BuffContextEntry(
    string Source,
    string StatusKey,
    int StatusId,
    double RemainingSeconds,
    double RemainingGcds,
    int Stacks);

/// <summary>状态上下文里的单个 DoT 条目（对照 output_context_builder 的 dots 段）。</summary>
public sealed record DotContextEntry(
    string DotKey,
    double RemainingSeconds,
    double RemainingGcds,
    double TickIntervalSeconds,
    double PotencyPerTick);

/// <summary>
/// 单个时刻的状态上下文原料（对照 output_context_builder.build_state_context 的返回字典）。
/// 历史条目与候选条目的 before/after 都复用这个类型。
/// </summary>
public sealed record StateContext(
    string JobTag,
    string JobName,
    PlayerContext Player,
    TargetContext Target,
    IReadOnlyDictionary<string, object> Resources,
    IReadOnlyList<BuffContextEntry> Buffs,
    IReadOnlyList<DotContextEntry> Dots)
{
    /// <summary>系统层独立测试/导出用的空状态上下文占位（对照 Python 侧直接传空 dict）。</summary>
    public static StateContext Empty() => new(
        JobTag: "",
        JobName: "",
        Player: new PlayerContext(
            TimeSeconds: 0.0, Mp: 0, MaxMp: 0, MpRatio: 0.0, GcdIndex: 0,
            CurrentGcdSeconds: 0.0, FightRemainingSeconds: 0.0, GcdRemainingSeconds: 0.0, CastRemainingSeconds: 0.0,
            GcdRemainingGcds: 0.0,
            WeaveWindowSeconds: 0.0, WeaveWindowGcds: 0.0, OgcdWindowSeconds: 0.0,
            OgcdWindowGcds: 0.0, BossTargetable: false, NextUntargetableInSeconds: null,
            NextUntargetableInGcds: null, DowntimeRemainingSeconds: 0.0,
            DowntimeRemainingGcds: 0.0, IsMoving: false, OgcdsWeaved: 0, MaxOgcdPerWindow: 0),
        Target: new TargetContext(
            CumulativePotency: 0.0, CumulativeDotPotency: 0.0,
            CurrentPotency: 0.0, CurrentGcdDotPotency: 0.0),
        Resources: new Dictionary<string, object>(),
        Buffs: Array.Empty<BuffContextEntry>(),
        Dots: Array.Empty<DotContextEntry>());
}

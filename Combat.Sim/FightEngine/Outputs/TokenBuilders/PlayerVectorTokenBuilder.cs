// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Config;

namespace Combat.Sim.Outputs.TokenBuilders;

/// <summary>
/// 玩家向量 token 装配器（对照 token_builders/player_vector_token_builder.py）。
/// 当前态与历史态共用同一个向量装配函数，历史 token 只是 before/after 两段拼接。
/// 字段顺序以 schema.yaml 的 state_vector_fields.player_state 为权威，构造时缓存一次。
/// </summary>
public sealed class PlayerVectorTokenBuilder
{
    private readonly string[] _featureKeys;
    private readonly string[] _historyFeatureKeys;

    public PlayerVectorTokenBuilder()
    {
        _featureKeys = SchemaConfigLoader.RequireStateVectorFields("player_state").ToArray();
        _historyFeatureKeys = _featureKeys
            .Select(key => $"before.{key}")
            .Concat(_featureKeys.Select(key => $"after.{key}"))
            .ToArray();
    }

    public IReadOnlyList<string> FeatureKeys => _featureKeys;

    public IReadOnlyList<string> HistoryFeatureKeys => _historyFeatureKeys;

    public Dictionary<string, object?> BuildCurrentToken(PlayerContext player) =>
        new()
        {
            ["feature_keys"] = _featureKeys.ToList(),
            ["vector"] = BuildVector(player).ToArray(),
        };

    public IReadOnlyList<double> BuildHistoryToken(PlayerContext before, PlayerContext after) =>
        BuildVector(before).Concat(BuildVector(after)).ToList();

    /// <summary>
    /// 单个玩家上下文的向量装配；当前态与历史态共用。
    /// 按 schema 字段逐项取值，未知字段显式报错（schema 增删字段不再静默漂移）。
    /// </summary>
    private IReadOnlyList<double> BuildVector(PlayerContext player) =>
        _featureKeys.Select(key => ValueFor(player, key)).ToArray();

    private static double ValueFor(PlayerContext player, string key) => key switch
    {
        "mp" => player.Mp,
        "max_mp" => player.MaxMp,
        "mp_ratio" => player.MpRatio,
        "time_seconds" => player.TimeSeconds,
        "current_gcd_seconds" => player.CurrentGcdSeconds,
        "gcd_index" => player.GcdIndex,
        "fight_remaining_seconds" => player.FightRemainingSeconds,
        "gcd_remaining_seconds" => player.GcdRemainingSeconds,
        "gcd_remaining_gcds" => player.GcdRemainingGcds,
        "weave_window_seconds" => player.WeaveWindowSeconds,
        "weave_window_gcds" => player.WeaveWindowGcds,
        "ogcd_window_seconds" => player.OgcdWindowSeconds,
        "ogcd_window_gcds" => player.OgcdWindowGcds,
        "boss_targetable" => player.BossTargetable ? 1.0 : 0.0,
        "next_untargetable_in_seconds" => player.NextUntargetableInSeconds ?? 0.0,
        "next_untargetable_in_gcds" => player.NextUntargetableInGcds ?? 0.0,
        "downtime_remaining_seconds" => player.DowntimeRemainingSeconds,
        "downtime_remaining_gcds" => player.DowntimeRemainingGcds,
        "is_moving" => player.IsMoving ? 1.0 : 0.0,
        "ogcds_weaved" => player.OgcdsWeaved,
        "max_ogcd_per_window" => player.MaxOgcdPerWindow,
        _ => throw new InvalidOperationException($"schema.yaml 中未知的玩家状态字段: {key}"),
    };
}

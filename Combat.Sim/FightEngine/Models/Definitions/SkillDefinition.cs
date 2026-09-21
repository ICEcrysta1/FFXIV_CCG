// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Models.Definitions;

/// <summary>
/// 技能静态定义（对照 models.SkillDefinition）。
/// <c>MpCostIsFull</c> 对应 Python 侧 <c>mp_cost == "full"</c> 语义，
/// <c>MpCost</c> 仅在非 full 时有效。
/// </summary>
public sealed record SkillDefinition
{
    public SkillDefinition(
        string Key,
        int GameId,
        string Name,
        ActionKind Kind,
        string Behavior,
        int Potency = 0,
        double Value = 1.0,
        double CastTime = 0.0,
        double RecastTime = 2.5,
        bool MpCostIsFull = false,
        int MpCost = 0,
        int MpCostFloor = 0,
        double Cooldown = 0.0,
        int Charges = 1,
        bool Enabled = true,
        int MaxTargets = 1,
        double AoeSecondaryReduction = 1.0,
        int DotPotency = 0,
        double DotDuration = 0.0,
        string? DotKey = null,
        bool RequiresTarget = false,
        IReadOnlyList<string>? AppliesStatuses = null,
        IReadOnlyList<string>? Tags = null)
    {
        this.Key = Key;
        this.GameId = GameId;
        this.Name = Name;
        this.Kind = Kind;
        this.Behavior = Behavior;
        this.Potency = Potency;
        this.Value = Value;
        this.CastTime = CastTime;
        this.RecastTime = RecastTime;
        this.MpCostIsFull = MpCostIsFull;
        this.MpCost = MpCost;
        this.MpCostFloor = MpCostFloor;
        this.Cooldown = Cooldown;
        this.Charges = Charges;
        this.Enabled = Enabled;
        this.MaxTargets = MaxTargets;
        this.AoeSecondaryReduction = AoeSecondaryReduction;
        this.DotPotency = DotPotency;
        this.DotDuration = DotDuration;
        this.DotKey = DotKey;
        this.RequiresTarget = RequiresTarget;
        this.AppliesStatuses = AppliesStatuses ?? Array.Empty<string>();
        this.Tags = Tags ?? Array.Empty<string>();
    }

    public string Key { get; init; }
    public int GameId { get; init; }
    public string Name { get; init; }
    public ActionKind Kind { get; init; }
    public string Behavior { get; init; }
    public int Potency { get; init; }
    public double Value { get; init; }
    public double CastTime { get; init; }
    public double RecastTime { get; init; }
    public bool MpCostIsFull { get; init; }
    public int MpCost { get; init; }
    public int MpCostFloor { get; init; }
    public double Cooldown { get; init; }
    public int Charges { get; init; }
    public bool Enabled { get; init; }
    public int MaxTargets { get; init; }
    public double AoeSecondaryReduction { get; init; }
    public int DotPotency { get; init; }
    public double DotDuration { get; init; }
    public string? DotKey { get; init; }
    public bool RequiresTarget { get; init; }
    public IReadOnlyList<string> AppliesStatuses { get; init; }
    public IReadOnlyList<string> Tags { get; init; }
}

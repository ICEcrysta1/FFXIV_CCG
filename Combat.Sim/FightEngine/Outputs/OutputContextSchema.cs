// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using Combat.Sim.Config;

namespace Combat.Sim.Outputs;

/// <summary>
/// canonical 输出 context schema 定义（对照 outputs/output_context_schema.py）。
/// 顶层 keys、状态向量分组与版本由 <c>config/schema.yaml</c> 提供
/// （C# 与 Python 单一事实来源）；使用前必须先调用 <see cref="SchemaConfigLoader.Load"/>。
/// </summary>
public static class OutputContextSchema
{
    private static IReadOnlyList<string> TopLevelKeys =>
        SchemaConfigLoader.Instance.CanonicalTopLevelKeys;

    public static int CanonicalContextSchemaVersion =>
        SchemaConfigLoader.Instance.CanonicalSchemaVersion;

    public static string SceneContextKey => TopLevelKeys[2];
    public static string SkillHistoryContextKey => TopLevelKeys[3];
    public static string StateHistoryContextKey => TopLevelKeys[4];
    public static string CandidateSkillContextKey => TopLevelKeys[5];
    public static string CandidateStateContextKey => TopLevelKeys[6];
    public static string StateContextTokenKey => SchemaConfigLoader.Instance.TokenKey;

    public static string[] CanonicalContextTopLevelKeys => TopLevelKeys.ToArray();

    /// <summary>状态向量分组 schema（对照 STATE_VECTOR_GROUP_SCHEMAS）。</summary>
    public sealed record StateVectorGroupSchema(string GroupKey, string FeatureKeysField, string? ContextKey);

    public static StateVectorGroupSchema[] StateVectorGroupSchemas =>
        SchemaConfigLoader.Instance.StateVectorGroups
            .Select(group => new StateVectorGroupSchema(
                group.GroupKey, group.FeatureKeysField, group.ContextKey))
            .ToArray();

    /// <summary>按统一 schema 裁剪 canonical 输出（对照 format_canonical_output_context）。</summary>
    public static Dictionary<string, object?> FormatCanonicalOutputContext(
        Dictionary<string, object?> outputContext) =>
        CanonicalContextTopLevelKeys.ToDictionary(key => key, key => outputContext[key]);

    /// <summary>提取状态上下文里每个向量分组的 feature key（对照 extract_state_feature_keys）。</summary>
    public static Dictionary<string, List<string>> ExtractStateFeatureKeys(
        Dictionary<string, object?> stateContext) =>
        StateVectorGroupSchemas.ToDictionary(
            group => group.GroupKey,
            group => ExpectFeatureKeys(stateContext, group.FeatureKeysField));

    /// <summary>构造 canonical 输出的正式 schema 元数据（对照 build_output_context_schema_metadata）。</summary>
    public static Dictionary<string, object?> BuildOutputContextSchemaMetadata(
        Dictionary<string, object?> outputContext)
    {
        var historyStateContext = ExpectDict(outputContext, StateHistoryContextKey);
        var candidateStateContext = ExpectDict(outputContext, CandidateStateContextKey);
        var schemaVersion = ExpectInt(outputContext, "schema_version");
        return new Dictionary<string, object?>
        {
            ["schema_version"] = schemaVersion,
            ["top_level_keys"] = CanonicalContextTopLevelKeys.ToList(),
            ["scene_context_key"] = SceneContextKey,
            ["skill_history_context_key"] = SkillHistoryContextKey,
            ["state_history_context_key"] = StateHistoryContextKey,
            ["candidate_skill_context_key"] = CandidateSkillContextKey,
            ["candidate_state_context_key"] = CandidateStateContextKey,
            ["state_context_token_key"] = StateContextTokenKey,
            ["state_vector_group_keys"] = StateVectorGroupSchemas.Select(group => group.GroupKey).ToList(),
            ["state_vector_feature_key_fields"] = StateVectorGroupSchemas.ToDictionary(
                group => group.GroupKey, group => group.FeatureKeysField),
            ["state_history_feature_keys"] = ExtractStateFeatureKeys(historyStateContext),
            ["candidate_state_feature_keys"] = ExtractStateFeatureKeys(candidateStateContext),
        };
    }

    private static Dictionary<string, object?> ExpectDict(
        Dictionary<string, object?> container,
        string key)
    {
        if (container[key] is not Dictionary<string, object?> value)
        {
            throw new InvalidOperationException($"{key} must be a dict");
        }

        return value;
    }

    private static int ExpectInt(Dictionary<string, object?> container, string key)
    {
        if (container[key] is not int value)
        {
            throw new InvalidOperationException($"{key} must be an int");
        }

        return value;
    }

    private static List<string> ExpectFeatureKeys(
        Dictionary<string, object?> stateContext,
        string key)
    {
        if (stateContext[key] is not List<string> value)
        {
            throw new InvalidOperationException($"{key} must be a list of strings");
        }

        return value.ToList();
    }
}

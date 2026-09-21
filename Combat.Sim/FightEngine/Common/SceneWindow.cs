// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using System.Collections.Concurrent;

namespace Combat.Sim.Common;

/// <summary>
/// scene window 的索引解析与绝对时间窗口公共逻辑（对照 scene_window.py）。
/// 供输出层与转换层按 feature key 列表解析窗口字段索引。
/// </summary>
public static class SceneWindow
{
    /// <summary>scene 窗口必填的三个 feature key（对照 REQUIRED_SCENE_WINDOW_FEATURE_KEYS）。</summary>
    public static readonly IReadOnlyList<string> RequiredSceneWindowFeatureKeys =
        new[] { "start_offset_seconds", "end_offset_seconds", "duration_seconds" };

    // Python 侧是 lru_cache(maxsize=32)，其内部有锁保证并发安全；
    // C# 侧用 ConcurrentDictionary 提供等价并发安全，缓存淘汰是性能细节不做容量限制。
    // 注意：缓存 key 是 IReadOnlyList<string> 引用，调用方不得在首次使用后修改列表
    // （配置派生的 feature key 实际不可变）。
    private static readonly ConcurrentDictionary<IReadOnlyList<string>, IReadOnlyDictionary<string, int>> IndexCache =
        new(SequenceComparer.Instance);

    /// <summary>缓存一个 scene feature key 到索引的映射（对照 feature_index_map）。</summary>
    public static IReadOnlyDictionary<string, int> FeatureIndexMap(IReadOnlyList<string> featureKeys) =>
        IndexCache.GetOrAdd(featureKeys, static keys =>
        {
            var mapping = new Dictionary<string, int>(StringComparer.Ordinal);
            for (var index = 0; index < keys.Count; index++)
            {
                mapping[keys[index]] = index;
            }
            return mapping;
        });

    /// <summary>
    /// 校验并返回 scene 窗口 start/end/duration 三个字段的索引（对照 resolve_scene_window_indices）。
    /// 缺失必填字段时抛 <see cref="InvalidOperationException"/>。
    /// </summary>
    public static (int Start, int End, int Duration) ResolveSceneWindowIndices(
        IReadOnlyList<string> featureKeys,
        string contextKey)
    {
        var indices = FeatureIndexMap(featureKeys);
        var missingFeatures = RequiredSceneWindowFeatureKeys
            .Where(featureKey => !indices.ContainsKey(featureKey))
            .ToList();
        if (missingFeatures.Count > 0)
        {
            throw new InvalidOperationException(
                $"scene window '{contextKey}' is missing required feature keys: " +
                string.Join(", ", missingFeatures));
        }
        return (
            indices[RequiredSceneWindowFeatureKeys[0]],
            indices[RequiredSceneWindowFeatureKeys[1]],
            indices[RequiredSceneWindowFeatureKeys[2]]);
    }

    /// <summary>按元素顺序比较字符串序列（Python lru_cache 的 tuple 哈希等价物）。</summary>
    private sealed class SequenceComparer : IEqualityComparer<IReadOnlyList<string>>
    {
        public static readonly SequenceComparer Instance = new();

        public bool Equals(IReadOnlyList<string>? x, IReadOnlyList<string>? y)
        {
            if (ReferenceEquals(x, y))
            {
                return true;
            }
            if (x is null || y is null || x.Count != y.Count)
            {
                return false;
            }
            for (var i = 0; i < x.Count; i++)
            {
                if (!string.Equals(x[i], y[i], StringComparison.Ordinal))
                {
                    return false;
                }
            }
            return true;
        }

        public int GetHashCode(IReadOnlyList<string> obj)
        {
            var hash = new HashCode();
            foreach (var item in obj)
            {
                hash.Add(item, StringComparer.Ordinal);
            }
            return hash.ToHashCode();
        }
    }
}

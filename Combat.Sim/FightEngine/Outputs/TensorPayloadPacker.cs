// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using System.Collections;

namespace Combat.Sim.Outputs;

/// <summary>
/// 把嵌套输出载荷递归打包成 torch 友好的紧凑结构（对照 outputs/tensor_payload_packer.py）。
/// C# 侧没有 torch：打包结果是与 torch.tensor(...).tolist() 等价的嵌套数值结构——
/// 混合数值/null 输出 {values, is_null}（null 用 0.0 占位 + is_null 掩码表达缺失），
/// 纯 bool 输出布尔嵌套结构，纯 int 输出整数嵌套结构；同键 mapping 列表转列式结构。
/// 数值按 <see cref="TensorPrecisionAdapter"/> 折算 dtype 语义。
/// </summary>
public static class TensorPayloadPacker
{
    public static object? PackPayload(object? payload, TensorPrecisionAdapter adapter) =>
        PackValue(payload, adapter);

    private static object? PackValue(object? value, TensorPrecisionAdapter adapter)
    {
        if (value is IDictionary dict)
        {
            var result = new Dictionary<string, object?>();
            foreach (DictionaryEntry entry in dict)
            {
                result[(string)entry.Key] = PackValue(entry.Value, adapter);
            }

            return result;
        }

        if (value is IList list)
        {
            var layout = InferNumericLayout(list);
            if (layout is not null)
            {
                return PackNumericSequence(list, layout, adapter);
            }

            if (IsMappingListWithSharedKeys(list))
            {
                var keys = ((IDictionary)list[0]!).Keys.Cast<string>().ToList();
                return keys.ToDictionary(
                    key => key,
                    key => PackValue(
                        list.Cast<object?>().Select(item => ((IDictionary)item!)[key]).ToList(),
                        adapter));
            }

            return list.Cast<object?>().Select(item => PackValue(item, adapter)).ToList();
        }

        return value;
    }

    /// <summary>
    /// 数值序列打包：含 null 或浮点（或空）→ {values, is_null}；纯 bool → 布尔结构；纯 int → 整数结构。
    /// null 位置用 0.0 占位而非 NaN：真正的"是否缺失"由 is_null 掩码表达，
    /// 避免 NaN 在下游归一化/前向计算里意外传播，也让同一份输入能稳定判等。
    /// </summary>
    private static object PackNumericSequence(
        IList value,
        NumericLayout layout,
        TensorPrecisionAdapter adapter)
    {
        if (layout.HasNull || layout.ScalarKinds.Contains(typeof(double)) || layout.ScalarKinds.Count == 0)
        {
            return new Dictionary<string, object?>
            {
                ["values"] = MapNested(
                    value,
                    item => item is null ? 0.0 : adapter.QuantizeFloat(Convert.ToDouble(item))),
                ["is_null"] = MapNested(value, item => item is null),
            };
        }

        if (layout.ScalarKinds.SetEquals(new[] { typeof(bool) }))
        {
            return MapNested(value, item => (bool)item!);
        }

        return MapNested(value, item => adapter.QuantizeInt(Convert.ToInt32(item)));
    }

    /// <summary>按嵌套结构逐元素映射（保持形状）。</summary>
    private static object MapNested(object? value, Func<object?, object> mapper)
    {
        if (value is IList list)
        {
            return list.Cast<object?>().Select(item => MapNested(item, mapper)).ToList();
        }

        return mapper(value);
    }

    /// <summary>推断嵌套列表的数值布局：形状一致且全部为数值/null 才视为数值序列。</summary>
    private static NumericLayout? InferNumericLayout(object? value)
    {
        if (value is IList list)
        {
            if (list.Count == 0)
            {
                return null;
            }

            var childLayouts = list.Cast<object?>().Select(InferNumericLayout).ToList();
            if (childLayouts.Any(layout => layout is null))
            {
                return null;
            }

            var firstShape = childLayouts[0]!.Shape;
            if (childLayouts.Skip(1).Any(layout => !layout!.Shape.SequenceEqual(firstShape)))
            {
                return null;
            }

            var scalarKinds = new HashSet<Type>();
            foreach (var layout in childLayouts)
            {
                if (layout is not null)
                {
                    scalarKinds.UnionWith(layout.ScalarKinds);
                }
            }

            return new NumericLayout(
                Shape: new[] { list.Count }.Concat(firstShape).ToArray(),
                HasNull: childLayouts.Any(layout => layout!.HasNull),
                ScalarKinds: scalarKinds);
        }

        if (value is null)
        {
            return new NumericLayout(Array.Empty<int>(), HasNull: true, ScalarKinds: new HashSet<Type>());
        }

        if (value is bool)
        {
            return new NumericLayout(Array.Empty<int>(), HasNull: false, ScalarKinds: new HashSet<Type> { typeof(bool) });
        }

        if (value is int)
        {
            return new NumericLayout(Array.Empty<int>(), HasNull: false, ScalarKinds: new HashSet<Type> { typeof(int) });
        }

        if (value is double)
        {
            return new NumericLayout(Array.Empty<int>(), HasNull: false, ScalarKinds: new HashSet<Type> { typeof(double) });
        }

        return null;
    }

    private static bool IsMappingListWithSharedKeys(IList value)
    {
        if (value.Count == 0 || value[0] is not IDictionary first)
        {
            return false;
        }

        var keys = first.Keys.Cast<string>().ToList();
        return value.Cast<object?>().All(item =>
            item is IDictionary dict && dict.Keys.Cast<string>().SequenceEqual(keys));
    }

    private sealed record NumericLayout(int[] Shape, bool HasNull, HashSet<Type> ScalarKinds);
}

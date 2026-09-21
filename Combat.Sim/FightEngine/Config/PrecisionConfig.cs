// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using System.Globalization;
using Combat.Sim.Common;
using Combat.Sim.Outputs;

namespace Combat.Sim.Config;

/// <summary>tensor 精度配置（对照 config.PrecisionConfig，dtype 以枚举承载）。</summary>
public sealed record PrecisionConfig(TensorDtype IntDtype, TensorDtype FloatDtype);

/// <summary>从 config/precision.yaml 加载统一 tensor 精度（对照 load_precision_config）。</summary>
public static class PrecisionConfigLoader
{
    /// <summary>整数 dtype 别名表（对照 _INT_DTYPE_ALIASES）。</summary>
    private static readonly Dictionary<string, TensorDtype> IntAliases = new(StringComparer.Ordinal)
    {
        ["int32"] = TensorDtype.Int32,
        ["int64"] = TensorDtype.Int64,
    };

    /// <summary>浮点 dtype 别名表（对照 _FLOAT_DTYPE_ALIASES；float64 与 Python 一致不在此列）。</summary>
    private static readonly Dictionary<string, TensorDtype> FloatAliases = new(StringComparer.Ordinal)
    {
        ["float32"] = TensorDtype.Float32,
        ["fp32"] = TensorDtype.Float32,
        ["float16"] = TensorDtype.Float16,
        ["fp16"] = TensorDtype.Float16,
        ["bfloat16"] = TensorDtype.Bfloat16,
        ["bf16"] = TensorDtype.Bfloat16,
    };

    public static PrecisionConfig Load(string projectRoot)
    {
        var path = Path.Combine(projectRoot, "config", "precision.yaml");
        var mapping = YamlConfig.LoadYamlMapping(path, "precision config");
        // Python 端 raw.get("precision", {}) 缺省空 dict 走默认值；存在但非 mapping 视为非法配置
        var precision = YamlValues.Get(mapping, "precision");
        var payload = precision is null
            ? new Dictionary<string, object?>()
            : precision as Dictionary<string, object?>
                ?? throw new InvalidOperationException("precision config 'precision' must be a mapping");

        return new PrecisionConfig(
            IntDtype: ResolvePrecisionAlias(YamlValues.Get(payload, "int_dtype", "int32"), "int_dtype", IntAliases),
            FloatDtype: ResolvePrecisionAlias(YamlValues.Get(payload, "float_dtype", "float32"), "float_dtype", FloatAliases));
    }

    /// <summary>解析 dtype 别名（对照 _resolve_precision_alias，大小写不敏感）。</summary>
    private static TensorDtype ResolvePrecisionAlias(
        object? rawValue,
        string fieldName,
        IReadOnlyDictionary<string, TensorDtype> aliases)
    {
        var rawText = (rawValue?.ToString() ?? "").ToLowerInvariant();
        if (aliases.TryGetValue(rawText, out var resolved))
        {
            return resolved;
        }

        // 错误消息按 Python repr 语义显示原始值：null 显示 None、字符串带单引号保留原大小写、
        // 其他值不加引号。Convert.ToString 对 null 返回空串，不能承担 null 兜底；
        // 数值经 InvariantCulture 显示，避免受本机 locale 干扰（repr 的 "1.0" 尾零格式等
        // 属非典型输入差异，dtype 别名均为短词不会触发）。
        var rawDisplay = rawValue is null
            ? "None"
            : rawValue is string text ? $"'{text}'" : Convert.ToString(rawValue, CultureInfo.InvariantCulture);
        var supported = string.Join(", ", aliases.Keys.OrderBy(k => k, StringComparer.Ordinal));
        throw new InvalidOperationException(
            $"precision.yaml: {fieldName}={rawDisplay} is invalid; supported aliases: {supported}");
    }
}

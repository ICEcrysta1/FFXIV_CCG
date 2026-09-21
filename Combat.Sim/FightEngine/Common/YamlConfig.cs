// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using System.Globalization;
using System.Text.RegularExpressions;
using YamlDotNet.Core;
using YamlDotNet.RepresentationModel;

namespace Combat.Sim.Common;

/// <summary>
/// YAML mapping 加载（对照 common/yaml_config.py）：按 UTF-8 读取，
/// 顶层必须是 mapping；空文档 / null 归一化为空 mapping，其他非 mapping 值抛错。
/// 解析树由 YamlDotNet 承担；Plain（未加引号）标量按 PyYAML 顺序做类型提升
/// （null → bool → int → float → string），数值统一使用 .NET 原生
/// <c>long.TryParse</c> / <c>double.TryParse</c> 得到与 Python 一致的 double 精度。
/// 带引号 / 块标量保持字符串（PyYAML 同样不对 quoted 标量做类型推断）。
/// 返回 Python 风格的嵌套结构：mapping → Dictionary&lt;string, object?&gt;，
/// sequence → List&lt;object?&gt;。
/// </summary>
public static class YamlConfig
{
    public static Dictionary<string, object?> LoadYamlMapping(string path, string description = "YAML config")
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"{description} not found: {path}", path);
        }
        string text;
        try
        {
            text = File.ReadAllText(path);
        }
        catch (Exception ex)
        {
            throw new IOException($"{description} read failed: {path}", ex);
        }

        YamlStream stream;
        try
        {
            stream = new YamlStream();
            using var reader = new StringReader(text);
            stream.Load(reader);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"{description} parse failed: {path}", ex);
        }

        var root = stream.Documents.Count == 0 ? null : stream.Documents[0].RootNode;
        if (root is null)
        {
            return new Dictionary<string, object?>(StringComparer.Ordinal);
        }
        if (root is not YamlMappingNode mapping)
        {
            throw new InvalidOperationException($"{description} must be a mapping: {path}");
        }
        return ConvertMapping(mapping);
    }

    private static Dictionary<string, object?> ConvertMapping(YamlMappingNode mapping)
    {
        var result = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var (rawKey, rawValue) in mapping.Children)
        {
            var key = rawKey is YamlScalarNode scalarKey ? scalarKey.Value ?? string.Empty : rawKey.ToString();
            result[key] = ConvertNode(rawValue);
        }
        return result;
    }

    private static object? ConvertNode(YamlNode node) => node switch
    {
        YamlMappingNode nested => ConvertMapping(nested),
        YamlSequenceNode sequence => sequence.Children.Select(ConvertNode).ToList(),
        YamlScalarNode scalar => ParseScalar(scalar),
        _ => node.ToString(),
    };

    private static readonly Regex LeadingZeroInteger = new(
        @"^[-+]?0[0-9]+$",
        RegexOptions.CultureInvariant);

    /// <summary>PyYAML（YAML 1.1）风格标量提升：null → bool → int → float → string。</summary>
    private static object? ParseScalar(YamlScalarNode scalar)
    {
        var value = scalar.Value;
        // 带引号 / 块标量保持字符串，不做类型推断（与 PyYAML 一致）。
        if (scalar.Style != ScalarStyle.Plain)
        {
            return value;
        }
        if (value is null or "" or "~" or "null" or "Null" or "NULL")
        {
            return null;
        }
        if (value is "true" or "True" or "TRUE" or "yes" or "Yes" or "YES" or "on" or "On" or "ON")
        {
            return true;
        }
        if (value is "false" or "False" or "FALSE" or "no" or "No" or "NO" or "off" or "Off" or "OFF")
        {
            return false;
        }
        // 前导零整数：PyYAML（YAML 1.1）按八进制解析（010 → 8），而 C# 十进制解析会静默得到 10。
        // 这里显式拒绝而不是静默按十进制解析，防止两套引擎跑出不同数值。
        if (LeadingZeroInteger.IsMatch(value))
        {
            throw new InvalidOperationException($"ambiguous leading-zero integer: {value}");
        }
        if (long.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var integer))
        {
            return integer;
        }
        if (double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var floating))
        {
            return floating;
        }
        return value;
    }
}

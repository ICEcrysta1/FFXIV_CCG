// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

using System.Globalization;

namespace Combat.Sim.Common;

/// <summary>
/// Python 语义的配置值访问与转换（对照 config.py / project_config.py 的取值方式）：
/// <c>dict[key]</c> 缺失抛错、<c>dict.get(key, default)</c>、<c>int()</c> 截断、
/// <c>float()</c>、<c>bool()</c>（非空/非零即真）等行为保持一致。
/// YamlDotNet 的类型推断可能产出 byte/int/uint/long/float/double 等任意数值类型，
/// 这里统一按数值族处理。
/// </summary>
public static class YamlValues
{
    /// <summary>对应 Python 的 <c>payload[key]</c>：缺失即抛错。</summary>
    public static object? RequireKey(Dictionary<string, object?> mapping, string key, string context)
    {
        if (!mapping.TryGetValue(key, out var value))
        {
            throw new InvalidOperationException($"{context}: missing key {key}");
        }
        return value;
    }

    /// <summary>对应 Python 的 <c>payload.get(key, fallback)</c>。</summary>
    public static object? Get(Dictionary<string, object?> mapping, string key, object? fallback = null)
    {
        return mapping.TryGetValue(key, out var value) ? value : fallback;
    }

    /// <summary>对应 Python 的 <c>int()</c>：float/double 截断、字符串解析整数。</summary>
    public static int ToInt(object? value, string context)
    {
        switch (value)
        {
            case string s:
                if (long.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed))
                {
                    return (int)parsed;
                }
                throw new InvalidOperationException($"{context}: invalid integer value {s}");
            case bool b:
                return b ? 1 : 0;
            case byte or sbyte or short or ushort or int or uint or long or ulong:
                return Convert.ToInt32(value, CultureInfo.InvariantCulture);
            case float or double:
                return (int)Math.Truncate(Convert.ToDouble(value, CultureInfo.InvariantCulture));
            default:
                throw new InvalidOperationException($"{context}: unsupported integer value {value}");
        }
    }

    /// <summary>对应 Python 的 <c>float()</c>。</summary>
    public static double ToDouble(object? value, string context)
    {
        switch (value)
        {
            case string s:
                if (double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed))
                {
                    return parsed;
                }
                throw new InvalidOperationException($"{context}: invalid float value {s}");
            case bool b:
                return b ? 1.0 : 0.0;
            case byte or sbyte or short or ushort or int or uint or long or ulong or float or double:
                return Convert.ToDouble(value, CultureInfo.InvariantCulture);
            default:
                throw new InvalidOperationException($"{context}: unsupported float value {value}");
        }
    }

    /// <summary>
    /// 对应 Python 的 <c>bool()</c>：None / 0 / 0.0 / 空字符串为 False，其余为 True。
    /// 配置里的布尔字段应为 YAML bool；此处保持 Python 语义一致性。
    /// </summary>
    public static bool ToBool(object? value)
    {
        switch (value)
        {
            case null:
                return false;
            case bool b:
                return b;
            case string s:
                return s.Length != 0;
            case byte or sbyte or short or ushort or int or uint or long or ulong or float or double:
                return Convert.ToDouble(value, CultureInfo.InvariantCulture) != 0.0;
            default:
                return true;
        }
    }

    /// <summary>对应 Python 的 <c>str()</c>。</summary>
    public static string ToText(object? value)
    {
        return Convert.ToString(value, CultureInfo.InvariantCulture) ?? string.Empty;
    }

    /// <summary>
    /// 对应 common/project_config.py 的 resolve_project_path：
    /// 绝对路径原样保留，相对路径解析到项目根目录。
    /// </summary>
    public static string ResolveProjectPath(object? value, string projectRoot)
    {
        var raw = ToText(value);
        return Path.IsPathRooted(raw)
            ? Path.GetFullPath(raw)
            : Path.GetFullPath(Path.Combine(projectRoot, raw));
    }
}

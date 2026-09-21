using Combat.Sim.Common;

namespace FightEngine.Tests.Common;

/// <summary>
/// YAML 标量解析边界测试：锁定与 PyYAML（YAML 1.1）一致或显式抛错的行为，
/// 防止静默错值（对照 MR 审核意见）。
/// </summary>
public class YamlConfigTests
{
    private static Dictionary<string, object?> Parse(string yaml)
    {
        var path = Path.Combine(Path.GetTempPath(), "yaml_scalar_" + Guid.NewGuid().ToString("N") + ".yaml");
        File.WriteAllText(path, yaml);
        try
        {
            return YamlConfig.LoadYamlMapping(path);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static object? ParseScalar(string value)
    {
        var parsed = Parse($"value: {value}");
        return parsed["value"];
    }

    [Fact]
    public void 常规标量解析与PyYAML一致()
    {
        Assert.Equal(42L, ParseScalar("42"));
        Assert.Equal(-7L, ParseScalar("-7"));
        Assert.Equal(0L, ParseScalar("0"));
        Assert.Equal(1.5, ParseScalar("1.5"));
        Assert.Equal(-0.25, ParseScalar("-0.25"));
        Assert.Equal(2.5, ParseScalar("2.5e0"));
        Assert.Equal(true, ParseScalar("true"));
        Assert.Equal(false, ParseScalar("false"));
        Assert.Equal(true, ParseScalar("yes"));   // YAML 1.1 布尔
        Assert.Equal(true, ParseScalar("on"));
        Assert.Null(ParseScalar("null"));
        Assert.Null(ParseScalar("~"));
        Assert.Equal("hello", ParseScalar("hello"));
    }

    [Fact]
    public void 前导零整数显式拒绝而非静默错值()
    {
        // PyYAML（YAML 1.1）把 010 解析为八进制 8，C# 十进制会静默得到 10；
        // 这里要求显式抛错，防止两套引擎跑出不同数值。
        var ex = Assert.Throws<InvalidOperationException>(() => ParseScalar("010"));
        Assert.Contains("leading-zero", ex.Message);
        Assert.Throws<InvalidOperationException>(() => ParseScalar("-010"));
        Assert.Throws<InvalidOperationException>(() => ParseScalar("00"));
    }

    [Fact]
    public void 十六进制与二进制前缀保持字符串()
    {
        // PyYAML 支持 0x/0b（010 八进制、0x10 十六进制、0b101 二进制），
        // C# 侧不推断这些写法：保留字符串，后续数值转换会显式抛错而非静默错值。
        Assert.Equal("0x10", ParseScalar("0x10"));
        Assert.Equal("0b101", ParseScalar("0b101"));
        // PyYAML（YAML 1.1）本身不识别 0o 前缀，同样返回字符串，行为一致。
        Assert.Equal("0o10", ParseScalar("0o10"));
    }

    [Fact]
    public void 下划线分隔数字保持字符串()
    {
        // PyYAML 支持 1_000 下划线分隔；C# 侧保留字符串，后续转换显式抛错。
        Assert.Equal("1_000", ParseScalar("1_000"));
    }

    [Fact]
    public void inf与nan保持字符串()
    {
        // PyYAML 把 .inf/.nan 解析为浮点特殊值；C# 侧保留字符串，
        // 配置校验（正有限性）会显式拒绝而非静默接受。
        Assert.Equal(".inf", ParseScalar(".inf"));
        Assert.Equal(".nan", ParseScalar(".nan"));
        Assert.Equal("-.inf", ParseScalar("-.inf"));
    }

    [Fact]
    public void 带引号标量保持字符串不做类型推断()
    {
        Assert.Equal("010", ParseScalar("\"010\""));
        Assert.Equal("1.5", ParseScalar("'1.5'"));
        Assert.Equal("true", ParseScalar("'true'"));
    }
}

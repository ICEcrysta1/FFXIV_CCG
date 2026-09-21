using Combat.Sim.Config;
using Combat.Sim.Outputs;

namespace FightEngine.Tests.Outputs;

/// <summary>
/// tensor 精度适配测试（对照 config.py 的 dtype 别名与 torch dtype 取值语义）。
/// 边界参考值由 torch.tensor(v, dtype=torch.bfloat16).tolist() 生成。
/// </summary>
public class OutputsPrecisionTests
{
    private static readonly (double Value, double Expected)[] Bfloat16BoundaryCases =
    {
        (0.0, 0.0),
        (-0.0, -0.0),
        (1.0, 1.0),
        (-1.0, -1.0),
        (0.4, 0.400390625),
        (-0.4, -0.400390625),
        (3.14159, 3.140625),
        (1e10, 9999220736.0),
        (-1e10, -9999220736.0),
        (255.5, 256.0),
        (256.5, 256.0),
        // 正规最小值附近
        (1.1754943508222875e-38, 1.1754943508222875e-38),
        (1.1663108012064884e-38, 1.1663108012064884e-38),
        // 次正规：2^-133 步长舍入
        (1.3775324423698682e-40, 1.8367099231598242e-40),
        (9.183549615799121e-41, 9.183549615799121e-41),
        (6.887662211849341e-41, 9.183549615799121e-41),
        (1.827526373544025e-40, 1.8367099231598242e-40),
        // 下溢为 0（保留符号）
        (4.591774807899561e-41, 0.0),
        (1.5, 1.5),
        (1.25, 1.25),
        (0.1, 0.10009765625),
        (12345.6789, 12352.0),
        (1e30, 1.0002555517425873e+30),
        (-1e30, -1.0002555517425873e+30),
        (65504.0, 65536.0),
        // 溢出边界：bfloat16 最大有限值 2^128-2^120，之上量化 +Inf（torch 实测参考）
        (Math.Pow(2, 128) - Math.Pow(2, 120), 338953138925153547590470800371487866880.0),
        (Math.Pow(2, 128) - Math.Pow(2, 119), double.PositiveInfinity),
        (Math.Pow(2, 128), double.PositiveInfinity),
        (-Math.Pow(2, 128), double.NegativeInfinity),
    };

    [Fact]
    public void QuantizeBfloat16MatchesTorchBoundaryCases()
    {
        var adapter = new TensorPrecisionAdapter(TensorDtype.Int32, TensorDtype.Bfloat16);

        foreach (var (value, expected) in Bfloat16BoundaryCases)
        {
            Assert.Equal(expected, adapter.QuantizeFloat(value));
        }
    }

    [Fact]
    public void QuantizeFloat16MatchesTorchRounding()
    {
        var adapter = new TensorPrecisionAdapter(TensorDtype.Int32, TensorDtype.Float16);

        // 参考值由 torch.tensor(v, dtype=torch.float16).tolist() 生成
        Assert.Equal(0.0999755859375, adapter.QuantizeFloat(0.1));
        Assert.Equal(0.39990234375, adapter.QuantizeFloat(0.4));
        Assert.Equal(65504.0, adapter.QuantizeFloat(65504.0));
        // 超出 float16 最大值 → Inf（.NET Half 转换语义与 torch 一致）
        Assert.Equal(double.PositiveInfinity, adapter.QuantizeFloat(1e10));
    }

    [Fact]
    public void PrecisionConfigLoaderResolvesAliasesAndDefaults()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"precision_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(Path.Combine(tempDir, "config"));
        try
        {
            // 缺省 precision 键走默认 int32/float32（对齐 Python raw.get("precision", {})）
            File.WriteAllText(Path.Combine(tempDir, "config", "precision.yaml"), "");
            var defaults = PrecisionConfigLoader.Load(tempDir);
            Assert.Equal(TensorDtype.Int32, defaults.IntDtype);
            Assert.Equal(TensorDtype.Float32, defaults.FloatDtype);

            // 别名 + 大小写不敏感（对齐 _FLOAT_DTYPE_ALIASES）
            File.WriteAllText(
                Path.Combine(tempDir, "config", "precision.yaml"),
                "precision:\n  int_dtype: INT64\n  float_dtype: bf16\n");
            var aliased = PrecisionConfigLoader.Load(tempDir);
            Assert.Equal(TensorDtype.Int64, aliased.IntDtype);
            Assert.Equal(TensorDtype.Bfloat16, aliased.FloatDtype);

            File.WriteAllText(
                Path.Combine(tempDir, "config", "precision.yaml"),
                "precision:\n  float_dtype: fp16\n");
            Assert.Equal(TensorDtype.Float16, PrecisionConfigLoader.Load(tempDir).FloatDtype);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void PrecisionConfigLoaderRejectsUnknownAliasWithPythonStyleMessage()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"precision_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(Path.Combine(tempDir, "config"));
        File.WriteAllText(
            Path.Combine(tempDir, "config", "precision.yaml"),
            "precision:\n  float_dtype: int8\n");

        try
        {
            var exception = Assert.Throws<InvalidOperationException>(() => PrecisionConfigLoader.Load(tempDir));
            Assert.Contains("float_dtype='int8' is invalid", exception.Message);
            Assert.Contains("bf16, bfloat16, float16, float32, fp16, fp32", exception.Message);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void PrecisionConfigLoaderRejectsNullAliasWithPythonStyleMessage()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"precision_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(Path.Combine(tempDir, "config"));
        File.WriteAllText(
            Path.Combine(tempDir, "config", "precision.yaml"),
            "precision:\n  float_dtype: ~\n");

        try
        {
            var exception = Assert.Throws<InvalidOperationException>(() => PrecisionConfigLoader.Load(tempDir));
            Assert.Contains("float_dtype=None is invalid", exception.Message);
            Assert.Contains("bf16, bfloat16, float16, float32, fp16, fp32", exception.Message);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }
}

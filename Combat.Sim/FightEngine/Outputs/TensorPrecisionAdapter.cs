// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Outputs;

/// <summary>tensor 打包用的 dtype 规格（对照 precision.yaml 的 int_dtype / float_dtype）。</summary>
public enum TensorDtype
{
    Int32,
    Int64,
    Float32,
    Float64,
    Float16,
    Bfloat16,
}

/// <summary>
/// tensor 输出适配层：把输出数值按目标 dtype 折算（对照 torch.tensor(..., dtype=...) 的取值语义）。
/// 量化逻辑集中在这里，打包器只按适配器折算，不散落各数值转换处；
/// 取消/调整 dtype 适配只需改这一个文件。
/// </summary>
public sealed class TensorPrecisionAdapter
{
    private readonly TensorDtype _intDtype;
    private readonly TensorDtype _floatDtype;

    public TensorPrecisionAdapter(TensorDtype intDtype, TensorDtype floatDtype)
    {
        _intDtype = intDtype;
        _floatDtype = floatDtype;
    }

    /// <summary>
    /// 浮点数值按 float dtype 量化：float32 经 IEEE-754 round-to-nearest-even 折算
    /// （float32(0.4) 即 0.4000000059604645），float16 经 .NET Half（binary16，同为
    /// round-to-nearest-even），bfloat16 按单次舍入截尾数到 7 位，使打包结果与
    /// Python 端 tolist() 逐位一致；float64 保持原值。
    /// </summary>
    public double QuantizeFloat(double value) =>
        _floatDtype switch
        {
            TensorDtype.Float32 => (double)(float)value,
            TensorDtype.Float16 => (double)(Half)value,
            TensorDtype.Bfloat16 => QuantizeBfloat16(value),
            _ => value,
        };

    /// <summary>整数数值按 int dtype 量化（int32/int64 均无损，保留原值）。</summary>
    public int QuantizeInt(int value) => value;

    /// <summary>
    /// double → bfloat16 单次舍入（round-to-nearest-even），对照 torch 由 double 直接
    /// 截尾数到 7 位；不做 double→float 的中间舍入，避免双重舍入在边界值上产生差异。
    /// 位布局：符号 1 位 + 指数 8 位（偏置 127）+ 尾数 7 位；次正规步长 2^-133，
    /// 最小正规 2^-126（与次正规 127 连续）。
    /// </summary>
    private static double QuantizeBfloat16(double value)
    {
        var bits = BitConverter.DoubleToUInt64Bits(value);
        var sign = bits >> 63;
        var exponent = (int)((bits >> 52) & 0x7FF);
        var mantissa = bits & 0xF_FFFF_FFFF_FFFFUL;

        if (exponent == 0x7FF)
        {
            return value; // NaN / Inf 原样保留（指数全 1）
        }

        var biasedExponent = exponent - 1023 + 127;
        if (biasedExponent >= 0xFF)
        {
            // 指数超出 bfloat16 范围：按符号溢出为 ±Inf
            return sign != 0 ? double.NegativeInfinity : double.PositiveInfinity;
        }

        if (biasedExponent > 0)
        {
            // 正规数：尾数 52 → 7 位（截断 45 位），round-to-nearest-even
            var lsb = (mantissa >> 45) & 1;
            var rounded = (mantissa + (1UL << 44) - 1 + lsb) >> 45;
            var resultExponent = biasedExponent;
            if (rounded >= 1UL << 7)
            {
                // 7 位小数位溢出进位到指数（rounded 不含隐式位，128 即进位标记）
                resultExponent++;
                rounded = 0;
                if (resultExponent >= 0xFF)
                {
                    return sign != 0 ? double.NegativeInfinity : double.PositiveInfinity;
                }
            }

            return BitConverter.UInt64BitsToDouble(
                (sign << 63) | ((ulong)(resultExponent + 896) << 52) | (rounded << 45));
        }

        // 次正规 / 下溢：v = 1.m × 2^(biased-127) 按 2^-133 步长舍入到整数 N
        // （N ∈ [1,127] 为次正规尾数，N == 128 恰好进位到最小正规 2^-126）
        var shift = 46 - biasedExponent; // 隐含 1 位 + 52 位尾数右移的位数（≥ 46）
        if (shift >= 54)
        {
            // 值 < 2^-133 的一半：舍入为 0（保留符号）
            return sign != 0 ? -0.0 : 0.0;
        }

        var significand = mantissa + (1UL << 52);
        var roundedN = (significand + (1UL << (shift - 1)) - 1 + ((significand >> shift) & 1)) >> shift;
        if (roundedN == 0)
        {
            return sign != 0 ? -0.0 : 0.0;
        }
        if (roundedN == 1UL << 7)
        {
            // 进位到最小正规 2^-126
            return BitConverter.UInt64BitsToDouble((sign << 63) | (897UL << 52));
        }

        // N×2^-133 规范化到 double 位域：最高位 k 决定指数位（890+k），余数 m 左移到尾数
        var topBit = 63 - global::System.Numerics.BitOperations.LeadingZeroCount(roundedN);
        var remainder = roundedN - (1UL << topBit);
        return BitConverter.UInt64BitsToDouble(
            (sign << 63) | ((ulong)(890 + topBit) << 52) | (remainder << (52 - topBit)));
    }
}

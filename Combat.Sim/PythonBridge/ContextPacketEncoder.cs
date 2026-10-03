// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only.

using System.Buffers.Binary;
using System.Collections;
using System.Numerics;
using System.Text;

namespace Combat.Sim.PythonBridge;

/// <summary>
/// 在托管线程中批量编码观测，不访问 Python 对象或获取 GIL。
/// 只生成 pickle 协议 2 的基础值指令，由 Python 标准库一次还原独立 dict/list；
/// 不生成全局对象、构造器或可执行指令，不保存为文件或接受外部 pickle 输入。
/// </summary>
public static class ContextPacketEncoder
{
    public static byte[] Encode(IDictionary value)
    {
        ArgumentNullException.ThrowIfNull(value);
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        var encoder = new Encoder(writer);
        writer.Write((byte)0x80); // PROTO
        writer.Write((byte)2);
        encoder.Write(value);
        writer.Write((byte)'.'); // STOP
        return stream.ToArray();
    }

    private sealed class Encoder(BinaryWriter writer)
    {
        // 仅复用不可变字符串；容器每次独立还原，调用方原地改写不会污染其他观测。
        private readonly Dictionary<string, int> _strings = new(StringComparer.Ordinal);

        public void Write(object? value, int depth = 0)
        {
            if (depth > 128)
                throw new NotSupportedException("状态机观测嵌套过深或包含循环引用");
            switch (value)
            {
                case null: writer.Write((byte)'N'); break;
                case bool flag: writer.Write(flag ? (byte)0x88 : (byte)0x89); break;
                case string text: WriteString(text); break;
                case char character: WriteString(character.ToString()); break;
                case byte number: WriteInteger(number); break;
                case sbyte number: WriteInteger(number); break;
                case short number: WriteInteger(number); break;
                case ushort number: WriteInteger(number); break;
                case int number: WriteInteger(number); break;
                case uint number: WriteInteger(number); break;
                case long number: WriteInteger(number); break;
                case ulong number: WriteInteger(number); break;
                case float number: WriteFloat(number); break;
                case double number: WriteFloat(number); break;
                case decimal number: WriteFloat((double)number); break;
                case IDictionary dictionary:
                    writer.Write((byte)'}'); // EMPTY_DICT
                    writer.Write((byte)'('); // MARK
                    foreach (DictionaryEntry entry in dictionary)
                    {
                        if (entry.Key is not string key)
                            throw new NotSupportedException("状态机观测字典的键必须是字符串");
                        WriteString(key);
                        Write(entry.Value, depth + 1);
                    }
                    writer.Write((byte)'u'); // SETITEMS
                    break;
                case IList list:
                    writer.Write((byte)']'); // EMPTY_LIST
                    writer.Write((byte)'(');
                    foreach (var item in list) Write(item, depth + 1);
                    writer.Write((byte)'e'); // APPENDS
                    break;
                default:
                    throw new NotSupportedException($"状态机观测包含不支持的 CLR 值类型：{value.GetType().FullName}");
            }
        }

        private void WriteString(string value)
        {
            if (_strings.TryGetValue(value, out var index))
            {
                writer.Write((byte)'j'); // LONG_BINGET
                writer.Write(index);
                return;
            }
            var bytes = Encoding.UTF8.GetBytes(value);
            writer.Write((byte)'X'); // BINUNICODE
            writer.Write(bytes.Length);
            writer.Write(bytes);
            writer.Write((byte)'r'); // LONG_BINPUT
            writer.Write(_strings.Count);
            _strings.Add(value, _strings.Count);
        }

        private void WriteInteger(BigInteger value)
        {
            if (value >= int.MinValue && value <= int.MaxValue)
            {
                writer.Write((byte)'J'); // BININT
                writer.Write((int)value);
                return;
            }
            var bytes = value.ToByteArray(); // 小端有符号二进制，保留 ulong 的最高位。
            writer.Write((byte)0x8a); // LONG1
            writer.Write(checked((byte)bytes.Length));
            writer.Write(bytes);
        }

        private void WriteFloat(double value)
        {
            writer.Write((byte)'G'); // BINFLOAT 规定使用大端 IEEE 754。
            Span<byte> bytes = stackalloc byte[sizeof(double)];
            BinaryPrimitives.WriteDoubleBigEndian(bytes, value);
            writer.Write(bytes);
        }
    }
}

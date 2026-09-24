// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only.

using System.Collections;
using Python.Runtime;

namespace Combat.Sim.PythonBridge;

/// <summary>将状态机的 CLR 容器直接映射成 Python 原生容器，不经过文本序列化。</summary>
public static class NativeContextMarshaller
{
    private static PyObject? _boolType;

    public static PyObject Convert(IDictionary value)
    {
        using var gil = Py.GIL();
        return ConvertValue(value);
    }

    private static PyObject ConvertValue(object? value)
    {
        if (value is null)
            return PyObject.FromManagedObject(null!);
        if (value is string text)
            return new PyString(text);
        if (value is char character)
            return new PyString(character.ToString());
        if (value is bool boolean)
            return ConvertBoolean(boolean);
        if (value is byte byteValue)
            return new PyInt(byteValue);
        if (value is sbyte signedByteValue)
            return new PyInt(signedByteValue);
        if (value is short shortValue)
            return new PyInt(shortValue);
        if (value is ushort unsignedShortValue)
            return new PyInt(unsignedShortValue);
        if (value is int integerValue)
            return new PyInt(integerValue);
        if (value is uint unsignedIntegerValue)
            return new PyInt(unsignedIntegerValue);
        if (value is long longValue)
            return new PyInt(longValue);
        if (value is ulong unsignedLongValue)
            return new PyInt(unsignedLongValue);
        if (value is float singleValue)
            return new PyFloat(singleValue);
        if (value is double doubleValue)
            return new PyFloat(doubleValue);
        if (value is decimal decimalValue)
            return new PyFloat((double)decimalValue);
        if (value is IDictionary dictionary)
            return ConvertDictionary(dictionary);
        if (value is Array array)
            return ConvertArray(array);
        if (value is IList list)
            return ConvertList(list);

        throw new NotSupportedException(
            $"状态机观测包含不支持的 CLR 值类型：{value.GetType().FullName}");
    }

    private static PyObject ConvertDictionary(IDictionary value)
    {
        var output = new PyDict();
        try
        {
            foreach (DictionaryEntry entry in value)
            {
                if (entry.Key is not string keyText)
                    throw new NotSupportedException("状态机观测字典的键必须是字符串");

                using var key = new PyString(keyText);
                using var item = ConvertValue(entry.Value);
                output.SetItem(key, item);
            }

            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }

    private static PyObject ConvertArray(Array value)
    {
        var output = new PyList();
        try
        {
            foreach (var item in value)
            {
                using var converted = ConvertValue(item);
                output.Append(converted);
            }

            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }

    private static PyObject ConvertList(IList value)
    {
        var output = new PyList();
        try
        {
            foreach (var item in value)
            {
                using var converted = ConvertValue(item);
                output.Append(converted);
            }

            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }

    private static PyObject ConvertBoolean(bool value)
    {
        if (_boolType is null)
        {
            using var builtins = Py.Import("builtins");
            _boolType = builtins.GetAttr("bool");
        }

        using var argument = new PyInt(value ? 1 : 0);
        return _boolType.Invoke(argument);
    }
}

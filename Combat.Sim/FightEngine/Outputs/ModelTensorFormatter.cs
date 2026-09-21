// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Outputs;

/// <summary>
/// 模型 tensor 输出翻译模块（对照 outputs/model_tensor_formatter.py）。
/// 输出层不直接依赖 torch：打包结果是与 torch.tensor(...).tolist() 等价的嵌套结构，
/// dtype 规格随构造参数显式传入（对照统一精度配置，禁止输出链路自己写死精度）。
/// </summary>
public sealed class ModelTensorFormatter
{
    private readonly TensorPrecisionAdapter _adapter;

    public ModelTensorFormatter(TensorDtype intDtype, TensorDtype floatDtype)
    {
        _adapter = new TensorPrecisionAdapter(intDtype, floatDtype);
    }

    public object? Format(Dictionary<string, object?> outputContext) =>
        TensorPayloadPacker.PackPayload(
            OutputContextSchema.FormatCanonicalOutputContext(outputContext),
            _adapter);
}

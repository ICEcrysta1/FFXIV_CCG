// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Outputs;

/// <summary>canonical 输出翻译模块（对照 outputs/model_vector_formatter.py）。</summary>
public static class ModelVectorFormatter
{
    /// <summary>把统一上下文裁剪成稳定的 canonical 输出。</summary>
    public static Dictionary<string, object?> Format(Dictionary<string, object?> outputContext) =>
        OutputContextSchema.FormatCanonicalOutputContext(outputContext);
}

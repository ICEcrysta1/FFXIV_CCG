// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// Additional permission: FightEngine GPLv3 Linking Exception, Version 1.0.
// See LICENSE and LICENSE-FightEngine-Linking-Exception in the repository root.

namespace Combat.Sim.Common;

/// <summary>
/// 仓库根目录定位（向上找含 config/default.yaml 的目录）。
/// 供缺省 projectRoot 的构造路径与宿主顶层入口复用。
/// </summary>
public static class RepoRootLocator
{
    public static string Find()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "config", "default.yaml")))
            {
                return dir.FullName;
            }
        }
        throw new InvalidOperationException("仓库根未找到（缺少 config/default.yaml）");
    }
}

<p align="center">
  <img src="preview/Logo+Title.png" alt="FFXIV_CCG" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.12-brightgreen?labelColor=555&logo=python" alt="Python >= 3.12" />
  <img src="https://img.shields.io/badge/torch-2.12.0%2Bcu132-ee4c2c?labelColor=555&logo=pytorch" alt="torch 2.12.0+cu132" />
  <img src="https://img.shields.io/badge/ONNX%20Runtime-1.27.0%20GPU-005ced?labelColor=555&logo=onnx" alt="ONNX Runtime 1.27.0 GPU" />
  <img src="https://img.shields.io/badge/CUDA-13.x-76b900?labelColor=555&logo=nvidia" alt="CUDA 13.x" />
  <img src="https://img.shields.io/badge/.NET-10.0-512bd4?labelColor=555&logo=dotnet" alt=".NET 10.0" />
  <img src="https://img.shields.io/badge/license-GPL--3.0--only-blue?labelColor=555" alt="License: GPL-3.0-only" />
</p>

## 📄 FFXIV CCG 文档

[FFXIV CCG](https://github.com/ICEcrysta1/FFXIV_CCG)（Context Combat Generator）是一个面向《最终幻想 XIV》战斗场景的生成式决策模型系列，覆盖 FFLogs 战斗日志采集、行为克隆预训练、GRPO 后训练、ONNX 部署导出、模型表征分析与自回归回放，并提供 PowerShell 命令行入口统一调用。模型以 Transformer 为骨干，依靠注意力机制对战斗上下文进行长程建模；在实时推理约束下，建议采用较小规模的层数与维度配置（例如 8×768、12×512），以在推理时延与表征能力之间取得平衡。

---

## 🖥️ 训练配置最低要求

以下是项目训练的最低硬件要求

| 项目 | 最低要求 | 说明 |
|---|---|---|
| GPU 显存 | `>= 4 GiB` | 低于此容量不属于明确支持范围。 |
| GPU 型号 | NVIDIA RTX 3050 或更新型号 | RTX 3050 已完成12x768x12x3072 256上下文训练验证。 |
| 系统内存 | `>= 16 GiB` | 用于 raw JSON、compiled cache、DataLoader 和训练进程。 |

当前 `artzip` 配置不是最低训练配置，不能按上述最低硬件要求直接运行；它需要更高的显存和计算资源。最低硬件测试应使用单独缩小后的模型与训练 YAML，不应把当前 `artzip` 配置的失败归因于最低硬件要求本身。

---

## 🚀 通过命令快速开始

请见下文快速启动安装和使用示例。有关FLogs 数据下载、行为克隆预训练、GRPO 后训练、ONNX 导出、模型分析与自回归回放的详细命令参数请参阅我们完整的 [说明文档](./docs/命令行使用说明.md)。

```pwsh
# 1) 一键准备环境：检查系统 Python >= 3.12，创建根目录 .venv，安装 CUDA/GPU 依赖
.\setup.ps1
# 如果执行策略拦截脚本，再执行：
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\setup.ps1

# 2) 按 setup.ps1 输出和 .env.example 的说明准备环境变量
if (-not (Test-Path .env)) { Copy-Item .env.example .env }

# 3) 构建 C# 状态机运行文件；Python.NET 随后在同一 Python 进程内加载 FightEngine
dotnet build Combat.Sim/SidecarHost/SidecarHost.csproj --configuration Debug

# 4) 启动工具菜单：输入 1～7 执行对应工具，0 退出
.\ffxiv_ccg.ps1
```

训练、转换和回放直接调用 C# 状态机，不会启动 SidecarHost 子进程或逐次传输 JSON。
SidecarHost 项目只用于生成 Python.NET 所需的 CoreCLR 配置与托管依赖文件，同时保留 JSON Lines 兼容入口。

本项目只支持 NVIDIA CUDA 环境，训练、导出和自回归不提供 CPU 支持。

---

## 📜 许可

Project-owned source code in this repository is licensed under the GNU
General Public License v3.0 only (GPL-3.0-only); see LICENSE. Third-party
materials remain under their own applicable licenses.

The FightEngine Library consists of the project-owned source files under
Combat.Sim/FightEngine/ included by
Combat.Sim/FightEngine/FightEngine.csproj. Those files also carry the
additional permission in
LICENSE-FightEngine-Linking-Exception, Version 1.0. The exception permits
combining the FightEngine Library with Independent Modules under its stated
conditions; it does not automatically apply to SidecarHost, tests, tools,
Python code, or other repository components.

Copyright notices for the relevant material identify ICE_crystal and, for
the original Machinist implementation, SpikeHS.

---

## 💬 反馈方式

有关 [FFXIV CCG](https://github.com/ICEcrysta1/FFXIV_CCG) 的 bug 与功能建议，请访问 [GitHub Issues](https://github.com/ICEcrysta1/FFXIV_CCG/issues)。

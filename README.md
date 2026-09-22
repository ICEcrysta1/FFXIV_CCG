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

[FFXIV CCG](https://github.com/ICEcrysta1/FFXIV_CCG) 是一个适用于FFXIV的生成式战斗 AI 模型系列（Context Combat Generator），支持 FFLogs 数据下载、行为克隆预训练、GRPO 后训练、ONNX 导出、模型分析与自回归回放，可通过 ps1 命令行脚本一键使用。FFXIV CCG 基于Transformer架构，使其具备高效的上下文理解能力，对于适用于实时推理的情况下建议设置较小的层数和维度，例如 8x768 或者 12x512 。

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
# 1) 准备虚拟环境：Python >= 3.12 均可，项目正式环境固定为根目录 .venv
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # 仅当 Activate.ps1 被执行策略拦截时
.\.venv\Scripts\Activate.ps1

# 2) 安装主依赖：固定 torch 2.12.0+cu132
# 可直接运行在 CUDA 13.x 驱动上
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3) 安装 ONNX GPU 依赖
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install -r requirements-onnx-gpu.txt

# 4) 确认 GPU 依赖就绪
# 应输出 torch 2.12.0+cu132 与 CUDAExecutionProvider
python -c "import torch, onnxruntime as ort; print(torch.__version__, torch.version.cuda); print(ort.__version__, ort.get_available_providers())"

# 5) 准备环境变量：按需填写 FFLogs 凭证、职业标签、模型变体和 checkpoint
Copy-Item .env.example .env

# 6) 启动工具菜单：输入 1～6 执行对应工具，0 退出
.\ffxiv_ccg.ps1
```

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

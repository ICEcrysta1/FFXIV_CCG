<p align="center">
  <img src="preview/Logo+Title.png" alt="FFXIV_CCG" />
</p>

<p align="center">
  <a href="https://github.com/ICEcrysta1/FFXIV_CCG/releases/latest"><img src="https://img.shields.io/github/v/release/ICEcrysta1/FFXIV_CCG?label=CCG&labelColor=555&color=blue&logo=github" alt="CCG 最新版本" /></a>
  <img src="https://img.shields.io/badge/VRAM-%3E%3D4Gi-76b900?labelColor=555" alt="VRAM >= 4Gi" />
  <img src="https://img.shields.io/badge/RAM-%3E%3D16Gi-ff6f00?labelColor=555" alt="RAM >= 16Gi" />
  <img src="https://img.shields.io/badge/python-%3E%3D3.12-brightgreen?labelColor=555&logo=python" alt="Python >= 3.12" />
  <img src="https://img.shields.io/badge/torch-2.12.0%2Bcu132-ee4c2c?labelColor=555&logo=pytorch" alt="torch 2.12.0+cu132" />
  <img src="https://img.shields.io/badge/ONNX%20Runtime-1.27.0%20GPU-005ced?labelColor=555&logo=onnx" alt="ONNX Runtime 1.27.0 GPU" />
  <img src="https://img.shields.io/badge/CUDA-13.x-76b900?labelColor=555&logo=nvidia" alt="CUDA 13.x" />
  <img src="https://img.shields.io/badge/Rust-2024-dea584?labelColor=555&logo=rust" alt="Rust 2024" />
  <img src="https://img.shields.io/badge/Node-22.x-339933?labelColor=555&logo=nodedotjs" alt="Node 22.x" />
  <img src="https://img.shields.io/badge/license-GPL--3.0--only-blue?labelColor=555" alt="License: GPL-3.0-only" />
</p>

## 📄 FFXIV CCG 文档

[FFXIV CCG](https://github.com/ICEcrysta1/FFXIV_CCG)（Context Combat Generator）是一个面向《最终幻想 XIV》战斗场景的生成式决策模型系列，覆盖 FFLogs 战斗日志采集、行为克隆预训练、GRPO 后训练、ONNX 部署导出、模型表征分析与自回归回放，并提供 Bash 命令行入口统一调用。模型以 Transformer 为骨干，依靠注意力机制对战斗上下文进行长程建模；在实时推理约束下，建议采用较小规模的层数与维度配置（例如 6×768、8×512），以在推理时延与表征能力之间取得平衡。

---

## 🖥️ 训练配置最低要求

| 项目 | 最低要求 | 说明 |
|---|---|---|
| GPU 显存 | `>= 4 GiB` | 低于此容量不属于明确支持范围。 |
| GPU 型号 | NVIDIA RTX 3050 或更新型号 | RTX 3050 已完成12x768x12x3072 256上下文训练验证。 |
| 系统内存 | `>= 16 GiB` | 用于 raw JSON、compiled cache、DataLoader 和训练进程。 |

当前 `artzip` 配置不是最低训练配置，不能按上述最低硬件要求直接运行；它需要更高的显存和计算资源。最低硬件测试应使用单独缩小后的模型与训练 YAML，不应把当前 `artzip` 配置的失败归因于最低硬件要求本身。

---

## 🚀 通过命令快速开始

项目开发与运行主线为 Linux／WSL2，统一使用 `python3` 和 Bash 入口。先安装 Python 3.12+（含 venv）、通过 rustup 管理的 Rust stable（含 Cargo）、C 编译器、Git、curl、tar、xz，以及支持 CUDA 13.x 的 NVIDIA 驱动；`setup.sh` 会安装项目专用 Linux Node 22、pnpm 10 和 Python 依赖，并以 release 模式构建 Rust 原生扩展。详细参数见 [命令行使用说明](./docs/命令行使用说明.md)。

```bash
# 1) 一键准备环境：创建根目录 .venv 和项目本地 .node，安装 Python GPU 与 Node 依赖
./setup.sh

# 2) 按 .env.example 配置职业、模型变体、FFLogs 凭证及本机参数
[ -f .env ] || cp .env.example .env

# 3) 修改 Rust 状态机或共享 schema 后，重新构建并安装原生扩展
.venv/bin/python3 -m pip install ./fight-engine/python

# 4) 验证配置与 Rust 状态机链路
.venv/bin/python3 main.py validate --job-tag black_mage
.venv/bin/python3 main.py smoke --job-tag black_mage

# 5) 启动工具菜单：输入 1～9 执行对应工具，10 退出
./ffxiv_ccg.sh
```

Windows 的 `.venv` 和 `.node` 不能搬入 Linux 复用，需要在 Linux 重新运行 `./setup.sh`。

首次训练按以下顺序准备数据；重复运行下载、标注和转换会复用经校验仍有效的产物：

| 顺序 | 菜单 | 产物与用途 |
|---|---|---|
| 1 | FFLogs 数据下载 | 国服分档训练日志与北美高分验证日志，保存为 `raw/**/*.json.br` |
| 2 | XIVanalysis 数据标记 | 从 raw 生成 annotated，保留训练分档与 `VAL/<副本>/` 层级 |
| 3 | 训练文件转换 | 按 `training.max_files` 和独立的 `training.validation_files` 准备完整 compiled cache |
| 4 | BC 预训练 / 恢复训练 | 只读取已有有效缓存，缺失或过期时提示先转换 |
| 5 | GRPO / ONNX 导出 / 分析 / 回放 | 使用与当前输入契约兼容的 checkpoint；ONNX 导出菜单包含正式发布门禁 |

训练配额和验证配额互不借用；验证副本不足时只在同副本内补位。下载器的 `validation_ratio` 决定准备多少验证日志，训练 YAML 的 `validation_files` 决定实际消费多少份，两者独立配置。算法、数据语义和训练参数以 YAML 为准，根目录 `.env` 管理本机配置；BC 与 GRPO 的优化器分别由 `optimizer.yaml` 的 `optimizers.bc` / `optimizers.grpo` 配置。

菜单中的 BC 训练与恢复训练按 `training.yaml` 顶部的 `training.tensorboard.enabled` 决定是否自动打开 TensorBoard；GRPO 按 `grpo.yaml` 顶部的 `grpo.tensorboard.enabled` 独立决定。服务就绪后会打开本机浏览器，端口由 `.env` 的 `TENSORBOARD_PORT` 指定。由 `ffxiv_ccg.sh` 启动的 Web 服务会在训练入口结束或中断后关闭；手动启动的后台服务仍需手动管理。若该端口上的旧服务读取其他模型的事件目录，启动器会报错并提示关闭旧服务或换端口。

新训练 run 会写入以下面板，已有日志不会自动补齐缺少的数据：

| 面板 | 数据与用途 |
|---|---|
| Scalars / Custom Scalars | 记录区间的样本加权损失、准确率、实际更新学习率、梯度范数及每轮回放 PPG；步号为累计优化器更新次数 |
| Graphs | 首个真实编码样本的 eval 前向结构图，仅代表该样本的形状与分支 |
| Histograms / Distributions | 权重、未缩放梯度、logits 与概率；GRPO 另记录奖励、优势、ratio、KL 等已有轨迹与更新数据 |
| Projector | 分别展示输入技能 embedding 和独立输出动作头，带动作名、raw id、vocab id 和 GCD 类型，排除 padding 行 |
| Images / PR Curves | BC 全验证集的动作混淆矩阵、逐动作 precision/recall/support，以及有界采样的一对其余类别 PR 曲线 |
| Text / HParams | 模型结构、实际运行配置、指标语义、混淆计数和本次运行的最终指标，供跨 run 对比 |

`tensorboard.graph` 控制结构图，`histogram_every_steps` 控制分布频率，`max_histogram_values` 限制每项分布的确定性采样量；`embedding_every_epochs` 控制向量快照频率，初始状态也会保存。BC 的 `pr_curves` / `pr_curve_max_samples` 控制 PR 曲线及其内存上限，混淆矩阵始终汇总全部有效验证样本。这些设置分别位于 BC / GRPO 的 YAML；诊断复用已有前向结果，不消耗训练随机数。HParams 在训练退出时写入最近一次完成的指标；训练中可先查看 Text 中的配置。

训练、转换和回放通过 PyO3 在当前 Python 进程内调用 Rust 状态机。核心与各职业为独立 crate，Python 原生扩展负责注册职业；执行和观测编码释放 GIL，各队列使用独立锁。观测通过基础值 pickle 包由 Python 标准库还原为独立字典和列表，转换子进程继续复用父进程中的引擎。

本项目只支持 NVIDIA CUDA 环境，训练、导出和自回归不提供 CPU 支持。

不符合当前缓存／转换契约的旧缓存需通过转换入口重建；模型输入契约变化时，旧 checkpoint 需重新训练，ONNX 包需重新导出，符合当前缓存契约的完整 PT 无须重新转换，旧 GRPO 轨迹需按新输入契约重新采样。单独生成 `model.onnx` 不代表部署验收完成：正式流程要求同一部署包通过空场景 128 GCD、真实场景至少 100 次执行动作的 PT/ORT parity，并校验契约、文件和证据指纹。只有当前加载器校验通过且状态为 `release_validated` 的包可用于正式部署；旧包内的同名状态不能替代当前版本验收。详见 [ONNX 导出与发布门禁](./docs/命令行使用说明.md#7-onnx-导出与发布门禁)。

---

## 📜 许可

离线动作质量标注的外部分析依赖以 Git 子模块保存在
`third_party/xivanalysis/`，版本由主仓库的子模块 commit 固定。已有仓库可运行
以下命令获取该版本；新克隆仓库也可以使用 `git clone --recurse-submodules`：

```bash
git submodule update --init -- third_party/xivanalysis
```

该依赖采用 MIT 许可，原始许可证保留在
[third_party/xivanalysis/LICENSE](./third_party/xivanalysis/LICENSE)，来源与完整许可声明见
[THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。离线桥接入口为
`python3 -m scripts.action_quality`，安装和输出契约见 [命令行说明](./docs/命令行使用说明.md#附录-d离线动作质量标注)。

Project-owned source code in this repository is licensed under the GNU
General Public License v3.0 only (GPL-3.0-only); see LICENSE. Third-party
materials remain under their own applicable licenses.

The Rust core and the Black Mage and Machinist library sources under
`fight-engine/`, together with their compiled code portions, have the
additional permission in [FightEngine GPLv3 Linking Exception, Version 1.1](./LICENSE-FightEngine-Linking-Exception).
Independent Modules need not disclose their source solely because they are
combined with the library; the GPLv3 source obligations for the library and
its modifications remain in effect. The three library Cargo packages carry
both the GPLv3 and exception texts.
The exception does not automatically apply to the Python binding code,
tests, tools, or third-party dependencies, nor to a combined native extension
as a whole. See the exception for the precise scope and conditions.

Copyright notices for the relevant material identify ICE_crystal and, for
the original Machinist implementation, SpikeHS.

---

## 💬 反馈方式

有关 [FFXIV CCG](https://github.com/ICEcrysta1/FFXIV_CCG) 的 bug 与功能建议，请访问 [GitHub Issues](https://github.com/ICEcrysta1/FFXIV_CCG/issues)。

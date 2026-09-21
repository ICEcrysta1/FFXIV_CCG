<p align="center">
  <img src="preview/Logo+Title.png" alt="FFXIV_CCG" />
</p>

<p align="center">
  <em>Context Combat Generator — 最终幻想 XIV 职业 AI 行为克隆项目</em>
</p>

---

## License

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


## 项目结构

```
common/        跨顶层模块复用的公共函数
Combat.Sim/    C# 战斗状态机、SidecarHost、CLI 导出宿主与 xUnit 测试
config/        模拟器、职业和 tensor 精度 YAML 配置
data/          训练数据 (不进入版本控制)
  human/job/<job>/
    raw/       FFLogs 原始 JSON，只读输入
    .cache/    raw JSON 直接编译出的最终 compiled cache
scripts/       辅助脚本 (FFLogs 抓取、日志转换、模型分析、模型回放)
training/      训练公共层与模型架构
tests/         测试目录
artifacts/     checkpoint、模型分析和回放等运行产物
preview/       README 和项目展示资源，不承载业务代码
```

训练和分析都直接消费 `raw/` 对应的 `.cache/`。项目不再保留 raw JSON 到训练 PT 之间的中间格式；`.cache/` 缺失或签名过期时由转换脚本生成。

## CLI 工具

### main.py — 状态机后端检查

```bash
# 打印配置概览
python main.py validate

# 列出当前职业所有已启用技能
python main.py list-skills

# 列出初始状态下所有合法动作
python main.py list-actions

# 跑一段最小循环验证主链路
python main.py smoke
```

可选参数 `--job-tag` 可指定职业路由，默认读取根目录 `.env` 中的 `FFXIV_JOB_TAG`。

### scripts/fflogs_scraper.py — FFLogs 数据拉取

通过 FFLogs V2 GraphQL API 拉取战斗报告事件数据。需要在 `.env` 中配置 `FFLOGS_V2_CLIENT_ID` 和 `FFLOGS_V2_CLIENT_SECRET`。

```bash
# 单报告下载 (URL 方式)
python scripts/fflogs_scraper.py "https://www.fflogs.com/reports/<CODE>?fight=6&source=10"

# 单报告下载 (参数方式)
python scripts/fflogs_scraper.py single --report <CODE> --fight 6 --source 10

# 批量下载高分报告 (按 encounter 排行)
python scripts/fflogs_scraper.py batch -e 1079 --spec-name BlackMage --max-pages 3

# 列出 zone 下的 encounters
python scripts/fflogs_scraper.py encounters -z 39

# 列出所有 zones
python scripts/fflogs_scraper.py encounters
```

子命令:

- `single` — 单报告下载，支持 `--events-only` `--damage-only` `--output` `--output-dir`
- `batch` — 按 encounter 排行批量下载，支持 `--bracket` `--metric` `--mode` `--output`
- `encounters` — 列出 zones 或 zone 下的 encounters

### Combat.Sim — 转换与回放的 C# 状态机后端

FFLogs 转换和自回归回放都通过 `SidecarHost` 驱动 C# 状态机。每次修改
`Combat.Sim/FightEngine`、`Combat.Sim/SidecarHost` 或
`config/schema.yaml` 后，必须在同一工作树重建宿主，再启动转换/回放：

```powershell
dotnet build Combat.Sim/SidecarHost/SidecarHost.csproj --configuration Debug
```

`SidecarHost` 与 Python 客户端都从 `config/schema.yaml` 读取并校验运行时契约版本。旧 DLL（例如仍输出没有 `value` 的技能 token）会在
`init` 阶段被 Python 客户端拒绝；新版 DLL 使用旧的 `config/schema.yaml` 时会因缺少必需的
`contracts.scene_epsilon` 或 `contracts.sidecar_contract_version` 直接失败。因此 DLL 与配置必须来自同一版本，不能只替换其中一部分；启动转换或回放前应先执行真实 Sidecar init 握手。

### scripts/convert_fflogs/cli.py — FFLogs 原始 JSON 直接编译最终训练缓存

把 `fflogs_scraper.py` 拉取的原始 JSON 文件直接转换为职业 `.cache` 下的最终 compiled cache。raw 目录只读，不保留中间训练 PT。

```bash
# 使用训练 YAML 中的 raw_data_dir，自动扫描 raw JSON，写入对应职业的 .cache
python -m scripts.convert_fflogs.cli

# 单文件转换
python -m scripts.convert_fflogs.cli data/fflogs_xxx.json

# 转换一个副本目录下的全部 raw JSON（脚本会递归扫描目录）
python -m scripts.convert_fflogs.cli data/human/job/black_mage/raw/FRU

# 指定最终缓存根目录
python -m scripts.convert_fflogs.cli data/fflogs_xxx.json --cache-root .tmp/black_mage_cache

# 并行转换整个职业 raw 目录
python -m scripts.convert_fflogs.cli data/human/job/black_mage/raw --workers 4
```

参数:

- `inputs` — 原始 FFLogs JSON 文件列表，可省略；省略时自动扫描 `.env` 职业目录下的 `raw/`
- `--job-tag` — 职业标识，默认依次读取 `.env` 和 `config/convert_fflogs/default.yaml`
- `--source` — 玩家 sourceID，默认优先读取 JSON 内的 `source_id`
- `--encounter` — 覆盖副本名
- `--cache-root` — 最终 compiled cache 根目录；省略时使用 `data/human/job/<job>/.cache`
- `--shard-size` — 覆盖最终缓存 shard 大小
- `--downtime-gap` — downtime 判定伤害间隙阈值 (默认 3.0)
- `--workers` — 并行进程数，`1` 强制串行

省略 `inputs` 时，只扫描训练 YAML 的 `raw_data_dir`；最终缓存统一写入
`data/human/job/<job>/.cache/`，例如 `raw/FRU/fight.json` 会生成对应的 compiled manifest 和 shard。缓存签名包含 raw 文件与转换参数，训练启动时也会自动调用同一脚本层入口补齐。单个 raw JSON 转换失败会记录错误并跳过；训练的 `--max-files` 配额会继续从同一副本目录的后备 JSON 补位，直到达到该副本的有效文件数要求，候选耗尽时才报告缺口并停止训练。

### training/train.py — 训练入口

训练职业行为克隆模型。

```bash
# 使用 .env 默认配置训练
python training/train.py

# 当前职业由 .env 的 FFXIV_JOB_TAG 自动选择；这里只覆盖数据目录
python training/train.py --raw-data-dir data/human/job/black_mage/raw

# 覆盖超参数
python training/train.py --epochs 50 --batch-size 32 --lr 1e-4 --device cpu

# 快速验证 (按副本比例选择 N 个 raw JSON；失败时从同副本后备文件补位)
python training/train.py --max-files 3

# 从已有 checkpoint 的下一轮继续训练
python training/train.py --resume artifacts/checkpoints/black_mage/artzip_bc/epoch_009_ppg_569.63.pt
```

参数:

- `--config` — 可选的显式模型 YAML；省略时读取根目录 `.env` 的 `FFXIV_JOB_TAG`，自动选择 `config/models/<job_tag>/*/config.yaml`
- `--raw-data-dir` — 覆盖 YAML 中的 raw JSON 目录
- `--output-dir` — 覆盖 checkpoint 输出目录
- `--epochs` — 覆盖训练轮数
- `--batch-size` — 覆盖 batch size
- `--lr` — 覆盖学习率
- `--max-files` — 按副本目录比例选择最多 N 个有效 raw JSON 文件；转换失败时从同副本后备文件补位 (smoke/test 用)
- `--device` — `cuda` 或 `cpu`
- `--resume` — 从指定 checkpoint 继续训练；恢复模型、optimizer、scheduler 和 best 指标，训练从 checkpoint 的下一轮开始

训练启动时会先调用 `scripts.convert_fflogs` 把选中的 raw JSON 编译到 `data/human/job/<job>/.cache/`，然后训练数据集直接读取 compiled cache；不会向 `raw/` 写入转换结果，也不会生成中间训练 PT。compiled cache 保存完整历史，模型配置中的 `model.history_capacity` 仅在读取样本时裁剪模型窗口，调整它不会触发 cache 重建。
checkpoint 会在每轮保存续训所需的 optimizer、scheduler、best 指标和随机状态；旧 checkpoint 缺少 scheduler 或随机状态时仍可按已完成轮次回退恢复，但要保证模型配置、数据 schema 和归一化契约与当前训练配置一致。
安全提示：续训需要使用 `torch.load(weights_only=False)` 恢复完整 checkpoint 元数据，因此只应加载可信来源的 checkpoint；不要对不明来源的文件执行 `--resume`。

### scripts/onnx_export — 独立 ONNX 部署包导出

ONNX 导出是 checkpoint 的独立后处理，不进入训练循环。正式部署精度与当前训练主线一致，固定为 BF16，并要求支持原生 BF16 的 NVIDIA GPU、PyTorch CUDA 和 ORT CUDA EP。CPU extras 只用于 FP32 小模型开发测试，不用于正式模型发布：

根目录提供 [onnx_pipeline.ps1](./onnx_pipeline.ps1) 作为 PT → ONNX → parity → 发布校验的一键入口。脚本本身不保存模型路径、opset、精度、Provider、容量或门禁步数，只负责调用 Python；所有协作环境差异统一写在根目录 `.env`：

```powershell
# 交互菜单：输入 1 执行完整流程，或输入 2～7 单独执行某一步
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\onnx_pipeline.ps1

# 无交互执行完整流程
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\onnx_pipeline.ps1 -Action all

# 也可以只执行某一步
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\onnx_pipeline.ps1 -Action env
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\onnx_pipeline.ps1 -Action export
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\onnx_pipeline.ps1 -Action empty-parity
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\onnx_pipeline.ps1 -Action scene-parity
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\onnx_pipeline.ps1 -Action verify
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\onnx_pipeline.ps1 -Action run
```

`all` 会依次检查 CUDA/BF16/ORT 环境、导出完整 BF16 ONNX、执行空 scene 与真实 scene parity，最后核验模型哈希和发布状态。两项 parity 即使第一项失败也会继续跑完并保留审计报告；失败时部署包保持 `parity_failed`，Python 返回专用退出码 `2`，PowerShell 输出简洁中文结论而不打印预期门禁失败的 traceback。是否替换已有包只由 `.env` 的 `ONNX_EXPORT_OVERWRITE` 控制。

```bash
# 先安装主依赖；这里固定 torch 2.12.0+cu132，ONNX extras 不重复声明 torch
python -m pip install -r requirements.txt

# 仅 FP32 小模型开发测试
python -m pip install -r requirements-onnx.txt
```

```bash
# 本地 NVIDIA；正式矩阵使用 CUDA 13.x / cuDNN 9 / ORT 1.27
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install -r requirements-onnx-gpu.txt
```

PyTorch `cu132` 和 ORT 1.27 均可运行在 CUDA 13.x 驱动上，不要求存在名称完全对应 CUDA 13.3 的 Python wheel。Windows 的纯 Python 回放可由 PyTorch 预加载匹配的 CUDA/cuDNN DLL；不加载 PyTorch 的 .NET 插件宿主仍需在系统 PATH 中提供 CUDA 13.x 与 cuDNN 9 DLL。

安装后可确认实际版本和 provider；GPU 环境应同时看到 `CUDAExecutionProvider` 和 `CPUExecutionProvider`：

```bash
python -c "import torch; import onnxruntime as ort; print(torch.__version__, torch.version.cuda); print(ort.__version__, ort.get_available_providers())"
```

配置 `.env` 后，无参数即可导出正式 checkpoint：

```powershell
python -m scripts.onnx_export
```

`AUTOREGRESSIVE_REPLAY_CHECKPOINT=artifacts/checkpoints/black_mage/artzip_bc/best.pt` 且 `AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE` 留空时，导出目录自动映射为 `artifacts/exports/black_mage/artzip_bc/`；回放插件也使用同一部署包路径。非标准 checkpoint 目录可显式设置共用的 `AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE`。

ONNX 导出与发布门禁 `.env` 参数：

| `.env` 参数 | 默认值 | 用法与约束 |
|---|---|---|
| `AUTOREGRESSIVE_REPLAY_CHECKPOINT` | 训练配置选择的 checkpoint | 导出与 PyTorch parity 共用的完整 checkpoint。模型结构、权重、输入契约和 `model_config.history_capacity` 全部从这里读取。 |
| `AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE` | 按 checkpoint 路径自动推导 | 导出目标与 ORT 回放共用的部署包目录；留空时自动从 `artifacts/checkpoints/...` 映射到 `artifacts/exports/...`。 |
| `AUTOREGRESSIVE_REPLAY_ORT_PROVIDER` | `CUDAExecutionProvider` | 导出验收、parity 和普通 ORT 回放共用的 Execution Provider。 |
| `AUTOREGRESSIVE_REPLAY_SCENE_JSON` | 当前职业首个 raw JSON | 普通回放和实战 parity 共用的 scene。 |
| `ONNX_EXPORT_DEPLOYMENT_PROFILE` | 按 checkpoint 职业选择内置 profile | 可选职业部署画像；留空时自动选择，不扫描 cache。 |
| `ONNX_EXPORT_OPSET` | `18` | 显式 ONNX opset；升级后必须重新执行 checker、ORT 和 parity。 |
| `ONNX_EXPORT_PRECISION` | `bf16` | 图中权重、浮点输入与输出精度；正式发布使用 BF16。 |
| `ONNX_EXPORT_VALIDATION_DEVICES` | `cuda` | 逗号分隔的 PyTorch padding 验收设备；正式 BF16 只能为 `cuda`。 |
| `ONNX_EXPORT_OVERWRITE` | `false` | 为 `true` 时，新包全部验证通过后原子替换已有包；失败时旧包不变。 |
| `ONNX_PARITY_EMPTY_MAX_STEPS` | `544` | 空 scene 门禁的动作安全上限。正式门禁要求至少 `4 × max_gcds + 32`，为强制首步、每个 GCD 的 weave 与额外调度余量留出空间；预算不足会在启动昂贵回放前直接拒绝。 |
| `ONNX_PARITY_EMPTY_MAX_GCDS` | `128` | 空 scene 正式门禁目标 GCD 数。 |
| `ONNX_PARITY_EMPTY_REPORT` | `artifacts/onnx_parity_bf16_empty_128gcd.json` | 空 scene parity 审计报告。 |
| `ONNX_PARITY_SCENE_MAX_STEPS` | `100` | 实战 scene 正式门禁的执行动作数，每个 `ReplayRow` 计一次；事件驱动后不等同于 GCD 数，旧报告与新报告应按 `output_gcds` 分开解读。 |
| `ONNX_PARITY_SCENE_REPORT` | `artifacts/onnx_parity_bf16_scene_100.json` | 实战 scene parity 审计报告。 |
| `ONNX_PARITY_TOLERANCE` | 按 manifest 精度选择 | 正式门禁只接受精度固定阈值（BF16 为 `0.25`）；建议留空。自定义宽松阈值只能通过普通 replay CLI 做调试，不能晋升发布状态。 |

命令行参数仍可临时覆盖导出 `.env`，但不再是必填项。scene 容量只来自 checkpoint 的 `model_config.scene_capacity`，由 deployment profile 的 `evidence.scene_length_max` 统计佐证下限；history 容量只来自 PT checkpoint 的 `model_config.history_capacity`，两者都写入 manifest。两种容量都不在 `.env` 或 PowerShell 中重复定义。

ORT provider 默认严格使用 `CUDAExecutionProvider`；CUDA EP 或其 CUDA/cuDNN 运行库不可用时直接失败，不会把模型放到 CPU。`--ort-provider CPUExecutionProvider` 只保留给显式的本地测试，`--ort-provider auto` 只保留给明确需要 fallback 的兼容场景。

pytest 中的“CPU 小模型门禁”仅指测试代码临时生成的夹具：`d_model=16`、2 层、2 头、`ff_dim=32`、3 个候选和 `scene/history=3/4`。它只在安装 ONNX extras 后快速检查 exporter、ONNX checker、ORT session 和 PT/ORT parity；依赖缺失时测试会明确 skip。仓库目前没有 `.gitlab-ci.yml`，因此这是本地 pytest 门禁，不宣称已接入 CI。它不会生成发布产物，也不允许代替真实 checkpoint 验收。正式导出始终读取 checkpoint 内的完整 `DataSpec`、`ModelInputContract`、模型配置和权重；当前黑魔模型是 `12×768×12`、`ff_dim=3072`、25 个候选、`history_capacity=384`，并由真实导出后的 padding/parity 验收拒绝缺权重或缩模产物。

BF16 ORT 使用 DLPack 与 I/O Binding 直接接收 CUDA `torch.bfloat16` Tensor，并在 CUDA 上返回 `tensor(bfloat16)`，不会经过不支持 BF16 的 NumPy，也不会混入 FP32/FP16/TF32。确定性 golden NPZ 对 BF16 使用显式 `uint16-little-endian-bfloat16-bits` 位模式，逻辑 dtype 和 shape 仍以 manifest Tensor 契约为权威。真实长上下文与自回归回放采用 BF16 专属 `max_abs_diff <= 0.25` 门槛，并继续强制有限值、padding 值不变、Top-1、Top-3 和宿主最终动作一致。

当前正式 BF16 导出矩阵固定为 `torch==2.12.0+cu132`、`onnx==1.22.0`、`onnxscript==0.7.1`、Python `onnxruntime-gpu==1.27.0` 与 .NET `Microsoft.ML.OnnxRuntime.Gpu==1.27.0`，并严格使用 `CUDAExecutionProvider`；workflow 会在导出前逐项校验版本。ORT 1.27 的官方 GPU 包使用 CUDA 13.0 / cuDNN 9，并通过 CUDA 次版本兼容运行在 CUDA 13.x 驱动上。升级矩阵后，旧 ORT 1.26/cu126 部署包及其 parity/release 证明不再代表当前正式环境，必须重新导出并重新执行空 scene 128 GCD 与真实 scene 100 执行动作门禁后，才能声明新的发布状态。

Parity 报告会记录 checkpoint、`model.onnx`、manifest 和部署 contract 的 SHA-256。只有 `onnx_pipeline.ps1` / `scripts.onnx_export.workflow` 的显式正式门禁入口能把结果登记到部署包；普通 `--parity-onnx-package` 只生成外部调试报告，即使步数与正式门禁相同也不会改变发布状态。正式证据还绑定门禁版本、manifest 精度和该精度固定容差。每次模型调用同时保存双方 raw logits 差异、按分数排序的 Top-1/Top-3、Top-3 集合是否一致及宿主最终动作。空 scene 128 GCD 与真实 scene 100 执行动作两项都通过后，`release_report.json` 晋升为 `release_validated`；它是当前发布状态的唯一来源和原子提交点。`export_report.json` 只作为导出验证快照，v3 门禁登记不再改写它，也不能用它判断当前发布状态。Parity 证据按内容哈希使用不可变文件名，重跑或 v1 迁移替换下来的证据会登记到 `superseded_parity_reports`，不会覆盖已有通过证据；可捕获的提交失败会删除本次未提交证据。普通 ONNX 回放只校验并返回当前生效证据，不暴露未经复核的 superseded 条目；正式登记与 workflow `verify` 才额外流式校验并返回全部历史，避免长期重跑后增加每次 ORT session 的启动成本。普通 ONNX 回放或 .NET 加载不需要额外传递发布参数；模型或 manifest 改变后，旧发布证明会因 SHA 不匹配而自动失效。

首版黑魔通过一次当前 1,912 场 raw 语料扫描得到 scene 实测最大值 160（P99=116），记录为 deployment profile 证据；模型配置声明 `scene_capacity: 200`（不小于实测最大值），导出时校验。日常导出不扫描 raw 或 compiled cache。history 容量直接读取 checkpoint 的 `model_config.history_capacity`，当前正式配置为 `384`，具体值仍以 checkpoint 的 `model_config` 为准。契约固定 `batch=1`、候选数和全部字段维度由 checkpoint 决定，scene/history 都在右侧补位；物理 token 总长自动计算为 `scene_capacity + history_capacity + candidate_count + CLS`，不再配置 `max_sequence_length`；补位位置的 mask 为 `False`，各语义段的 position id 分别从 0 起算，超出容量直接报错，不会静默截断。CLI 使用 `torch.onnx.export(..., dynamo=True)` 并显式设置 opset，生成 `model.onnx`、完整签名 manifest、容量证据、确定性 golden 输入/输出和两类导出报告。发布前会执行 ONNX checker、shape inference、ORT session、PT/ONNX 参数体量审计和空/满/临界 padding/parity 矩阵；参数体量比例保留在报告中用于审计，不把易受导出器常量折叠影响的固定比例作为单独失败条件。全部验证通过后才原子替换目标目录。

真实正式 checkpoint 的完整导出是显式慢速门禁，不放进普通 pytest：

```bash
$env:RUN_REAL_ONNX_EXPORT="1"
python -m pytest tests/scripts/onnx_export/test_real_checkpoint_padding.py -q
```

该命令会真实读取职业 profile 与 checkpoint 的 `model_config` 容量，验证 vocab 行数与 `total_token_count == scene_capacity + history_capacity + candidate_count + 1`，并以 BF16/CUDA 执行完整 export/checker/ORT/padding 流程。manifest v6 统一容量契约（物理 token 总长按 `scene_capacity + history_capacity + candidate_count + 1` 换算，移除 `max_sequence_length`），与旧 v1-v5 部署包不兼容；升级后必须重新导出，loader 会给出明确的 `re-export` 错误。

模型小于 2 GiB 时强制使用单个 `model.onnx`；如果后续大模型需要 external data，manifest 会记录配套文件。部署包只包含推理权重和输入契约，不写入 checkpoint 中的 optimizer、scheduler 或随机状态。

### scripts/model_analysis — 模型分析 PNG 生成

```bash
python -m scripts.model_analysis --loss-landscape
```

参数:

- `--checkpoint` — 模型 checkpoint 路径
- `--model-config` — 模型 YAML 配置
- `--raw-json` — 分析用 raw JSON；对应 compiled cache 从职业 `.cache` 读取，缺失或过期时自动调用 `scripts.convert_fflogs.cli` 编译
- `--output` — 输出目录 (默认 `artifacts/model_analysis`)
- `--max-samples` — 分析样本数 (默认 256)
- `--max-tokens` — 每层保留最大 token 数 (默认 20000)
- `--loss-landscape` — 在同一轮分析中追加导出逐 Transformer 层损失地图
- `--loss-landscape-resolution` — 每层损失地图单轴采样点数，必须是奇数 (默认 21)
- `--loss-landscape-radius` — filter-normalized 参数扰动半径 (默认 0.5)
- `--loss-landscape-max-samples` — 损失地图固定样本数；默认复用 `--max-samples`
- `--loss-landscape-seed` — 逐层正交参数方向的可复现随机种子 (默认 3407)
- `--device` — `auto` / `cpu` / `cuda`

基础生成结果包括 hidden 分布、每层 hidden 2D/3D PCA、skill/pair embedding 和 attention。传入 `--loss-landscape` 后，loss 会和这些图由同一条命令导出，并保存逐 Transformer 层的 300 DPI 3D 曲面、2D 等高线、float64 `.npz` 原始网格和方向元数据。所有模型前向统一使用模型配置 `model.yaml` 的 `precision`（当前主配置为 BF16），不提供按输出拆分的精度开关；loss 汇总值单独以 float64 累加。为控制峰值显存，程序会在基础图完成后释放 hidden/attention/PCA 上下文，再以相同 checkpoint、数据和精度执行 loss 阶段。该步骤会在固定样本集上反复执行前向计算，计算量约为 `层数 × 分辨率² × ceil(样本数 / batch-size)`，分辨率每增加一倍，计算量约增加到四倍。

## 环境配置

复制 `.env.example` 为 `.env` 并按需修改:

- `FFXIV_JOB_TAG` — 项目统一职业标签；转换、训练、模型分析、自回归回放和状态机路由共用
- `config/models/<job_tag>/*/config.yaml` — 训练、模型分析和回放共用的配置清单；由 `.env` 的 `FFXIV_JOB_TAG` 自动选择，清单分别引用 `model.yaml`、`training.yaml` 和 `grpo.yaml`
- `TRAINING_CACHE_ROOT` — compiled cache 根目录；默认 `data/human/job`
- `TRAINING_MODEL_CHECKPOINT` — checkpoint 文件名
- `TRAINING_DEVICE` — 训练设备 (cuda/cpu)
- `AUTOREGRESSIVE_REPLAY_SCENE_JSON` — 自回归回放使用的 raw JSON；为空时自动选择当前职业 raw 目录中的第一个文件
- `AUTOREGRESSIVE_REPLAY_SCENE_MODE` — `cache` 使用对应 compiled cache，`empty` 使用空 scene；默认 `cache`
- `FFLOGS_V2_CLIENT_ID` / `FFLOGS_V2_CLIENT_SECRET` — FFLogs API 凭证

## 测试

```bash
pytest
```

## scripts/autoregressive_replay：模型自回归回放

使用真实 `CombatStateMachine` 推进状态，可选择 PyTorch checkpoint 或已验证 ONNX 部署包执行每一步候选打分。两个 backend 共用同一 `LiveBatchBuilder`、完整绝对时间 scene、重复惩罚、合法 mask、temperature/top-p 解码和状态推进；ORT 根据 manifest 自动路由职业并恢复正式容量，不复制状态机逻辑。

### 推荐命令

```bash
# PyTorch：使用根目录 .env 的默认 checkpoint、CUDA 和回放配置
python -m scripts.autoregressive_replay

# ONNX Runtime：严格使用 CUDA EP，不允许模型回落 CPU
python -m scripts.autoregressive_replay --backend onnxruntime --onnx-package artifacts/exports/black_mage/artzip_bc --ort-provider CUDAExecutionProvider --max-steps 100

# 正式门禁一：从 .env 读取固定策略，并把空 scene 128 GCD 证据登记到部署包
python -m scripts.onnx_export.workflow empty-parity

# 正式门禁二：登记真实 scene 100 执行动作证据；两项都通过后自动标记 release_validated
python -m scripts.onnx_export.workflow scene-parity

# 调试 parity：可自定义容差，只写指定 JSON，不修改部署包发布状态
python -m scripts.autoregressive_replay --backend pytorch --scene-mode empty --max-steps 100 --parity-onnx-package artifacts/exports/black_mage/artzip_bc --parity-tolerance 1.0 --parity-output artifacts/onnx_parity_debug.json
```

配置优先级统一为“命令行参数 > 根目录 `.env` > 程序默认值”。相对路径均相对项目根目录解析。普通模型回放推荐保持 `CUDAExecutionProvider`，不要使用允许 fallback 的 `auto`。

### 命令行参数

| 参数 | 对应 `.env` | 默认值 | 用法与约束 |
|---|---|---|---|
| `-h` / `--help` | 无 | 无 | 打印当前 CLI 参数列表并退出，不启动模型。 |
| `--backend {pytorch,onnxruntime}` | `AUTOREGRESSIVE_REPLAY_BACKEND` | `pytorch` | 选择推理后端。`pytorch` 加载 checkpoint；`onnxruntime` 加载部署包并校验 manifest。别名 `torch` / `ort` 仅配置解析层支持，CLI 请使用表中的完整值。 |
| `--checkpoint PATH` | `AUTOREGRESSIVE_REPLAY_CHECKPOINT` | 训练配置解析出的 checkpoint | PyTorch backend 的模型文件。仅影响 PyTorch 普通回放和 parity 的参考模型；ORT 普通回放不读取 checkpoint。 |
| `--onnx-package PATH` | `AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE` | 按 checkpoint 路径自动推导 | ORT 部署包。可以传部署包目录或包内 `model.onnx`；都未设置时从共用 checkpoint 路径推导。 |
| `--ort-provider NAME` | `AUTOREGRESSIVE_REPLAY_ORT_PROVIDER` | `CUDAExecutionProvider` | ORT Execution Provider。推荐并默认严格使用 CUDA；CUDA EP、CUDA/cuDNN DLL 或图分配失败时直接报错。`auto` 会允许 CUDA→CPU fallback，只用于明确需要兼容回落的测试；`CPUExecutionProvider` 只用于显式 CPU 测试。 |
| `--parity-onnx-package PATH` | 无 | 无 | 开启 PT/ORT 双后端验收，而不是普通回放。当前 `--backend` 应为 `pytorch`，并同时提供可用 checkpoint；参数值是待验收 ONNX 部署包。两个 backend 在同一状态机决策点比较 raw logits、Top-1、Top-3 和宿主最终动作。 |
| `--parity-output PATH` | 无 | `--output` 改为 `.parity.json` 后缀 | parity JSON 审计报告路径。成功和失败都会写完整固定参考轨迹报告，但 replay CLI 永远不会把它登记为正式发布证据；正式门禁必须使用 `scripts.onnx_export.workflow`。仅与 `--parity-onnx-package` 一起使用。 |
| `--parity-tolerance X` | 无 | 按 manifest 精度选择；BF16 为 `0.25` | 普通调试 parity 的 raw logits 最大绝对误差上限。可以覆盖，但报告不会晋升发布状态；正式 workflow 固定使用 manifest 精度阈值。Top-1、Top-3 和宿主最终动作仍是独立硬门禁。 |
| `--output PATH` | `AUTOREGRESSIVE_REPLAY_OUTPUT` | `artifacts/autoregressive_rollout.md` | 普通回放 Markdown 输出路径；history ablation 也以它作为基准文件名。parity 未显式传 `--parity-output` 时，用它推导 JSON 路径。 |
| `--scene-json PATH` | `AUTOREGRESSIVE_REPLAY_SCENE_JSON` | 当前职业 raw 目录排序后的第一个 JSON | 指定提供完整绝对时间 scene 的 raw FFLogs JSON。其职业必须与 checkpoint/manifest、`FFXIV_JOB_TAG` 和训练模型配置一致。 |
| `--scene-mode {cache,empty}` | `AUTOREGRESSIVE_REPLAY_SCENE_MODE` | `cache` | `cache` 使用 raw JSON 对应的 compiled cache scene；cache 缺失或过期时按正式契约重建。`empty` 使用空 scene，适合木桩和空场景验收。 |
| `--scene-sample N` | `AUTOREGRESSIVE_REPLAY_SCENE_SAMPLE` | `0` | 从 scene 时间线第 N 个样本开始，必须 `N >= 0`。它改变 scene 起点，不改变 scene 容量。 |
| `--max-steps N` | `AUTOREGRESSIVE_REPLAY_MAX_STEPS` | `100` | 最多执行 N 次动作决策，必须 `N >= 1`。这是动作安全上限，不等于 GCD 数量，因为 oGCD 也会占用一步。 |
| `--max-gcds N` | `AUTOREGRESSIVE_REPLAY_MAX_GCDS` | 不限制 | 达到 N 个输出 GCD 后提前停止，必须 `N >= 1`。如果先达到 `--max-steps` 仍不足 N 个 GCD，命令会失败，不能把短轨迹误当成验收通过。 |
| `--top-k N` | `AUTOREGRESSIVE_REPLAY_TOP_K` | `8` | 每个决策点写入 Markdown 的合法候选数量，必须 `N >= 1`。只影响报告展示，不改变动作选择。 |
| `--temperature X` | `AUTOREGRESSIVE_REPLAY_TEMPERATURE` | `0.0` | `0` 使用确定性的合法 Top-1；`X > 0` 时按 temperature softmax 采样。parity/确定性验收应保持 `0`。 |
| `--top-p X` | `AUTOREGRESSIVE_REPLAY_TOP_P` | `1.0` | 核采样阈值，范围 `(0,1]`。仅在 `--temperature > 0` 时影响选择；`1.0` 表示不截断候选概率质量。 |
| `--max-history N` | 无 | checkpoint/manifest 的正式容量 | 仅用于显式历史截断实验。正常 PyTorch 回放读取 checkpoint，ORT 回放读取 manifest，不允许 `.env` 再定义一份正式容量。 |
| `--history-ablation N [N ...]` | 无 | 无 | 先生成一条完整自回归轨迹，再在相同状态快照上分别只给模型最近 N 条历史；截断预测不会反过来污染后续状态。会额外生成 `_history_N.md` 文件。每个 N 必须 `>= 0` 且不能重复；N 大于当前可用历史时等价于保留全部可用历史，通常应选择不超过基准 `--max-history` 的值。 |
| `--device {cuda,cpu}` | `AUTOREGRESSIVE_REPLAY_DEVICE`，再回退 `TRAINING_DEVICE` | `cuda` | 只控制 PyTorch backend 的模型和 tensor 设备。BF16 checkpoint/parity 必须使用 CUDA，程序不再静默改成 FP32。ORT backend 的执行位置由 `--ort-provider` 决定；live batch 可在 CPU 装配，但正式 BF16 输入通过 DLPack/I/O Binding 送入 CUDA。 |
| `--use-kv-cache` | `AUTOREGRESSIVE_REPLAY_USE_KV_CACHE` | PyTorch 开启，ORT 关闭 | 显式开启 PyTorch 模型内部 scene/history 前缀 KV cache。新轨迹或上下文截断时自动 reset。首版 ONNX backend 不支持 KV cache，开启会报错。 |
| `--no-use-kv-cache` | `AUTOREGRESSIVE_REPLAY_USE_KV_CACHE` | 同上 | 显式关闭 PyTorch KV cache。PT/ORT parity 必须关闭，确保参考路径与无 cache 的 ONNX 图一致。 |

### 常用组合

指定 scene、输出和步数：

```bash
python -m scripts.autoregressive_replay --scene-json data/human/job/black_mage/raw/FRU/fight.json --scene-mode cache --output artifacts/autoregressive_rollout.md --max-steps 100 --top-k 8
```

固定完整轨迹并比较 10/20/30/50 条历史：

```bash
python -m scripts.autoregressive_replay --max-history 240 --history-ablation 10 20 30 50 --output artifacts/autoregressive_rollout_history_ablation.md
```

随机采样仅在明确需要探索时开启：

```bash
python -m scripts.autoregressive_replay --temperature 0.8 --top-p 0.9 --max-steps 100
```

### ORT CUDA 的 ScatterND 警告

当前正式 ONNX 图在 CUDA EP 初始化时可能输出：

```text
ScatterND with reduction=='none' only guarantees to be correct if indices are not duplicated.
```

这是 ORT CUDA 对 `ScatterND(reduction="none")` 的通用提示，不表示当前运行已经发现重复索引，也不表示模型回落到了 CPU。正式黑魔图中的 9 个 `ScatterND` 都来自输入编码器构造 `valid` / attention mask 时的固定切片赋值；索引是导出期常量，并已逐组验证无重复：scene 为 `0..199`，history 为 `200..583`，candidate 为 `584..608`，完整 attention 行为 `0..609`。因此当前图不满足警告所描述的危险条件，保留该警告即可。实际模型执行位置以输出 Markdown 的 `execution provider` 为准；严格 CUDA 应显示 `CUDAExecutionProvider`。

普通 ORT 回放默认显式请求 `CUDAExecutionProvider`，并通过 ORT session 配置禁止节点静默回落 CPU；初始化或图分配失败会直接报错。`export_report.json` 分别记录请求链、ORT 注册链和 `ort_cpu_fallback_disabled`，避免把 ORT 自动显示的 CPU provider 误读为实际允许回落。

当前 `scripts.autoregressive_replay` 是仓库内的状态机回放/验收入口，不是只携带 ONNX 部署包即可运行的独立宿主。即使选择 ORT backend，它仍会读取由 `FFXIV_JOB_TAG` 自动选择的配置清单及其 `model.yaml`、`training.yaml` 中的 raw 数据目录和 compiled cache 分片参数，并要求 `model.history_capacity` 与 manifest 一致。未来若向仓库外分发模型，需要另建只依赖 manifest 和宿主输入契约的部署运行时，不能直接复制本 CLI。

### 仅通过 `.env` 配置的参数

以下配置当前没有对应 CLI 参数：

| `.env` 参数 | 默认值 | 用法与约束 |
|---|---|---|
| `FFXIV_JOB_TAG` | 项目职业配置 | 状态机职业，并自动选择 `config/skills/<job_tag>.yaml`、`config/models/<job_tag>/*/config.yaml`；checkpoint、manifest 和 scene/cache 的职业必须与它一致。 |
| `config/models/<job_tag>/*/config.yaml` | 按 `FFXIV_JOB_TAG` 自动选择 | 配置清单；`model.yaml` 提供 raw/output、模型架构和统一 precision，`training.yaml` 提供训练/DataLoader/增强/序列过采样，`grpo.yaml` 提供 GRPO 采样与更新参数。PT/ORT 回放共用合并后的视图，避免生成两套 cache 签名。 |
| `TRAINING_MODEL_CHECKPOINT` | `.env.example` 为 `best.pt` | `AUTOREGRESSIVE_REPLAY_CHECKPOINT` 和 `--checkpoint` 都未设置时，用于从训练配置的 `output_dir` 选择 checkpoint；相对路径相对该输出目录解析。 |
| `TRAINING_DEVICE` | `cuda` | `AUTOREGRESSIVE_REPLAY_DEVICE` 和 `--device` 都未设置时，作为 PyTorch 回放设备。不会控制 ORT EP。 |
| `AUTOREGRESSIVE_REPLAY_INITIAL_ACTION` | `fire_iii` | 模型开始预测前强制执行的首步技能；设为空字符串则不强制首步。输出 Markdown 会把它标记为“强制”。 |
| `AUTOREGRESSIVE_REPLAY_INITIAL_TIME` | 自动推导 | 首步开始时间，必须 `<= 0`。留空时按首步实际读条/占用时间自动计算负预读时间。 |
| `AUTOREGRESSIVE_REPLAY_BASE_GCD` | `config/system.yaml` | 回放使用的基础 GCD 秒数，例如 `2.17`，必须 `> 0`。 |

Markdown 输出会记录 backend、实际 Execution Provider、模型来源、scene raw JSON、输入 tensor 设备、history 上限、PPG、延迟/内存指标，以及每一步选择的技能、Top-1 概率和合法候选 Top-K；ORT 模式不会伪造 `checkpoint: None`。parity JSON 额外记录每个决策点、raw logits 最大误差、Top-1/Top-3/最终动作一致率、首个分叉、四类产物 SHA 和目标运行库；即使中途发现差异也会继续走完同一条 PyTorch 参考轨迹，写出完整统计和发布状态后再让 CLI 非零退出。强制首步会标记为“强制”。回放时间线只采用事件驱动：GCD 读条结束、oGCD 动画锁结束、主动结束 weave 后的 GCD 窗口结束，以及会改变状态向量或候选合法性的 scene 边界都会重新调用模型；普通回放无合法动作时推进到最近的锁、GCD、冷却、状态、DoT、回蓝或 scene 事件，不会直接终止轨迹。同一份完整绝对时间 scene tensor 会同时提供给模型，并按训练口径把 Boss 可选中、强制移动、团辅窗口和目标数同步到状态机。验证 PPG 额外采用失败闭合语义：候选全非法时先尝试由 `DecisionScheduler` 推进；若已无法推进，则该副本返回全 0 的 PPG 结果并仍参与副本平均，已执行的伤害不保留到该失败结果中。“跳过本副本”不表示从平均值排除。

# Changelog

本文件记录项目主要变更。

## [Unreleased]

### Added

- 训练新增 `training.max_files`：在模型 YAML 中限制参与训练与验证的 raw JSON 文件数，`null` 表示使用 raw 目录下全部有效文件；选择时按副本子目录比例分配，再按 `train_split` 划分训练集与验证集。一键脚本无需额外参数即可控制数据规模，命令行 `--max-files` 仍然优先，启动日志会打印实际取值与来源；当前黑魔 `artzip` 配置固定为 `1280`。
- 训练新增 `training.activation_checkpoint_attention_block`（默认 `false`）：置 `true` 时整块 attention（norm、Q/K/V 投影、三段 SDPA、merge、out_proj）作为一次 checkpoint 重算，反向多算一遍投影换取显存。RTX 3050 4GB batch 20 实测：`false` 1218 ms / 2930 MiB，`true` 1288 ms / 2006 MiB（省约 900 MiB）；开启确定性算法后两种粒度的前向输出与全部 253 个参数梯度逐位一致。
- 新增 `common/policy/model/activation.py`：把 `model.transformer_activation` 的取值表、门控判定、隐层宽度折算与门控合成收敛为唯一实现，该配置现在同时驱动主干 FFN、候选打分头与 pair 融合；GELU/ReLU 走单条隐藏层，SwiGLU 走 `down(SiLU(gate(x)) * up(x))` 三投影并按矩阵参数量近似相等的口径折算隐层宽度，切换取值后需要重新训练。

### Changed

- 根目录 `onnx_pipeline.ps1` 改造为通用工具入口 `ffxiv_ccg.ps1`：中文编号菜单只保留训练（BC 预训练）、GRPO 后训练、ONNX 导出、模型分析图生成、模型自回归回放和 FFLogs 数据下载六项，并支持 `-Action` 无交互调用；ONNX 导出沿用 `scripts.onnx_export.workflow all` 的完整 parity 门禁与发布校验，模型分析不再生成逐层损失地形图，FFLogs 下载在菜单内交互输入报告 URL 或报告码，其余业务参数统一读取 `config/` 与根目录 `.env`；`README.md` 与 `docs/项目各文件说明.md` 同步更新入口说明。
- split attention 的可见性 mask 改为每次前向只构造一次并跨层复用：prefix / candidate / CLS 三段的允许矩阵、全屏蔽行安全列与 `is_causal` 判定在进入层循环前算好，KV-cache 解码路径同样一次算好候选与 CLS 两段；`force_explicit_mask` 仍保持完全不读 device 取值的静态控制流，ONNX 导出路径不受影响。实测3050下训练 step 1246→1218 ms、峰值显存 2976→2930 MiB、单步 device 到 host 同步 38→2，KV 解码步 43.98→40.50 ms、同步 29→5，`logits` / `hidden` / 12 层 attention 与重构前逐位一致。
- 候选打分头由固定 ReLU 的 `scorer.network` 两层 MLP 改为按配置解析的 `gate_proj`/`up_proj`/`down_proj` 布局；加载旧打分头 checkpoint 时直接提示按当前激活配置重新训练，ONNX 真实 checkpoint 回归测试同步按该条件跳过。
- 按功能组整理 `scripts/convert_fflogs` 目录：缓存编译、配置常量、日志提取与战斗载荷装帧、scene window、raw source、训练样本分别归入 `cache/`、`config/`、`extraction/`、`scene/`、`source/`、`training/` 子包，包根只保留 `__init__.py`、`cli.py`、`pipeline.py` 与共享 helper `utils.py`；`scripts.convert_fflogs.cache`、`scripts.convert_fflogs.config` 继续作为子包门面导出原有入口，全部内部导入、测试引用与 `docs/项目各文件说明.md` 目录树同步更新。
- 重构 `scripts/convert_fflogs` 后与重构前 `main` 对照重跑 M5s 全量转换：100 个 raw JSON 成功 97 个、失败 3 个（`not_enough_mp` 与两个 `requires_polyglot`），失败文件、失败原因与 step/request_time 逐项一致；97 份 compiled shard 字节级相同，manifest 除路径派生的 `history_bank_id` 外逐字段一致。
- 按功能组整理 `scripts/onnx_export` 目录：将导出执行模块平铺到 `export/`，并将配置、部署契约、运行时、发布、policy 与产物 I/O 分别归入独立子目录；删除旧的巨型 `exporter.py` 入口，迁移内部导入和测试，CLI 使用方式保持不变。
- 拆分 ONNX 导出器的环境门禁、checkpoint/部署契约加载、ONNX 盖章、运行时验证、产物文档和 manifest 装配，保持原有导出行为与验收顺序不变，并整理 Ruff 检查。
- 训练程序化入口在模型初始化前统一拒绝缺失或空白的 `model_variant`，避免日志输出 `None` 与保存端校验时机不一致，并补充入口负例测试。
- 补充 ONNX manifest 的 `contract`、`model`、`capacity`、`history_capacity` 缺失字段负例测试；训练模型初始化与完成日志输出 `model_variant`，便于区分多变体训练。
- 新增 `FFXIV_MODEL_VARIANT` 配置：`FFXIV_JOB_TAG` 选择职业，`FFXIV_MODEL_VARIANT` 选择职业目录下的模型变体；训练、转换、回放、模型分析、GRPO 和 ONNX 流程统一解析 `config/models/<job_tag>/<variant>/config.yaml`，不再依赖职业目录下唯一变体自动扫描。
- 统一显式职业与模型变体参数的空值语义；checkpoint、GRPO checkpoint 和 ONNX manifest 记录并校验 `model_variant`，禁止同职业不同变体的模型产物混用，并将 ONNX manifest 升级到 v7。

- 保存 checkpoint 前强制要求 `model_variant` 非空；补充 ONNX manifest 变体不匹配、resume 变体不匹配和直接保存契约测试，旧 manifest 缺少变体时提示重新导出。
- 修复 ONNX manifest 缺少 `contract`、`job_tag` 或其他路由字段时被错误归因成缺少 `model_variant` 的诊断问题，并补充实际缺失字段回归测试。

## [0.1.0] - 2026-09-21

### Added

- 新增 GRPO 逐轨迹磁盘 rollout 存储：沿用根目录 `.env` 的 `TRAINING_CACHE_ROOT` 与职业路由，将采样决策保存到 `data/human/job/{job_name}/grpo/`，并在策略更新阶段按 minibatch 流式读取，避免整轮自回归记录持续占用内存。
- `scripts/model_analysis` 新增 `--loss-landscape` 逐层损失地图：在同一次模型分析中，对每个 Transformer layer 使用 filter-normalized 正交参数方向计算固定样本集交叉熵，统一网格导出 300 DPI 三维曲面、二维等高线、float64 原始数值网格和方向元数据；每张图复用共享近正方形布局并包含全部层，不拆成逐层文件。
- 损失地图分析支持 `--loss-landscape-only` 独立执行，默认采样分辨率提升为 31；普通 Split Encoder 复用前层编码结果并按层保存 partial 网格，元数据补充耗时、checkpoint、source 与缓存路径，Full AttnRes 继续保持完整前向语义。
- 新增调用方统一绝对时间调度：GCD（含瞬发）在读条结束后等待 0.05 秒再决策；oGCD 按调用方配置的间隔循环插入，窗口不足时提前 0.05 秒提交下一 GCD，普通回放、PPG 与 GRPO 共用同一套候选阶段规则。
- 新增 `scripts/common/scene_state.py`：把绝对时间 scene 窗口转换为外部事实流，并按场景上下文在每个变化时刻提交 Boss 可选中、目标数与团辅窗口事实；训练样本输出仍由输出层合成移动字段并应用滑步豁免。
- Python 调用方迁移到绝对时间协议：`scripts/common/cs_backend.py` 整体重写为 `init`、`submit_action`、`validate_at`、`advance_to`、`record_policy_action`、`apply_external_event`、`observe_at`、`close`，删除状态镜像类、相对时间推进与全部旧 op 客户端；宿主新增 `validate_at` 只读探测，`observe_at` 响应补 `next_scheduled_event_time` 供调用方调度下一次决策，`sidecar_contract_version` 提升到 8。
- 完成阶段 5 的 C# Sidecar 绝对时间分块协议：新增 `SidecarSession`，只暴露 `init`、`submit_action`、`advance_to`、`record_policy_action`、`apply_external_event`、`observe_at` 与 `close`；policy 决策独立写入模型历史，Boss 可选中、移动、目标数与团辅窗口作为带绝对时间戳的类型化外部事实提交，动画间隔由调用方调度，snapshot/fork 继续作为 FightEngine 进程内能力。
- 新增独立 `SidecarHost.Tests` 测试工程，锁定轻量动作/外部事实响应、policy history、七块/28 候选/147 维完整输出、负时间初始化，以及非法外部事实不推进时间。
- 完成时间线驱动状态机阶段 4：`JobSimulator` 只保留 `SubmitAction(timestamp, skillKey)`、`AdvanceTo`、`ObserveAt`、`ValidateActionAt`、`AvailableActionKeysAt`、外部事实提交与 snapshot/fork 等绝对时间协议；动作接受、读条完成和效果结算均由时间线事件驱动，动作历史记录 request/cast/effect 时间及 `ActionInstanceId`。容量 1 动作队列会在入队前校验 MP、通晓等非时间资源，并在 GCD、读条或冷却进入配置窗口时等待至真实 ready_at 自动执行；oGCD 动画间隔由调用方负责。
- 新增独立 policy action/context/history 层：`ogcd_wait` 从系统技能、`SkillBook`、状态机校验和游戏动作历史中移除，由 `config/policy_actions.yaml` 定义模型控制动作，模型候选仍保持 28；policy 决策只记录调用方选择及其前后观测，不推进或修改主状态机。Python 训练公共层新增同源 policy action 加载器，技能词表把 raw id 0 映射为独立的非 padding token，不再误与补位 id 混用。
- 新增时间线重复派发守卫：同一个到期事实连续派发超过阈值即报错，把“处理器不消费事实”的实现从原地挂死变成明确失败；同时补充职业周期资源“结算时刻不依赖推进切分方式”的回归测试，固定一次推进与分段推进必须得到相同的结算时刻。
- 新增时间线架构守卫：扫描生产程序集中的 `Advance*` 入口，只允许 `CombatTimelineRuntime.AdvanceTo` 及明确委托它的绝对时间门面；旧 `Step*`、`AdvanceTime*`、`StepResult`、直接场景写入和临时职业计时器桥接类型必须不存在，不再提供过渡登记白名单。
- 新增时间线驱动状态机阶段 2 的公共时基资源迁移：玩家 GCD/动画锁、冷却、Buff、DoT 与 MP 统一通过 `CombatTimelineRuntime` 的绝对截止时间和事件队列推进；输出层从 deadline 派生 remaining，资源刷新、冷却缩短、快照/fork 与同刻事件顺序均有回归覆盖。
- 阶段 2 首次按严格绝对 deadline 运行 M5S 全量转换时，195 个 raw 文件仅成功 103 个（52.8%）；排查确认 FFLogs 动作时间包含游戏动作队列的临界等待。转换适配层现按请求时资源校验、容量 1 的冷却队列等待 ready_at，重新转换成功 142/195（72.8%），与迁移前基线一致；结果保留在 `.tmp/m5s_conversion_queue_20260913`，剩余 53 个失败仍由通晓、团辅或 MP 等资源条件触发。
- 新增可通过 `model.transformer_activation` 切换的 SwiGLU FFN；保留 GELU/ReLU 实现，SwiGLU 使用 `down(SiLU(gate(x)) * up(x))` 三投影结构，并按 `ff_dim` 的 2/3 计算实际隐藏宽度以近似匹配传统 4×主维度 FFN 的参数量；训练日志、checkpoint、KV-cache、Full AttnRes 与 ONNX 导出均保留该架构选择。
- 新增可配置的 grouped-query attention：`model.num_kv_heads` 默认值为 `1`，当前 Artzip 使用 6 个 Q 头共享 1 组 K/V；训练、trace、推理 KV-cache 与 ONNX 路径均覆盖对应头数契约。
- 新增 GPT-like RoPE 位置编码模块与按样本有效长度生成逻辑位置的回归测试，覆盖混合长度 batch、padding 不变性、KV-cache 和 BF16/ONNX 导出路径。
- 新增独立 GRPO 后训练入口 `python -m grpo`：从 BC/GRPO checkpoint 读取真实训练集场景 token，按 `--max-files` 限制场景文件数量；每个场景先执行贪心 PPG 基线，再采样多条轨迹，以相对基线的 PPG 差值训练，并在贪心 PPG 下降时自动回滚本轮更新。
- 新增时间线驱动状态机的阶段 1 内核：`CombatTimelineRuntime.AdvanceTo(timestamp)` 作为绝对时间推进与到期事件排空的唯一新入口，支持稳定优先级/sequence 排序、事件取消、`TimelineMutation`、完整 `SimulationSnapshot` 及 fork；新增单调时间、分段推进、同刻排序和时钟权威性回归测试。旧状态机调用路径暂未迁移，后续阶段将逐步接入公共资源与职业规则。
- 完成时间线驱动状态机阶段 3：新增 `JobTimelineRegistry`，将黑魔通晓与 MP tick modifier、机工连击/过热/野火计时迁移为职业注册资源和到期事件；删除 `IJobStateMachine` 的三个旧按秒推进接口，并补充注册键唯一性、职业周期/倒计时驱动和接口边界测试。
- 新增 `StatusTimelineRuntime.RegisterExpiryHook(statusKey, handler)` 状态到期钩子契约：职业状态在移除之前回调职业处理器，使"过期时的残留层数"统计不必再另排一个倒计时事件去争同刻派发顺序；补充状态到期钩子的执行顺序、重复注册报错，以及职业倒计时与状态同刻到期时结算不被取消的回归覆盖。
- 新增 `PendingSettlementQueue` 待结算队列，以及事件与待结算事实共用的 `TimelineEventOrder` 排序键：领域只声明"什么时候到期"，越过到期点的到期事实毕业进队列后与普通时间线事件按同一套键排序派发，同一目标只保留一条；待结算事实随 `SimulationSnapshot` 一起保存，fork 与恢复不会重复或丢失结算，后续需要"已计算、稍后应用"的延迟结算可直接登记到该队列。
- 完成 `combat_sim/jobs/machinist.py` 到 `Combat.Sim` C# 状态机的迁移，覆盖热量、电池、连击、过热、野火、起爆、替代技能与机工冷却缩减。
- 新增机工 Python/C# golden 双跑场景，逐字段验证资源、状态、Buff、威力、野火延迟结算与 downtime 行为。

### Changed

- 拆分策略模型的注意力职责：将 `common/policy/model/split_encoder.py` 中的注意力变体、张量整形和 mask 构建分别移入 `attention_variants.py`、`attention_utils.py` 与 `attention_masks.py`，保留旧入口兼容；同时将 C# `ProjectConfigLoader` 的 runtime、engine、system、job 装配与跨配置校验拆为独立阶段。
- 重新整理预训练目录：训练入口迁移为 `training/train.py`，训练配置、数据管线、运行时诊断和训练循环分别归入 `training/config/`、`training/data/`、`training/runtime/`、`training/loop/`；删除 `training/common/`、`training/models/common/` 和 `training/scripts/` 旧目录及兼容入口。
- 训练与 GRPO 配置完成目录迁移：职业技能配置统一位于 `config/skills/`，策略模型与变体配置位于 `config/models/<job_tag>/<variant>/`，共享归一化配置位于 `config/policy/`；训练、回放、模型分析和 ONNX 流程统一读取根目录 `.env` 的 `FFXIV_JOB_TAG` 自动选择对应职业配置，不再依赖 `TRAINING_MODEL_CONFIG` 或 `training/` 下的旧配置路径。
- GRPO 命令行入口由 `grpo/cli.py` 更名为 `grpo/grpo.py`，同步更新模块入口、测试和文档；删除预训练层遗留的配置兼容导出，保持 GRPO 与预训练模块的真实职责边界。
- 将 GRPO 后训练配置、rollout/优化算法、CLI 与测试迁移到根目录 `grpo/` 与 `tests/grpo/`；预训练公共层不再持有 `GrpoConfig` 或 `RunConfig.grpo`，GRPO 入口统一为 `python -m grpo`。
- 将跨预训练、GRPO、回放和 ONNX 共用的策略模型、DataSpec、输入契约、归一化器、compiled cache、policy action、候选顺序与技能词表迁移到根目录 `common/policy/{data,model}`，将共享指标迁移到 `common/training/`；删除 `training/` 中对应的旧实现与导出，预训练层仅保留自身配置、数据管线和训练流程。
- GRPO rollout 采样温度从 `1.0` 提高到 `1.3`，同步更新默认配置与 Artzip 主线配置，增加采样分布熵回归测试以保留更高的探索随机性。
- GRPO 每个场景的默认采样组从 4 条提升到 16 条；自回归采样与 C# 状态机奖励标注统一运行在 inference-only 阶段，全部轨迹完成后才进入真实 GRPO 反向更新，避免采样阶段保留激活导致显存峰值膨胀。自回归轨迹不再按步数或 GCD 数量截断，只由 `grpo.max_duration_seconds` 控制，当前 Artzip 配置为 1200 秒。
- 职业模型配置拆分为 `config.yaml` 清单、`model.yaml` 模型契约、`training.yaml` 预训练/DataLoader/增强/序列过采样配置和 `grpo.yaml` 后训练配置；模型 precision 统一由 `model.yaml` 提供，当前 Artzip 使用 BF16，并保留旧单文件配置的读取兼容。

- 修复自回归回放跨轨迹复用 `SceneTemplateProvider` 时外部事实签名缓存未重置的问题：每条新轨迹首帧都会重新提交 Boss 可选中、目标数与团辅窗口事实，避免沿用上一条轨迹的状态。

- 修复计划文档 §6 的候选契约说明编码损坏；根目录 CLI 的 `list-actions` 与 `smoke` 已切换到 `observe_at`、`submit_action`、`advance_to` 绝对时间协议，并补充回归测试。
- 黑魔模型候选集合移除 `retrace`、`manaward`、`surecast`，并删除状态机中仅服务于 `retrace` 的 `ley_lines_utility` 注册、校验和空应用分支；同步将输入契约提升至 v5、compiled cache 提升至 v12、部署契约提升至 v9，25 候选的模型配置、导出 profile 与测试已对齐，旧 checkpoint 和缓存需要重新训练或转换。
- `decision_timing.ogcd_interval_seconds` 从 0.4s 调整为 0.1s：动画间隔由调用方负责，缩短默认预留时间以匹配实际游戏中的常见操作延迟；C# 状态机不消费该配置。
- `action_queue_window_seconds` 分两步调整：先从 0.05s 修正为游戏动作队列约 0.5s，再从 0.5s 调整为 0.6s 以覆盖严重网络延迟。转换器把它作为请求配对/顺序容差，状态机把它作为容量一队列窗口；按当前配置重跑 M5S 全量 195 个 raw JSON，成功 190/195（97.4%），失败 5 个：2 个 `not_enough_mp`、2 个 `requires_polyglot`、1 个请求顺序超过 0.6s 队列窗口。
- 自回放、PPG 与 GRPO 调用方已完全切换到 Sidecar 绝对时间协议：真实技能使用 `submit_action`，`ogcd_wait` 使用 `record_policy_action`，场景改变使用 `apply_external_event` 提交 Boss 可选中、移动、目标数与团辅窗口事实，观测使用 `observe_at`；自回放移动窗口在结束前 0.5 秒进入滑步豁免，训练样本转换仍由输出层合成移动字段；三者共用 `DecisionScheduler` 按绝对时刻调度，并移除回放层对旧相对时间 API 和 Python state 镜像的依赖。
- FFLogs 回放与 FightEngine 的 GCD 时序现按动作请求时刻锚定：排队 GCD 的 recast 不再从实际接受时刻重新起算，容量队列等待期间仍可执行合法的即时 oGCD；转换层同时把 `begincast.duration` 推导的实际读条时长传入状态机，保持读条锁与效果结算各自按真实时间触发。
- 修复自回归 scene 移动事实未进入状态机的问题：`SceneTemplateProvider` 现在同步 `movement_changed`，移动期间拒绝新的非瞬发读条，已经开始的读条不因后续移动事实取消；硬读条效果统一按实际读条时长减去固定 0.5 秒滑步窗口结算，GCD 缩放不改变这段绝对提前量，并补充 C#/Python 回归测试。
- FFLogs 转换层的基础 GCD 检测改为优先使用黑魔纹外的探针样本：先用 `begincast.duration` 按 `probe_skill_cast_time_seconds` 与技能表基础 GCD 做时长校准，再用黑魔纹外间隔的主峰实际样本中位数；仅在外部样本不足时回退到黑魔纹内间隔并按 `ley_lines_haste_multiplier` 反推，最后才使用全量间隔兜底。
- 转换配置加载现在要求 `probe_skill_cast_time_seconds`（若配置）必须为正数，避免以 0 静默关闭时长校准或以负值生成无意义的基础 GCD。
- FFLogs 转换层的请求时刻改为直接使用配对的 `begincast`；没有 `begincast` 的瞬发或瞬发化动作使用 `cast`，只有开怪首个硬读条因窗口裁剪才按实际 GCD 与滑步窗口回拨，随后整场统一平移到最早请求为 0。该规则在 M5S 全量回放中由 99/195（50.8%）提升到 102/195（52.3%）。
- FFLogs 转换链路迁移到绝对时间内核：硬读条请求时刻改为 `cast − 实际读条时长 + slidecast_window`，读条时长优先取可按 duration 闭环的 `begincast.duration` 并标记来源（`begincast_duration` / `instant_skill` / `prepull_estimated` / `instant_cast_inferred`），随后把动作与场景事实整体平移到最早请求为 0。滑步窗口统一读取 schema 中的游戏规则 0.5s，不硬编码日志实测约 0.543s；多出的网络偏差只交给容量一动作队列。转换器删除 `replay_timing.py`、`training_sync.py`、隐藏时间修补和按 reason 重试，真实动作严格按归一化请求时间提交，`ogcd_wait` 改走 `record_policy_action`。
- 停手窗口改为开区间建模，且两端参照时刻不同：起点取最后一击的生效时刻，终点取恢复后第一击的请求时刻（合法性校验发生在请求时刻，读条技能的请求时刻天然早于生效时刻），边界再内缩一个容差覆盖端点动作被状态机推迟执行的最坏情况。`config/default.yaml` 的 `action_queue_window_seconds` 先从 `0.05` 修正为游戏动作队列约 `0.5s`，随后调整为 `0.6s` 以覆盖严重网络延迟；转换器与状态机共用该窗口，取值过小会把本可被排队吸收的偏差退化成"拒绝 + 调用方补偿 + 累积漂移"。
- FightEngine 把硬读条的服务器效果结算与完整读条锁结束拆开：`ActionEffect` 在 `request + max(actual_cast − slidecast_window, 0)` 触发，`CastCompleted` 仍位于完整读条末端，GCD/读条锁和容量一队列不提前释放；动作历史分别保留 request、effect 与 cast-complete 时间。M5S 严格全量复验为 99/195（50.8%），剩余失败主要暴露在 0.5s 队列边界及加速状态时序，不再以日志专属常数掩盖。
- compiled cache 格式与转换版本提升到 v11、fight data schema 提升到 4、sample schema 提升到 7，`INPUT_CONTRACT_VERSION` 保持 4：归一化请求原点、提前效果结算、policy 词表与真实效果历史增长方式都会改变训练样本，旧缓存不可复用。
- history bank 改为按状态机实际效果历史的单调增长构造，不再假设“每个监督样本恰好新增一条游戏动作历史”；相邻决策共享同一历史前缀、一次观测间结算多条效果以及独立 policy 决策都能生成正确的 end-exclusive 引用。
- 统一 `scripts/model_analysis` 全部输出图的布局：多子图网格改用近正方形列数规则（先最小化 |行数 − 列数|，同一差异下取更少列），20 层的 hidden 分布与注意力矩阵图由 2 列 10 行变成 4 列 5 行、8 个 head 由 4 列 2 行变成 3 列 3 行，空位统一隐藏并保留占位以维持网格对齐（此前 3D PCA 的空位用二维轴占位、按层索引与列数手算，空位与网格不对称）。原先 6 处网格各自硬编码列数（固定 2 列 / 上限 3 列 / 上限 4 列）、6 套 `figsize` 基元、3 套布局引擎（`constrained_layout`、`tight_layout(rect=...)`、`subplots_adjust`）和 2 套空位隐藏写法（`axis("off")`、`set_visible(False)`）全部收敛到 `common.py` 的 `grid_shape`、`create_grid_figure`、`create_figure`、`hide_empty_tiles`、`save_figure`；决策变量染色图的手算 `subplots_adjust` 与硬编码 `fig.add_axes` colorbar 改为共享 colorbar，跨图图例改用 `outside` 位置由约束布局自动让位。单轴图尺寸仍按内容推导（候选注意力图继续按候选数计算），只统一布局引擎与 `savefig` 留白，因此仅图幅与排列变化，图内数值与语义不变。
- 删除失去 Python 状态机对照端后的 C#/Python 状态机 golden 导出器、fixture 与 `CliHost` 项目；固定 seed 随机回放改为 FightEngine 进程内确定性测试，仍有 PyTorch/ONNX 比较对象的部署 golden 保持不变。
- 收紧 FightEngine 动作边界：`ActionRequest` 只携带 `Timestamp` 与 `SkillKey`，`ActionSubmissionResult` 只返回接受/排队、稳定 reason、动作实例和关键时间，不再混入来源、观测标识或完整 `CombatState`；新增程序集级协议守卫防止这些字段回流。
- `SidecarHost` 改为 `Program` 只负责 JSON Lines 循环、`SidecarSession` 负责会话协议；宿主删除旧 `step`、`advance`、`inject_scene`、`validate`、`get_state`、`format_vector_state`、`resolve_cast_seconds`、`advance_hidden` 与 `history_mode` 路径。`init`、动作、policy 和外部事实命令只返回轻量元数据，完整状态或 canonical 七块模型上下文只由 `observe_at` 返回；Python 调用方留待后续迁移。
- 职业时间资源改为绝对截止时刻驱动：`JobTimelineRegistry` 只声明周期与倒计时的截止时刻，不再按 elapsed 递减或累积剩余量，随之删除 `AdvanceResources`、内核的按秒资源推进回调与两个职业的推进方法；黑魔通晓的 tick 时刻由“进入元素态时刻 + 周期”唯一确定，机工连击与野火直接记录绝对到期时刻，活跃条件改为截止时刻的存在性判断。`JobTimerProjection` 仅在输出时把 deadline 投影为 `polyglot_timer` / `combo_remaining` / `wildfire_remaining` 相对秒数，旧计时器反向写入和 `TempJobTimerLegacyBridge` 已删除。
- `CombatStateMachine`、候选预演、`ReplayStateCache`、`SequenceRunner` 和固定随机回放统一复用完整时间线快照及 `SubmitAction + AdvanceTo`；删除 `CombatStateMachine.Step*` / `AdvanceTime*`、`StepResult`、`SystemSkillRuntime`、公共 `CombatState.Time` setter 和旧相对时间窗口结算契约，不再保留双执行内核。
- 删除 `SystemStateMachine` 上阶段 3 遗留的 `resolveTickAmount` 三参回调：职业 MP tick 修正统一走 `JobTimelineRegistry.RegisterMpTickModifier`，`MpTick` 事件只剩一条结算路径；`SystemGoldenExporter` 与系统层测试同步改用真实注册入口，不再通过与生产装配平行的回调传入固定回蓝量。该改动为纯清理，六个 golden 导出逐字节不变。
- 解耦历史容量与 compiled cache：缓存编译阶段固定保存完整 history bank，训练/分析读取时再按 `model.history_capacity` 裁剪；调整模型历史容量不再触发 cache 重建，旧缓存因格式升级仅需迁移重建一次。
- 统一模型上下文容量配置：使用 `model.scene_capacity` 与 `model.history_capacity` 描述两个输入块的最大容量，物理 token 总长自动按 `scene + history + candidate + CLS` 换算；移除 `model.max_sequence_length`，ONNX contract/manifest 升级到 v6。
- 彻底移除已由 `Combat.Sim`/`SidecarHost` 接管的 Python `combat_sim/` 状态机及其专属测试、Python golden 双跑脚本；`main.py`、训练测试 fixture 与回放测试统一使用 `common/` 静态契约和 C# 状态机后端。
- 将 Artzip 当前主线基底模型调整为 `d_model=512`、`n_layers=20`、`n_heads=8`、`num_kv_heads=1`、`ff_dim=2048` 的 MQA 配置，`history_capacity` 调整为 `384`，并关闭训练集候选顺序随机化；旧的 768/6 层/12 头 checkpoint 不与该模型架构混用，GRPO 应从匹配的新 BC checkpoint 开始。训练主线输出目录统一为 `artifacts/checkpoints/black_mage/artzip_bc`；RoPE、MQA 和 SwiGLU 模型变体共用该主线，模型分析默认路径与真实 checkpoint padding 测试同步指向该目录，不再为当前训练维护独立的 `artzip_rope_bc` 输出目录。
- 优化训练热路径：split encoder 将三段共享的 LayerNorm、QKV、输出投影和 FFN 合并执行，同时保留 prefix/candidate/CLS 各自的 SDPA mask；重复动作惩罚缓存候选索引并在 batch 搬运前生成 mask；BC/GRPO 指标改为设备端累积后集中读取；变长 history/scene padding 改用 PyTorch `pad_sequence` 和批量长度 mask。

- 修正 grouped attention 在 `batch_first=False` 时注意力权重的输出布局，保持 `MultiheadAttention` 约定的 `[batch, query, key]` / `[batch, heads, query, key]`，并补充对应回归测试。

- 候选 Transformer 固定改用 GPT-like RoPE：删除 learned absolute `pos_embed`，由输入编码器按样本有效 scene/history 生成逻辑位置，并让普通 Attention、Full AttnRes、trace、KV-cache 与 ONNX 共用 Q/K 旋转语义；新训练输出隔离到 `artzip_rope_bc`，原 `artzip_bc` baseline 保持不变。
- grouped attention 针对 PyTorch 2.12 的 `enable_gqa` 会物化重复 K/V 的行为，改为按 KV 组调用普通 SDPA；单 KV 通过零 stride 的 `expand` 一次处理全部 Q 头，多 KV 每个 KV 组调用一次，降低训练、验证、PPG 与直接推理的调用次数，同时保留压缩 K/V 存储；trace 仍仅在显式 attention 权重结果上物化展开头数。
- `num_kv_heads` 同步进入模型结构校验、旧 RoPE checkpoint 兼容解析和推理 KV-cache；缺少该字段的旧 checkpoint 按 `n_heads` 恢复传统 MHA，避免误加载为 MQA。
- ONNX 部署 contract/manifest 升级到 v5，位置契约改为“右侧 padding 不计入的按样本逻辑位置”；BF16/FP16 导出保留 RoPE 所需的 FP32 频率常量，最大物理 token 序列容量改由 `max_sequence_length` 校验。
- 新增 `Combat.Sim/FightEngine/Jobs/JobResourceAccessors` 公共职业量谱访问基类，统一资源读取、最大值、类型转换与写入；黑魔和机工通过 `Bind` 复用，消除职业间重复实现并为后续职业扩展建立公共入口。
- 修复 Sidecar 多阶段 `init` 丢失 `max_history` 的问题：提取阶段与训练阶段重新初始化时继续沿用构造时的历史窗口，避免增量 history bank 在第 769 个样本处误判并跳过 FRU raw 文件。
- compiled cache 现在会拒绝结构损坏的 history bank manifest，并统一回退到重编译路径，避免非 tensor 字段或异常 `action_keys` 使训练启动直接崩溃。
- compiled cache manifest 现在校验 history bank 行数与 `num_samples` 一致，并区分结构损坏异常与 `MemoryError` 等运行时错误，补齐错位字段和缺失字段的回归覆盖。
- `Combat.Sim/tools/run_golden.sh` 的真实 Sidecar init 握手现在同时覆盖 `black_mage` 与 `machinist`，确保机工注册和资源路径也经过 Sidecar 契约回归检查。
- Sidecar 的 `advance` / `advance_hidden` 支持可选 `targetable_windows`，由 Python 转换层提取场景窗口并交给 C# 状态机执行延迟伤害判定。
- `Combat.Sim/tools/run_golden.sh` 扩展为七组双跑，并在执行前自动重建 SidecarHost/CliHost，再通过 `dotnet run --no-build` 执行 CliHost；默认 Debug，可用 `COMBAT_SIM_CONFIGURATION=Release` 切换配置，不再要求手动预构建或依赖固定 TFM/DLL 路径；已有 NuGet assets 时不再无条件触发离线 restore。
- Sidecar 新增运行时契约版本握手：Python 与 C# 统一从 `config/schema.yaml` 读取版本，转换/回放在 `init` 阶段拒绝旧 DLL，避免旧版本技能 token 缺少 `value` 时静默分叉；`run_golden.sh` 现在包含真实 Sidecar init 握手；修改 C# 宿主或 schema 后须先重建 `SidecarHost` 与 `CliHost`，并保证 DLL 与 schema 同版本部署。
- 机工资源上限统一由 `machinist.yaml.resources.*.max_value` 提供，Python/C# 机工逻辑不再维护 100/100/6 的重复常量；通用资源注册器继续消费各资源的上限以保持归一化语义。
- 机工配置加载新增 `timing.wildfire_duration` 与 `resources.wildfire_remaining.max_value` 一致性校验；保留资源上限供通用归一化使用，但不再允许两份配置静默漂移。
- scene 浮点容差上收到 `config/schema.yaml`，Python 场景查询与 C# Sidecar 延迟伤害边界统一使用 `1e-6`；回放层将公开的 token 窗口解析 helper 与按 scene 的窗口缓存用于每次推进复用。
- 回放场景窗口缓存的返回列表明确为只读契约，防止未来调用方原地修改共享缓存对象。
- 机工整备改为职业层动态 `value`：下一个符合条件的 GCD 武器技能 `value` 解析为 `2.0`；候选/历史 token 与训练 compiled cache 使用运行时值，不再输出或解析 `guaranteed_critical_hit` / `guaranteed_direct_hit`。
- 训练 compiled cache 升级为 v7 增量历史 bank：每个 source 只保存一份按动作递增的历史行，样本只保存 `history_end` / `history_length` 引用；collator 不再在 CPU 重建重复历史前缀，模型输入层在目标设备上 gather 原有 dense 窗口。旧 compiled cache 会因格式与转换版本变化自动失效并需要重建；历史输入值与模型/ONNX dense 接口保持不变。
- history bank manifest 现在使用 `mmap=True` 加载，并通过稳定的 `history_bank_id` 在 batch 内去重，降低超长 source 和多 DataLoader worker 场景下的常驻内存放大；旧 v7 manifest 缺少该 ID 时回退使用 cache 路径作为标识。
- 技能 token 的 `kind` 统一编码为数值维度（`gcd=1`、`ogcd=0`），同时供模型 `skill_features` 与指标计算读取；compiled cache 升级为 v8 / 转换版本 v5，并为独立验证 PPG 保存历史技能直接威力和累计 DoT 威力。
- 训练指标拆分为真实验证副本自回归 `val_ppg` 与空场景模型自回归 `none_ppg`：`val_ppg` 为模型从每个验证 source 的首个缓存状态和完整 scene token 自循环到最后一个 Boss 可选中窗口结束后，按执行历史的直接威力、累计 DoT 威力和 GCD 数计算单副本 PPG，再对副本取平均；不读取候选技能伤害，也不与 teacher-forced 的 top1/top3 共用计算路径。

### Fixed

- 修复 ONNX exporter 测试残留的裸 `torch.load(weights_only=False)`，统一改用 `common.torch_serialization.safe_torch_load`，恢复 checkpoint 读取的 fail-closed 约定。
- 修复模型分析相较训练异常占满显存和系统内存的问题：所有模型前向统一读取模型 YAML 的 `training.precision` 并以 BF16 加载/执行，逐批释放完整 hidden/attention trace 和 CUDA allocator 缓存，PCA 改用 CPU float64 特征协方差分解以避免 CUDA 完整 SVD 工作区；基础图完成后释放重上下文，再以相同 checkpoint、数据和精度加载轻量 loss 上下文，loss batch 按需流式搬入 GPU，float64 只用于损失累计和导出数值而不形成分散的输出精度配置。
- 修复训练启用技能价值辅助损失时遗漏 policy 候选 `ogcd_wait` 的问题：运行时 action value 回退现在同时加载职业技能和独立 `policy_actions.yaml`；回放 batch 在 GCD 已就绪时屏蔽 `ogcd_wait`，避免把它当作合法等待动作。
- 修复黑魔 Firestarter（火苗）与 Swiftcast/Triplecast 的瞬发资源消耗：火苗瞬发 Fire III 不再消耗通用瞬发层，并将通用瞬发优先级统一为 Swiftcast > Triplecast；补充对应回归测试，修复 FFLogs M5S 回放中后续动作被误判为 cast_locked。

- 修复目标数事实只发"进入多目标"、不发"离开多目标"的问题：状态机的 `CombatState.TargetCount` 是持久字段，而目标数量 token 只描述多目标区间（区间之外隐含单目标），旧实现按 token 值去重后从第一次多目标起就再也不发归位事件，导致停手或回到单目标后的 AoE 技能（`flare`、`flare_star`、`foul`、`freeze`、`high_fire_ii` 等）一直被叠加 AoE 衰减倍率。现按分段常量时间线在每个变化点发事实（区间内取 token 目标数、区间外回到单目标），并补充 scene 事实流回归测试；本地 50 个 raw 文件中原本有 25 个受影响、366 次 AoE 动作的倍率错配，现已归零。
- 修复 `common/config.py` 仍按旧键名读取 `cooldown_ready_tolerance_seconds` 导致 `load_project_config()` 直接 KeyError 的问题：该键在阶段 4 已重命名为 `action_queue_window_seconds`，Python 侧未同步，`skill_vocab`、`normalizer` 与模型分析等所有依赖配置加载的路径都会在启动时失败。
- 修复负时间初始化时读条锁未对齐起始时刻的问题：`SystemStateMachine.InitialState` 补齐 `CastEndsAt = startTime`，与 GCD、动画锁、weave、自然回蓝与停手截止时刻保持同一时基。旧实现下用负 `initial_timestamp`（prepull 起手）初始化时，`CastEndsAt` 会保持默认 `0.0` 并在负时间轴上变成"未来锁"，导致首个动作恒定返回 `cast_locked`。
- 修复绝对时间门面在未知技能或非法外部事实上先推进时间再抛错的问题：技能和外部事件载荷现在会在触碰时间线前完成解析与校验，失败后逻辑时钟保持不变；补充对应回归测试。
- 修复直接构造 `SkillDefinition` 时省略 `applies_statuses` / `tags` 会留下 null 并在状态机装配时触发空引用的问题，两个集合统一归一化为空只读集合。
- 修复动作效果事件化后读条期间仍可提交后续动作的问题：玩家态新增绝对 `CastEndsAt` 锁，读条结束前统一返回 `cast_locked`；进入动作队列窗口后允许容量 1 的请求等待至读条结束，并保证同刻 readiness 事实先于队列动作接受处理。
- 修正职业时间资源历史快照绕过绝对截止时刻投影的问题：`SystemHistoryRuntime` 现在复用系统层资源转换入口，`polyglot_timer`、`combo_remaining` 与 `wildfire_remaining` 的 `job_resources_before/after` 会按当前时间正确投影；补充黑魔通晓计时器历史回归测试。
- 修正动作历史快照里职业计时器滞后一拍的问题：旧实现的计时器只在 `AdvanceTo` 的事件循环内累积，`Step` 路径不经过它，因此 `ActionHistoryEntry` 的 `job_resources_before/after` 记录的是累积前的值（黑魔通晓计时器与机工连击、野火剩余量恒为 0）。改用绝对时基即时投影后，历史快照正确反映当时的剩余量，黑魔与机工 golden 中该批字段随之修正。
- 修复时间线内核用资源重新描述的结果取消已到期资源事件、导致到期结算静默丢失的问题：时钟已经越过到期点的资源事件会从资源描述"毕业"进待结算队列，之后不再参与重新描述，因此不会被同刻其他事件触发的资源同步取消。该缺陷会让机工野火在到期时刻与停手开始、状态过期或自然回蓝并发时丢失整次结算、`wildfire_active` 永久卡住（后续野火与起爆判定随之污染）；修复后 `JobTimerExpired` 回到计划 §3.3 的到期档位，不再依赖与状态过期事件的相对派发顺序。
- 修复阶段 3 机工过热到期统计依赖 `JobTimerExpired` 与 `StatusExpired` 同刻派发顺序的隐式耦合：统计改由状态到期钩子承担，并与状态移除落在同一次过期结算内，职业不再自行删除系统状态。
- 修复阶段 3 机工过热倒计时在到期前临界同步窗口被取消、导致 `wasted_overheated_stacks` 静默丢失的问题；倒计时仅在真实到期后停止排程，过热到期统计与状态移除合并为一次性处理，并补充尾部窗口回归测试。
- 修复转换器临界冷却动作队列的复验结果丢失：等待至 `ready_at` 后若动作仍因通晓、MP 等资源条件非法，现在返回结构化结果并由标注/训练调用方显式失败，不再把已推进时间误记为未处理；新增该分支回归测试。
- 更新 `engine_timing.cooldown_ready_tolerance_seconds` 注释，明确其仅作为转换器容量 1 动作队列窗口，C# 引擎不再消费该配置。

- 修正黑魔悖论的元素转换条件：AF 仅在 AF III 进入转换时授予悖论，UI 仅在 UI III 且拥有 3 层灵极魂时授予；AF I/UI I 或灵极魂不足不再错误生成悖论；绝望从 AF I 施放后仍恢复 AF III。
- 修复候选 oGCD 预览把技能冷却错误投影到下一 GCD，导致自回归提前选择尚未转好的星灵移位；候选合法性现在读取当前状态，Sidecar 也会在 timing/effect 修改前拒绝非法动作。同步升级 Sidecar 至 v5、checkpoint 输入契约至 v3、compiled cache 转换版本至 v8、ONNX deployment contract 至 v8，旧产物需重建或重新导出。
- 同步提升状态机语义变更涉及的运行时与产物契约：Sidecar 至 v4、checkpoint 输入契约至 v2、compiled cache 转换版本至 v7、ONNX deployment contract 至 v7；旧 DLL、checkpoint、缓存和导出包会被拒绝或自动失效，需重建、重训或重新导出。

- 修复训练验证 PPG 仍读取已移除的顶层 `RunConfig.max_history` 导致首轮验证崩溃的问题，统一从 `model.history_capacity` 读取，并补充验证入口回归测试。

- 恢复候选 Transformer 的统一 RoPE 位置语义：所有有效 token 按 `scene -> history -> candidate -> CLS` 连续编号并共用同一套 Q/K 旋转，删除 `position_id_semantics` 配置开关；旧 candidate-shared checkpoint 直接拒绝加载，训练集候选顺序随机化固定为 `probability: 1.0`，验证集与在线推理继续使用 canonical 顺序。
- 修复 GRPO 共享 replay session 接入时遗留的 `normalizer` 局部变量引用，确保真实训练启动时使用 session 持有的归一化器；同时补充 Sidecar/cache 复用、独立轨迹 reset、`history_state_null_mask` 右侧填充为 `True` 的回归覆盖。
- 修复 GRPO/replay session 在自有回放构造失败、训练更新/指标计算或 checkpoint 写盘异常时未统一回收 SidecarHost 的生命周期漏洞：自有 session 构造失败会立即关闭，GRPO 训练主体统一通过 `finally` 回收共享 session，并补充异常路径回归测试。
- 修复 GRPO 多条轨迹复用 SidecarHost 与 compiled cache 时的生命周期边界：每条轨迹显式 reset，训练结束或异常时统一回收共享 session，避免状态串轨迹及重复启动进程。

- 修复 GRPO 贪心回滚快照常驻 GPU 显存的问题：第 1 轮回滚恢复启动 checkpoint，第 2 轮起直接从上一轮 `latest.pt` 恢复模型、optimizer 和 scheduler 状态，移除完整模型权重的 GPU `deepcopy`；新增磁盘回滚恢复回归测试。
- 修复 GRPO 贪心基线优势在组内 PPG delta 方差极小时被 `1/std` 过度放大的问题：增加与 delta 绝对量级相关的 scale 下限，并将最终 advantage 对称裁剪到 `[-5, 5]`；保留以贪心 PPG 为零点的正负奖励语义，同时记录 `advantage_abs_max` 便于监控裁剪情况。
- 修复训练 YAML 中已移除的 `model.scorer_use_raw_projection` 只通过 `True` 身份比较导致 `1`、`"true"` 和 `"yes"` 被静默忽略的问题；现在严格解析布尔别名，启用值明确拒绝，禁用值仅执行兼容迁移。
- 旧 learned-absolute checkpoint 不再静默兼容：缺少新 RoPE 模型契约的 checkpoint 会在加载阶段明确拒绝，避免位置编码架构与权重错配。
- 修复 `build_position_ids` 对右侧 padding 的隐含依赖：scene/history 现在按 mask 的有效计数生成逻辑位置，左侧或中间存在空洞时不会造成 RoPE 角度错位；补充非右填充回归测试。
- 修复 `ogcd_wait`（模型显示为 `-`）在下一个 GCD 边界使用 oGCD 的时序问题：Python/C# 状态机在边界返回 `ogcd_after_wait`，执行下一个 GCD 或越过边界后解除限制；autoregressive replay 同步屏蔽 wait 后的 oGCD 候选。
- 收敛 `ogcd_wait` 后 oGCD 禁止规则的真值来源：Sidecar 快照暴露 C# 状态机的边界 pending 标记，autoregressive replay 和 history ablation 均读取该标记；补充越过边界、GCD 清除及快照传递回归测试。
- 升级 Sidecar 运行时契约版本以覆盖 `ogcd_wait_boundary_pending` 快照字段；Sidecar 在 init 返回旧契约版本时会在初始化阶段拒绝，若旧 DLL 随当前磁盘 schema 返回新版本但 Snapshot 缺少该字段，则在回放首次决策时显式报错，避免静默回退为未屏蔽 oGCD。
- 修复 ONNX 固定容量 padding 门禁将 trace 手工 attention 与正式 SDPA logits 混比，导致 BF16 近似并列候选误报 argmax mismatch；现在两条路径分别进行同路径 padding 不变性校验，并补充回归测试与真实 checkpoint 验收。
- 修复覆盖已有 ONNX 部署包时 Windows 临时 `WinError 5` 导致发布中断；部署目录备份、发布和恢复的改名操作增加有限重试，并在最终失败时保留旧包。
- 修复 RoPE 旋转方向与主流约定相反的问题：由负旋转 R(-θ) 改为与 LLaMA/Hugging Face/GPT-NeoX `rotate_half` 一致的标准正旋转 R(+θ)，避免外部复刻或独立数值对照时 sin 项反号形成镜像差异；补充旋转方向锁定测试；此前按负旋转训练的 checkpoint 不再兼容，需重新训练。

## [0.0.8] - 2026-08-17

### Added

- 新增 `Combat.Sim/` C# 状态机搬迁工程骨架：net10.0 类库 `FightEngine`（Models/Common/System/Jobs/Config/Outputs/Facade 分层，最终 DLL 供游戏插件引用）+ `CliHost`/`SidecarHost` 宿主壳 + xUnit 测试工程 + `tools/` golden 工具；职业与系统 YAML 配置权威来源不变，C# 直接复用 `config/` 下配置，Python 状态机保持权威、golden 双跑逐字段验证。
- `Combat.Sim` 配置加载已按 Python `load_project_config` 1:1 迁移：YamlDotNet 解析树 + Plain 标量类型提升（null→bool→int→float→string，数值统一 double），覆盖职业路由、技能/状态/量谱构建、mp_cost 语义与跨配置冲突校验；通过 17 项 xUnit 测试与 Python 逐字段 JSON 对比（整数/布尔精确、浮点 1e-9 相对容差）验证完全一致。
- `Combat.Sim` 深入职业/系统配置收尾完成：新增 `Common/SceneContracts`（scene 契约常量：窗口 context key、绝对时间模式、统一滑步窗口 0.5s，对照 contracts.py）与 `Common/SceneWindow`（feature key 索引映射缓存与窗口索引解析，对照 scene_window.py）；技能索引 `Skills/SkillBook`（阶段 4 因门面强依赖提前落地）；配置校验点与 Python `config.py` 全部对齐（跨配置重复状态/技能、grant_status 缺 applies_statuses、mp_cost 语义、资源 max_value 等）；6 项 scene 测试全绿并与 Python 权威端逐用例一致，累计 107 项 xUnit 测试全绿。
- `Combat.Sim` 系统状态机已按 Python `combat_sim/system/` 迁移完成：门面 `SystemStateMachine`（五段时间推进编排、威力解析与全部委托入口）+ 10 个运行时分类组织（`Registries/` 注册中心、`Timeline/` 时间推进、玩家态/回蓝/历史/系统技能）；Models 分类重组为 `Definitions/`（静态定义）与 `Combat/`（运行时值）；`advance_time` 五段顺序、冷却双阈值容差、DoT 窗口多段 tick、量谱 round(4) 与 consumed 差分等语义与 Python 一致；66 项 xUnit 测试全绿，25 步操作序列 golden 双跑与 Python 逐字段完全一致。
- `CliHost` 增加正式 golden 导出命令（`export-config` / `export-system-golden`），与 Python 权威导出脚本同构输出；新增 `tools/run_golden.sh` 一键端到端双跑（Python 导出 → C# 导出 → 逐字段对比），使「1:1」验证在仓库内可复现、可回归。
- `Combat.Sim` 门面层已按 Python `state_machine.py` / `factory.py` / `candidate_preview.py` / `replay_cache.py` 迁移完成：`Facade/CombatStateMachine`（职业路由与构造校验、step/advance/validate/available、set_target_count、技能索引、候选预演入口）+ `ReplayStateCache` 增量回放缓存 + `CandidatePreviewBuilder` 候选预演；新增 `Jobs/` 职业子状态机契约 `IJobStateMachine`（Python getattr 可选方法以默认接口实现表达）与 `JobMachineRegistry` 注册中心、`Skills/SkillBook` 技能索引、`Models/Combat/StepResult` DTO 与 `Common/GcdUnits` 换算；新增 C# 侧设计 `Facade/JobSimulator` 插件外壳（状态快照/执行/推进/候选预演/场景注入/系统层直接操作，Python 无对应物）；`CliHost` 新增 `export-facade-golden` 命令与 golden 最小职业，`tools/` 新增 `facade_golden_python.py`（生成最小内嵌配置 + 注册行为等价最小职业）并把 `run_golden.sh` 扩展到三段双跑；97 项 xUnit 测试全绿，18 步决策序列（GCD/oGCD 时序、weave 窗口、空转、读条、冷却锁定与恢复、爆发药增伤、AoE 目标聚合）golden 双跑与 Python 权威端逐字段完全一致。
- 新增自回归回放 `PolicyBackend` 边界和 ONNX Runtime backend：PyTorch 与 ORT 共用同一 `CombatStateMachine`、`LiveBatchBuilder`、候选顺序、重复惩罚、合法 mask、temperature/top-p 解码和轨迹推进；ORT 在创建 session 前严格校验部署 manifest、文件哈希、Tensor 契约和职业路由，并按 manifest 的正式 scene/history 容量执行固定右侧补位。
- 新增 `--parity-onnx-package` 完整轨迹验收和 JSON 报告：同一决策点逐步比较 PT/ORT raw logits、Top-1、Top-3 集合及宿主后处理后的最终动作，同时记录首个分叉、动作序列、PPG、p50/p95/p99 延迟和峰值内存；支持 `--max-gcds 128` 固定木桩验收终止条件。
- 新增 `requirements-onnx-gpu.txt` 本地 NVIDIA 可选依赖；模型导出和回放默认严格使用 CUDA EP，运行库不完整时直接失败，只有显式选择 `auto` 才允许 CPU fallback。pytest tiny checkpoint 只作为安装 ONNX extras 后的本地快速门禁，不宣称仓库已接入 CI，也不会替代真实 46.9M 参数 checkpoint 的显式慢速全链路导出验收或缩小发布模型。
- 新增独立的 `scripts/onnx_export` 纯 Tensor 候选打分 adapter：从已有模型复用 input encoder、Transformer 和 scorer 权重，显式执行 Q/K/V attention，固定使用 eval 且关闭内部 KV cache，只输出合法性、重复惩罚和采样之前的 raw logits；导出脚本职责不进入训练包或训练循环。
- 新增可选 Full Attention Residual 模型路径：以零初始化的深度 pseudo-query 和逐子层 RMSNorm，对 embedding、Attention/FFN 输出及最终输出执行沿深度的 softmax 聚合；`full_attention_residuals` 已接入训练配置、split encoder、prefix KV cache、trace 和 ONNX `torch.export`，黑魔 Artzip 实验配置曾启用、现因 OOM 暂时关闭；启用该架构时需要重新训练 checkpoint，当前关闭时沿用非残差 checkpoint，训练/推理/部署对照及初始化、反向传播测试全部覆盖。
- 强化 Full AttnRes 实验门禁：`full_attention_residuals` 采用严格布尔解析，避免引号 `"false"` 静默启用架构；depth-attention logits 增加 `1/sqrt(d_model)` 缩放；ONNX `torch.export` 图在 residual 开关两种模式下实际执行并与 PT logits 对齐；补充非追加历史触发 KV-cache 重建的 residual 回归测试。
- 优化 Full AttnRes 训练显存：启用已有 Attention/FFN activation checkpoint 时，对整个 residual encoder 使用 non-reentrant checkpoint，释放 forward 中长期持有的多层 `sources` 激活，并以 forward/反向梯度对照测试保证数学语义不变。
- 优化 Full AttnRes 深度聚合的临时显存：改为逐 source 计算 key/logit 并只堆叠深度 logits，避免复制完整的 value/key 历史激活；新增与原堆叠公式的输出及梯度等效性测试。
- 新增 `python -m scripts.onnx_export` 独立导出 CLI 和 `requirements-onnx.txt` 可选依赖：无参数运行时从根目录 `.env` 读取共用 checkpoint、部署包、opset、精度、职业部署 profile 与 ORT EP，CLI 仅用于临时覆盖；原子生成单文件 `model.onnx`、JSON Schema 约束的完整签名 manifest、容量证据、确定性 golden 输入/输出和导出报告，并在发布前执行 ONNX checker、shape inference、ORT session、PT/ONNX 参数体量审计及 CPU/CUDA/ORT padding 矩阵验收，失败不会覆盖已有有效产物。
- 新增根目录 `onnx_pipeline.ps1` 中文菜单和 `scripts.onnx_export.workflow` 编排入口：PowerShell 只选择完整流程、环境检查、导出、两项 parity、发布校验或普通 ORT 回放，不保存模型路径或数值参数；一键流程即使首项 parity 失败仍会完成后续报告，最终通过专用非零退出码报告 `parity_failed`，不再把预期门禁失败打印成 Python traceback。
- `Combat.Sim` 黑魔职业子状态机已按 Python `combat_sim/jobs/black_mage.py` 迁移完成：新增 `Jobs/Black.Mage/` 五文件（`BlackMageConstants` 常量表 / `BlackMageJobStateMachine` 主类 / `BlackMageValidators` / `BlackMageApplications` / `BlackMageElemental`），static Tag + Bind 注入装配，内置职业由 `JobMachineRegistry` 静态构造期反射自动发现注册（扫描程序集 `IJobStateMachine` 实现，注册中心不引用具体职业，等价 Python 模块自动发现）；18 个行为分派与 `unknown_behavior` 防御、元素极性 MP/威力解析（AF 双倍耗蓝、灵极魂 0.5 折扣、UI 免费冰系）、满层异极性减读条 0.5、火苗免费瞬发、三连咏唱层数消耗、通晓 30s 计时、AF 自然回蓝屏蔽、醒梦 tick +550、灵极魂命中回蓝（满层回满/每层 2500）、stance 过渡悖论/雷云授予等语义与 Python 逐条一致；34 项 xUnit 测试全绿（含人工循环整条回放），`CliHost` 新增 `export-black-mage-golden` 命令，`tools/` 新增 `black_mage_golden_python.py`，`run_golden.sh` 扩展到四段双跑，黑魔 34 步决策序列（含 AoE 目标聚合、DoT tick、通晓增长与停手灵极魂）golden 双跑与 Python 权威端逐字段完全一致。
- `Combat.Sim` 输出层已按 Python `combat_sim/outputs/` 迁移完成：`Outputs/` 二十余文件覆盖状态上下文装配（`StateContextBuilder`）、canonical schema 与元数据（`OutputContextSchema`）、scene 空壳（`SceneContextSchema`）、秒制/GCD 制快照（`SecondsStateFormatter` / `GcdStateFormatter`，seconds 量谱在 GCD 制下转 `*_gcd`）、token 装配（`SkillTokenBuilder` 单一 build 函数同时服务技能历史与候选技能，`StateTokenBuilder` 单一 build 同时服务状态历史与合法候选、非法候选保留真实 before 并把 after/consumed 写 null，四个向量 builder 当前/历史共用同一向量函数）、历史/候选上下文装配（含条目引用增量缓存与状态上下文引用缓存）、canonical/tensor 输出（`TensorPayloadPacker` 无 torch 递归打包为 `{values,is_null}` 结构，`TensorPrecisionAdapter` 独立适配层按 dtype 量化数值使 float32 输出与 `torch.tensor(...).tolist()` 逐位一致，精度从 `config/precision.yaml` 懒加载）；门面 history 条目的 `state_before`/`state_after` 由占位空字典替换为真实 `StateContext`，并新增 `format_state` / `format_step` / `format_vector_state` / `format_tensor_state` 输出入口；`CliHost` 新增 `export-outputs-golden` 命令，`tools/` 新增 `outputs_golden_python.py`，`run_golden.sh` 扩展到五段双跑；22 项输出层 xUnit 测试全绿（累计 166 项），7 步序列的秒/GCD 快照、canonical 与 tensor 输出 golden 双跑与 Python 权威端逐字段完全一致。
- `Combat.Sim` 新增序列回放辅助 `Facade/SequenceRunner`（对照 `combat_sim/sequence_runner.py`：自动补推进 GCD 锁/动画锁 + 精简动作审计 + 只保留最终 canonical 输出），并配套 2 项 xUnit 测试；`CombatStateMachine.AdvanceTime` / `AdvanceTimeInPlace` 与 `IJobStateMachine.AdvanceTimeBeforeSystem` 签名扩展 `targetable_at` 场景查询回调透传（对齐 Python 原生 `advance_time(targetable_at=...)`，可选参数不破坏既有调用），门面新增公共 `BuildPreviewTiming` 纯时序查询（读条/有效 GCD/窗口元信息，不执行动作、无副作用，供外部编排使用）；用 FFLogs 人类序列 `fflogs_AprDNvVJ6T8Xm4Gc_f14_N'lya_Tarin.json`（871 步）驱动 C# 状态机逐步回放并与 Python 权威端逐字段对比，动作序列、决策时刻、合法性、时序协调与秒制快照共 34 个字段完全一致，验证状态机被外部调用时行为逐步对齐；累计 168 项 xUnit 测试全绿。
- `Combat.Sim` 新增固定 seed 随机序列双跑验证载体：`Facade/RandomSequenceReplay` 统一承载决策循环（每步枚举合法动作，按 key 排序并排除 sentinel 系统技能；无合法动作时推进 `max(gcd_remaining, animation_lock_remaining)` 到最近可恢复事件，动作可执行要求 GCD 与动画锁都结束，min 会在一边归零时推 0 卡死；有合法动作时用 SplitMix64 固定 seed 随机选择，与 Python 端 ulong 运算逐位一致），`CliHost` 新增 `export-random-sequence-golden` 命令，`tools/` 新增 `random_sequence_golden_python.py`，`run_golden.sh` 扩展到六段双跑；100 步随机决策（63 动作 + 37 推进，覆盖 23 种技能）的合法列表、选中下标、执行结果与状态快照 golden 双跑与 Python 权威端逐字段完全一致；新增 `FightEngine.Tests/Golden/` 回归测试，提交 Python 权威导出的 fixture（`random_sequence_black_mage.json`）并重跑决策循环逐步对比（容差对照 `compare_config.py`），fixture 缺失或配置漂移时给出重新生成命令；累计 175 项 xUnit 测试全绿。
- `scripts/convert_fflogs` 转换链路已切换为 C# 状态机后端（方案 A：SidecarHost 长驻进程，不耦合训练/回放，后端可替换）：新增 `Combat.Sim/SidecarHost` 交互宿主，通过 stdin/stdout JSON Lines 协议暴露 init / advance / inject_scene / validate / step / get_state / format_vector_state / resolve_cast_seconds / get_resource / set_resource / grant_status / clear_status / set_state / advance_hidden / close 命令（step 在宿主内完成时序应用、GCD 扣减、推进到生效时刻、效果与 history 记录，history after 按转换语义推进扣减后的 gcd_remaining/animation_lock，advance_hidden 推进连续资源并恢复 polyglot/mp tick 时基资源），并新增 `scripts/convert_fflogs/cs_backend.py` 客户端（StateSnapshot 属性镜像保持 `state.mp` 等驱动语法、seq 匹配校验与响应同步）；转换流水线（pipeline / raw_source / replay_timing / training / training_sync / utils）全部改为后端驱动，不再保留 Python 状态机兼容模式，`CombatStateMachine` 的 `BuildSkillSnapshot` / `BuildActionMetrics` / `BuildNextGcdWindowSeconds` 提升为 public 供宿主调用；早期迁移基线的 M5s 全量人类数据转换通过率为 188/195（96.4%）与 Python 状态机逐文件一致，42 步标准木桩循环回放效率约 4.5 倍；后续按当前 0.6s 默认值重跑的验收结果见上方 190/195 条目。
- 转换测试同步切换为 C# 后端：新增 `tests/scripts/convert_fflogs/conftest.py`（`cs_backend` / `cs_skill_book` fixture，SidecarHost 未构建时跳过），raw 转换与训练样本测试全部改由 `SidecarBackend` 驱动，机工相关测试在 C# 机工状态机迁移完成前显式跳过。
- `scripts/autoregressive_replay` 已切换为 C# 状态机后端（SidecarHost），生产脚本不再实例化 Python 状态机：SidecarHost `step` 命令返回执行元信息（skill_key / skill_kind / actual_occupancy_seconds / next_gcd_window_seconds / gcd_unit_seconds）并支持 `history_mode`（`replay` 转换语义 / `standard` 自回归回放语义），状态快照 dots 补充 `next_tick_in_seconds`；`cs_backend` 新增 `StepResult` 镜像保持 `skill.kind.value` 驱动语法；回放核心（replay.py）全部由后端驱动（step/advance/inject_scene/format_vector_state），历史消融改为缓存决策时刻 canonical 上下文做截断重算（不驱动后端、避免大结构状态跨进程序列化），scene 同步统一走 `inject_scene`（团辅窗口经 raid_buff_remaining），PPG 评估（ppg.py）由训练 CLI 回调经 SidecarHost 执行；`cs_backend` / `cs_skill_book` 测试 fixture 上移到 `tests/scripts/conftest.py` 供转换与回放共用。

### Changed

- 暂时关闭黑魔 Artzip 配置中的 `full_attention_residuals`：Full AttnRes 训练发生 OOM 后触发显存到系统内存的分页，训练耗时从原本约 10 分钟增加到接近 1 小时，第一轮也无法完成；待 Block AttnRes 或更低显存方案验证后再重新启用。
- 清理注意力热图渲染内部接口：`_attention_color_norm` 固定使用 0~1 线性色阶并移除无效动态限值参数，同时删除矩阵热图中已由形状校验保证冗余的 `np.broadcast_to`。
- `scripts/model_analysis` 的开场候选注意力图 `06_opener_attention_candidates.png` 与 `07_opener_attention_by_layer.png` 统一使用 `magma` 线性色阶和 0~1 逐行最大值归一化；移除旧的蓝红均匀基线差值渲染，colorbar 改为明确表示 row-relative attention。
- `scripts/model_analysis` 的完整注意力矩阵与最后一层逐 head 热图改用线性色阶，并按每个 query 行的最大有效权重独立归一化；masked 区域继续灰显，colorbar 明确标注相对权重，便于观察不同 query 行的注意力分布。角色块汇总仍保留原始注意力权重语义。
- 模型注意力正式调整为共享权重的 split attention：scene/history 组成严格因果的 prefix，推理时缓存各层 KV；当前候选 `C1...C28` 只保留一份，候选通过标准非因果 SDPA 同时读取 prefix KV 与全部候选 KV，CLS 单独聚合且候选不读取 CLS。移除 `C1...C28; C28'...C1'` 双遍布局，职业 YAML 的人工 canonical 顺序包含 `ogcd_wait`，训练阶段仍可选随机打乱，在线回放和 ONNX 始终按单份人工顺序对齐；该结构变更需要重新训练 checkpoint，compiled cache 无需重建。
- `Combat.Sim` precision 错误消息 null 兜底修复：`Convert.ToString(object, IFormatProvider)` 对 null 返回空串而非 null，原 `?? "None"` 兜底失效，precision.yaml 显式 null 标量（`float_dtype: ~`）下错误消息恢复为 `float_dtype=None is invalid`（对照 Python `{raw_value!r}` 的 None 显示）；新增 null alias 错误消息测试锁定该路径，累计 175 项 xUnit 测试全绿。
- `Combat.Sim` 保真度与一致性收尾（审核备注闭环）：黑魔满 UI 回蓝由字面量 10000 改为 `nextState.MaxMp`（对照 Python `next_state.max_mp`），`CurrentGcdDuration` 的 `_project!` 空包 NRE 统一为与 `Job`/`System` 一致的未 Bind 守卫；`JobMachineRegistry` 反射自动发现与 `Register&lt;T&gt;()` 共用同一写路径，自动发现后再手动注册同一类型幂等返回（修复前因 `_factories` 占用而抛错）；`PrecisionConfigLoader` 对齐 Python：`precision` 键缺省走默认值（对照 `raw.get("precision", {})`）、dtype 别名表支持 `fp32` / `fp16` / `bf16` / `float16` / `bfloat16`（大小写不敏感）并移除 Python 同样不支持的 `float64`、错误消息改为与 Python 同构；`TensorDtype` 新增 `Float16` / `Bfloat16`，`TensorPrecisionAdapter` 支持 float16（.NET Half）与 bfloat16（double 单次舍入，对照 torch 直接截尾数到 7 位）量化，323 个边界/随机值与 `torch.tensor(dtype=torch.bfloat16).tolist()` 逐位一致；`unsupported output mode` 支持列表顺序对齐 Python（`gcd, seconds`）；`FormatTensorState` 构造注释声明 `projectRoot` 前置条件；新增 `OutputsPrecisionTests`（4 项）与注册幂等对称测试，累计 173 项 xUnit 测试全绿。
- `Combat.Sim` 修复 `JobMachineRegistry.Register&lt;T&gt;()` 的互斥回归：`_types` 检查后补回 `_factories` 占用检查，工厂先占标签后类型注册同样抛 `duplicate job state machine tag`，不再静默覆盖工厂；类型/工厂互斥恢复对称，并新增"工厂先占标签后类型注册抛错"测试锁定该方向（测试职业独立实现接口，避免派生类静态成员不参与接口 `static virtual` 解析的陷阱）；注册表并发边界与 `SceneWindow` 缓存 key 不可变约束写入注释；109 项测试全绿。
- `Combat.Sim` 职业注册与索引缓存并发安全收尾：`IJobStateMachine.Tag` 改为静态成员（`static virtual`，实现类必须提供），`JobMachineRegistry.Register&lt;T&gt;()` 注册路径完全零实例化（幂等/冲突判断直接走静态 Tag，只有 `Get` 才构造实例），并新增"注册路径不构造实例"测试锁定；注册表全局 tag 空间约束写入类注释；`SceneWindow.IndexCache` 由 `Dictionary` 换成 `ConcurrentDictionary`（对齐 Python `lru_cache` 内部锁的并发安全）；108 项测试全绿。
- `Combat.Sim` 职业注册中心 `JobMachineRegistry.Register&lt;T&gt;()` 修正为同一类型重复注册幂等（对齐 Python `register_job_state_machine` 语义），不同类型或工厂占用同一标签才抛 `duplicate job state machine tag`；`IJobStateMachine.AdvanceTimeBeforeSystem` 注释记录 `targetable_at` 场景查询回调的迁移边界（scene 层迁移时扩展签名）；新增 4 项注册中心测试（幂等/冲突/工厂占用/查询排序）。
- `Combat.Sim` YAML 标量解析对前导零整数显式拒绝（`010` / `-010` / `00` 抛 `ambiguous leading-zero integer`），不再静默按十进制解析——PyYAML（YAML 1.1）会把 `010` 解析为八进制 8，静默错值会导致两套引擎跑出不同数值；`0x`/`0b`/下划线/`.inf`/`.nan` 保持字符串（后续数值转换显式抛错），并新增 6 项边界测试锁定行为。
- 自回归回放改为单一事件驱动时间线：在 GCD 读条结束、oGCD 动画锁结束、主动结束 weave 后的 GCD 窗口结束及 scene 状态边界重新调用模型；删除 `legacy_gcd_remaining` 与 `AUTOREGRESSIVE_REPLAY_TIMING_MODE`，`max_steps` 只限制实际执行动作，等待和边界唤醒由独立安全上限保护。正式空 scene parity 的动作预算按 `4 × max_gcds + 32` 动态派生，128 GCD 默认使用 544，给强制首步、多次 weave 和额外调度留出安全余量。
- 自回归 Markdown 将宿主控制动作 `ogcd_wait` 统一显示为 `-`，包括动作列和 Top-K 候选；模型词表、历史、重复惩罚与 ONNX 输入语义保持不变。
- ONNX BF16 正式运行矩阵迁移到 CUDA 13.x：PyTorch 固定为 `2.12.0+cu132`，Python/.NET ONNX Runtime 同步固定为 `1.27.0`，ONNX `1.22.0` 与 ONNXScript `0.7.1` 暂时保持不变；旧 cu126/ORT 1.26 部署包必须重新导出并重跑发布门禁。严格 CUDA 小模型测试改为引用统一版本常量，避免未来升级时遗漏同步字面量。
- 整理 ONNX 导出内部职责：新增独立 `artifact_io.py` 统一负责 SHA-256、确定性 golden NPZ 和 PyTorch 导出报告归一化，部署 manifest 校验不再反向导入 exporter；`precision.py` 改为单一精度规格表，集中派生 PyTorch、ORT、ONNX TensorProto、manifest dtype 与数值容差，并使用 ONNX 官方常量替代数值魔数。稳定全遮罩 softmax 同时升级为公开 adapter 契约，相关测试不再导入私有实现。
- ONNX 导出与回放统一以根目录 `.env` 为协作配置入口：`AUTOREGRESSIVE_REPLAY_CHECKPOINT`、`AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE`、ORT Provider 和实战 scene 在两条链路间复用，部署包留空时自动把 `artifacts/checkpoints/<job>/<run>/*.pt` 映射到 `artifacts/exports/<job>/<run>/`；导出、空场景门禁和实战门禁的其余参数分别由 `ONNX_EXPORT_*` / `ONNX_PARITY_*` 管理。
- ONNX 正式部署精度切换为 BF16 主线：导出 CLI 默认 `bf16` + CUDA，manifest/contract 升级为 v3 并声明 `tensor(bfloat16)`；ORT 导出验收与回放通过 DLPack/I/O Binding 保留 CUDA BF16 输入输出，不再经过 NumPy 或静默提升 FP32。BF16 golden 以可审计的 `uint16` 原始位模式保存，真实 checkpoint padding、ORT 矩阵和 rollout parity 使用 `max_abs_diff <= 0.25` 的 BF16 专属门槛，并继续强制 Top-1、Top-3 和宿主最终动作一致。
- ONNX 发布闭环现在把 parity 报告绑定到 checkpoint、model、manifest 和部署 contract 的 SHA-256；正式空 scene 128 GCD 与真实 scene 100 决策两项都通过后，部署包自动晋升为 `release_validated`，普通推理无需额外发布参数。失败报告同样原子登记到包内并标记 `parity_failed`，模型或 manifest 变化会使旧证明自动失效；正式运行库固定为 Python `onnxruntime-gpu==1.27.0` 与 .NET `Microsoft.ML.OnnxRuntime.Gpu==1.27.0`、`CUDAExecutionProvider`。
- 重新对当前 `best.pt` 与 BF16 ONNX 包执行完整门禁：空 scene 128 GCD 共 147 次决策，最终动作一致率 `93.20%`、Top-3 集合一致率 `87.76%`、最大 logits 误差 `1.31640625`；M5S 真实 scene 100 次决策最终动作一致率 `95%`、Top-3 集合一致率 `89%`、最大误差 `2.6796875`，首个动作分叉在决策 6。两项均未达到既定门槛，因此当前包保持可运行但发布状态为 `parity_failed`，取代旧的第 31 步短路结论。
- 扩充自回归回放 README：逐项记录全部 CLI 参数、对应 `.env`、默认值、后端适用范围、history/device/KV-cache 特殊语义和 PT/ORT parity 组合；明确推荐 ONNX 命令严格使用 `CUDAExecutionProvider`，不把 `auto` 当作正式 GPU 推理入口。
- 记录 ORT CUDA 的 `ScatterND(reduction="none")` 通用警告：正式黑魔 ONNX 图中的 9 组索引均来自固定容量 valid/attention mask 切片，已验证每组常量索引无重复；该提示不表示 CPU fallback，也不需要修改模型图，实际 EP 以回放报告为准。
- 收紧低风险验收边界：零决策 parity 报告不再把 `passed` 或动作序列误标为通过；真实 checkpoint 慢速导出测试从 profile/checkpoint 读取权威容量；回放 Markdown 元数据不再依赖固定下标插入；畸形 `manifest_version` 统一转换为 `ValueError`。README 同时明确当前 ORT 回放 CLI 仍依赖仓库训练配置，不属于可脱离仓库分发的纯部署宿主。
- ONNX manifest 现在保存 checkpoint 的规范化重复惩罚配置；部署图继续只输出 `raw_logits`，重复惩罚与合法候选屏蔽由两个回放 backend 共用的宿主后处理执行，避免把 Python 字符串策略重新耦合进 ONNX 图。
- ONNX 导出和回放默认严格使用 `CUDAExecutionProvider`，不再因 CUDA 环境不完整而静默执行 CPU 模型；CUDA EP 显式关闭 TF32，Markdown 回放结果会记录实际 Execution Provider、模型来源、PPG 和 backend 延迟/内存指标。FP32/FP16 仅保留为兼容实验精度，不作为当前正式发布口径。
- ONNX 部署 manifest 升级为 v3；旧 v1/v2 部署包必须重新导出。显式非 CPU ORT provider 通过 session 配置禁止节点静默回落 CPU，导出报告分别记录请求链、ORT 注册链与 fallback 状态；BF16 必须显式使用 CUDA EP 和 CUDA PyTorch 参考，FP16 导出仍拒绝使用 CPU EP 验证。PyTorch 多份导出报告会合并为单一审计文件，initializer 体量比例保留为审计信息而不再作为依赖导出器常量折叠行为的独立失败阈值。
- 固化首版黑魔 ONNX 输入契约为 `batch=1`、`scene_capacity=160`、`history_capacity=768` 和右侧 padding；scene 容量在首次语料分析后写入稳定职业部署 profile，正常导出不扫描 raw/compiled cache，history 容量只从 checkpoint 的 `run_config.max_history` 读取并写入 manifest，删除 `.env` 与 CLI 中重复的正式容量定义；position id 在 scene/history/candidate 各语义段分别从 0 起算，超容量输入显式拒绝而不静默截断。
- scene 类型投影改为固定计算全部类型后通过 Tensor `gather` 选择，移除依赖输入值的 Python 分支；模型参数名、shape 和 checkpoint 契约不变，现有 checkpoint 无需重新训练。
- `scripts/autoregressive_replay` 新增 `AUTOREGRESSIVE_REPLAY_TOP_P` / `--top-p` 核采样参数；温度大于 `0` 时在屏蔽非法候选并完成 temperature softmax 后保留累计概率达到阈值的最小候选集合，再重新归一化采样。默认 `1.0` 不截断，根目录 `.env.example`、README 和专项测试同步更新。
- 按职责整理 `training/models/common`：将配置与 `DataSpec`、模型结构、运行时诊断、训练数据/循环分别归入 `config/`、`model/`、`runtime/`、`training/`，保留公共导出与旧训练测试注入边界。
- 优化 FFLogs scene targetable 查询：合并当前窗口与下一次 downtime 的 token 遍历，减少重复扫描。
- 统一模型分析图表配色：连续数值图、hidden/PCA/embedding、角色图和 Attention 角色块统一使用黑色到白金的 `magma` 色阶；保留需要表达正负基线差异的发散色阶。
- 模型分析新增完整 Attention 矩阵诊断：输出逐层 head-mean、最后一层逐 head 和 scene/history/candidate/CLS 角色块汇总；mask 区域灰显，并使用幂律色阶增强低权重区域的可读性，新增 PNG 路径同步写入分析 metadata。
- 新增可选 `training.runtime_debug` 运行时显存调试，按 batch 搬运、输入编码、每层 Attention/FFN、scorer/loss、反向、梯度裁剪和优化器阶段记录显存峰值、耗时与 batch 形状，并输出 JSONL 原始及聚合数据；默认关闭。
- 补充 Transformer 层级显存调试埋点、配置解析和训练运行时测试覆盖。
- 新增 `training.activation_checkpoint_attention`，训练时对 Transformer Attention 中间激活做 checkpoint 并在反向传播阶段重算；当前黑魔 Artzip 配置已开启该选项以降低长历史训练峰值显存。
- 训练模型新增 `training.activation_checkpoint_ffn` 开关；开启后仅在训练反向传播阶段对 Transformer FFN 的中间激活做 checkpoint 并重算，验证和推理不启用，当前黑魔 Artzip 配置已开启该选项以降低长历史训练的峰值显存。
- 训练入口新增 `--resume` 断点续训参数，可从已有 checkpoint 的下一轮继续训练；同时保存并恢复 optimizer、scheduler、best 指标和随机状态，旧 checkpoint 缺少 scheduler/随机状态时按已完成 batch 和当前指标安全回退。
- 训练 checkpoint 新增自描述 `input_contract`，保存实际归一化配置、职业资源/Buff 上限、state/scene schema 和字段布局；自回归部署与缺失 cache 重建直接从 checkpoint 恢复归一化规则，不再依赖外部重新扫描 YAML。
- 拆分 FFLogs compiled cache 的 raw 路径选择、编译编排和缓存加载职责，保留 `scripts.convert_fflogs.cache` 兼容门面，降低缓存格式变更的回归范围。
- 拆分公共训练层的 DataLoader、checkpoint 生命周期和训练循环职责，保留 `training.models.common.training` 兼容门面，并同步迁移相关测试注入点。
- 模型注意力改为完全因果：结构化分块 mask（scene 双向、history 块内因果、candidate 双向、CLS 汇总）替换为单一完整下三角 mask，scene/history/candidate/CLS 全序列严格因果；position id 从每块内部从 0 起算改为全局递增连续编号，role/segment embedding 保留。模型行为改变，旧 checkpoint 必须重新训练；mask 与 position id 由模型在运行时按序列长度生成，训练集 compiled cache 无需重建。
- KV-Cache 推理注意力改用标准 SDPA kernel：前缀追加从逐 token 逐层循环改为逐层批量推进，additive mask 采用 `(B, H, N, L)` 形状（前缀全部可见 + 当前块内部严格因果），bf16 CUDA 下命中 memory-efficient / cuDNN 标准 kernel；训练路径经 `nn.MultiheadAttention` 在 bf16 CUDA 下自动调度到 cuDNN fused causal kernel。
- ONNX 部署契约与 manifest 升级为 v4：position id 语义声明为全局递增；验收体系从"动态长度与固定容量逐 token 等价"改为"同容量 parity + padding 屏蔽 + padding 值不变性"，动态侧只验证有限性与同输入一致性；旧 v3 部署包被 manifest 版本校验显式拒绝，需重新导出。
- parity 双跑统一在固定容量输入上进行（张量放在 PT reference 的 device），与 ORT 部署图的容量语义一致。
- 回放测试从单个臃肿文件拆分为 `tests/scripts/autoregressive_replay/` 子目录：按 config / context / replay / parity / ppg 主题拆分，公共 dotenv 隔离与文件构造 fixture 移入 conftest；`test_autoregressive_ppg.py` 一并移入同目录。
- 统一 scene 容量权威来源：`ModelConfig` 新增 `scene_capacity` 字段并在职业模型配置中显式声明，随 checkpoint 的 `model_config` 进入 ONNX 部署契约；部署 profile 不再维护 scene 容量，只保留技能 vocab 映射与语料统计证据，导出时校验配置容量必须 ≥ 语料实测 `scene_length_max`；旧 checkpoint 缺失该字段时与训练路径恢复逻辑一致回退默认值。
- parity 固定容量输入改为直接构造在 reference 的 device 上，消除 CUDA 下"先构造 CPU 再搬回 GPU"的重复搬运；KV-cache 的全屏蔽行检查同步上提到每决策一次，回放路径（前缀全有效）不再每层注意力各做一次 CUDA→CPU 同步，含 padding 时原置零保护逻辑不变。
- 黑魔 Artzip 实验模型配置调整：`pair_embedding_dim` 384→256（模型结构变化，旧 checkpoint 权重形状不匹配，必须重新训练）、`max_position_embeddings` 2048→1024、`scene_capacity` 声明为 200；1024 与 scene(200)+history(768)+candidate(28)+CLS=997 仅余 27 token，后续 scene 容量或 max_history 上调需同步加大 `max_position_embeddings`。
- 生产代码的 Python 状态机实例依赖清零：`main.py` 命令行入口切换为 C# 状态机后端（validate / list-skills 为纯配置与静态技能索引，list-actions / smoke 经 SidecarHost 驱动），未迁移的机工职业保留 Python 权威状态机例外路径；`tests/training/_common_fixtures.py` 的演示动作生成与训练样本构建全部改为 C# 后端驱动（构建样本前重新 init 清空动作生成产生的历史），训练测试数据生成不再实例化 Python 状态机。
- 共享 schema 定义单一事实来源：新增 `config/schema.yaml`，承载共享常量（滑步窗口/绝对时间模式/scene context keys）、scene context 窗口字段模板与 segment kinds、canonical 输出顶层 keys 与状态向量分组、状态向量字段模板（player 23 字段/buff/target_buff）；C#（`SchemaConfigLoader`，状态机构造与 CliHost/测试入口装配）与 Python（`common/schema_config.py`）都从它加载，修改定义只改一处，动态部分（职业量谱/Buff/DoT 列表）仍由运行时推导；公共层（contracts/scene_window/models/config/skills/gcd_utils/scene_context_schema/output_context_schema）物理上移到 `common/`，combat_sim 原位置保留 re-export 兼容层，`registered_job_tags` 改为配置驱动（`resolve_registered_job_tags`）；golden 六段双跑逐字段一致。
- 共享 schema 消费闭环（MR 审核收尾）：`SchemaConfigLoader.Load` 加锁消除并行竞态，并新增 `RequireStateVectorFields` 严格字段组访问（schema 缺失字段组显式报错，不再静默返回空数组）；C# 玩家/Buff/目标 Buff 三个向量 builder 改为按 schema 字段逐项动态装配（未知字段 fail-loud），Python 侧三个对应 builder 同步动态化，两侧以 schema.yaml 为唯一字段顺序权威；`RaidBuffWindowFeatureKeysFor` 改为消费 `window_extra_fields.raid_buff` 的 `{key}` 占位模板（对照 Python `field.format(key=...)`），模板改动 C# 同步跟随；`state_vector_groups.target_buff_state` 补充与 `state_vector_fields` 字段组的按序拼接注释；新增 8 项 schema 加载与模板展开测试（含「未 Load 时访问抛友好错误而非类型初始化异常」的行为锁定，测试程序集禁用并行以隔离单例状态重置），累计 183 项 xUnit 测试全绿，golden 六段双跑逐字段一致。
- 共享 schema 并发与一致性收尾（第二轮 MR 审核）：`SchemaConfigLoader._instance` 声明为 `volatile`，读端（`Instance` / `RequireStateVectorFields`）在跨 root 重载窗口不再撕裂读或抛空值；`SceneContextSchema` 窗口额外字段组缺失改为显式报错（与 `RequireStateVectorFields` fail-loud 哲学一致）；Python 侧 Buff/目标 Buff 向量 builder 的字段取值改为未知字段恒抛 KeyError（与 C# 恒抛对称，不再在 buff/dot 为空时对未知字段静默返回 0.0）；`PlayerVectorTokenBuilder` 字段数组改为构造时一次性缓存（热路径不再每次查 schema 表）；`CombatStateMachine` 构造器 XML doc 明确省略 `projectRoot` 时要求可执行文件位于仓库目录树内（游戏插件等独立嵌入场景必须显式传 `projectRoot`）；新增同根重复加载幂等与跨根重载后 Instance 语义两项确定性测试；累计 185 项 xUnit 测试全绿，golden 六段双跑逐字段一致。
- 共享 schema 字段装配一致性收尾（第三轮 MR 审核）：`BuffVectorTokenBuilder` / `TargetBuffVectorTokenBuilder` 的字段模板从 static 属性改为构造时一次性缓存的实例字段，Python `target_buff_vector_token_builder` 同步构造时缓存字段组（`_build_vector` 不再每次调 `load_schema_config()`），热路径消除重复查表；Python 侧新增「未知 schema 字段恒抛 KeyError」行为锁定测试（直接调用 player/buff/dot/target 四个字段取值器断言 KeyError，与 C# 恒抛对称，schema 增删字段未同步实现取值器时立即失败）；`config/schema.yaml` 头部补充字段联动维护说明（新增字段必须同步实现 C# switch 臂与 Python 取值器，否则运行期显式报错）；跨根重载测试的临时目录清理改为嵌套 finally（恢复 Load 失败也不泄漏）；测试程序集串行注释数字同步为 185；累计 185 项 xUnit 测试全绿，Python 441 项全绿，golden 六段双跑逐字段一致。

### Fixed

- 修复 `python -m scripts.model_analysis` 在增量 history bank 样本上选择空历史样本作为完整 attention 图代表样本的问题：代表样本选择现在读取 `history_length`，并补充回归测试，确保 `history_pair` 正确出现在分析图中。

- 修复 Full AttnRes checkpoint 判定使用 `any()` 导致部分层配置被整体化的问题；只有所有层的 FFN/Attention checkpoint 配置一致时才启用外层 checkpoint，否则保留各层独立的 checkpoint 粒度。
- 修复 Full AttnRes 整段 checkpoint 与层内 Attention/FFN activation checkpoint 嵌套导致的重复重算；外层 checkpoint 的前向和反向重算期间会临时跳过层内 checkpoint，并新增 SDPA 调用次数回归断言。
- 修复 `split_encoder.finish_layer()` 对 PyTorch `TransformerEncoderLayer._ff_block()` 的 FFN 输出重复应用 `dropout2` 的问题；训练态现在保持与原生 Transformer 语义一致，并新增非零 dropout 的 pre-LN / post-LN 回归测试。
- 修复 `python -m scripts.model_analysis` 未读取职业 YAML `candidate_order_file` 的问题；分析数据现在与 checkpoint 使用同一份人工候选顺序，并增强 `DataSpec` 不匹配错误以列出具体差异字段。
- 修复候选顺序严格全等校验的连锁诊断问题：`candidate_permutation` 现在报告 `missing` / `unknown` 技能差集；live 回放在候选重排前明确拒绝空候选；candidate order 配置注释与严格全等实现一致，并提示候选集合变化后更新 YAML、频率变化后重新统计排序；新增相关回归测试。
- 修复事件驱动 ONNX parity 可能在达到空 scene 128 GCD 前耗尽动作预算的问题：正式门禁现在于昂贵回放前统一校验 GCD 目标与动作额度；训练转换、训练同步和自回归 scene 调度同时复用同一 0.5 秒滑步窗口契约，战斗结束后的无合法动作也会给出明确终局原因。
- 修复自回归回放在每个动作后直接跳完整 GCD、导致每个 GCD 最多只能执行一个 oGCD，以及无合法动作时直接终止的问题；现在按动作实际完成时间推进，空输出等待可被 scene 边界中断，无合法动作会继续推进到最近的锁、GCD、冷却、状态、DoT、回蓝或 scene 恢复事件。
- 修复 cache scene 只作为模型 tensor 输入、却没有完整同步到回放状态机的问题；同一份绝对时间 scene 时间线现在按训练口径同步 Boss 可选中、强制移动、团辅窗口和目标数，并把对应边界纳入事件调度，保证候选合法性、状态向量和模型 scene 输入使用同一语义。
- BF16 导出现在先校验显式 CUDA provider 与验证设备，再检查完整依赖版本矩阵，避免 Torch/ORT 版本不匹配掩盖 `auto` 或 CPU provider 本身不受支持的参数错误。
- 修复无参数自回归回放在根目录 `.env` 加载前解析 backend、导致 `AUTOREGRESSIVE_REPLAY_BACKEND=onnxruntime` 被忽略并错误回退到 PyTorch 的问题；现在 backend、ONNX 部署包和 scene 模式会在同一次 dotenv 装配后解析，并新增仅依赖 `.env` 启动的回归测试。
- 统一 ONNX 环境失败和直接导出边界：完整 workflow 的环境检查失败现在以中文摘要和退出码 `1` fail-fast；BF16 的 Torch/ONNX/ONNXScript/ORT 固定版本矩阵校验下沉到 exporter，使无参数 CLI、参数化 CLI 与 Python API 都无法绕过，并把 ONNXScript 版本写入新部署 manifest 及其 JSON Schema。
- 固化 v1 legacy ONNX 发布证明的降级语义：正式加载优先明确拒绝并要求重新执行显式门禁，审计加载则继续校验证据哈希后降级为 `parity_in_progress`；补充有效证据、缺失证据和两种加载模式的直接回归测试。
- 加固 ONNX 发布工作流的低风险边界：`all` 导出失败时以中文摘要和退出码 `1` fail-fast；BF16 导出前严格核对 Torch/ONNX/ONNXScript/ORT 固定版本矩阵；正式 empty/scene 门禁共用集中阈值并在昂贵回放前拒绝不足的 GCD/动作数，同时补充 scene-first 对称晋升测试。
- 修复 ONNX 发布门禁可被普通或宽松容差 parity 运行满足、以及已有通过证据会被后续失败调试覆盖的问题：普通 `--parity-onnx-package` 现在只生成外部审计报告，只有显式 `scripts.onnx_export.workflow` 能登记正式证据；正式报告绑定门禁版本、manifest 精度和固定容差，发布状态会从证据重新计算，旧 v1 证明需重新执行正式门禁，已通过证据不会被失败重跑覆盖。
- 修复 ONNX 发布状态同时改写 `release_report.json` 与 `export_report.json`、无法形成单一原子提交点的问题：发布报告升级为 v3，并以 `release_report.json` 作为唯一当前状态源；`export_report.json` 保持导出验证快照，parity 证据改用内容哈希不可变文件名，被替换或 v1 迁移的旧证据显式登记到 `superseded_parity_reports`。新增提交中断回归，保证状态写入失败时旧发布证明和证据仍可验证。
- 完善 ONNX 发布报告兼容与历史审计边界：新增 v2 `release_report` 的 `release_validated` / `parity_failed` 双状态合成回归，并覆盖 v2 携带 `superseded_parity_reports` 的完整历史审计；可捕获的发布提交失败会清理本次未登记的内容寻址 parity 文件，同内容证据已存在时则不会误删。异常附注通过可选 `add_note` 调用兼容 Python 3.10，不支持该接口时改写 stderr，避免掩盖原始提交异常；普通 ORT 加载只哈希当前有效证据，并以深拷贝返回不含未经复核 superseded 条目的运行时视图，正式登记与 workflow `verify` 才完整校验并返回全部历史，避免历史证据线性增长拖慢每次 session 初始化。
- 收口 ONNX/scene 低风险维护项：parity 环境变量测试改用满足正式动作预算的 129 GCD/548 steps 组合；全仓确认无调用方后删除 `_find_next_window_start` 及其兼容导出，避免未来误用已废弃的查询入口。
- 完善 parity 失败诊断：逐决策报告现在保存 PyTorch/ORT 各自按分数排序的 Top-3 候选，失败摘要明确列出数值阈值、Top-1、Top-3 集合和宿主最终动作各项结果；修复空 scene 首次失败只有 Top-3 不一致时，终端摘要却只显示双方相同 Top-1/最终动作而无法解释失败原因的问题。
- 收紧自回归 parity 与低精度运行边界：`--parity-tolerance` 现在拒绝 `NaN`、正负无穷和负数，避免无穷容差绕过 raw logits 数值门；FP32/FP16/BF16 默认容差由专项测试锁定。PyTorch 回放新增显式 `--precision` 覆盖，可通过 `--device cpu --precision float32` 分析 BF16 checkpoint，同时 ONNX 后端和 parity 仍以 manifest 精度为唯一权威。补充 ORT CPU I/O Binding、bool NumPy 回退和 CUDA BF16 golden 字节确定性的职责级测试。
- 修复 parity 首个不一致直接丢失报告、`first_divergence`/动作一致字段不可达或硬编码的问题；失败时保留逐决策 logits、候选、动作与 partial metrics 后再非零退出。修复 parity 忽略 `.env` ORT provider、未达到 `--max-gcds` 仍成功、ORT Markdown 输出 `checkpoint: None`，以及 ONNX/PyTorch backend 使用不同 raw/cache 分片配置的问题。
- 修复部署 profile 默认路径可被不可信 `job_tag` 穿越、`padding_fill=zero` 未清零离散 padding 载荷，以及显式 CUDA session 的报告可能与实际 active EP 不一致的问题。
- 修复空 scene 或固定容量 padding 产生全遮罩 attention 行时的非有限值：稳定 masked softmax 让全遮罩行输出零权重，并保证所有层的候选与 CLS 都不能读取 padding key；真实 46.9M 参数 checkpoint 已覆盖 CPU/CUDA 的空、满和双侧临界组合，动态长度与固定容量 raw logits、argmax 和随机 padding 值保持一致。
- 修复 scene 投影的 `if type_mask.any()` 和模型前向字符串策略阻塞稳定导出图的问题；独立 adapter 仅暴露 12 个 Tensor 输入并绕开 `nn.TransformerEncoder` 的 mask 数据探测，`torch.export.export()` 可完整捕获无字符串候选打分图。
- 修复训练续训、模型分析、自回归回放和 compiled cache 使用不安全 pickle 加载的问题：统一通过安全的 `weights_only=True` 入口读取，仅允许必要的路径和 schema 类型；compiled cache 改用安全 unpickler 兼容的 pickle protocol 2，旧缓存可从 raw JSON 重建。
- 修复 FFLogs 单报告下载的 `--report` 文件名路径遍历风险，报告编号现在仅允许字母和数字。
- 加固 FFLogs 批量下载文件名：batch 路径复用报告编号校验并过滤角色名中的 `/`、`\\` 和空格；同时忽略非数字的 URL `source` 参数，避免未捕获的整数转换异常。
- 为 FFLogs 事件分页增加 `MAX_EVENTS=100_000` 累计上限，达到上限后截断当前页并停止请求，避免异常分页导致内存无界增长。
- 校验 FFLogs 排行查询的 `spec_name` / `class_name` 仅允许字母和数字，避免特殊字符破坏或注入 GraphQL 查询。
- 完善 FFLogs 排行查询 GraphQL 参数防护：限制 `metric` 白名单，并严格校验 `encounter_id`、`bracket` 和 `page` 为安全整数。
- 补充 FFLogs 事件查询 `fight_ids` 的逐项整数校验，避免列表参数注入 GraphQL；同时修正分页截断日志，使其记录实际追加的事件数量。

- 修复标准 Attention 分析代表样本选择对非映射样本静默退化的问题；现在会显式校验样本结构和 token 字段长度，避免错误选中短上下文样本。
- 修正 FFN checkpoint 的实现与测试：委托 PyTorch `TransformerEncoderLayer` 的父类 FFN，覆盖带 dropout 的前向/反向一致性，并验证 checkpoint 会在反向阶段重算 FFN。
- 补充断点续训的 checkpoint、scheduler、best 指标和随机状态恢复测试；CUDA RNG 状态在恢复时校验 checkpoint 与当前 GPU 数量，避免设备数量变化导致隐式失败。
- 修复 parity 固定容量 batch 被强制搬到 CPU、CUDA 环境下 PT reference 收到 CPU 输入直接 device mismatch 崩溃的问题；固定容量张量现在统一放到 reference 的 device，ORT 侧仍自行搬回 CPU。
- 修复 KV-Cache 全屏蔽注意力行在 math 后端 softmax 产生 NaN 的问题：先解禁全屏蔽行避免 NaN，再把对应输出显式置零，与导出路径 `stable_masked_softmax` 的行为一致。
- 统一 parity 契约守卫判断，并补齐动态路径序列长度守卫：scene + history + candidate + CLS 总长超过 `max_position_embeddings` 时显式报错，不再等位置 embedding 查表越界才失败；当前黑魔 160+768+28+1=957 满足 2048 上限。
- 自回归回放配置的设备解析改为复用公共 `resolve_training_device()`，不再直接读取 `TRAINING_DEVICE` 环境变量，与训练/转换/分析入口共用同一套 dotenv 加载与 cuda/cpu 校验。
- 回放测试 conftest 移除不生效的 `run_config.max_history` 字段：`load_replay_config` 的 max_history 以职业 YAML 为准，checkpoint 该字段不参与解析，删除误导性配置。
- 修复 C# 后端转换链路中 ogcd_wait 检查推进后 Python 镜像未同步导致的量谱计时漂移：`_build_injected_ogcd_wait_action` 预推进决策下限后只更新局部变量，主循环镜像 state 保持旧时间，后续 `advance_state_to_time` 的 delta 按旧镜像计算而多推进 0.05s（action_effect_settle_seconds），polyglot_timer 随之每决策多累积 0.05s；现在该函数返回推进后的决策状态并由主循环同步，新旧链路 483 样本 label 与 context 逐字段一致。
- 修复 MR 审核发现的转换链路两个边界问题：滑步修正路径 `_reinject_scene` 从 sync 前的陈旧镜像读 `raid_buff_window` 回灌，团辅窗口刚生效/刚过期时会清除或复活后端窗口状态导致威力倍率丢失或放大，滑步分支现在先同步后端权威镜像再注入；`_find_candidate_index` 恢复 fail-loud 语义，候选集缺失目标动作（含注入的 ogcd_wait）时抛错而不是静默返回 0 错标训练标签；新增 3 项回归测试锁定（还原修复可复现失败）。
- 修复 `SidecarBackend` 构造失败泄漏 SidecarHost 子进程的问题：init 被后端拒绝（如不支持的 job_tag）或进程启动即崩溃时，异常从 `__init__` 直接传播而 `_proc` 从未回收，并行转换下每个失败文件都会泄漏一个孤儿进程；现在构造失败路径先 `close()` 回收子进程再抛出，`close()` 超时 `kill()` 后补 `wait()` 完整回收，并新增 4 项进程生命周期测试（还原修复可复现泄漏）。
- 加固 `SidecarBackend` 构造失败清理路径：`close()` 自身抛意外异常时不再覆盖原始构造异常（清理是尽力而为），并新增异常遮蔽回归测试（去掉隔离可复现遮蔽）。
- 修复 `tests/training/_common_fixtures.py` 在转换链路切换 C# 后端后仍用旧签名调用 `build_training_samples(machine, ...)` 的问题：改为经 `SidecarBackend` 构建训练样本（动作序列仍由 Python 权威状态机生成），训练公共层测试恢复全绿。
- `SidecarBackend` 构造失败清理路径不再静默丢弃 close 异常细节：被抑制的清理异常通过 `logger.warning` 记录类型与消息（仍不覆盖原始构造异常），并新增 caplog 回归测试。
- 统一 SidecarHost 依赖测试守卫与加固回放边界：`tests/training/_common_fixtures.py` 的样本构建复用 `tests/scripts/conftest.py` 的 `_require_sidecar_host()`，SidecarHost 未构建时训练测试组 skip 而非全红（报错信息可操作）；SidecarHost `step` 的 `history_mode` 显式校验 `replay|standard`，非法值不再静默回退（新增拒绝测试）；`replay.py` 缓存一次 `load_project_config` 供 skill_book 与 mp tick 复用，`mp_recovery` 缺失时 mp tick 等待事件安全跳过（未来机工迁移不 AttributeError）。
- 修复 `CombatStateMachine` 无 `projectRoot` 构造路径的输出层回归：此前构造可成功但首次 Step / Format / FormatVectors 抛 `SchemaConfigLoader.Load 必须先调用`（构造函数 XML doc 声称「projectRoot 仅供 tensor 输出，省略时 canonical / 快照输出不受影响」，与实现不符），FacadeGoldenExporter 等直接构造路径依赖 CliHost 顶层 Load 才能工作；现在构造器缺 `projectRoot` 时经新增公共 `Common/RepoRootLocator` 按仓库根兜底加载共享 schema（CliHost 顶层入口同步复用该定位器），并移除 `PlayerVectorTokenBuilder` 在 Load 前首次访问触发 TypeInitializationException 的 static readonly 初始化器（改惰性属性，未 Load 时抛友好错误），新增「未 Load 时行为」测试锁定错误路径。

### Migration

- 完全因果注意力改变模型行为：现有 checkpoint 语义失效，必须重新训练；结构兼容可加载但行为与训练时不一致，不建议续训混用。训练集 compiled cache 无需重建，mask 与 position id 由模型在运行时按序列长度生成，不落盘。
- ONNX 部署契约升级为 v4：旧 v3 部署包被 manifest 版本校验显式拒绝，需按完全因果语义重新导出并重跑发布门禁；BF16 正式环境版本矩阵保持不变（torch 2.12.0+cu132 / ORT 1.27.0 / CUDA EP）。

## [0.0.7] - 2026-07-21

### Changed

- 将历史上下文上限改为职业模型配置的 `training.max_history` 来源；当前 Artzip 使用 `768`，状态机通过注入参数保留对应数量的动作历史，转换 cache 与自回归回放沿用同一配置，不再依赖全局历史长度常量。
- 统一职业路由配置：转换、训练、模型分析、自回归回放和状态机现在都读取根目录 `.env` 的 `FFXIV_JOB_TAG`，删除 `FFLOGS_CONVERT_JOB_TAG`、`CONVERT_FFLOGS_JOB_NAME` 和 `AUTOREGRESSIVE_REPLAY_JOB_TAG` 三个旧环境变量。
- 迁移说明：已有 `.env` 中的 `CONVERT_FFLOGS_JOB_NAME` 不再生效，请删除该字段；转换输出职业目录现在统一由 `FFXIV_JOB_TAG` 决定，不再单独维护落盘目录名。
- 删除旧的单职业配置入口 `runtime.active_job` / `runtime.job_config`；`config/default.yaml` 现在只注册 `runtime.job_configs`，由统一职业标签选择对应的职业 YAML，不再保留旧配置回退解析。
- 统一运行时职业标识为 `job_tag`，并校验训练模型配置路径、checkpoint、scene compiled cache 与 `.env` 职业标签一致，避免不同工具加载不同职业模型或状态机。
- 重构 raw JSON 到最终 compiled cache 的转换边界：`scripts/convert_fflogs` 现在在内存中完成 source 构建并流式写入 `.cache` manifest/shard，raw 目录只读，不再落盘中间训练 PT；单个样本只提取一次历史/候选技能行，技能特征矩阵一次性构造，状态分组避免重复 tensor 拷贝，scene window 按窗口批量构造，raw JSON 可并行编译。
- 编译缓存格式升为 `raw_json_compiled_samples_v6`，不保留旧格式兼容读取；磁盘 shard 使用职业 YAML 的分片大小，Tensor 通过 `mmap` 延迟读取，sampler 保证 batch 不跨 shard，DataLoader 按 batch 批量访问并让同一 shard 只查找一次。
- 编译缓存写盘使用 Pickle protocol 5，降低 shard 和 manifest 的 Python 容器序列化开销；过采样预扫描改为事件流匹配，不再把单个 raw JSON 的全部展开样本同时留在内存。
- 删除旧的中间格式与训练层构建模块：`scripts/convert_fflogs/io.py`、`parallel.py`，`combat_sim/outputs/training_dataset_formatter.py`、`training_pt_contract.py`，以及 `training/common` 中的 cache compiler/manager、PT reader 和转换 sample helper；不保留旧入口或兼容转发。
- 明确跨模块公共函数边界：新增根目录 `common/`，只承载无具体业务归属且被多个顶层模块复用的函数；训练、回放和模型分析统一复用 `torch_runtime.py`，项目配置入口统一复用 `project_config.py`，模块内部 helper 不再上移。
- 优化模型分析 PCA：逐层 PCA 与候选 feature PCA 复用投影结果，CUDA 分析时使用 GPU SVD，减少重复 CPU 计算和大矩阵分析耗时。
- 统一转换与回放的嵌套数值展开逻辑：新增 `common.numeric.flatten_numeric_mapping()`，由调用方传入 `ignored_keys`，移除两处重复实现并保持各模块自行定义字段过滤。
- 将训练启动时的 raw→cache 选择、失败补位和编译编排移入 `scripts/convert_fflogs/cache.py` 的 `prepare_training_caches()`；`training/models/common/training.py` 只消费已准备的 raw/cache 路径，拆除 `training → scripts → training` 的反向依赖。
- 黑魔 Artzip 实验模型配置调整为 `d_model=768`、`pair_embedding_dim=384`、`n_layers=6`、`n_heads=6`、`ff_dim=3072`、`batch_size=16`、`max_epochs=8`、`max_history=768`；仅影响模型/训练运行，不触发 raw JSON compiled cache 重建。
- 训练模型新增 `transformer_norm_first` 与 `transformer_activation` 配置；黑魔 Artzip 当前使用 Pre-LN + GELU，并在 Transformer encoder 末尾增加 LayerNorm，候选 scorer 的激活函数保持独立。
- 统一模型输入内容嵌入：scene、history pair、candidate pair 与 CLS 先经过共享的 `token_embedding`（`Linear + LayerNorm`）再进入 Transformer；旧 checkpoint 需要重新训练。
- 统一 scene 绝对时间归一化：在 `Normalizer` 集中处理 `start/end/duration`，将 `scene_time_max` 更名为 `fight_time_max=1800`，缓存构建时将战斗绝对时间裁剪并线性归一化到 `0.0~1.0`，旧 compiled cache 会因签名变化自动重建。
- 统一职业量谱上限来源：职业 YAML 的 `resources.<key>.max_value` 同时驱动状态机资源写入截断、训练 state/skill 归一化和 compiled cache 签名；资源注册层负责通用装配，当前黑魔已配置 AF/UI、心极、星灵魂、通晓及相关计时/触发资源。
- 黑魔耀星资源不足时改用通用错误码 `insufficient_astral_soul`，避免资源上限调整后仍显示硬编码的六层灵极魂提示。
- 训练归一化的剩余秒数上限改为从对应职业 YAML 的最大技能 `cooldown` 自动解析，并在每个 `Normalizer` 实例中按职业缓存，不再把 `120s` 写死在公共训练 YAML；即时威力归一化上限提升为 `current_potency_max=2500`，减少多目标和职业倍率下的输入饱和。
- 完善训练归一化规则：`fight_remaining_seconds` 与玩家绝对时间统一使用 `fight_time_max=1800`，最大技能冷却同时扫描启用的系统/职业技能，累计威力支持配置的 `cumulative_potency_mode`，`current_gcd_dot_potency` 使用即时威力上限，系统与职业 Buff 的 `stacks` 按各自 YAML 的 `max_stacks` 归一化并纳入 cache 签名。
- scene 的多目标辅助 token 增加 `target_count_max=3` 配置；转换阶段和状态机始终先使用实际目标数量计算 AoE 威力，最终 compiled cache 再把 scene `target_count` 与目标状态威力统一归一化；首拍预读动作按实际生效时刻重新同步目标数量。
- 稳定 scene 上下文：训练样本和自回归回放现在始终携带整场绝对时间 scene token，不再按当前时间重基准、扣除已过时间或自动销毁窗口；模型推理时可直接复用固定 scene/history 前缀的 KV-Cache。
- 将 KV-Cache 嵌入 `CandidateTransformerModel`：训练模式和默认完整 `forward` 不启用缓存，评估/部署通过模型自身的 `enable_kv_cache()` / `reset_kv_cache()` 管理运行时状态，调用方不再持有或传递 K/V 对象；候选技能、候选状态和 CLS 每步重新计算，历史前缀被调用方截断后模型自动重建缓存。
- KV-Cache 只缓存 scene/history 前缀的 K/V；移除无效的 layer output 缓存，候选位置索引不再覆盖完整输入 schema，并统一追加历史与候选路径的 Transformer 残差、归一化和 FFN 后半段逻辑。
- 玩家状态新增 `time_seconds`，使用独立的 `fight_time_max` 归一化；保留实时 `current_gcd_seconds`，技能 token 仅移除 `cast_time.gcds` 与 `gcd_window.gcds` 输出，内部动作历史仍保留 GCD 计算所需字段。
- 多目标 scene token 调整为回放辅助标记：仅从实际双目标命中事件生成 `target_count >= 2` 的绝对时间区间，双目标事件间隔超过 `2s` 重新开始区间，最后一次事件后延长 `1.5s`；不再生成单目标或停机填充 token，也不由标记直接承担 AoE 倍率逻辑。转换训练回放和自回归回放仍按标记同步 `CombatState.target_count`，实际倍率继续由状态机计算。
- 训练验证新增 128 个输出 GCD 的自回归木桩 PPG：从空 scene、满 MP、空初始历史开始执行合法 Top-1 动作，按 `(累计直接威力 + 累计 DoT 威力) / GCD 数` 计算 PPG，并以 `mean(top1, ppg / 1000)` 作为 best 主评分。
- best checkpoint 的决胜顺序统一为综合评分、top1、top3、较低的 `value_preference_loss`；训练额外保存每轮带 PPG 的 checkpoint，例如 `epoch_001_ppg_768.00.pt`。
- 根目录 `.env` 的加载统一使用 `python-dotenv`，移除公共配置模块的自定义解析逻辑，保持转换、抓取和其他工具的解析行为一致。
- 删除转换层的 `RAID_BUFF_WINDOW_DURATION` 硬编码及导出；团辅窗口时长统一以 `config/system.yaml` 的 `raid_buff_window.duration_seconds` 为权威配置。
- 统一状态机配置与 FFLogs 转换配置的 YAML mapping 加载，集中处理 UTF-8 读取、文件缺失和顶层类型校验。
- 统一训练模块的 YAML mapping 加载：公共训练配置、职业运行配置和过采样配置现在复用 `common.yaml_config.load_yaml_mapping()`；过采样配置缺失时仍保持跳过该配置的原有语义。
- 模型分析的向量、role 和 metadata 收集统一使用带显式空形状与 dtype 的数组拼接 helper，移除三份重复实现。

### Added

- 新增脚本层 cache 构建入口 `scripts/convert_fflogs/cache.py`、`cache_writer.py`、`raw_source.py`、`source_reader.py`、`source_helpers.py` 和 `sample_builder.py`；训练启动会调用同一入口补齐最终 cache，`training/common` 只负责只读 cache、Dataset、schema、归一化和 batch 读取。
- 新增 `common/project_config.py`、`common/torch_dependencies.py` 和 `common/torch_runtime.py`，统一提供项目路径/`.env` 解析、PyTorch 延迟导入、batch 搬运、模型 dtype 与 autocast；删除 `combat_sim/config_utils.py`、`combat_sim/torch_utils.py`，不保留旧路径兼容转发。
- 新增 `common/yaml_config.py`，为状态机和转换脚本提供统一的 YAML mapping 加载与顶层类型校验。
- 新增 `common/cache_compilation.py`，统一提供模型分析与自回归回放调用正式转换 CLI 编译 raw JSON cache 的公共入口。
- 新增 raw JSON → compiled cache 测试，验证转换不会修改 raw 文件，并覆盖 cache manifest、shard、worker 和训练 Dataset 读取链路。
- 补充 `training/models/common` 公共模型与训练流程覆盖率测试：覆盖配置错误、模型输入校验、DataLoader 分支、训练循环、职业路由防护、价值辅助损失和 checkpoint 落盘；`training.py` 单模块覆盖率达到 100%。
- 补充 `scripts/autoregressive_replay` 与 `scripts/model_analysis` 测试：覆盖回放配置/scene 模板/live batch、温度采样、历史消融、Markdown/CLI 输出，以及 token 元数据、职业标签、PCA、hidden/attention/pair/skill 图表和分析 metadata。
- 技能 YAML 与 `SkillDefinition` 新增 `guaranteed_critical_hit` / `guaranteed_direct_hit` 固定命中属性；候选技能和技能历史 token、compiled cache 技能特征都会保留这两个字段，`value` 仍只用于训练辅助 loss。
- 新增技能 token 字段会改变训练输入特征维度；重新训练前需要从 raw JSON 重建 compiled cache 并重新生成 checkpoint。
- 新增完整绝对时间 scene token、玩家 `time_seconds` 和技能输出字段调整，canonical、训练样本与转换数据 schema 已同步升级；重新训练前需要重建 compiled cache 并重新生成 checkpoint。
- 补充多目标 scene 标记的转换与回放回归测试，并用 FRU 训练 raw 验证双目标 token 能与玩家状态历史的绝对时间对齐。
- 新增职业技能 `value` 决策价值字段：默认值为 `1.0`，黑魔当前全部为 `1.0`，机工的固有直击暴击技能按 `2.0` 标注；该字段只保留在职业 YAML 和运行时技能定义中，作为训练辅助信号，不进入技能 token，不改变技能 `potency` 或实际伤害计算。
- 新增职业模型 YAML 可配置的技能价值辅助排序损失：基础交叉熵仍为主损失，只对标签技能与更低价值的合法候选施加小权重约束；价值在训练 batch 中从职业 YAML 注入，不作为模型输入，也不会在推理阶段硬推高价值技能。
- 新增训练 PPG 配置与回放评估入口：职业 YAML 可配置 `ppg.enabled`、`ppg.gcd_count` 和 `ppg.normalization`，训练 CLI 通过验证指标回调调用 `scripts/autoregressive_replay/ppg.py`，训练核心不承载回放实现。
- 新增模型推理 KV-Cache 模块和回归测试，覆盖历史追加、候选状态变化、上下文重建、Pre-LN/Post-LN、训练模式禁用以及回放默认开启和模型内置开关。
- 新增机工职业状态机与职业配置：机工职业资源、连击、过热、野火计数、起爆、工具技能和职业冷却缩减现在由 `combat_sim/jobs/machinist.py` 与对应 YAML 负责；转换层同时提供机工专用 GCD 探针配置。
- 新增机工野火延迟威力测试，覆盖自动起爆、起爆时增伤、不可选中窗口丢失威力和职业范围隔离。

### Fixed

- 修复 `common.yaml_config.load_yaml_mapping()` 使用 `or {}` 把顶层 `0`、`False`、空字符串或列表误判为空 mapping 的问题；现在仅空文档/`null` 转为空 mapping，其他非 mapping 顶层值会正确报错，并补充回归测试。
- 修复 `scripts/model_analysis` 选中的 raw JSON 缺少或过期 compiled cache 时直接失败的问题；现在会调用 `scripts.convert_fflogs.cli` 自动编译后重试，并修正转换 sample builder 的类型检查导入路径。
- 修复 `scripts/autoregressive_replay` 选中的 raw JSON 缺少或过期 compiled cache 时直接失败的问题；现在会调用正式转换 CLI 自动编译后重试，避免回放因缓存未预先构建而中断。
- 修正训练配置中 `raw_data_dir` 的数据目录语义：它现在明确指向只读 raw JSON 输入，不再指向转换后的 PT 目录；转换输出统一落在职业 `.cache` 目录。
- 修正模型分析与自回归回放的数据入口：两者现在从 raw JSON 定位对应的 compiled cache，不再消费中间 scene PT；回放支持 `--scene-json` 和 `AUTOREGRESSIVE_REPLAY_SCENE_JSON`。
- 删除转换 CLI 的 `--output-root` / `--output-format` 中间格式选项，改为直接写最终 cache，并移除训练、分析、回放对旧中间 PT 读取路径的依赖。
- 修复 Windows pytest 测试进程共用临时目录时可能出现的权限冲突；测试夹具现在为每个进程分配独立的项目内临时目录，并在退出时清理。
- 移除 `requirements.txt` 文件开头的 UTF-8 BOM，依赖内容不变，统一使用 UTF-8 无 BOM 编码。
- 修正 FFLogs 训练回放在推进到技能生效时间、构造历史 `after` 状态和时序调和时传递 `scene_context`；野火等延迟威力现在按实际伤害发生时刻查询 Boss 是否可选中，不再由状态机预测未来场景。
- 修正测试配置 fixture 与当前 `job_tag` / `job_configs`、统一 `FFXIV_JOB_TAG` 和爆发药 `1.08` 倍率契约一致，并补充转换层延迟威力回归测试。
- 修复自回归回放目标数量查询的 scene 时间坐标不一致：查询时间和 compiled cache scene 时间字段现在都通过统一 `Normalizer` 使用 `fight_time_max` 归一化，并拒绝误传原始秒值的 scene token。
- 修复首拍预读 AoE 在多目标窗口起点生效时仍沿用负时间单目标状态的问题；动作提交前会按实际生效时刻同步目标数量，确保 raw 状态威力先按真实 `2/3` 目标计算，再由 compiled cache 归一化。

### Migration

- 技能 `value` 不再是 skill 数值特征，不写入技能 token 或 compiled cache；修改职业 YAML 中的 value 不需要重新转换 raw JSON 或重建 compiled cache，但需要重新训练 checkpoint 才能让模型吸收新的偏好。旧 checkpoint 仍需重新训练，不能与新的模型输入结构混用；旧编译缓存会按 `raw_json_compiled_samples_v6` 直接失效并重建，不保留旧格式兼容读取。
- scene 上下文由按时间重基准迁移为整场绝对时间模板；旧的 `scene_time_max` 配置名改为 `fight_time_max`。旧 compiled cache 会因签名/schema 变化自动失效并重建，旧 checkpoint 因玩家状态维度和技能特征布局变化不能直接复用，必须重新训练。
- Transformer 由旧的 Post-LN + ReLU 改为可配置的归一化顺序与 FFN 激活，默认黑魔 Artzip 为 Pre-LN + GELU，并新增 encoder 末尾 LayerNorm；旧模型 checkpoint 不应直接续训，需重新训练并单独比较效果。
- 现有训练数据迁移为：保留 `data/human/job/<job>/raw` 原始 JSON，删除或忽略旧的 `pt/` 中间输出，运行 `python -m scripts.convert_fflogs.cli` 生成 `data/human/job/<job>/.cache`。训练、模型分析和回放均使用 raw JSON 定位最终 cache，不提供旧 PT 兼容层。
- 转换命令改用 `--cache-root`；训练命令的数据覆盖参数改为 `--raw-data-dir`；回放命令的 scene 参数改为 `--scene-json`。旧的 `CONVERT_FFLOGS_OUTPUT_ROOT`、`CONVERT_FFLOGS_OUTPUT_FORMAT` 和 `AUTOREGRESSIVE_REPLAY_SCENE_PT` 不再生效。
- 项目内部引用 `combat_sim.config_utils` / `combat_sim.torch_utils` 的代码需要改用 `common.project_config` / `common.torch_dependencies`；训练、回放和分析中的重复精度与 batch helper 统一改用 `common.torch_runtime`，不提供旧私有函数兼容层。

## [0.0.6] - 2026-07-16

### Fixed

- 修复 `scripts/autoregressive_replay` 的历史长度为 `0` 时 Python 负零切片会错误保留全部历史的问题；现在 `0` 明确表示不提供技能/状态历史。
- 修复 `scripts/model_analysis` 在 pair token 架构下仍按旧的技能/状态双 token role 解析的问题；PCA 候选筛选和 role 图例现在统一使用 `scene`、`history_pair`、`candidate_pair`、`cls`。
- 修复两个技能 embedding PCA 图的标签：现在统一从系统/职业配置读取实际中文技能名，不再显示 action key 或 raw skill ID。
- 修复训练公共层读取 `candidate_skill_dynamic` 的按样本嵌套 nullable 字段时错误复用候选索引的问题；`cast_time`、`gcd_window` 等技能动态特征现在会从原始 PT 正确传递到未缓存 Dataset、编译缓存和 batch，避免静默退化为 `0`。
- 编译缓存格式升级为 `training_compiled_samples_v3`；旧格式缓存会自动失效并按修复后的读取逻辑重建，避免继续复用错误的候选技能特征。
- 新增原始 PT、PT reader、未缓存 Dataset、编译缓存和 collator 之间的动态技能特征一致性测试，以及旧缓存格式自动重建测试。
- 训练 Dataset 现在会显式校验 sample builder 已由参考 reader 初始化；异常路径会给出明确错误，而不是留下半初始化对象。
- 模型分析的职业状态标签现在按 `job_tag` 路由；黑魔 AF/UI/MP 规则移入职业专用模块，未知职业返回 `unknown`，不会误套黑魔资源字段。
- `scripts/autoregressive_replay` 与 `scripts/model_analysis` 现在统一通过 checkpoint 配置解析入口恢复模型配置，旧 checkpoint 缺少新增配置字段时不会因直接展开字典而加载失败。

### Added

- `scripts/autoregressive_replay` 新增固定轨迹历史消融实验：先生成完整自回归轨迹，再在相同状态快照上分别只保留最近 N 条历史，输出完整轨迹动作与截断后预测动作对照，避免截断后的动作改变后续轨迹而污染比较结果。

### Changed

- 职业 YAML 的动作序列过采样改为按单场 PT 的完整 label 时间线匹配：命中规则后，首尾动作及其中间的所有训练样本都会获得权重；`ignored_actions` 不参与序列匹配，但仍会随命中区间一起加权。过采样不会跨 PT 文件拼接，重叠规则取最高权重。
- 黑魔 Artzip 当前实验训练配置调整为 `d_model=512`、`n_layers=4`、`n_heads=4`、`ff_dim=1536`、`batch_size=64` 和最多 7 轮；该配置变化需要重新训练，不会改变 PT 或编译训练缓存。
- 训练公共层职责进一步拆分：Dataset、sample builder、单 Dataset 缓存管理和并行缓存编译分别由独立模块负责；并行缓存 worker 不再反向导入 Dataset，降低隐式循环依赖风险。
- 模型公共层职责进一步拆分：输入编码、候选评分和 Transformer trace 分别由 `input_encoder.py`、`candidate_scorer.py` 和 `trace.py` 负责，模型入口只负责编排；分析工具复用这些正式接口和编码器实际 token 位置。
- 模型输入新增可配置的 pair embedding：历史技能与对应状态、候选技能与候选状态分别在模型内部融合为单个 token；默认 `pair_embedding_dim=384`，Transformer 仍使用职业配置的 `d_model`，序列长度和注意力计算量随之下降。
- 模型注意力改为固定的结构化 history 因果 mask：同一决策 step 的全部候选技能/状态保持双向可见，候选可以读取完整 scene 和 label 之前的 history；history 内部只能读取自己及更早的 history，scene 继续双向读取。该 mask 不改变 PT/cache 契约，但模型需要重新训练。
- 跨状态机、输出和训练层共用的 PT/scene 契约常量统一移入 `combat_sim/contracts.py`，训练层不再依赖输出层模块获取基础契约常量。

### Migration

- 模型内部模块拆分后，旧 checkpoint 的 `state_dict` 参数路径不再保证兼容，需要重新训练；PT 文件、编译缓存签名、manifest 和 shard 格式未因本次职责拆分改变，已有训练缓存可以继续复用。
- pair embedding 改变模型 token 布局和输入层参数，旧 checkpoint 无法直接复用；PT 文件和编译训练缓存未改变，可以继续复用。

## [0.0.5] - 2026-07-13

### Fixed

- 修复开场候选注意力图右上角图例与 colorbar 重叠导致标注被遮挡的问题；图例现移动到图表上方并预留布局空间。
- 修复训练 PT 读取器对 nullable `steps.time_offset` 的解析：现在能正确恢复每个样本的真实决策时间，避免所有 scene window 在 rebasing 时错误地落在起点。
- 训练新增职业 YAML 可控的历史前缀截断增强：仅训练集按概率移除早期历史并保留最近状态，技能/状态历史成对截断；候选技能、候选状态和可信 scene 上下文始终保持完整，验证集不启用该增强。
- 修复 `scripts/autoregressive_replay` 未加载 checkpoint `model_state_dict` 的问题；此前回放实际使用随机初始化权重，导致即使输入、空场景和旧时间推进方式一致，输出也无法复现旧脚本。
- 修正模型侧技能威力显示：候选技能和技能历史现在使用职业状态机解析后的实际威力，黑魔 AF1/AF2/AF3 火系技能分别应用 1.4/1.6/1.8 倍率；`Paradox` 不应用火系倍率，历史技能使用施法前状态计算，避免状态切换后错误套用新状态倍率。
- 修正黑魔 `Blizzard III` 的冰状态回蓝：从非 UI3 状态进入冰状态时按前一状态的 UI 层数恢复 MP；只有前一状态为 UI3 时才恢复满 MP。
- 修正冰状态 `Paradox` 不应恢复 MP 的状态机逻辑，并补充 AF/UI 极性校验：`Fire IV` 不能在 UI 下使用，`Blizzard IV` 不能在 AF 下使用。
- 黑魔读条时间现在由职业状态参与计算：AF3 下冰系硬读条减半，UI3 下火系硬读条减半，UI1 下 `Blizzard III` 保持 3.5 秒；状态机、候选预演与 FFLogs 回放统一使用该规则。
- 以上状态机和读条规则变更不会自动更新已有训练数据或 checkpoint；重新训练前必须重新导出 `.pt` 训练集，再生成新的 checkpoint。
- 训练数据集不再因为“当前合法标签在下一 GCD 投影候选中被标为非法”而中断；候选合法性 mask 仍保留给模型学习非法动作。
- 状态机和 FFLogs 回放现在对职业可选的 `cast_time_multiplier` 使用 `1.0` 通用回退，未实现该方法的职业不会因读条计算触发 `AttributeError`。
- 训练低精度校验现在拒绝 CPU 上的 `bf16` / `float16` autocast，避免配置与设备不匹配时静默进入不受支持的训练路径。
- FFLogs 转换输入文件解析现在按规范化路径去重，避免显式文件、目录和 glob 重叠时重复转换同一个 JSON。
- 修复目标数量 scene window 在相邻窗口边界处优先读取前一段的问题，确保 AoE 动作在目标数量变化的生效时刻使用新窗口。

### Added

- 新增 `05_pair_embedding_pca.png`：使用真实 PT 候选的技能与候选状态融合向量绘制 pair embedding PCA；原 `05_skill_embedding_pca.png` 继续用于观察原始技能 ID embedding。
- 新增目标数量 scene context：转换器从支持多目标技能的 FFLogs 命中事件推导目标数量窗口，并与 targetable 窗口叠加后写入 `.pt`；训练读取器将其作为第四类 scene window 向量提供给模型。该信息仍属于战斗场景，不写入玩家状态向量。
- 新增技能级 `aoe_secondary_reduction` 配置：单目标技能使用 `1.00`，无衰减 AoE 使用 `0.00`，其他技能按游戏描述填写额外目标的降低比例。
- 新增 `config/system.yaml` 的 `raid_buff_window` 配置：可指定团辅标记技能列表与窗口时长；转换器会按配置生成动态来源维度，并对同一窗口内的多个标记时间取平均。
- 新增 `scripts/autoregressive_replay/` 状态机驱动的自回归回放工具：从 `.env` 读取 checkpoint、职业、首步、基础 GCD、scene 模式和时间推进模式，并支持空 scene 与旧脚本时间轴对照实验。
- `scripts/model_analysis` 现在会对 Transformer 各层输出多份 PCA 2D 分析图，分别按战斗步数、候选技能、职业状态、有效性、无效原因、标签排名和 model logit 着色，用于检查模型是否过度利用战斗步数；分析同时保留原有的 role PCA 2D/3D 图。
- 模型分析 PT 样本现在保留候选 `invalid_reason` 和逐 token 决策元数据，便于比较 step、AF/UI、MP、合法性与最终 logit 对 hidden 表征的影响；空数据集会给出明确错误，而不是触发索引异常。

### Changed

- `scripts/autoregressive_replay` 新增 `AUTOREGRESSIVE_REPLAY_TEMPERATURE` / `--temperature`：默认 `0` 保持贪心 Top-1，大于 `0` 时仅在合法候选中按 softmax 概率采样，方便区分稳定评估与随机探索回放。
- 训练公共层新增职业 YAML 可配置的候选顺序随机化：仅训练集按概率同步打乱候选技能、候选状态、合法性、非法原因与 label 索引，验证集和推理保持固定候选顺序；该增强只作用于 batch collate，不改变 `.pt` 数据契约或编译缓存。
- `scripts/model_analysis` 现在会从真实 PT 开场决策中提取每层、每个 head 的 `CLS → 候选技能` 注意力，输出候选注意力热图和按 Transformer 层汇总的偏离均匀基线热图；默认取前 28 个决策，可通过 `--attention-steps` 调整。
- 过采样规则现在支持职业 YAML 的 `oversampling.ignored_actions`；公共层始终忽略 `ogcd_wait`，黑魔额外忽略 `high_thunder` / `high_thunder_ii`，避免 DoT 插入任意位置导致目标序列匹配失败。
- 训练公共层新增连续重复动作软惩罚策略：职业 YAML 可在 `training.repetition` 中二选一使用 `whitelist` 或 `blacklist`，重复动作只降低候选 logit，不改变合法性；黑魔默认允许连续使用 `fire_iv`、`xenoglossy`、`foul` 和 `flare`。
- `scripts/model_analysis` 的 MP PCA 着色不再使用五档 MP bucket，而是还原为 `0–10000` 的实际 MP，并使用与 `step_index` 相同的连续线性色带。
- 训练职业 YAML 的过采样序列现在支持 `技能(0)`（仅瞬发）和 `技能(1)`（仅读条）标签；未标注的技能不限制读条状态，且不改变 PT 或编译缓存契约。
- 训练 CLI 的 `--max-files` 现在按 `data/human/job/<job>/pt` 下各副本目录的 PT 文件数量比例选取，并在目录内均匀间隔抽样；不再按全路径排序后直接截取前 N 个文件。
- 状态机、FFLogs 训练回放和自回归回放现在会消费当前 scene 的 `target_count`：直伤与 DoT snapshot 按技能级 AoE 降低比例计算，并写入目标状态的当前/累计威力字段；同一 FRU 样本重新转换后，多目标阶段的目标累计威力已实际增加。
- 继续收敛模块职责：新增候选预演、回放缓存、scene builders/queries、训练决策同步和 PT 提取辅助模块，原入口文件保留公共导出 facade；黑魔重复行为 wrapper 合并，`.pt` / cache 契约和运行时行为保持不变。
- 统一配置与路径解析 helper，移除 convert_fflogs、训练配置和入口层的重复状态机/路径逻辑；scene window 重基准现在共享同一套时间数学、一次性索引解析和缓存的 feature key 映射，减少大规模转换时的重复查找；不改变 `.pt` 序列化契约。
- 清理训练与输出公共层的低耦合问题：删除重复的 `outputs/torch_utils.py` 和 bootstrap 转发 wrapper，scene window 索引改为一次性 schema 校验，normalizer 改用职业无关的 `_ready` / `_timer` 规则，测试默认 MP/GCD 提取为共享常量；不改变 `.pt` 序列化契约。
- 收敛状态机与 FFLogs 回放的动作执行核心：时序占用、冷却/职业资源消耗、系统/共享/职业行为和目标威力记录现在由 `CombatStateMachine` 统一实现，回放只保留自身的生效时间与历史构造逻辑，避免两套执行逻辑漂移。
- 输出层新增公共数值转换与 GCD 单位转换 helper，状态向量分组 key 和 scene context rebasing mode 改为引用统一 schema/常量；统一玩家 MP 特征命名为 `mp`，并让测试复用正式 scene token 构造器。
- `training/common/compiled_cache.py` 将训练缓存改为 manifest + 分片 PT：职业 YAML 可配置 `compiled_cache_shard_size` 与 `compiled_cache_max_shards`，初始化时逐源文件编译，训练时按需加载 shard，并在所有源文件之间共享有界 LRU，避免加载大规模训练集时把全部展开样本留在内存。
- 训练缓存首次编译现在按源 PT 文件使用独立进程并行处理，职业 YAML 可通过 `compiled_cache_workers` 配置 worker 数量；调整驻留 shard 上限或编译 worker 数不会使已有缓存失效，只有源 PT、分片大小、历史长度、dtype 或归一化配置变化才会重新编译。
- 新增 `training/common/sampler.py` 的 shard-aware batch sampler：训练集在 shard 内随机、shard 之间随机，但每个 batch 保持在单个 shard 内；训练和验证仍完整遍历全部样本，不改变 PT/cache 契约，减少随机访问造成的 shard LRU 抖动。
- 训练公共层现在支持从职业模型 YAML 读取变长动作序列过采样规则；黑魔通过最长后缀匹配提升 `transpose → paradox` 与 `paradox → fire_iii` 决策链，权重只在 shard 内重复训练索引，不改变 PT 契约或候选顺序。
- 训练编译缓存现在按 shard 流式展开和写入，普通 Dataset 路径与多进程 worker 都不再一次性持有单个 PT 的全部展开样本；缓存格式不变，已有缓存可以继续复用。
- `scripts/convert_fflogs/cli.py` 现在允许省略输入参数：会按 `.env` 职业自动扫描 `data/human/job/<job>/raw`，并把输出按相对目录写入 `data/human/job/<job>/pt`；`.env.example` 不再预置冗余的 `CONVERT_FFLOGS_OUTPUT_FORMAT=pt`，默认格式仍为 `pt`。
- 编译缓存启用时不再叠加样本级 LRU，样本索引改为按源文件累计偏移二分定位；Artzip 保持 `bf16` 与 `num_workers=0`，避免 Windows worker 重复加载训练缓存。
- `scripts/convert_fflogs/cli.py` 现在与并行转换入口统一解析单文件、目录和 glob（包括递归 `**`）输入；README 中的命令统一改为可正确处理包内相对导入的 `python -m scripts.convert_fflogs.cli`。
- `training/models/common/model.py` 现已移除了候选非法屏蔽：`forward()` 不再对非法候选做 `masked_fill(-inf)`，而是直接用全体候选的 raw logits 作为 `logits` 计算 `cross_entropy` loss 和 accuracy。模型必须自己学会给非法动作打低分，而不是依赖训练时外部注入的 `candidate_legal_mask` 兜底。`predict()` 同样不再依赖 masked logits。
- `training/models/common/training.py` 的 `train_epoch()` / `validate()` 和训练日志同步移除了 `legal_prediction_rate` 指标（全量候选下无意义），只保留 `loss`、`top1_accuracy`、`top3_accuracy`。
- `training/common/normalizer.py` 与 `training/common/pt_reader.py` 现在将 null 状态值编码为 `-1.0`，并继续保留 null mask；模型可以同时利用数值哨兵和显式缺失标记识别不可用状态。
- `training/common/dataset.py` 现在支持按职业配置限制已展开样本的 LRU 缓存；黑魔默认最多缓存 `768` 个样本，避免跨 epoch 重复重建压缩 PT 历史，同时避免无上限占用内存。
- 性能验证确认压缩后的 `timeline + ends + lengths` PT 契约以磁盘空间换取读取时的 CPU 重建成本；后续训练优化应优先采用按 PT 文件分块展开、连续 Tensor 批量索引的路径，避免逐样本 Python 重建和 collator 等待 GPU。

## [0.0.4] - 2026-07-10

### Migration

- 训练 CLI 默认使用 CUDA，也可通过根目录 `.env` 的 `TRAINING_DEVICE` 或 `--device` 覆盖；首次使用仍建议执行 `copy .env.example .env`（PowerShell 可使用 `Copy-Item .env.example .env`）并按本机修改。`convert_fflogs` 的 `--output-format` 默认值已从 `json` 改为 `pt`，依赖旧默认行为的脚本请显式传入 `--output-format json`。

### Added

- 新增 `training/models/common/` 公共训练模型层：从 PT 契约推导候选数量、状态/scene/skill 向量维度和候选顺序，提供通用 Transformer Candidate Scorer、数据规格和公共训练循环。
- 新增 `training/models/jobs/black_mage/artzip/config.yaml` 与 `training/scripts/train.py`；训练和科研可视化现在按根目录 `.env` 的 `TRAINING_MODEL_CONFIG` 读取职业模型配置，模型架构大小与 checkpoint 输出目录由职业 YAML 决定。
- 新增公共模型测试 `tests/training/test_models.py`，覆盖 PT 动态维度、`job_tag` 职业路由、语义 position 编码和无 label 推理路径。
- checkpoint 顶层新增 `job_tag`，与 `data_spec.job_tag` 保持一致，便于外部工具直接识别职业；`scripts/fflogs_scraper.py` 统一只读取项目根目录 `.env`。
- 新增训练设备和 FFLogs 转换输出的根目录 `.env` 配置：`TRAINING_DEVICE`、`CONVERT_FFLOGS_JOB_NAME`、`CONVERT_FFLOGS_OUTPUT_ROOT` 与 `CONVERT_FFLOGS_OUTPUT_FORMAT`，转换默认输出为职业目录下的 PT 数据集。
- `config/system.yaml` 新增 `mp_recovery.tick_interval_seconds` 与 `mp_recovery.in_combat_amount`，并新增 `combat_sim/system/mp_recovery.py` 统一承载系统层公共自然回蓝 tick 运行时；当前默认战斗内自然回蓝为 `3s / 200 MP`。
- `scripts/convert_fflogs/parallel.py` 新增并行转换编排模块，支持按 `--workers N` 把输入文件分发给 N 个独立进程并行转换，每个进程内部串行处理，进程间互无共享状态；文件数不足 worker 数时自动缩减。
- `config/convert_fflogs/default.yaml` 新增 `default_worker_count: 6`，统一管理默认并行进程数。
- `config/default.yaml` 新增 `engine_timing.cooldown_ready_tolerance_seconds`，把 FFLogs 回放里“剩余极小冷却是否视为已转好”的边界容差收敛到统一全局时序配置。
- `config/convert_fflogs/` 新增 FFLogs 转换专用配置层：`default.yaml` 统一维护默认职业与 GCD 检测公共默认值，`jobs/black_mage.yaml` 维护黑魔的探针技能与需排除的加速 Buff 规则；`scripts/convert_fflogs/config.py` 负责加载这套配置并解析根目录 `.env` 的本机默认职业。
- 黑魔职业技能表补齐 `flare(162)`，FFLogs 抽取与训练回放链路不再把核爆当作未知动作跳过；对应黑魔规则层新增核爆的合法性校验、`+3 astral_soul`、灵极心清空与 AoE GCD 支持。
- `config/default.yaml` 新增 `engine_timing.action_effect_settle_seconds`，把训练回放里“动作结果落点后的最小决策间隔”收敛到统一时序配置，而不是散落在 FFLogs 转换脚本里硬编码。
- `scripts/convert_fflogs/replay_timing.py` 新增共享回放时序辅助，统一承载 FFLogs 动作序列的真实读条时长推导、首拍预读回拨、时序补推进与“两段式动作提交”逻辑，供训练样本回放和原始动作预处理共同复用。
- 新增 [docs/m5s-dancinggreen-replay-failures-2026-07-05.md](./docs/m5s-dancinggreen-replay-failures-2026-07-05.md)，记录移除“非首拍动态回拨”后的 `M5S_DancingGreen` 整目录回放失败样本、原因分组与优先排查方向，后续可直接点查 `polyglot_capped` / `requires_polyglot` / `requires_ley_lines` 三类主问题，不再重复全量盲扫。
- 黑魔技能表补齐 `high_fire_ii(25794)`、`freeze(159)`、`high_blizzard_ii(25795)`、`foul(7422)` 与 `high_thunder_ii(36987)`；其中 `high_thunder` / `high_thunder_ii` 现在支持通过共享 `dot_key` 复用同一个目标雷 DoT 槽位，而不是在目标向量里分裂成两份同义 DoT。

### Changed

- `training/common/dataset.py` 现在校验多 PT 文件的职业标识、候选数量、候选顺序和监督 label 合法性；`training/common/normalizer.py` 同步归一化 skill 数值特征。
- 公共模型现在使用共享 history step / candidate index position，显式消费 state null mask，并输出 Top-1、Top-3 与未屏蔽预测合法率；候选 logits 仍统一屏蔽非法动作。
- 临时的 `training/models/black_mage/artzip/` 模型实现和 `training/scripts/train_bc.py` 已移除，不再保留旧路径兼容层。
- 训练入口不再使用 `FFLOGS_CONVERT_JOB_TAG` 推导模型职业，而是从 `TRAINING_MODEL_CONFIG` 的 `training/models/jobs/<job_tag>/...` 路径解析职业，并继续与 PT 契约和状态机路由校验。
- `scripts/convert_fflogs/` 的输出现在按 `CONVERT_FFLOGS_OUTPUT_ROOT/<job_name>` 组织，路由用的 `FFLOGS_CONVERT_JOB_TAG` 与落盘目录名 `CONVERT_FFLOGS_JOB_NAME` 分离；转换配置只读取项目根目录 `.env`。
- 训练数据目录调整为 `data/human/job/<job_name>/`，黑魔 Artzip PT 输入迁移到 `data/human/job/black_mage/pt/FRU`；Git 忽略规则同步收敛到 `data/human/job/`，并清理旧的目录占位文件。
- 黑魔 Artzip 训练配置现在使用 `training.precision: bf16`：模型权重保存为 bf16，输入张量保持 fp32；训练日志和 checkpoint 会记录训练精度与 CUDA 设备。
- 训练 DataLoader 现在支持职业 YAML 配置的 worker 预取、pinned memory 和异步 CUDA 拷贝；黑魔默认使用 2 个 worker、`prefetch_factor=2` 和 `persistent_workers`，并将 `zero_grad` 改为 `set_to_none=True`。
- `combat_sim/state_machine.py`、`combat_sim/system/machine.py` 与 `combat_sim/models.py` 现已把自然回蓝从职业内部隐藏计时器提升为系统层正式时间轴：`CombatState` 显式维护 `natural_mp_tick_progress`，系统层统一推进公共 MP tick，职业层只负责返回特化修正。
- 黑魔职业状态机不再依赖旧的 `ui_mp_tick_progress` 被动回蓝语义；当前改为由系统层提供公共自然回蓝，而黑魔层只在 `Astral Fire` 下屏蔽该公共回蓝，并把 `Blizzard III` / `High Blizzard II` / `Blizzard IV` / `Freeze` / 冰悖论 / `Umbral Soul` 的 MP 恢复统一收敛到“按当前 UI 层数结算冰法命中回蓝”这一套规则。
- `combat_sim/system/mp_recovery.py` 现在只负责共享 `3s` MP server tick 调度，不再持有 `lucid_dreaming` 的数值与职业语义；黑魔职业层接管 `Lucid Dreaming` 的 `550 MP / 3s` 结算、`Astral Fire` 屏蔽与 tick 边界判定，系统层只把公共自然回蓝基值与“到当前 tick 为止已流逝秒数”交回职业层解析，因此不会额外补齐持续结束时的最后一跳。
- `scripts/convert_fflogs/config.py` 的 `ConvertFflogsConfig` 新增 `default_worker_count` 字段，配置加载从 `default.yaml` 读取默认值。
- `scripts/convert_fflogs/cli.py` 的 `--workers` 参数默认值改为从配置文件读取（`default=None`，实际走 `default_worker_count`）；多输入时自动走并行路径，并支持目录递归扫码输入。
- `scripts/fflogs_scraper.py` 的批量查询(`--batch`)新增 `--metric` 参数（默认 `rdps`），支持 dps / rdps / ndps / adps 四种排行指标；`get_high_score_reports()` 同步新增 `metric` 形参。同时简化 `_load_dotenv()` 去掉多余的 `return` 与"未找到 .env"的 debug 日志。
- `combat_sim/outputs/history_context_builders/` 的 `SkillHistoryContextBuilder` / `StateHistoryContextBuilder` 现在按 `ActionHistoryEntry` 对象身份增量缓存历史 token：逐决策回放里相邻样本的历史是滑动窗口，绝大多数 entry 重复出现，缓存持有上一次 build 的 `(entry, token)` 并按 `id(entry)` 复用（同时持有 entry 引用防止截断滑出后 id 复用串味），每步只为新增 entry 构建 token，把 `state_token_builder.build` 的调用量从整场约 96k 次降到约 8k 次、单场转换耗时约减半；`skill_history` / `state_history` 输出逐张量保持不变。
- `combat_sim/outputs/training_dataset_formatter.py` 与 `training/common/pt_reader.py` 现在把 `.pt` 里的 `skill_history` / `state_history` 从“每步重复存整段前缀”改为“单条 timeline + 每步 `ends` / `lengths` 切片”的 `O(T)` 存储；`combat_sim/outputs/training_pt_contract.py` 新增共享二进制契约常量，序列化格式同步升级为 `pt_dataset_v3`，读取端仍按样本索引还原完整历史上下文，旧 `.pt` 需重新导出后再交给训练公共层读取。
- `scripts/convert_fflogs/training.py` 的逐步样本展开现在会在一条 weave 链明确结束、且下一条真实动作不再继续当前 GCD 窗口内 oGCD 时，注入一条合成 `ogcd_wait` 正样本；该动作语义与状态机系统层保持一致，表示“主动结束当前 weave window，直接进入下一个真实动作”，而不是按剩余理论槽位补多个等待标签；对应 `sample_schema_version` 同步从 `3` 提升到 `4`。
- `scripts/convert_fflogs/training.py` 的 FFLogs 训练回放入口现已彻底收敛到单一 `logged` 语义：除首拍预读外，后续动作统一把 raw `time_offset` 当作 FFLogs `cast` 的生效时刻回放，不再保留 `hardcast_time_mode` / `allow_hardcast_fallback` 这套 logged/backshift 双口径与 CLI 开关。
- `scripts/convert_fflogs/training.py` 清理了 `_resolve_decision_time_offset()` 里的硬读条回拨分支与死参数，`build_training_samples()` 也收敛成单一路径实现；同时修正 `_reconcile_logged_timing()` 注释，明确这里只推进真实等待时间，不再混入第二套“转换层回拨”语义。
- `scripts/convert_fflogs/extraction.py` 的基础 GCD 检测不再硬编码 `Fire IV(3577)` 与 `Ley Lines(1000737)`；当前改为由职业级 `config/convert_fflogs/jobs/*.yaml` 提供探针技能与需排除的 FFLogs Buff ID，后续新增职业只需补配置即可接入同一条转换链路。
- `scripts/convert_fflogs/cli.py` 与 `scripts/convert_fflogs/utils.py` 现在按 `--job-tag > 根目录 .env > config/convert_fflogs/default.yaml` 的优先级解析转换职业，本机默认职业选择不再写死在代码里。
- `combat_sim/config.py` 与 `combat_sim/models.py` 现在支持显式 `mp_cost` 语义值；黑魔的 `despair` / `flare` 改为使用 `mp_cost: full` 搭配 `mp_cost_floor` 表达“真实耗蓝语义”和“最低可施放门槛”，避免配置层写名义 `800`、职业规则层再单独覆盖为“全蓝/半蓝”的双重语义。
- FFLogs 训练样本回放现在除首拍预读外，后续动作统一直接使用 raw `time_offset` 作为技能生效时刻输入状态机，而不是技能开始时刻；读条、GCD 启动与动画锁推进全部交给状态机内部时序处理，减少转换层重复回拨造成的时间轴漂移。
- FFLogs 训练样本回放的“两段式提交”口径已与 FFLogs `cast=生效时刻` 对齐：`replay_timing.commit_replay_action()` 在当前时间瞬时结算动作结果，并对 GCD 技能额外扣除已流逝的 `actual_cast_seconds`，使读条后的冷却技能不会再因重复计算读条而误判锁定。
- `scripts/convert_fflogs/replay_timing.py` 的动作提交现已显式区分 `decision_time` 与 `effect_time`：普通动作继续使用当前 logged `cast` 作为生效时刻，首拍预读动作则保留 `decision=-cast` 但把 `effect` 对齐回原始 logged offset，避免把首个读条技能的职业状态提前到负时间生效。
- `scripts/convert_fflogs/cli.py` 与 `scripts/convert_fflogs/pipeline.py` 不再透传 `--hardcast-time-mode` / `--allow-hardcast-backshift-fallback` 及对应参数；训练样本转换入口正式只保留单一回放口径。
- `scripts/convert_fflogs/pipeline.py` 现在会在构建 `fight_payload` 前先按真实战斗 GCD 创建临时时序状态机，给原始动作补充动态 `actual_cast_seconds`，再据此刷新移动标签与滑步窗口；`training.py` 也同步复用这份共享时序 helper，不再自己维护第二套同义提交逻辑。
- 黑魔职业状态机现在把 AF/UI 的火冰威力倍率、AF 火系耗蓝倍率与 `Paradox` 的火/冰极性解析统一收敛到职业层 helper，并把倍率常量前置到文件顶部；既有单体技能和新增 AoE 技能不再各自散落写一套火冰耗蓝/威力特判。
- `combat_sim/state_machine.py` 的直伤记录现在会先复用职业层 `resolve_potency()` 做黑魔火冰属性倍率解析，再交给系统层统一叠加爆发药和团辅窗口倍率；`high_thunder` / `high_thunder_ii` 的 DoT snapshot 也同步复用同一份职业层威力解析入口。
- `combat_sim/system/dot_timeline.py` 现在按 `skill.dot_key or skill.key` 注册目标 DoT；`combat_sim/config.py` 和 `combat_sim/models.py` 也同步把 `dot_key` 纳入正式技能契约，避免同类 DoT 技能必须各自占一套目标状态维度。
- `combat_sim/system/resource_state.py` 现在会在职业资源注册期预构建快照顺序与按 `vector_group` 分组的稳定 key 视图；输出层的 `buff/resource/target` token builder 与 `output_context_builder` 同步复用这份缓存，去掉热路径里的重复排序、分组过滤与候选预演重复状态序列化，降低 FFLogs 转换和向量输出阶段的上下文装配开销。
- 测试目录 `tests/` 从扁平结构重构为按模块拆分的三级目录结构：`combat_sim/`（状态机测试）、`scripts/convert_fflogs/`（FFLogs 转换测试）和 `training/`（训练公共层测试）。其中最大的单体文件 `test_training_sample_builder.py` 拆分为 `test_training_builder.py`（样本构建流程与落盘）、`test_replay_timing.py`（回放威力与时序调和）以及 2 个黑魔职业测试并入 `test_black_mage.py`；`test_config.py` 也按模块拆分为 `combat_sim/test_config.py` 和 `scripts/convert_fflogs/test_config.py`。共享的 `_write_yaml` 辅助函数迁移到 `tests/helpers.py`。

### Fixed

- 修复训练公共配置的 `.env` 引号解析：只剥离匹配的外层单引号或双引号，避免不对称引号被静默吞掉。
- 修复科研模型分析在空 PT 数据集上的 `IndexError`，技能 embedding 标签提取现在会抛出带上下文的空数据集错误。
- `scripts/convert_fflogs/training.py` 与 `scripts/convert_fflogs/replay_timing.py` 的 FFLogs 时序调和路径现已改为显式保留 `natural_mp_tick_progress`，不再硬依赖已删除的黑魔内部 `ui_mp_tick_progress` 资源，训练回放与原始日志预处理链路继续复用同一套系统级自然回蓝语义。
- `combat_sim/outputs/tensor_payload_packer.py` 现在把可空数值序列里 `null`（缺失）位置的占位值从 `float("nan")` 改为 `0.0`，缺失与否仍由 `is_null` 掩码表达：既消除“同一份输入导出的 `.pt` 因 `NaN != NaN` 被 `torch.equal` 误判为不可复现”的假象，又避免 `NaN` 在下游归一化/前向计算里意外传播污染梯度；`.pt` 结构与 `serialization_format` 不变，读取端仍按 `is_null` 掩码把缺失位覆盖为 `0.0`，旧 `.pt` 向后兼容。
- `training/common/schema.py` 现在会在多 `.pt` 文件装载阶段显式校验 `sample_schema_version`；带 `ogcd_wait` 注入的新训练集与旧语义 `.pt` 不会再静默混装，避免同一批训练同时混入“无结束标签”和“显式 weave 结束标签”两套监督口径。
- `combat_sim/models.py` 把公共 `max_ogcd_per_window` 默认值从 `2` 调整为 `3`，系统层校验与训练回放不再把黑魔起手常见的 `ley_lines + swiftcast/amplifier + potion` 三插误判为 `ogcd_limit`；对应系统层测试也补上了“三个 oGCD 合法、第四个被拦截”的回归覆盖。
- `scripts/convert_fflogs/fight_payload.py` 现在会把预处理阶段解析出的 `actual_cast_seconds` / `actual_is_instant` 一并保留到回放载荷；`scripts/convert_fflogs/replay_timing.py` 则把 FFLogs `cast` 时间戳视为动作生效时刻，并在 GCD 提交后扣除已经流逝的 `actual_cast_seconds`，修复 `fire_iii -> manafont` 这类读条后紧接冷却技能仍被误判 `cooldown_locked` / `gcd_locked` 的问题。
- 首拍预读硬读条动作现在不会再在负时间提前生效职业状态：`training.py` 会保留原始 `_logged_time_offset`，`commit_replay_action()` 则按显式 `effect_time_offset` 在首个技能真正生效时才写入 AF/UI、天语与通晓计时相位，修复 `polyglot_timer` 提前约一个首拍读条时长起步、进而导致 `requires_polyglot` 大量误报的问题。
- `scripts/convert_fflogs/training.py` 的训练回放标签现在改为直接使用当前决策态 `validate_action()` 结果，而不是误用候选预演里“投影到下一 GCD 窗口”的合法性字段；`_reconcile_logged_timing()` 在 logged 模式下会用 `advance_time` 把阻塞锁态与连续的冷却、Buff、职业资源计时按真实等待时间 `hidden_elapsed` 一并推进。
- `combat_sim/system/cooldown_runtime.py` 现在会在校验、快照与消耗三条路径统一应用 `cooldown_ready_tolerance_seconds`，把 near-zero 剩余冷却规范化为真正可用充能，避免“验证已转好但消耗时没扣冷却”的边界不一致。
- 黑魔 `flare` 的真实耗蓝规则现在与旧状态机语义对齐：无灵极心时消耗当前全部 MP，有灵极心时消耗当前一半 MP，并消耗全部灵极心；对应技能索引、配置加载、状态机估算耗蓝和单元测试已同步覆盖。
- 训练回放首拍预读 GCD 现在仍可保留负 `time_offset`，后续动作的最小决策落点与真实状态时间推进分离；`action_effect_settle_seconds` 不再直接污染 `polyglot_timer` 等连续计时轴，只用于约束下一拍决策排序。
- 黑魔火冰属性技能现在会稳定吃到统一的 AF/UI 威力与耗蓝规则：AF 下火系威力提升并提高耗蓝，UI 下火系免费但威力受罚，冰系在 AF 下威力受罚且在 AF/UI 中切姿态技能免费；`Paradox` 也不再依赖零散特判，而是按当前极性自动归入火/冰语义。
- combat_sim/system/player_state.py 移除 oGCD 的 no_weave_window 校验：FFXIV 中只要动画锁结束即可使用 oGCD，编织窗口不足时下一个 GCD 自然延后（clipping），不需要提前拒绝。
- _resolve_bootstrap_hidden_elapsed 新增长读条（cast > recast）场景下的 no_weave_window 修正：当 GCD 已转完但动画锁仍在时，推进到动画锁结束后再 + epsilon，让长读条后的 oGCD 编织合法。
- FFLogs 转换的强制移动窗口现在与 `cast=生效时刻` 语义对齐：`detect_forced_movement_windows()` 对带读条动作回拨到 `prev_timestamp - 0.5`，即“前一个动作生效前 0.5 秒”的滑步起点，不再错误使用旧公式 `prev_timestamp + prev_cast_time - 0.5`。
- `MOVEMENT_MERGE_GAP` 从 `2.5` 降低到 `0.1`：滑步修正后每个移动时间戳代表一次独立的滑步事件（约每 GCD 一次），不再跨 GCD 合并成覆盖多个技能施放时间的大窗口。
- 训练样本回放 `_sync_state_for_decision` 新增滑步容差：如果当前移动窗口即将结束（≤ 0.5s），自动解除 `is_moving` 标记，允许玩家在滑步阶段开始读条。
- `_reconcile_logged_timing` 新增 `movement_locked` 精确判定：当状态机仍报 `movement_locked` 时，用具体技能的 `cast_time` 检查当前移动窗口剩余时间是否允许滑步（`movement_remaining ≤ cast_time - 0.5`），避免全局 0.5s 阈值不够精确时误拒合法动作。
- `scene_context.py` 新增 `_find_next_window_start` 和 `_find_current_window_end` 两个查询辅助，供滑步容差检查使用。
- FFLogs 的移动标注与强制移动窗口现在优先使用状态机回放得到的动态 `actual_cast_seconds`，不再只看技能表静态 `cast_time`；像 `triplecast/swiftcast/firestarter` 下的瞬发 `blizzard_iii/fire_iii`，后续位移不再被误标为读条中强制移动，也不会再生成错误的滑步窗口。
- `_reconcile_logged_timing` 现在在推进隐藏耗时前保存 `polyglot` / `polyglot_timer` / `ui_mp_tick_progress` 量谱状态，并在推进后恢复，避免模型校正时间（`gcd_locked` / `animation_locked` 的 `hidden_elapsed`）错误推进按真实墙钟累积的通晓计时器和 UI 回蓝进度，修复 `polyglot_timer` 领先 `state.time` 导致 tick 错位的问题；FRU 转换通过率从 53.9% 提升到 66.7%。
- `_reconcile_timing_for_annotation`（`replay_timing.py`）同步加上 polyglot / polyglot_timer / ui_mp_tick_progress 保护，确保动作标注阶段的瞬发判断等不因 annotate 路径的 reconcile 而产生 polyglot 漂移。
- 黑魔 `_validate_fire_iv` 的星极火要求从 AF≥3 放宽为 AF≥1：炽炎在火 1/2/3 档均可施放，只需处于星极火状态即可。
- `scripts/convert_fflogs/parallel.py` 的 `total_fights` 统计现在与串行路径对齐：零样本文件（无可识别动作）不再计入战斗数，避免并行/串行路径日志口径不一致。
- `scripts/convert_fflogs/config.py` 的 `default_worker_count` 校验现在只接受 `int` 类型，拒绝 `float` 以避免静默截断；同时修复校验报错里引用了未定义变量 `source` 的 bug。
- `scripts/convert_fflogs/parallel.py` 的 `_process_single_file` 现在把 `json.loads` 和初始解析也纳入 `try/except Exception`，避免无效 JSON 文件导致 worker 进程未捕获异常。

### Removed

- `combat_sim/system_skills.py`：系统技能兼容门面，真实实现早已迁到 `system/system_skills.py`，已无外部引用。
- `combat_sim/system_state_machine.py`：系统状态机兼容门面，真实实现早已迁到 `system/machine.py`，已无外部引用。

## [0.0.3] - 2026-07-05

### Added

- `config/precision.yaml` 新增统一 tensor 精度配置，集中控制状态机 tensor 输出和训练样本 `.pt` 导出的 `int_dtype` / `float_dtype`，默认使用 `int32` / `float32`。
- `combat_sim/torch_utils.py` 新增项目级 torch 导入辅助，统一训练与输出链路缺失 PyTorch 依赖时的报错入口。
- `training/common/` 新增训练公共层：`pt_reader.py` 负责读取正式 `.pt` 契约，`dataset.py` / `collator.py` 负责多文件样本展开与 batch padding，`skill_vocab.py` 改为只允许从正式项目配置构建词表，`normalizer.py` 负责按状态 feature key 和 `null mask` 归一化状态向量。

### Changed

- `combat_sim/config.py` 新增 `PrecisionConfig` 与 `load_precision_config()`，输出层和训练样本导出链路不再各自维护一份 dtype 约定。
- `combat_sim/outputs/router.py` 的 `format_tensors()` 现在会懒加载 tensor formatter，并统一从 `config/precision.yaml` 读取精度配置，避免非 tensor 路径提前 import torch。
- `combat_sim/outputs/model_tensor_formatter.py`、`tensor_payload_packer.py` 与 `training_dataset_formatter.py` 现在统一显式传递 `int_dtype` / `float_dtype`，不再在各自模块里硬编码 `int64` / `float32`。
- `combat_sim/outputs/torch_utils.py` 现在收敛为兼容转发入口，`combat_sim/outputs/tensor_payload_packer.py` 改为直接复用项目级 `combat_sim.torch_utils.import_torch`，避免 outputs 子目录继续维护一份重复实现。
- `scripts/convert_fflogs/io.py` 的单文件 `.pt` 导出现在与状态机 tensor 输出共用同一份精度配置，导出结构和运行时输出的 dtype 口径保持一致。
- `scripts/convert_fflogs/io.py` 现在也直接复用项目级 `combat_sim.torch_utils.import_torch`，FFLogs `.pt` 导出链路不再额外依赖 outputs 子目录下的兼容 helper。
- `scripts/convert_fflogs/training.py`、`scripts/convert_fflogs/io.py` 与 `combat_sim/outputs/training_dataset_formatter.py` 已移除遗留空字段 `bootstrap_action`，训练样本顶层 schema 因此从 `sample_schema_version=2` 提升到 `3`。
- 训练公共层不再把 history / candidate / scene 强行拼成一条模型专用大序列；新的 `TrainingDataset` 直接输出按语义分组的 structured tensor，并在加载多 `.pt` 文件时显式校验状态 schema、scene schema、候选数量和技能数值特征布局。

### Fixed

- `main` 上原本半落地的 tensor 精度改造现已补齐：`write_training_sample_pt_dump()` 不再因为 `TrainingDatasetFormatter` 构造参数不一致而在导出 `.pt` 时抛错。
- `tests/test_config.py`、`tests/test_output_contexts.py` 与 `tests/test_training_sample_builder.py` 现已覆盖统一精度配置的加载、状态机 tensor 输出和 `.pt` 导出链路，避免后续再次出现某一条输出路径仍偷写死 dtype。
- `combat_sim/config.py` 的 `resolve_int_dtype()` / `resolve_float_dtype()` 现在复用项目级 `combat_sim.torch_utils.import_torch`，缺失 PyTorch 依赖时会与其他训练/导出路径统一抛出同一份 `RuntimeError`，不再裸露 `ModuleNotFoundError`。
- 单样本 `.pt` 与多样本 `.pt` 现在可以在训练公共层下共同装载：candidate skill 的 static / dynamic 存储切分不再被误当成跨文件契约差异，空 history / 空 state token 也会被显式解读为零长度张量而不是脏列表。
- 训练公共层的 `SkillVocab` 现在会把 `raw_skill_id=0` 的系统动作 `ogcd_wait` / 空输出映射为真实 `skill_vocab_id>=1`，同时 history / candidate 里的未知 `raw_skill_id` 不再静默降级成 `NON_SKILL/PAD`，而是直接显式报错，避免把真实动作误训练成填充位。
- `training/common/normalizer.py` 的 `_infer_rule_type()` 现在只把以 `_seconds` 结尾的字段归类为秒数归一化规则，不再对中间仅包含 `_seconds` 的未来字段做宽泛误匹配；对应的训练公共层测试也已补上防回归覆盖。
- 修复 `scripts\convert_fflogs\replay_timing.py` 没有同步状态机的 `base_potency=skill.potency`

## [0.0.2] - 2026-07-04

### Added

- `combat_sim/outputs/model_tensor_formatter.py` 与 `tensor_payload_packer.py` 新增模型侧 tensor 导出链路：前者复用 canonical 上下文结构，后者把嵌套数值递归打包成 `torch.Tensor` 或 `{values,is_null}` 结构，供单文件 `.pt` 导出直接复用。
- `combat_sim/outputs/output_context_schema.py` 新增 canonical 输出 schema 定义，集中维护顶层 keys、状态向量分组、feature key 字段名以及正式 schema 元数据提取入口，避免多个 formatter 和训练 `.pt` 装配器各自硬编码一份输出结构。
- `combat_sim/outputs/torch_utils.py` 新增共享 torch 导入辅助，统一 `.pt` 导出链路里缺失依赖时的报错入口，避免 `tensor_payload_packer.py` 与 `scripts/convert_fflogs/io.py` 各自维护一份 `_import_torch()`。
- `combat_sim/outputs/token_builders/target_buff_vector_token_builder.py` 新增目标 Buff 向量装配器，统一把目标侧持续效果导出为 `target_buff_state`；当前除黑魔 `high_thunder` 的 `active` / `remaining_seconds` / `remaining_gcds` / `stacks` 外，也统一承载累计直伤、累计 DoT、当前直伤和当前 GCD 内 DoT 结算威力四个目标威力字段。
- `combat_sim/outputs/scene_context_schema.py` 新增 `scene_context` 统一 schema 定义，集中维护三类窗口上下文的 `feature_keys`、稳定空壳和通用装配入口，避免运行时输出、FFLogs 转换和测试辅助各自维护一套结构。
- `config/system.yaml` 新增系统共享状态 `raid_buff_window`，用于把背刺、占卜、连祷等团辅窗口统一折算成一个系统 Buff，后续无论接训练数据还是实时游戏输入，都只需要由系统层注入这一种共享状态。
- `scripts/fflogs_scraper.py` 新增 FFLogs 抓取脚本，用于从 FFLogs API 拉取原始报告、战斗与事件数据，供训练数据转换前置使用。
- `scripts/.env.example` 新增 FFLogs 抓取脚本的环境变量示例，约定本地凭据和访问配置的键名。
- 新增 [docs/scene_context_plan.md](./docs/scene_context_plan.md)，用于约束后续 `scene_context` 的职责边界：当前只预留结构，真正的场景字段由训练数据转换脚本设计和填充，不由状态机运行时直接生成。

### Changed

- `CombatStateMachine.from_default_config()` 现在支持注入 `actual_base_gcd`，训练样本回放会按真实战斗记录里的基础 GCD 缩放，而不是直接把技能表参考值当成运行时基准。
- `combat_sim/outputs/router.py` 不再维护单独的人类侧 formatter；`format_step()` 与 `format_vector_state()` 现在统一复用同一份 canonical formatter，输出契约只保留 canonical 与 tensor 两条链路。
- `combat_sim/outputs/model_vector_formatter.py`、`model_tensor_formatter.py` 与 `training_dataset_formatter.py` 现在统一复用 `output_context_schema.py` 暴露的正式 schema 元数据，不再分别手写顶层字段集合、状态向量分组和 feature key 字段名。
- `combat_sim/outputs/router.py` 的模块说明现已与实现对齐：当前只保留状态快照输出，以及 canonical / tensor 两类上下文输出，不再继续保留“三类输出”的过时描述。
- `CombatStateMachine` 新增 `ReplayStateCache` 与 `create_replay_cache()`，用于在整场战斗增量回放时复用中间状态，避免训练数据转换阶段每一步都从头重放。
- `CombatState.clone()` 不再递归 `deepcopy` 整棵状态对象树，改为按字段显式复制 `job_resources`、`cooldowns`、`statuses`、`dots` 与历史列表，减少候选预演和增量回放时的大量对象重建。
- `config/system.yaml` 新增 `potency` 配置段，系统层统一从这里读取爆发药和团辅窗口倍率；直接威力与 DoT 威力都复用同一套倍率解析入口，不再各走一条散落逻辑。
- `CombatState` 新增 `cumulative_potency`、`cumulative_dot_potency`、`current_potency` 和 `current_gcd_dot_potency` 四个目标威力累计字段；输出层会把它们统一拼进 `target_buff_state`，供状态历史和候选状态上下文共同复用。
- 模型侧状态向量从原先的三组 `player_state` / `buff_state` / `resource_state` 扩展为四组，其中新增 `target_buff_state` 承载目标侧持续效果；对应的 `state_history_context` 与 `candidate_state_context` 也同步复用同一组目标 Buff 特征定义。
- `state_history_context` 与 `candidate_state_context` 的 `after.*` 语义现已统一：两者都表示“动作生效后，再推进该动作对应 GCD / oGCD 占用时间后的状态”，不再出现历史输出和候选输出对同一个 `after` 使用两套时间口径的情况。
- `candidate_skill_context` 现在改为按稳定顺序输出全部已启用技能，不再只保留当前合法技能；`skill_history_context` 与 `candidate_skill_context` 也统一复用同一套技能 token 字段，其中新增 `is_legal`、`invalid_reason`、`next_cooldown_seconds`、`available_charges` 和 `max_charges`，用于同时表达技能合法性与共享冷却快照。
- 非法候选技能对应的 `candidate_state_context` 现在保留真实 `before.*`，并统一把 `after.*` 与 `consumed.*` 写成 `null`，显式表示该候选没有预演结果，而不是复用数值哨兵。
- canonical 输出顶层的 `scene_context` 与 FFLogs 转换导出的场景数据现已统一为 `targetable_window_context`、`forced_movement_context` 和 `raid_buff_window_context` 三组窗口上下文；每组都固定为 `feature_keys + tokens` 的向量结构，运行时继续只输出稳定空壳，训练导出的 `data_schema_version` / `sample_schema_version` 同步提升到 `2`。
- 训练数据导出链路已收敛为单脚本 `scripts/convert_fflogs.py`：现在直接从原始 FFLogs JSON 提取动作、构建 `scene_context`、驱动统一状态机并输出逐步 dump，不再保留 `combat_sim/training_sample_builder.py` 与 `scripts/build_training_samples.py` 这层中间壳。
- `scripts/convert_fflogs.py` 拆分为 `scripts/convert_fflogs/` 包后，模块职责进一步收敛：`extraction` 负责动作提取、移动标注、强制移动窗口与 GCD 检测，`downtime` 只负责 Boss 停手窗口检测，`scene_context` 统一负责三类场景窗口的向量化构造、重基准和查询；对外 `__init__.py` 继续保持兼容导出，原文件保留为 `_convert_fflogs_original.py` 做对照。
- FFLogs 转换阶段不再按长时间空窗把一整把战斗切成多场，而是保留完整战斗并把停手/不可选中窗口折算进 `scene_context`，避免绝伊甸等长时间转场导致训练起手状态错位。
- FFLogs 转换 CLI 现在支持 `--output-format pt`，可把整场战斗的逐步样本直接从内存打包成单个 `.pt` 文件；该链路不再先落逐步 JSON 再做二次转换，避免重复序列化和大量小文件写入。

### Fixed

- FFLogs 训练样本回放现在会把首个记录在 `0s` 的有读条 GCD 回拨到真实起手时刻（负 `time_offset`），不再用 `bootstrap_action` 吞掉首拍样本；因此战前预读 `fire_iii` 会保留在 `resolved_sequence` 与逐步导出里，后续动作仍按真实决策间隔推进。
- 黑魔的暮冰被动回蓝、`Umbral Soul` 叠层和 `Flare Star` 就绪后的 `Fire IV` 判定已和真实循环对齐，训练样本导出不再在后半段误判非法动作。
- 黑魔 `high_thunder` 现在改为走系统层注册的目标 DoT 通道，由系统统一维护目标侧 DoT 的挂载、剩余时间推进与移除，不再由职业状态机直接改写 `state.dots`。
- 训练样本回放现在会把 `raid_buff_window_context` 同步注入系统层共享 Buff，导出的 `buff_state` 不再只有 schema 维度而没有真实激活值。
- 目标威力输出现在会正确吃到 `burst_potion` 与 `raid_buff_window` 倍率；其中 DoT tick 采用施放时 snapshot 后的每跳威力，后续 3 秒结算不再退回裸 potency。
- 系统层 DoT 时间推进现在会按真实 `3s` tick 节奏结算每一跳威力，并同步更新当前 GCD 内 DoT 威力与累计 DoT 威力，不再只递减剩余持续时间而没有伤害侧输出。
- 系统层技能结算现在会把本拍直接威力同步写入目标累计威力，训练导出的候选状态上下文因此可以稳定看到“本技能威力 + 本 GCD 内 DoT 跳数”的结果。
- 训练样本构建阶段现在会在读取当前步标签前显式断言历史技能序列、状态历史长度和候选技能/候选状态数量对齐关系，并按系统历史上限自动对齐尾序列，防止当前步标签被意外提前写进上下文造成 label leak，同时避免长战斗因历史截断产生误报。
- 候选预演、`ReplayStateCache` 和时间推进链路现在复用轻量复制 / 原地推进路径，不再为每个候选技能重复整状态深拷贝，训练样本导出性能从深拷贝瓶颈中解放出来。
- `tests/test_output_contexts.py` 的 schema 对齐回归测试不再依赖本地 `.tmp/sequence_context_dump_f3_high_thunder_amplifier/` 导出文件，现已改为复用仓库内的固定动作序列常量，避免 CI 或其他开发机因缺少临时目录而直接失败。
- `main.py` smoke 命令修复 `state.gauge.xxx` → `state.job_resource("xxx")`，适配 MR !2 重构后的 `CombatState` 模型接口，`python main.py smoke` 不再抛 `AttributeError`。
- `combat_sim/system/buff_state.py` 与 `combat_sim/system/machine.py` 的共享状态授予接口现在支持直接写入 `remaining` / `stacks`，便于把训练数据或实时环境里的外部 Buff 剩余时间映射进系统层，而不把环境细节塞进职业状态机。
- `buff_state` 输出与测试现已覆盖统一团辅窗口，以及三连咏唱 / 即刻咏唱这类真实 Buff 的剩余秒数、剩余 GCD 和层数字段，避免共享 Buff 与真实职业 Buff 在导出层缺项。
- 配置加载阶段现在会拦截 `max_stacks < 1` 的异常状态定义（`_validate_status_definitions`），`StatusDefinition` 模型层也同步增加 `__post_init__` 校验，防止不合法的状态定义进入运行时系统。

## [0.0.1] - 2026-07-01

### Added

- 新增 `data/` 原始训练数据仓库，按副本分组存放人工战斗记录（FFlogs JSON），当前包含 FRU/M11S/M12S/格莱杨拉波尔/欧恩 等副本数据
- 新增 `training/` AI 训练管线目录占位（尚未开始搭建），`models/black_mage/artzip/` 为内部代号 artzip 模型定义目录
- 新增 `artifacts/` 训练运行时产物目录，含 checkpoints（断点续训）和 exports（ONNX/TorchScript 导出）
- 新增 `combat_sim/state_machine.py` 统一战斗状态机入口，外部通过 `job_tag` 路由职业子状态机，而不是直接依赖黑魔实现类。
- 新增 `combat_sim/system_state_machine.py`，把 GCD、动画锁、公共冷却、共享状态推进和停手时间轴从职业逻辑中独立出来。
- 新增 `combat_sim/outputs/` 输出层，提供秒制输出和 GCD 制输出两种状态格式，方便后续模型训练和调试。
- 新增 `combat_sim/outputs/candidate_context_builders/`，把候选技能上下文与候选状态上下文分别拆成独立装配模块，并复用现有 `skill_token_builder.py` 与 `state_token_builder.py`。
- 新增 `combat_sim/sequence_runner.py` 序列回放导出模块，专门负责把外部动作序列喂给统一状态机，并输出精简的最终快照。
- 新增系统技能抽象，当前包含空转 `ogcd_wait` 和爆发药 `potion`，统一放在 `config/system.yaml` 与 `combat_sim/system_skills.py` 中管理。
- 新增根目录 `CHANGELOG.md`，作为当前仓库的正式变更记录入口。
- 新增 `combat_sim/outputs/token_builders/` 与 `history_context_builders/`，把玩家/Buff/量谱三组向量、单技能 token、状态 token、技能历史上下文和状态历史上下文分别拆成独立装配模块。
- 新增 `combat_sim/system/cooldown_runtime.py`、`dot_timeline.py` 和 `status_timeline.py`，把共享冷却、DoT 推进和 Buff 推进从公共玩家态模块中继续拆开。

### Changed

- 人类侧与模型侧输出现在统一消费同一份 canonical 上下文，顶层共同保留 `job_tag`、`schema_version`、`skill_history_context`、`state_history_context`、`candidate_skill_context` 和 `candidate_state_context`，不再维护两套分裂的正式输出结构。
- `skill_token_builder.py` 现已收敛为单一技能 token 构造入口，技能历史与候选技能上下文都只负责准备参数，再复用同一个 `build()`。
- `state_token_builder.py` 现已收敛为单一状态 token 构造入口，状态历史与候选状态上下文都统一复用同一个 `build()`，三组高维向量的组装职责不再散落在其他模块。
- 玩家状态向量里的 MP 特征统一改名为 `mp`，对应的历史维度名称也统一为 `before.mp` / `after.mp`，不再混用 `before.current_mp` / `after.current_mp`。
- 重构战斗模拟核心结构，明确拆分为系统状态机、职业子状态机、输出层、技能索引层和配置层。
- 黑魔状态机改为职业子模块，只保留 AF/UI、悖论、通晓、雷云、火苗等黑魔专属规则，不再负责公共时间轴。
- GCD 基准时序参数迁移到系统配置，`base_gcd` 与技能表参考基准 `skill_table_base_gcd` 统一由系统层管理，而不是放在职业配置里。
- 状态机步进输出里的技能对象现在额外带 `skill_name` 中文字段，方便人工查看；模型侧仍可继续只消费 `skill_key` / `skill_id`。
- 候选技能上下文与技能历史上下文现在统一复用单技能 token 装配骨架，显式包含 MP 前后值、实际 MP 消耗、读条占用和 GCD 窗口占用。
- 命令行入口 `main.py` 改为走统一状态机入口，并支持通过 `--job-tag` 选择路由职业。
- 测试改为覆盖统一入口架构，验证路由、输出模式和黑魔职业行为，而不是只验证旧的单体黑魔状态机。
- 原先臃肿的黑魔单文件测试已按职责拆分为系统状态机、黑魔专属、统一输出上下文、序列回放和共享辅助模块，系统逻辑与职业逻辑不再混堆在同一个测试文件里。
- 旧实现继续保留在 `old/` 目录，作为历史归档和迁移参考，不再作为新代码入口。
- 序列导出结果改为只保留每一步的执行审计信息和最后一步的候选技能快照，不再把所有历史候选技能列表重复堆进输出文件。
- 系统层内部实现已进一步整理到 `combat_sim/system/` 子包，公共玩家态、Buff 注册中心、职业量谱注册中心、动作历史和系统技能不再继续堆在单文件里。
- 原 `combat_sim/system/registry.py` 已拆成 `buff_state.py` 与 `resource_state.py`，明确把 Buff 管理与职业量谱管理硬拆开。
- `combat_sim/system_state_machine.py` 与 `combat_sim/system_skills.py` 现在保留为兼容门面，真实实现分别迁到 `combat_sim/system/machine.py` 与 `combat_sim/system/system_skills.py`。
- `combat_sim/state_machine.py` 已直接依赖新的 `combat_sim.system` 门面，而不是继续通过旧兼容壳间接访问系统层实现。
- 职业注册入口改为 `combat_sim/jobs/__init__.py` 自动发现和注册职业状态机，不再继续把职业类型硬编码在统一状态机入口里。
- `JobConfig` 里的职业 timing 改为通用字典读取，公共配置层不再继续写死黑魔专属字段名。
- 秒制/GCD 制状态快照不再依赖职业子状态机手写通用输出字段，统一改为由系统层量谱快照和输出层自动拼装。
- 输出层改为先由 `output_context_builder.py` 统一装配上下文，再分别交给 `human_readable_formatter.py` 与 `model_vector_formatter.py` 翻译，避免人类侧和模型侧重复维护两套格式。
- 模型侧的 `state_history_context` 现在按历史 token 输出三组高维向量：`player_state`、`buff_state`、`resource_state`；其中玩家向量承载 MP before/after，量谱向量承载 before/after/consumed，当前状态约定体现在最后一个历史 token 的 `after` 段。
- 模型侧正式输出现在额外包含 `candidate_skill_context` 与 `candidate_state_context`，两者按同下标一一对应；其中候选状态上下文复用与 `state_history_context` 相同的三组高维向量定义。
- 候选动作预演链路现在由 `combat_sim/state_machine.py` 统一构造共享原料，再分别服务于人类侧 `legal_actions` 和模型侧候选上下文，避免两边各自重复装配。
- 模型正式输出不再重复暴露顶层 `player_state`、`buff_state`、`resource_state`，避免把已经包含在历史上下文里的当前状态再额外复制一遍。
- 人类侧继续保留可对照的 `state_snapshot_history_context`，模型侧独立消费向量化的 `state_history_context`，两条历史上下文不再混合输出。
- 训练向量的特征顺序现在改为确定性排序，避免 YAML 或注册顺序变化导致列索引漂移。

### Fixed

- 黑魔职业状态机 `validate_action` 对 `_behaviors` 中注册但缺少 `_validate_*` 方法的 behavior 不再静默通过，改为显式拒绝并返回 `unknown_behavior:<behavior>` 原因，防止开发时漏写校验规则导致校验/执行行为不一致。
- 技能合法性判断统一由系统层和职业层串联处理，避免公共规则散落在职业文件里。
- 黑魔技能表与系统技能表分离后，爆发药、空转、共享状态授予逻辑不再需要在黑魔内部重复定义。
- 配置加载阶段现在会显式拦截 system/job 跨配置的技能 `key` 与 `game_id` 冲突，不再只靠 `SkillBook` 合并时兜底报错。
- 系统层的零值浮点阈值已经统一，去掉了单独的 `1e-6` 写法。
- 星灵移位现在符合游戏内规则：无火冰状态时不能使用，必须先处于元素状态。
- GCD 技能的下一拍推进时间现在按“读条时间”和“GCD 窗口占用时间”取更大值计算，避免长读条技能的投影时点偏早。
- 输出里的 `cast_time` 现在会正确反映即刻咏唱、三连咏唱等瞬发修正；同时秒制与 GCD 制会始终成对显示，不再一会只给秒、一会只给 GCD。
- 系统层目录整理后，现有单步输出样例结构保持不变，回归样例导出的哈希结果一致。
- `火苗预备` 现在会在 `三连咏唱` / `即刻咏唱` 覆盖下施放 `爆炎` 时正确消耗，不再错误保留到后续动作。
- 共享 `grant_status` 行为不再短路职业层效果，后续职业技能可以同时吃系统共享 Buff 授予和职业专属量谱/状态变更。
- 配置装配阶段现在会提前拦截 `applies_statuses` 指向未注册状态，以及技能 `behavior` 未被系统层或职业层识别的情况，避免运行时才晚失败。
- `requires_target` 已变成显式技能契约，不再继续用 `potency > 0` 近似判断是否需要目标。
- `skill_history_context`、`state_history_context` 和 `state_snapshot_history_context` 不再默认截断为 8 条，当前会完整透出系统层保留的历史记录。

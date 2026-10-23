# 公共

路径相对仓库根目录。返回[项目目录](./index.md)。

本文件记录跨顶层模块共用的组件、策略模型与输入契约，以及共享配置、数据目录和仓库公共文件。模块内部的辅助代码仍在所属模块中说明。

## 共享组件

```text
FFXIV_CCG/
└── common/                                           跨顶层共享组件，按职责分包；模块内部 helper 留在所属顶层模块
    ├── configuration/                                配置基础设施，不依赖游戏领域或 PyTorch
    │   ├── __init__.py                               轻量包入口，不隐式导入其他组件
    │   ├── environment.py                            项目路径、python-dotenv、职业/模型变体、并发数与 TensorBoard 端口解析
    │   └── yaml.py                                   UTF-8 YAML mapping 加载与顶层类型校验
    ├── context/                                      引擎输出、转换与模型输入共用的上下文契约
    │   ├── __init__.py                               轻量包入口
    │   ├── contracts.py                              从共享 YAML 装配 scene、滑步与桥接版本常量
    │   ├── output.py                                 canonical 输出 schema、状态向量分组和有序元数据
    │   ├── scene.py                                  scene_context 窗口字段模板与稳定空壳
    │   ├── schema.py                                 config/schema.yaml 的唯一加载器与进程内缓存
    │   └── window.py                                 scene 窗口 feature key 索引缓存与必填字段校验
    ├── data/                                         跨下载、标注、转换和模型缓存复用的数据目录约定
    │   ├── __init__.py                               轻量包入口
    │   └── layout.py                                 raw/annotated/.cache 阶段、副本/百分位目录、Brotli JSON 发现与输出路径映射
    ├── game/                                         跨顶层复用的游戏静态配置、定义与索引，不承载运行时战斗状态
    │   ├── __init__.py                               轻量包入口
    │   ├── config.py                                 系统/职业 YAML 配置装配、资源与状态/技能契约校验
    │   ├── models.py                                 静态技能、状态数据类与动作类型
    │   └── skills.py                                 按 key/game_id 查询技能，保留声明顺序和启用过滤
    ├── policy/                                       跨预训练、GRPO、回放和 ONNX 共用的策略模型、输入契约与数据组件
    │   ├── data/                                     按输入契约、编码和缓存分类的策略数据组件
    │   │   ├── contracts/                            固定动作词表、输入字段和 checkpoint 输入契约
    │   │   │   ├── __init__.py                       轻量包入口
    │   │   │   ├── action_space.py                   启用技能与 policy 动作的固定输出顺序、词表行映射和 GCD 类型元数据
    │   │   │   ├── context_fields.py                 上下文字段的阶段、dtype、逻辑轴/shape、padding 与 cache/raw 命名映射；数据读取、合批、编码及 ONNX 共用
    │   │   │   ├── input_contract.py                 checkpoint 输入契约、独立 token 编码描述、schema 与归一化规则的序列化/恢复
    │   │   │   ├── policy_actions.py                 独立 policy 控制动作配置加载器
    │   │   │   ├── skill_vocab.py                    正式技能与 policy 动作组成的模型技能词表
    │   │   │   └── spec.py                           从保存的 StateFeatureLayout 推导并核验基础/合并宽度和动作规格
    │   │   ├── encoding/                             模型外共享输入编码、归一化与历史窗口
    │   │   │   ├── __init__.py                       轻量包入口
    │   │   │   ├── context_encoding.py               模型外共享 FP32 编码：窗口 gather、绝对状态、时间重锚与有符号 DELTA、场景裁剪差分；缩放统一委托 normalizer
    │   │   │   ├── history_window.py                 完整历史游标到分段窗口长度的纯函数；场景、状态／动作和死亡事件共享预算，超限保留最近 K 条动作后重新累积
    │   │   │   ├── normalization.py                  归一化配置数据类、FP32 有效尺度范围与 YAML 加载
    │   │   │   ├── normalizer.py                     字段归一化规则与统一 FP32 截断/除法入口；保存相对时间与绝对时长尺度、职业单位及独立技能状态的注册充能上限
    │   │   │   └── state_features.py                 canonical 状态的唯一 reader 与编码后唯一拼接器；拒绝错序、非有限值和非二值可用性
    │   │   ├── cache/                                compiled cache 读取与转换、训练共享的数据来源选择
    │   │   │   ├── __init__.py                       轻量包入口
    │   │   │   ├── compiled_cache.py                 raw FP32 ABS/DELTA 完整 bank 的只读 reader、签名与目录定位；统一比较缓存签名与生效配置依赖，拒绝旧平铺和归一化状态缓存
    │   │   │   ├── prepared_sources.py               BC 与 GRPO 共享的只读训练数据选择器，并按验证副本固定配额校验 VAL compiled cache；训练进程不编译
    │   │   │   └── source_selection.py               转换与训练共用的副本比例配额、区间均分和同区间失败补位候选顺序
    │   │   ├── __init__.py                           策略数据契约导出入口
    │   │   └── schema.py                             稳定 schema 类型：有序 StateFeatureGroup、快照顺序与 StateFeatureLayout；raw source、cache 和模型共用，保留序列化模块路径
    │   ├── model/                                    按注意力、网络层和执行路径分类的策略模型组件
    │   │   ├── attention/                            注意力计算、可见性 mask 与位置编码
    │   │   │   ├── __init__.py                       轻量包入口
    │   │   │   ├── attention_masks.py                因果可见性、padding、cache 后缀 mask 与分析矩阵构建
    │   │   │   ├── attention_utils.py                Q/K/V 投影、头维度整形与 Transformer 层收尾辅助
    │   │   │   ├── attention_variants.py             统一因果 attention 的手工/SDPA 与重算执行路径
    │   │   │   ├── grouped_attention.py              MQA/GQA 投影与按 KV 组 SDPA 适配
    │   │   │   └── position_encoding.py              唯一 RoPE 实现；一次编码共享 FP32 三角函数表
    │   │   ├── layers/                               输入投影、Transformer 层、激活与残差结构
    │   │   │   ├── __init__.py                       轻量包入口
    │   │   │   ├── activation.py                     Transformer 主干 FFN 激活算法：选项校验、门控判定、隐层宽度折算与门控合成
    │   │   │   ├── attention_residual.py             Full Attention Residual 深度聚合器
    │   │   │   ├── hyper_connections_cuda.py         CUDA 训练专用 HC 系数与 Sinkhorn 的 Triton 前后向核；由残差模块延迟导入
    │   │   │   ├── input_encoder.py                  技能、状态与 scene 独立内容投影；死亡使用独立 FP32 embedding 并按动作边界插入，content 与 role 相加后统一无参数 RMSNorm，历史/当前状态共享参数，生成交错布局和逻辑位置
    │   │   │   ├── residual_mix.py                   Hyper-Connections 多路残差、动态 pre/post 混合、Sinkhorn 归一化；CUDA 合并 batch/序列轴进行融合前后向与重算，CPU 保留局部重算，供 attention/FFN 共用
    │   │   │   └── transformer.py                    可学习 RMSNorm 及 Transformer 层、FFN 激活/门控、CUDA 门控前后向融合、CPU 局部重算、整块重算开关和逐层运行诊断，不依赖分析 trace
    │   │   ├── execution/                            训练、KV-cache 与分析共用的模型执行组件
    │   │   │   ├── __init__.py                       轻量包入口
    │   │   │   ├── causal_encoder.py                 单一因果层/encoder 执行、Hyper-Connections 多路残差与跨子层 pre 传递、激活重算及 Full AttnRes；训练、cache 与 trace 共用
    │   │   │   ├── kv_cache.py                       推理专用 scene/history prefix 分层 KV-Cache，分别记录历史动作数与独立 token 数
    │   │   │   └── trace.py                          hidden 与 attention head 分析 trace
    │   │   ├── __init__.py                           策略模型组件导出入口
    │   │   ├── model.py                              固定动作因果策略模型；独立无 bias 动作头、FP32 softcap 与 checkpoint 架构校验，输入技能 embedding 和输出头分别训练
    │   │   └── repetition.py                         连续重复动作软惩罚策略
    │   ├── __init__.py                               策略共享层入口，不隐式导入可选依赖
    │   ├── config.py                                 策略模型配置与模型/设备/checkpoint/cache 路径解析；按 `.env` 的 `FFXIV_JOB_TAG` + `FFXIV_MODEL_VARIANT` 解析 config.yaml，并合并 model/training/optimizer/grpo/action_quality 子配置，不依赖 training
    │   └── replay.py                                 回放配置数据类；不包含回放执行实现
    ├── pytorch/                                      PyTorch 依赖、精度、运行时和安全序列化工具
    │   ├── __init__.py                               轻量包入口，不提前导入 PyTorch
    │   ├── dependencies.py                           PyTorch 延迟导入与缺失依赖错误
    │   ├── precision.py                              统一 tensor 精度 YAML、别名与 dtype 解析
    │   ├── runtime.py                                batch 搬运、模型 dtype 与 autocast 上下文
    │   └── serialization.py                          weights_only 安全读取、必要 Path 类型允许列表与保存前路径字符串化
    ├── training/                                     跨 BC 与 GRPO 共用的优化器、训练指标与监控组件
    │   ├── __init__.py                               轻量包入口，不隐式加载计算或监控依赖
    │   ├── master_weights.py                         低精度模型的 FP32 主权重与优化器状态，保留小更新、兼容旧状态迁移并完整保存恢复累计值
    │   ├── metrics.py                                设备端指标累积与集中 CPU 读取
    │   ├── optimizer.py                              BC/GRPO 共用优化器装配；线性矩阵、归一化、bias/scale、embedding/输出头按职责分组，组合 Muon/AdamW/Sinkhorn，统一调度与状态保存恢复
    │   ├── optimizer_config.py                       BC/GRPO 的 Muon、AdamW、Sinkhorn 独立配置与数值校验，解析各组学习率、衰减及算法参数，拒绝重复配置权威
    │   ├── sinkhorn.py                               embedding 与输出头的动量、近零行屏蔽及交替行列 L2 归一化更新，不施加权重衰减
    │   ├── tensorboard.py                            BC 与 GRPO 共用的监控配置与可选 SummaryWriter，按模型变体共享 `<variant>_tensorboard` 事件根目录并写入有限数值标量；配置读取不加载 PyTorch 或模型诊断
    │   ├── tensorboard_labels.py                     BC/GRPO 共用的中文展示标签；按实际参数路径动态生成层号，保留原始参数路径与动作标识，不修改训练指标键
    │   └── tensorboard_monitor.py                    BC/GRPO 模型诊断：真实输入结构图、确定性有界权重/梯度分布、输入技能/独立输出头 Projector、配置 Text/HParams 和 BC 验证混淆矩阵/PR 曲线
    └── __init__.py                                   共享层入口，不隐式导入可选依赖
```

## 配置、数据与仓库文件

```text
FFXIV_CCG/
├── .agents/                                          Codex/agent 本地辅助目录，不承载项目业务代码
├── .codex/                                           Codex 本地运行目录，存放会话和本地辅助数据，不属于项目实现
├── .git/                                             Git 元数据目录，记录版本历史、分支和对象数据，不手动修改内部内容
├── .node/                                            项目本地 Node 22、pnpm 10 与包缓存，setup.sh 自动安装并复用，不提交
├── .tmp/                                             临时输出目录，当前主要放 pytest 缓存和状态机示例导出文件
├── .venv/                                            本地 Python 虚拟环境目录，不承载项目业务代码
├── .vscode/                                          VS Code 本地工作区配置目录，不属于项目实现
├── .zcode/                                           本地辅助运行目录，存放 plans 等本地辅助数据，不属于项目实现
├── artifacts/                                        训练运行时产物，自动生成不手动维护
│   ├── checkpoints/                                  训练中间检查点（断点续训用）
│   ├── exports/                                      导出文件（ONNX/TorchScript 等部署格式）
│   └── model_analysis/                               科研模型分析 PNG 与 metadata 输出
├── config/                                           模拟器 YAML 配置目录，存放项目级、系统级和职业级静态定义
│   ├── convert_fflogs/                               FFLogs 转换专用配置目录，集中维护转换脚本默认职业与按职业拆分的日志解释规则
│   │   ├── action_quality/                           各职业评估标签进入训练目标的准入配置；不按模型变体拆分
│   │   │   ├── black_mage.yaml                       黑魔建议、循环码和事件原因开关，仅允许可精确关联成功施法的建议进入 PT
│   │   │   └── machinist.yaml                        机工建议和通用事件原因开关，当前全部关闭技能负标签
│   │   ├── jobs/                                     FFLogs 转换职业配置目录，每个职业一个 YAML，只写该职业的日志解释与 GCD 探针规则
│   │   │   ├── black_mage.yaml                       黑魔 FFLogs 转换配置，定义基础 GCD 探针技能、`probe_skill_cast_time_seconds` 与需排除的加速 Buff ID
│   │   │   └── machinist.yaml                        机工 FFLogs 转换配置，定义机工基础 GCD 探针技能与职业专用日志解释参数
│   │   └── default.yaml                              FFLogs 转换全局 GCD 检测与连续速度阈值/采样参数；职业标签及转换 worker 分别读取根目录 `.env` 的 `FFXIV_JOB_TAG` 和 `CONVERT_FFLOGS_WORKERS`
│   ├── fflogs_scraper/                               FFLogs 下载目标配置目录
│   │   ├── batch.yaml                                批量下载时验证集目标占训练集目标的比例
│   │   └── encounters.yaml                           副本代号到 Zone/Encounter ID 的映射：FRU、M1s～M11s、M12sI/II 与 DancingMad
│   ├── models/                                       训练与推理共用的策略模型配置，按 `.env` 的 `FFXIV_JOB_TAG` 与 `FFXIV_MODEL_VARIANT` 选择职业目录下的模型变体
│   │   └── <job_tag>/<variant>/                      模型变体配置目录；`action_quality.yaml` 定义动作质量监督的轻微/中等/严重等级权重，由 `config.yaml` 的 `action_quality_config` 引用并合并，与 `training.yaml` 的排名衰减曲线共同生成 BC 主交叉熵和价值辅助损失的逐样本学习权重；`config.yaml` 是清单，`model.yaml` 提供模型架构、输入数据阶段/output 与统一 precision（黑魔 Artzip 默认使用 annotated），`training.yaml` 提供预训练/DataLoader/增强/序列过采样，`optimizer.yaml` 由清单的 `optimizer_config` 引用，通过 `optimizers.bc` / `optimizers.grpo` 独立选择方法，阶段内按 `muon`／`adamw`／`sinkhorn` 分类维护各组学习率、衰减和算法参数，warmup 由阶段统一管理；`grpo.yaml` 提供 GRPO rollout 与更新参数（当前 Artzip 每场 16 条样本、仅按 1200 秒时间上限终止），固定动作顺序由启用技能和 policy 配置生成并写入 cache/checkpoint
│   ├── policy/                                       跨训练与策略运行时共享的策略输入配置
│   │   └── normalizer.yaml                           训练、回放和部署共用的归一化尺度：MP 10000、相对时间与绝对时长分别为 270 秒、目标数 5、单次威力 2000
│   ├── skills/                                       职业配置目录，每个职业一个 YAML，只写该职业自己的静态定义
│   │   ├── black_mage.yaml                           黑魔职业配置，定义 timing、量谱上限与时间单位、statuses、统一字段的 skills；绝望/核爆最低 MP 均为 800
│   │   └── machinist.yaml                            机工职业配置，定义机工资源、状态、技能决策 `value`、野火延迟威力和冷却/连击规则所需的静态参数
│   ├── action_quality.yaml                           离线动作质量桥接运行配置，维护外部分析源码路径和单份分析超时；批量并发数读取根目录 `.env`，等级权重属于职业模型配置
│   ├── default.yaml                                  项目总配置入口，只注册 `runtime.job_configs` 职业配置映射和系统配置路径；当前职业严格由根目录 `.env` 的 `FFXIV_JOB_TAG` 选择，不再声明 `active_job` 或单一 `job_config`
│   ├── policy_actions.yaml                           独立 policy 控制动作配置；`ogcd_wait` 保留 raw id 0，但不进入 FightEngine SkillBook
│   ├── precision.yaml                                tensor 精度配置入口；`int_dtype` 用于技能 ID 与质量等级，`index_dtype` 用于交叉熵标签和 gather 索引，`float_dtype` 用于执行统计、监督值和权重；模型数值输入与归一化固定 FP32，mask 固定为 bool
│   ├── schema.yaml                                   共享 schema 定义（Rust 与 Python 单一事实来源）：共享常量（滑步窗口/绝对时间模式/scene context keys）、scene context 窗口字段模板与 segment kinds、canonical 输出顶层 keys 与状态向量分组、状态向量字段模板；动态部分（职业量谱/Buff/DoT 列表）仍由运行时推导
│   └── system.yaml                                   系统级共享配置，定义基础 GCD、技能表基准、战斗内自然回蓝 tick、目标威力倍率、爆发药及统一团辅窗口（含 `raid_buff_window.marker_skills` 与权威的 `duration_seconds`）
├── data/                                             原始训练数据目录，大文件不进版本控制
│   └── human/                                        人工战斗记录根目录
│       └── job/                                      按职业目录名存放训练数据
│           └── {job}/                                具体职业训练数据目录，转换输出与训练缓存按职业隔离
│               ├── .cache/                           最终 compiled cache，按 <副本>/<区间>/ 保存 manifest 与分片；独立或旧未分类输入保持对应层级，读取兼容旧平铺缓存
│               ├── annotated/                        原始 raw 与离线分析结果合并的 `.json.br`，复用公共目录映射保留 <副本>/<区间>/，不替换原始数据
│               ├── grpo/                             GRPO 自回归 rollout 持久化目录，按 run/iteration 保存 manifest 与轨迹 tensor，不纳入版本控制
│               └── raw/                              FFLogs 原始 `.json.br` 输入，只读，不接收转换输出
│                   └── .../                          各个副本原始日志目录
├── docs/                                             项目文档目录；agents/ 面向智能体，其他文件保留使用说明与时间戳分析记录
│   ├── agents/                                       智能体按任务查阅的项目目录与职责说明
│   │   ├── grpo.md                                   GRPO 后训练目录与职责
│   │   ├── tests.md                                  Rust 与 Python 测试目录及覆盖职责
│   │   ├── common.md                                 共享组件、配置、数据目录及仓库公共文件
│   │   ├── fight-engine.md                           Rust 核心、职业和原生绑定及状态机运行边界
│   │   ├── scripts.md                                数据下载、标注、转换、回放、分析与导出
│   │   ├── training.md                               BC 预训练数据管线与训练循环
│   │   └── index.md                                  类索引，按任务定位相关文档
│   ├── fflogs-timestamp-semantics-2026-09-14.md      FFLogs raw JSON 时间戳真实语义实测：begincast=读条开始+实际读条时长、cast=服务器提前结算、damage=落地按技能固定偏移、DoT 3s 节拍与 prepull 裁剪无负时间戳；转换层正常硬读条直接取 begincast，瞬发取 cast，仅开怪首个预读按配置 0.5s 滑步窗口恢复请求，随后统一归零
│   └── 命令行使用说明.md                               各命令入口与完整参数说明：统一入口、FFLogs 下载、数据标记、独立转换、训练、GRPO、ONNX 导出、模型分析和自回归回放
├── old/                                              旧实现归档区，保留历史状态机、训练代码、数据和文档，不作为新代码入口
│   ├── data/                                         旧训练数据、战斗记录和回放样本
│   ├── docs/                                         旧文档目录，包含人工循环等参考资料
│   ├── State_Machine/                                旧状态机实现与相关资源
│   ├── Tools/                                        旧工具脚本和辅助程序
│   └── Training/                                     旧训练代码与配置
├── preview/                                          README 和项目展示使用的预览资源目录，不承载业务代码
├── third_party/                                      外部项目源码目录，子模块版本由主仓库固定，我们的适配代码放在子模块外
│   └── xivanalysis/                                  MIT 许可的外部日志分析器 Git 子模块，用于后续离线动作质量标注；保留上游结构与 LICENSE
├── .env                                              本机环境变量覆盖文件，维护 FFLogs 路由、职业与模型变体、TensorBoard Web 端口、raw 数据根目录、compiled cache、模型 checkpoint、训练设备和回放解码参数等本地默认值，默认不提交
├── .env.example                                      项目根目录环境变量示例，统一约定转换与标注并发数、训练、TensorBoard、回放、ONNX 导出与 parity 门禁参数；导出与回放共用模型路径和 Provider
├── .gitattributes                                    Git 文本属性配置，约束仓库里的换行和文本处理规则
├── .gitignore                                        Git 忽略规则，过滤缓存、临时文件和不应提交的本地内容
├── .gitmodules                                       Git 子模块路径和上游地址声明，记录 third_party/ 下的外部依赖
├── .node-version                                     项目要求的 Node 主版本，setup.sh 首次安装时选择最新 22.x
├── AGENTS.md                                         本地协作规范文件，记录开发约束、发布约定和项目架构说明
├── CHANGELOG.md                                      项目变更说明文件，记录当前重构后的里程碑和后续变更
├── package-lock.json                                 离线动作质量桥接的 DOM 依赖锁文件，Node 依赖统一安装在项目根目录
├── package.json                                      离线动作质量桥接的独立 DOM 依赖声明（jsdom），Node 依赖统一安装在项目根目录
├── README.md                                         项目命令说明文件
├── requirements-onnx-gpu.txt                         本地 NVIDIA CUDA 13.x ONNX 导出与 ORT GPU 回放依赖，与 CPU 版依赖二选一安装
├── requirements-onnx.txt                             独立 CPU ONNX 导出与 ONNX Runtime 校验依赖，不耦合训练主环境安装流程
├── requirements.txt                                  项目 Python 运行依赖与训练、TensorBoard、FFLogs 抓取及测试所需依赖
└── THIRD_PARTY_NOTICES.md                            第三方组件来源、使用范围及原始版权和许可文本
```

## 上下文与缓存契约

内部推进与公开 `AdvanceTo` 共用同一个事件内核。阶段 3 的历史和最新状态统一为上一动作后与当前请求快照，按请求身份关联并冻结；wait 的动作后状态来自决策落实时，历史裁剪不丢失已知前序快照。真实动作前后统计与模型状态独立保存。阶段 4 已删除技能绝对时间 `time_seconds`，技能数值特征为 16 维，两段基础状态继续保留时间，移除两段累计威力及冗余 mp_ratio 后共 80 维；独立技能状态注册表按两段快照、固定动作顺序、可用性／当前层数／恢复时间展开，使当前黑魔 25 个动作的状态值为 80 + 25 × 2 × 3 = 230 维，null/reset 为 80 维；在线历史身份从当前请求状态获取时间，并逐字段确认复用。阶段 5 编码顺序为场景、状态、技能交错、最新状态，所有有效场景、历史状态/动作、死亡事件及最新状态共用 model.context_capacity token 总预算，默认 1024；每条死亡标记占一个 token，各样本有效前缀压紧后统一补齐，物理序列也不超过预算。窗口重置和额外裁剪保留所选动作窗口首尾边界的死亡标记，同边界多次死亡保持原顺序，零动作历史也可包含死亡 token；历史游标仍只统计动作。当前 canonical 17、桥接 20、checkpoint 输入 27、样本 schema 12、cache v26/转换 v29、部署 29/manifest 17、GRPO rollout 7；字段归一化契约为 4。

完整历史 bank 保存原始 FP32 ABS、相邻 DELTA、null 和 reset；当前状态也保存 raw ABS/DELTA，场景保存完整绝对时间。ContextEncoder 在模型外统一进行窗口 gather、ABS 重锚、场景裁剪排序及有符号归一化，数值输入保持 FP32。基础状态只有 time_seconds 以首个可见 request_state.time_seconds 为原点，首条使用偏移、后续使用 DELTA，缺失恢复和重锚仅为时间列设置 ABS reset；MP、布尔、层数、量谱和有上限的剩余时长逐步使用归一化绝对值，技能三元组也保持绝对值。战斗时间和场景起止坐标使用保存的 normalizer 契约 time_scale，GCD、冷却、Buff/DoT 剩余时间、明确声明秒单位的量谱及场景持续时长使用 duration_scale，两者默认均为 270 秒；时间坐标和场景持续时长不 clipping。MP/耗蓝除以 10000，计数资源和已注册 Buff 层数除以各自上限；场景目标数截断到 0～5 后除以 5，单次威力截断到 0～2000 后除以 2000。状态中的当前层数除以保存的技能注册上限，可用性保持 0/1，下一层恢复秒数除以 duration_scale；历史技能移除充能与恢复时间两列，保留的 max_charges 仍除以冻结记录中的自身原值，累计威力不进入模型，只在真实执行 metadata 中用于 PPG。模型新增零初始化的 state_reset_proj；超过共享 token 预算时保留 K=8 条完整状态/动作记录并重新累积；每次重置按新历史锚点重新计算有效场景，离线随机读取和实时回放共用确定性的分块窗口规则，真实引擎时间、伤害与 PPG 分母不重置。ONNX 宿主执行同一编码，图接收已编码的 14 个输入，包含独立的 `history_dead_positions`／`history_dead_mask`；重锚后 KV prefix 必须重建。

上下文字段声明集中于 context_fields.py；raw batch 的场景使用 scene_abs_values，编码后只保留 scene_vectors。dataset 在读取边界显式映射现有 cache 磁盘场景字段，本次模型字段、技能归一化和绝望 MP 门槛变化必须一次性重建旧产物，后续读取参数调整仍复用同一 raw cache。Normalizer 只提供正式状态编码元数据与技能归一化，不再保留旧状态 normalize/inverse 接口。LiveBatchBuilder 的 history_cursor 通过独立 context_metadata 保存，窗口长度由 history_mask 得出，起点由 cursor−length 得出；GRPO rollout 7 同样独立序列化元数据，并在文件头/manifest 保存唯一输入契约、校验内外动作顺序；死亡位置与 mask 随 prepared batch 保存，旧输入契约的 rollout 需重新生成。此次技能状态表、历史特征、mp_ratio 移除及绝对状态/双时间尺度变化必须从 raw 重新转换旧 PT；checkpoint、GRPO 轨迹和部署包也需按新契约重新生成。历史消融先按模型 context_capacity/history_reset_keep 选择正式窗口，再按 max_history 保留最近 N 条，消融长度不改变正式重置周期。

## 模型结构与产物兼容性

输入编码取消各类独立 LayerNorm，在 content 与 role 相加后统一执行一次无参数 RMSNorm，epsilon 为 1e-5；Transformer 主干统一要求 PreNorm，两处子层归一化及最终读出采用可学习 RMSNorm，epsilon 固定为 1e-20，在 FP32 中统计并归一化，乘通道尺度 γ 后恢复输入 dtype。γ 初始化为全 1，保存在各层 `norm1.weight`、`norm2.weight` 及 `encoder.norm.weight`；当前 6 层、384 维共 13 组、4992 个参数。`full_attention_residuals=false` 使用 Hyper-Connections：初始 token 复制成 `hc_mult` 路（当前 4 路），每层 attention 与 FFN 各有一组 `fn/base/scale`，在 FP32 中由展平残差流动态计算 pre、post 与残差混合矩阵；混合投影的 RMS 统计固定使用 1e-20，pre 为 sigmoid 加 `hc_eps`，post 为两倍 sigmoid，矩阵先逐行 softmax 再按列及交替行列做 Sinkhorn 归一化，当前 `hc_sinkhorn_iters=20`、`hc_eps=1e-6`。首个 attention 选择第一路，之后每个子层使用前一子层给出的 pre，末端按最后 FFN 的 pre 加权求和后执行最终 RMSNorm；四个旧 `residual_mix_*` 配置已删除。新训练的 fn 使用 Xavier 随机初始化，scale 为 0.01，pre base 为 logit(1/hc_mult)，post base 为 0，残差混合 base 对角为 4、其余为 0；这些是本项目从头训练的初始化选择。Full AttnRes 使用独立深度聚合路径，不创建 HC 参数。Q/K 在 RoPE 后逐 head 执行无参数 RMSNorm（`eps=None`），各自乘以保存的 `model.qk_norm_scale`（黑魔 Artzip 当前为 1.2）；KV-cache 只对新 K 归一化一次，缓存仍为单组 MQA K/V。

SwiGLU 的 up 与 gate 投影在 FP32 中裁剪：up 限制在 `[-swiglu_limit, swiglu_limit]`，gate 在 SiLU 前仅限制上界 `swiglu_limit`，再执行 `SiLU(gate)*up` 并恢复投影精度进入 down；当前阈值为 10.0。`ff_dim=1536` 仍按三投影参数预算折算为实际 1024 维隐藏层，GELU/ReLU 不执行该裁剪。SwiGLU 阈值、HC 路数、Sinkhorn 迭代数及 epsilon 均保存在模型配置与 checkpoint 中，不从本机 YAML 覆盖已保存的执行契约。可学习主干 RMSNorm 的参数、epsilon 与计算精度纳入输入契约，本分支仅由 26 升至 27，部署 schema 同步拒绝旧描述；旧无参数主干的 checkpoint 需重新训练，部署包与 GRPO 轨迹需重新生成。本次主干变化不改变输入数值、raw bank 或 compiled cache，既有数据缓存可继续复用。

CUDA 训练且启用梯度时，HC 的系数生成、读流及残差合成使用独立的 `torch.compile` 入口，将局部 checkpoint 与反向一起编译；系数与 Sinkhorn 使用延迟导入的 Triton 核，在片上迭代小矩阵，前向不保存逐轮中间量，反向临时重算后解析求导。编译入口显式关闭 autocast，保证 HC 投影及反向矩阵乘仍为 FP32。加权读流与紧随的 RMSNorm 合并编译，保留读流转回原 dtype 的舍入，并按统一公式在 FP32 中归一化、乘 γ 后恢复输入 dtype。残差合成融合乘加、归约与精度转换，不物化五维广播乘积。RoPE 每次编码只计算一份 FP32 cos/sin，供 Q/K、所有层及重算共用；KV-cache 只计算本次新增位置的三角函数。SwiGLU 使用融合前后向，只保存原始 up/gate，不再叠加 CUDA 门控局部 checkpoint；CPU 保留 HC 与 SwiGLU 局部 checkpoint，整 FFN 重算启用时不再嵌套 CPU 门控重算。局部重算均使用 `use_reentrant=False`、`preserve_rng_state=False`，确定性计算内部没有 dropout；dropout 保留在外层，验证、推理与 ONNX 导出直接执行原公式。上述训练融合不替换模型模块，不额外改变参数身份或已声明的模型契约。

动作读出使用独立无 bias 输出头，初始化按 `action_to_vocab_id` 顺序复制技能 embedding 的动作行，后续独立训练；以 FP32 执行 `s*tanh(logits/s)`，s 由保存的 `model.logit_softcap` 提供，黑魔 Artzip 当前为 15.0。训练、trace、KV-cache、Full AttnRes 与 ONNX 共用动作头和 softcap；ONNX 适配层在 softcap 后将输出恢复为部署目标精度，图内不包含宿主重复惩罚和合法性过滤。checkpoint 恢复已学习的独立头与保存的两种尺度，不从本机 YAML 补默认值。黑魔 Artzip 主线为 6 层、d_model 384、6 个 Q head、1 个 KV head、ff_dim 1536。此次输入归一化与状态字段修改需一次性重建 raw 双 bank cache；后续容量、保留数及随机裁剪变化复用同一 cache；time_scale 或 duration_scale 变化进入缓存签名，需重新编译 cache。旧共享头、未归一化 Q/K、旧普通相加残差、旧 LayerNorm 主干或旧输入编码的 checkpoint 和部署包拒绝复用，需重新训练、导出；旧技能时间字段或旧状态语义产物仍须重建。

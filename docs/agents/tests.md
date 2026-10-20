# tests

路径相对仓库根目录。返回[项目目录](./项目目录.md)。

本文件集中记录 `fight-engine/` 的纯 Rust 测试、`Combat.Sim/FightEngine.Tests/` 的现有 xUnit 测试、`tests/` 下的 Python 与 Node 测试，以及 pytest 配置和测试产物。目录按原有位置展示。Rust 测试消费真实 C# 生成的 JSON 基线，不包含或调用 C#；临时差分探针仅放在忽略目录 `.tmp/rust-migration/`。

## 目录与覆盖职责

```text
FFXIV_CCG/
├── fight-engine/
│   ├── tests/                                       核心配置、状态、系统规则、职业契约、动作生命周期与上下文对比；不依赖具体职业 crate
│   │   ├── common.rs / common/jobs.rs               通用比较及职业回放驱动，直接放实现，不使用 mod.rs
│   │   └── fixtures/                                JSON 基线记录来源提交、输入和预期结果
│   └── jobs/{black-mage,machinist}/tests/            各职业注册后独立验证；核心可在无职业代码时运行系统测试
├── Combat.Sim/FightEngine.Tests/                     xUnit 测试工程，按 Common/Config/Facade/Jobs/Outputs/Policy/Sessions/Skills/System 分组
│   ├── Common/                                       公共层测试目录
│   │   ├── ProjectConfigLoaderTests.cs               配置加载测试：真实仓库解析 + 校验错误路径
│   │   ├── SceneWindowTests.cs                       scene 契约常量/索引映射缓存/窗口索引解析/缺失字段抛错（常量来自 config/schema.yaml）
│   │   └── YamlConfigTests.cs                        YAML 标量解析边界测试：前导零拒绝/hex/bin/下划线/inf/引号
│   ├── Config/                                       共享 schema 测试目录
│   │   └── SchemaConfigLoaderTests.cs                未 Load 行为锁定（反射重置单例）、严格字段组访问、团辅窗口模板展开
│   ├── Facade/                                       门面测试目录
│   │   ├── CombatStateMachineTests.cs                门面：职业路由/构造校验/决策循环/冷却/可用技能/回放缓存/历史
│   │   ├── FacadeKit.cs                              最小内嵌配置与测试职业注册辅助
│   │   ├── JobSimulatorTests.cs                      绝对时间门面：提交/队列/推进/观测/外部事实/snapshot/fork 及失败不推进，覆盖请求时刻锚定 GCD 与队列期间立即 oGCD
│   │   ├── RandomSequenceReplayTests.cs              固定 seed 绝对时间随机回放及共享引擎 16 队列串行/并行一致性测试
│   │   ├── SequenceRunnerTests.cs                    序列回放辅助：绝对时间请求、最终 canonical 输出、完整历史、16 队列并发隔离与异常释放
│   │   ├── SkillAvailabilityBuilderTests.cs          冻结接受表、132 维布局、同刻批次原子性、父子队列及 policy 布局隔离回归
│   │   └── TestJobStateMachine.cs                    测试最小职业，供门面组合逻辑复用
│   ├── Jobs/                                         职业注册中心与职业状态机测试目录
│   │   ├── Black.Mage/                               黑魔职业状态机测试目录
│   │   │   └── BlackMageJobStateMachineTests.cs      黑魔规则：AF/UI/火苗/悖论/通晓/雷云/灵极魂/回蓝/人工循环回放
│   │   ├── Machinist/                                机工职业状态机测试目录
│   │   │   └── MachinistJobStateMachineTests.cs      机工规则：资源/连击/过热/替代技能/野火/起爆/downtime
│   │   ├── JobMachineRegistryTests.cs                同类型幂等/冲突抛错/工厂占用/查询与排序/自动发现后手动注册同类型幂等
│   │   ├── JobResourceAccessorsTests.cs              公共量谱访问基类：资源读写、最大值、类型转换与未绑定 fail-fast
│   │   └── RegistryTestJobs.cs                       注册测试专用职业 A/B（同 tag 不同类型）
│   ├── Outputs/                                      输出层测试目录
│   │   ├── ModelStateSemanticsTests.cs               跨步状态、回退、请求冻结、统一 wait、排队/同戳、窗口外前序及 fork/restore 回归
│   │   ├── OutputsHistoryTests.cs                    历史上下文：MP/量谱前后与消耗、系统/职业 Buff、DoT、威力缩放、秒/GCD 快照
│   │   ├── OutputsPrecisionTests.cs                  tensor 精度：bfloat16/float16 量化 torch 边界参考、dtype 别名/缺省键/错误消息
│   │   ├── OutputsSchemaTests.cs                     canonical/tensor 结构、空 scene、schema 元数据、2.17 GCD 缩放、precision 解析
│   │   ├── OutputsTestKit.cs                         输出层测试共享：仓库根定位/状态机构造/向量取值辅助
│   │   └── OutputsTokensTests.cs                     历史/当前状态 token、稳定数值字段、动作顺序/合法性/价值及禁止未来预演测试
│   ├── Policy/                                       模型控制动作边界测试目录
│   │   ├── PolicyContextBuilderTests.cs              wait 不进入状态机、固定输出动作空间、policy 决策无游戏副作用及冲突校验
│   │   └── PolicySequenceReplayTests.cs              批量与逐步回放逐值等价、同刻事实、等待窗口、排队与失败停止
│   ├── Sessions/                                     多队列宿主测试目录
│   │   ├── SimulationDeltaObservationTests.cs        增量与完整观测逐值等价、完整保留约束、游标校验与错误零变更
│   │   └── SimulationEngineTests.cs                  多队列并发与独立模拟器等价、生命周期、错误隔离、基础 GCD 隔离和长程记录边界
│   ├── Skills/                                       技能索引测试目录
│   │   └── SkillBookTests.cs                         按 key/id 查询/重复抛错/启用过滤/项目合并顺序及技能可选集合默认值
│   ├── System/                                       系统状态机测试目录
│   │   ├── CombatTimelineRuntimeTests.cs             时间线内核：绝对时间、稳定顺序、mutation、快照/fork
│   │   ├── PlayerStateRuntimeTests.cs                玩家态：时序应用/时间推进/停手切换/六项校验
│   │   ├── PublicTimelineResourceTests.cs            公共资源：分段等价、截止时间、刷新取消、MP/DoT 边界与绝对时间原点
│   │   ├── RegistryTests.cs                          Buff 与量谱注册中心测试
│   │   ├── SystemStateMachineTests.cs                门面：五段编排/回蓝/威力解析/历史
│   │   ├── SystemTestKit.cs                          测试共享配置与技能构造辅助
│   │   ├── TimelineAuthorityTests.cs                 时间线 API 权威性守卫：唯一推进内核、稳定优先级、轻量动作协议及阶段 4 旧接口/兼容类型零残留
│   │   └── TimelineTests.cs                          状态/DoT/冷却时间轴测试
│   ├── GlobalUsings.cs                               测试程序集共享命名空间导入
│   ├── SystemTimelineTestDriver.cs                   系统规则测试专用的绝对时间线驱动器
│   ├── TestSchemaInit.cs                             测试程序集加载时统一初始化共享 schema（ModuleInitializer，config/schema.yaml）；程序集级禁用并行（单例状态重置测试隔离）
│   └── TimelineTestDriver.cs                         职业/输出规则测试专用的绝对时间驱动器
├── tests/                                            测试目录，按源代码模块拆分，并包含根目录公共工具回归测试
│   ├── fixtures/                                     固定提交采集的回归基线
│   │   └── state_refactor_non_state_baseline.json.br  重构前 canonical/raw/model 非状态字段及来源提交、源码哈希
│   ├── grpo/                                         独立 GRPO 后训练测试
│   │   ├── test_grpo.py                              GRPO 配置、优势/loss、变长 batch、CLI 和优化器更新测试
│   │   └── test_optimizer.py                         GRPO 优化器的磁盘与 CPU 快照回滚、FP32 主权重、双优化器状态和调度恢复测试
│   ├── scripts/                                      脚本模块测试子目录
│   │   ├── autoregressive_replay/                    自回归回放测试子目录
│   │   │   ├── __init__.py                           子包入口
│   │   │   ├── conftest.py                           回放测试共享 fixture：dotenv 隔离与最小 checkpoint/scene 文件构造
│   │   │   ├── test_config.py                        回放配置加载测试，覆盖 dotenv 装配、env 覆盖、ONNX manifest 路由和非法模式/上限拒绝
│   │   │   ├── test_context.py                       scene 上下文测试，覆盖 scene 模板归一化/目标数解析、targetable/移动/团辅同步和 live batch 构造
│   │   │   ├── test_parallel.py                      并行调度容量、顺序、合批 mask、失败唤醒、固定 batch 后端、共享缓存和有界性能统计测试
│   │   │   ├── test_parallel_integration.py          真实 C#、Transformer FP32/CUDA BF16、KV cache、PPG 与 GRPO 随机流的串行／并行等价测试
│   │   │   ├── test_parity.py                        PT/ORT parity 测试，覆盖 backend 对比失败报告、固定容量 batch device、完整失败轨迹审计、有限容差和 parity CLI
│   │   │   ├── test_ppg.py                           自回归 PPG 的 GCD 计数、直接/DoT 威力累计、当前状态 GCD 恢复与固定动作顺序校验测试
│   │   │   ├── test_replay.py                        回放核心逻辑测试，覆盖预测模式、解码、KV-cache 开关、历史消融、事件时间线、无合法动作推进、共享 session/cache 生命周期和 Markdown/CLI 输出
│   │   │   ├── test_scheduler.py                     共享 `DecisionScheduler` 的绝对时间推进、scene 边界同步、读条/GCD/oGCD 调度与无法动作时间调度测试
│   │   │   └── test_shared_entrypoints.py            冷缓存并行补编译与回放共用单引擎、显式资源归属、16 队列自检、历史消融及多场景 CLI/验收报告隔离测试
│   │   ├── convert_fflogs/                           FFLogs 转换测试子目录
│   │   │   ├── __init__.py                           子包入口
│   │   │   ├── _non_state_baseline.py                从指定源码提交采集固定非状态 canonical/raw/model 证据；拒绝覆盖冻结基线
│   │   │   ├── test_action_quality_labels.py         评估标签准入、精确原始事件关联及 PT 落盘测试
│   │   │   ├── test_cache.py                         raw JSON → compiled cache 测试，覆盖 raw 只读、cache 写入、worker 和训练读取
│   │   │   ├── test_cache_config.py                  缓存公共配置比较与转换 YAML 依赖回归，覆盖依赖增删、重要字段及顺序变化、注释不变性和无关配置排除
│   │   │   ├── test_config.py                        FFLogs 转换配置测试，覆盖统一 `FFXIV_JOB_TAG`、职业级探针技能和显式参数优先级
│   │   │   ├── test_convert.py                       FFLogs 转换脚本测试，覆盖动作提取、begincast.duration 基础 GCD 校准、整场战斗保留、多目标辅助标记、场景上下文装配和训练样本构建链路
│   │   │   ├── test_cs_backend.py                    InProcessBackend 动作校验、读条覆盖、无子进程调用和 Python 原生观测类型测试
│   │   │   ├── test_history_deduplication.py         冻结历史前缀复用、完整 bank、公开观测和样本修改隔离及存储规模回归
│   │   │   ├── test_incremental_history.py           增量游标与三通道一致性、批量回放和逐步回放逐值等价及分片批量编码回归
│   │   │   ├── test_movement.py                      连续速度移动窗口回归，覆盖长观测间隔、二维速度、贝塞尔交点、GCD 合并、实际读条与瞬发保留、最短长度和正式转换场景平移
│   │   │   ├── test_process_compile.py               Python 编译进程与共享状态机请求通道、错误语义、借用引擎隔离及进程/队列/管道回收回归
│   │   │   ├── test_timestamp_normalization.py       FFLogs 请求时间恢复与全局归零测试，覆盖硬读条 0.5s 提前结算、瞬发原时刻、场景同步平移、取消读条配对拒绝和小范围动作顺序稳定化
│   │   │   ├── test_training_builder_basic.py        raw 转换训练样本基础时序与回放测试，覆盖 GCD 覆盖、首拍预读、ogcd_wait 注入、施法时间戳和冷却回放
│   │   │   ├── test_training_builder_label.py        raw 转换训练样本 label 测试，覆盖动作前上下文、直接合法性校验和 label leak 断言
│   │   │   └── test_training_builder_scene.py        raw 转换训练样本 scene 测试，覆盖绝对时间场景上下文、运行时标记和停手边界
│   │   ├── fflogs_scraper/                           FFLogs 下载工具包的职责测试
│   │   │   ├── __init__.py                           测试子包入口
│   │   │   ├── conftest.py                           共用报告、玩家、宠物和敌人实例元数据
│   │   │   ├── test_api.py                           GraphQL 参数校验、事件完整分页与历史排名查询口径测试
│   │   │   ├── test_batch_config.py                  批量下载比例 YAML 校验与向上取整配置测试
│   │   │   ├── test_cli.py                           模块入口、命令分发与根目录 .env 定位测试
│   │   │   ├── test_contracts.py                     报告与事件适配、玩家归属和训练字段兼容测试
│   │   │   ├── test_download.py                      下载模式、分档保存、同档补位、已有文件完整性与提前停止测试
│   │   │   ├── test_encounters.py                    下载目标 YAML、单战斗 Zone 及 M12s 上下场映射测试
│   │   │   ├── test_io.py                            URL、文件名安全和 UTF-8/LF 原子写入测试
│   │   │   └── test_sampling.py                      十档配额、百分位边界、非最佳历史记录、匿名/隐藏排除、记录去重和 API 分区一致性测试
│   │   ├── onnx_export/                              独立 ONNX 导出脚本测试子目录
│   │   │   ├── __init__.py                           子包入口
│   │   │   ├── test_config.py                        ONNX `.env` 合并、checkpoint→部署目录映射、无重复 history 容量和 parity 参数校验
│   │   │   ├── test_exporter.py                      固定容量/全 dtype 零补位、完整 BF16 版本矩阵、公共确定性 golden I/O、TensorProto dtype 常量、精度专属 parity 容差、产物 SHA 绑定、发布状态晋升、失败不覆盖、profile 路径、ORT provider、报告兼容，以及安装 ONNX extras 后本地执行的 tiny checkpoint 导出/checker/PT-ORT parity 快速门禁
│   │   │   ├── test_policy.py                        纯 Tensor policy 测试，覆盖 scene 类型投影、稳定 masked softmax、逐层 padding 矩阵、raw logits 对齐、策略边界和 `torch.export` 图捕获
│   │   │   ├── test_real_checkpoint_padding.py       本地真实 46.9M 参数 BF16 checkpoint 的 profile/vocab/容量与 CUDA padding 回归；`RUN_REAL_ONNX_EXPORT=1` 时额外执行 BF16 export/checker/ORT 全链路慢速门禁，artifact 缺失时跳过
│   │   │   └── test_tensor_runtime.py                ORT Tensor I/O 职责级测试，覆盖 CPU I/O Binding、bool NumPy 回退、非 bool DLPack 路径和 CUDA BF16 golden NPZ 字节确定性
│   │   ├── __init__.py                               子包入口
│   │   ├── action_quality_runtime.test.cjs           Node 职业路由、实例事件匹配、消息序列化与机工量谱循环引用边界测试
│   │   ├── conftest.py                               scripts 测试公共 fixture：cs_backend（托管状态机运行文件未构建时跳过）与 cs_skill_book，转换与自回归回放测试共用
│   │   ├── test_action_quality.py                    离线标注 raw 保留、目录映射、失败保护、.env 和职业权重隔离测试
│   │   ├── test_action_quality_pool.py               常驻 Node 客户端与线程池的并发、复用、异常、取消和资源回收测试
│   │   ├── test_context_packet.py                    批量桥接的基础类型、数值边界、可变容器隔离、并发编码、GC 生命周期及异常输入回归
│   │   ├── test_dataset_compression.py               旧战斗数据压缩迁移、标注来源一致性和验证后删除测试
│   │   ├── test_dataset_layout.py                    数据阶段路径映射、区间目录校验、旧目录兼容与转换按副本/区间分组测试
│   │   ├── test_file_inventory.py                    持久校验复用、依赖变化、索引损坏回退和 2000 份已有文件中只校验 200 份变化与 200 份新增的测试
│   │   ├── test_inprocess_engine.py                  16 个 Python 线程共用引擎、6 个转换任务完整历史等价，以及容量、失败重置和关闭隔离测试
│   │   ├── test_model_analysis.py                    科研模型分析工具的独立 token 元数据、显式决策位置、职业标签、固定动作/DataSpec 对齐、PCA、hidden/attention/state/skill PNG 输出和 CLI metadata 测试
│   │   ├── test_resume_menu.py                       Linux 菜单分发与交互续训在验证份数变化或旧 checkpoint 缺字段时的确认测试
│   │   ├── test_scene_source.py                      参考场景选择测试：高百分位优先、无效缓存回退及训练/VAL/回放脱离转换配置的只读复用
│   │   └── test_scene_state.py                       scene 事实流端点、状态各段自身时间、feature key 重排、原始时间精度和非法时间拒绝回归
│   ├── training/                                     训练模块测试子目录
│   │   ├── __init__.py                               子包入口
│   │   ├── _causal_fixtures.py                       固定动作 DataSpec、新输入契约和因果 batch 的测试夹具
│   │   ├── _common_fixtures.py                       训练公共层测试共享的 PT、compiled cache 和演示战斗构造辅助；训练样本经 C# 状态机后端构建
│   │   ├── test_action_space.py                      固定动作空间的启用排序、共享 embedding 行和保存 GCD 类型契约测试
│   │   ├── test_attention_residual.py                Full AttnRes 的均匀初始化与 pseudo-query 反向传播测试
│   │   ├── test_bc_monitoring.py                     BC 区间样本加权、实际更新学习率、真实编码输入/输出复用和全验证集诊断接入测试
│   │   ├── test_bc_optimizer.py                      共享 Muon/AdamW 参数分组、原生更新等价性、调度、状态恢复与 BC 训练集成测试
│   │   ├── test_bc_optimizer_checkpoint.py           BC 优化器续训契约测试，拒绝优化器切换或 Muon 参数不匹配并兼容旧 AdamW
│   │   ├── test_bc_optimizer_config.py               BC/GRPO 独立优化器 YAML 配置、默认 AdamW、重复和非法参数校验测试
│   │   ├── test_causal_data_contract.py              当前状态、固定动作、旧产物拒绝、完整 history bank 和窗口复用测试
│   │   ├── test_common_norm.py                       训练公共层归一化与 scene schema 测试，覆盖时间/威力/资源/状态上限和 scene 窗口索引
│   │   ├── test_common_pt.py                         训练公共层 PT、compiled cache、DataLoader 和 batch padding 测试
│   │   ├── test_common_vocab.py                      训练公共层 shard batch sampler 与序列过采样测试
│   │   ├── test_context_encoding.py                  raw 双 bank 编码、字段恢复、分段窗口、场景差分与 FP32 精度测试
│   │   ├── test_delta_initialization.py              新增零初始化 reset 投影不改变公共参数和初始化随机状态的测试
│   │   ├── test_grouped_attention.py                 MQA/GQA 投影、头数配置、按 KV 组 SDPA、零 stride K/V 广播、mask/梯度和 trace 权重形状测试
│   │   ├── test_independent_token_encoder.py         独立技能/状态投影、共享状态编码、因果顺序、RoPE padding、容量、梯度和旧融合 checkpoint 拒绝测试
│   │   ├── test_input_contract.py                    checkpoint 输入契约与归一化契约的序列化、恢复和缺失契约校验测试
│   │   ├── test_input_encoder_normalization.py       已编码 content 与 role 合成后单次 RMSNorm、空上下文、padding 独立性和各输入梯度测试
│   │   ├── test_kv_cache.py                          模型推理 KV-Cache 测试，覆盖历史追加、当前状态变化、上下文重建、Pre-Norm/Post-Norm、Full AttnRes、训练模式禁用、配置 K/V 头数和 checkpoint 参数稳定性
│   │   ├── test_losses.py                            主交叉熵 FP32 口径、动作质量等级与排名衰减、主/价值损失同步降权、可选辅助损失装卸测试
│   │   ├── test_model_helpers.py                     公共模型与训练流程边界测试，覆盖配置错误、模型输入校验、因果 attention/FFN dropout、训练循环、DataLoader 分支、职业路由防护、价值辅助损失、checkpoint 落盘和断点续训
│   │   ├── test_models.py                            公共模型测试，覆盖 compiled cache 动态维度、职业路由、position 编码、训练精度配置、技能价值辅助损失、推理路径和脚本层副本目录比例选 raw JSON
│   │   ├── test_optimizer_master_weights.py          BF16 小更新累计、FP32 原生参考、梯度生命周期、新旧主权重 checkpoint 恢复与 CUDA 更新测试
│   │   ├── test_output_head.py                       独立动作头复制初值、动作行顺序、FP32 softcap 与数值边界、分离梯度及低精度主权重连续恢复测试
│   │   ├── test_policy_actions.py                    独立 policy 动作配置与训练词表测试，覆盖 raw id 0 的 `ogcd_wait` 非 padding 映射、固定动作顺序及非法配置拒绝
│   │   ├── test_position_encoding.py                 共享 RoPE 公式、位置编号和各设备/精度一致性测试
│   │   ├── test_residual_mix.py                      正式残差混合的初始化、真实 skip、初始 token 梯度、trace、KV 和激活重算一致性测试
│   │   ├── test_runtime_debug.py                     训练运行时诊断测试，覆盖 RuntimeDebugRecorder 的 batch 上下文写入、Attention/FFN 阶段记录与事件聚合
│   │   ├── test_state_features.py                    有序状态布局、唯一读取/拼接、二值表、错序与坏原料拒绝回归
│   │   ├── test_swiglu.py                            SwiGLU 三投影门控公式、参数预算、GELU/ReLU 兼容性、激活配置 checkpoint 往返、主干激活与独立输入投影、旧融合布局拒绝测试
│   │   ├── test_tensorboard.py                       TensorBoard 配置校验、BC/GRPO 独立开关与共用事件根目录、启动器 `.env` 自动选择、可选依赖提示和标量过滤测试
│   │   ├── test_tensorboard_graph.py                 CPU FP32/CUDA BF16 真实模型结构图、重复惩罚和 RNG/参数/模式/KV/诊断状态恢复测试
│   │   ├── test_tensorboard_monitor.py               真实面板事件、Projector 行映射、有限有界分布、确定性 PR reservoir 和禁用路径测试
│   │   ├── test_train_cli.py                         训练 CLI 直接执行入口和 `--resume` 参数转发测试
│   │   └── test_training_optimizations.py            因果层等价性、重复惩罚缓存、padding、指标累积和 checkpoint/梯度等价性测试
│   ├── __init__.py                                   测试包入口，显式声明 `tests` 为可导入包
│   ├── conftest.py                                   pytest 公共引导与共享夹具，保证项目根目录可稳定导入、隔离 Windows 临时目录权限并提供默认黑魔/机工状态机
│   ├── helpers.py                                    测试辅助函数模块，集中封装自动推进、向量取值、复用正式 scene token 构造器的场景上下文构造、canonical 输出断言和 YAML 写入
│   ├── test_black_mage_skill_config.py               黑魔技能字段一致性及显式耗蓝、DoT、Buff、充能配置回归
│   ├── test_common_boundaries.py                     冷启动验证基础组件与共享训练配置不隐式加载脚本、训练或 PyTorch，并约束 common 不反向导入业务入口
│   ├── test_main.py                                  根目录 CLI 回归测试，锁定共享引擎与绝对时间自检命令
│   ├── test_policy_config.py                         策略配置清单测试，覆盖动作质量权重合并、相对路径解析与旧清单兼容
│   ├── test_torch_serialization.py                   根目录公共 PyTorch 安全文件读取测试，覆盖恶意 pickle 拒绝和旧 checkpoint Path 元数据兼容
│   └── test_yaml_config.py                           根目录公共 YAML mapping loader 测试，覆盖空文档/`null` 归一化和 falsey 非 mapping 顶层值校验
├── .pytest_cache/                                    pytest 默认缓存目录，属于测试副产物
├── .coverage                                         coverage 本地产物文件，属于测试副产物，不参与项目实现
└── pytest.ini                                        pytest 配置文件，指定测试目录和缓存目录
```

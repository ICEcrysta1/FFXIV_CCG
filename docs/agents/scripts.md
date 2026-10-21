# 脚本

路径相对仓库根目录。返回[项目目录](./index.md)。

本文件记录数据下载、离线标注、转换编译、模型回放、分析、ONNX 导出和根目录命令入口。共享模型与数据契约见[公共](./common.md)，底层引擎规则见[状态机](./fight-engine.md)。

## 目录与职责

```text
FFXIV_CCG/
├── scripts/                                          辅助脚本目录，负责 FFLogs 抓取、原始日志转换、训练样本导出、模型回放分析和独立 ONNX 导出
│   ├── action_quality/                               离线动作质量桥接，自动选择职业/副本模块，原始 JSON 加 analysis 后保存到 annotated 对应层级
│   │   ├── runtime/                                  Node 无界面分析运行层，不修改上游代码
│   │   │   ├── extract/                              通用与职业专属结构化证据提取器
│   │   │   │   ├── black_mage.cjs                    黑魔循环、Proc、DoT 和资源证据；展示优先级不当作严重度
│   │   │   │   ├── common.cjs                        通用读条中断、插入和连击问题定位
│   │   │   │   └── machinist.cjs                     机工野火/超荷/整备/机器人窗口、冷却偏移和量谱历史
│   │   │   ├── analyze.cjs                           调用上游适配器与 Parser，选择模块并构造统一分析契约
│   │   │   ├── attribution.cjs                       捕获规则实际命中的动作，在错误类别下保存事件引用并沿用整场等级
│   │   │   ├── bootstrap.cjs                         复用上游 Babel 变换，补齐 DOM 与样式加载环境；禁用 jiti 共享磁盘缓存以避免权限和并发写入故障
│   │   │   ├── pool.cjs                              单个 Node 宿主的有界工作线程池：请求分发、任务超时、退出回收与 RSS 软预算收缩
│   │   │   ├── pool_worker.cjs                       常驻分析线程：只初始化一次模块，每份日志独立分析，完成后回收临时对象
│   │   │   ├── references.cjs                        原始事件精确索引匹配，保留 actor 实例、重复和未匹配语义
│   │   │   ├── routing.cjs                           根据日志职业和副本选择模块，未知职业拒绝降级
│   │   │   ├── run.cjs                               Node 入口，兼容单请求与常驻线程池模式，经标准输入输出交换分析 JSON
│   │   │   ├── serialize.cjs                         消息、Map 和证据数据序列化，不执行网页组件
│   │   │   └── time.cjs                              标注统一使用战斗起点相对毫秒，事件、窗口和已声明证据时刻共用换算
│   │   ├── __init__.py                               包入口
│   │   ├── __main__.py                               python3 -m scripts.action_quality 入口
│   │   ├── cli.py                                    批量输入发现、清单预检后仅调度过期或缺失标注、进度、失败汇总与退出状态
│   │   ├── config.py                                 复用 .env、策略模型 YAML 和职业严重度权重，按 CPU、待处理数量与内存软预算限制并发
│   │   ├── node_pool.py                              常驻 Node 宿主客户端：有界并行请求、按身份分发响应、超时取消与进程/管道回收
│   │   ├── runner.py                                 调用 Node、校验源文件身份并原子保存合并 JSON；规范产物以字节前缀校验原始字段，旧排版回退语义比较
│   │   └── scan.py                                   标注索引预检与有界进程校验，未变化文件只查元数据，定期保存进度并支持取消后续扫
│   ├── autoregressive_replay/                        状态机驱动的模型自回归回放包，通过共用 `PolicyBackend` 在 PyTorch checkpoint 与已验证 ONNX 部署包之间切换；Rust 状态机由进程内 `InProcessBackend` 直接调用，模型后端共用 live batch、重复惩罚、合法 mask、采样和轨迹推进，并支持逐决策 PT/ORT parity、性能/PPG 报告及固定完整轨迹上的历史长度消融；并行轨迹各持有独立 `AutoregressiveReplaySession`，共用一个多队列引擎与有界 `ReplayCacheStore`，单 session 也支持按轨迹 reset 复用
│   │   ├── outputs/                                  回放结果输出模块目录
│   │   │   ├── __init__.py                           回放输出导出入口
│   │   │   └── markdown.py                           输出 backend、实际 Execution Provider、模型来源、PPG、性能指标及逐步动作/合法 Top-K，宿主控制动作 `ogcd_wait` 简写为 `-`；历史消融结果额外输出完整轨迹动作对照列
│   │   ├── __init__.py                               回放包导出入口
│   │   ├── __main__.py                               `python3 -m scripts.autoregressive_replay` 入口
│   │   ├── backends.py                               PyTorch/ORT 共用 policy backend 契约；ORT 在创建 session 前严格校验部署包和已有发布证明，集中执行固定容量补位、raw logits 与有序 Top-3 parity、宿主策略选择一致性及延迟/峰值内存统计
│   │   ├── batch_replay.py                           多场景回放与历史消融入口 run_replays：单模型、共享引擎和有界缓存；每批先用同一引擎并行补编译，再按输入顺序输出结果
│   │   ├── config.py                                 回放配置加载，按 `.env` / CLI 选择 PyTorch checkpoint 或 ONNX 部署包；ORT 从 manifest 自动恢复职业和正式 history 容量，同时读取 scene、解码和时间轴参数；PyTorch 可显式覆盖分析精度，ORT 精度始终以 manifest 为准
│   │   ├── context.py                                使用 Rust 状态机后端实时历史/当前状态上下文，按绝对时间 cache scene 与空 scene 提交类型化外部事实；通过 `apply_external_events` 整批同步 Boss 可选中、移动、目标数与团辅窗口，移动窗口在结束前 0.5 秒进入滑步豁免；通过 `observe_at` 构建 live batch，将实际死亡回执映射至所选历史窗口；`build_from_canonical` 接收快照冻结的死亡标记，支持历史消融旁路重算
│   │   ├── main.py                                   回放 CLI 编排入口，支持 backend/ORT EP、PyTorch 显式分析精度、temperature/top-p、历史消融，以及 `--parity-onnx-package` 双后端完整轨迹验收
│   │   ├── parallel.py                               有界多队列调度：工作线程并行推进 Rust，调用线程合批模型输入；按场景、动作和死亡事件各自的轴补齐变长 mask，处理批次 KV 生命周期、任务顺序与失败唤醒；PPG、GRPO、普通回放及 parity 共用；支持批次准备、独立验收记录器和单轨迹推理失败隔离
│   │   ├── parity.py                                 多轨迹共用 PT/ORT 模型和引擎，每条轨迹独立记录逐决策结果；固定单条推理契约，发布状态由调用线程串行登记；拒绝非有限或负数容差，中途不因差异短路，成功或失败都输出绑定四类产物 SHA、目标运行库、raw logits、双方有序 Top-1/Top-3、宿主选择、首个分叉、性能、动作序列和 PPG 的完整 JSON；仅显式 workflow 门禁可登记部署包，普通 CLI 始终只写调试报告
│   │   ├── ppg.py                                    训练期间的 PPG 评估：验证副本通过共享引擎并行回放和模型合批，none_ppg 使用同一接口运行单队列；保留完整时长、失败计零与逐副本平均语义
│   │   ├── replay.py                                 状态机自回放与队列 session，显式借用共享引擎和模型后端，使用有界线程安全 cache；真实技能、策略等待及 scene 事实共用绝对时间调度，只有历史消融保存全程观测快照
│   │   └── scheduler.py                              单轨迹绝对时间决策调度：动作占用、场景边界与无合法动作时的下一事件推进
│   ├── common/                                       脚本包公共辅助目录：承载跨脚本复用的进程内 Rust 状态机适配器、结果类型与场景事实/输出改写层
│   │   ├── __init__.py                               公共辅助包入口，不隐式导入可选依赖
│   │   ├── file_inventory.py                         下载、标注与转换共用的持久校验索引；未变化文件只检查元数据，业务契约由所属流程校验，索引损坏自动重建
│   │   ├── inprocess_backend.py                      Rust 原生多队列适配器：惰性加载扩展，InProcessEngine 共享规则，InProcessBackend 持有队列句柄；Rust 释放 GIL 执行与编码，标准库还原独立 Python 容器
│   │   ├── json_io.py                                下载、离线标注与转换共用的 `.json.br` 识别、Brotli 解压读取及原子写入；普通 JSON 写入仅供其他小型产物使用
│   │   ├── numeric.py                                转换与回放共用的嵌套数值映射展开，字段过滤由调用方指定
│   │   ├── scene_source.py                           分析与回放共用的参考场景选择，只读校验已有 compiled PT，按百分位从 90-100 优先选择
│   │   ├── scene_state.py                            共享 raw FP32 场景执行视图、同刻事实批次和纯 ETA/停机剩余合成；两段使用各自冻结时间，不回写移动/scene token
│   │   └── state_machine_types.py                    进程内 Rust 状态机 API 的不可变返回类型：时间线位置、动作提交/校验、外部事实、policy 决策和观测结果
│   ├── convert_fflogs/                               FFLogs 原始 JSON 转换包，负责提取动作、构建场景上下文、驱动状态机回放并直接编译最终训练 cache
│   │   ├── cache/                                    compiled cache 子包：编译编排、加载与 shard 写入；路径选择位于共用数据层
│   │   │   ├── __init__.py                           compiled cache 门面，转发 raw 路径选择、缓存编译和缓存加载入口，不承载具体实现
│   │   │   ├── cache_compile.py                      raw JSON → compiled cache 的编译编排，训练副本/区间与 VAL 副本同批处理，各组先复用有效 PT、再从组内补编译，最后统一报告缺额；单路直接执行，多路通过有界 Python 进程共享父进程的多队列引擎，进程数不超过可用 CPU、在途任务不超过用户并发上限；编译固定保留完整状态机历史与完整 history bank，每个样本只保存历史引用
│   │   │   ├── cache_config.py                       转换缓存配置依赖声明，收集实际移动/GCD、标签准入、状态机、当前职业、schema 和转换选项 map；训练窗口与本机并发不参与
│   │   │   ├── cache_load.py                         raw JSON 对应 compiled cache 的签名校验与加载，以及转换预检索引；索引绑定源文件、manifest、分片和正式配置契约，只读取最终 cache，不触发 raw 转换；支持由 replay session 注入共享 shard cache
│   │   │   ├── cache_writer.py                       最终 compiled cache manifest/shard 写入器，只写 `.cache`，不写 raw 或中间训练格式；PT 内保存配置依赖 map 供转换阶段判断重建；manifest 持有 source 级 history bank 与稳定 `history_bank_id`，shard 样本只写 `history_end` / `history_length`
│   │   │   └── process_compile.py                    转换专用 spawn 进程池与有界请求通道：子进程解析、编码、写缓存，父进程独立服务线程操作共享状态机队列；仅传请求和增量观测包，子进程 Torch 内部线程固定为 1，异常及退出时回收队列、管道和进程
│   │   ├── config/                                   转换配置子包：YAML 与根目录 `.env` 加载、GCD 参数校验和共享常量
│   │   │   ├── __init__.py                           转换配置门面，导出 `config/convert_fflogs/` 与根目录 `.env` 加载入口
│   │   │   ├── config.py                             FFLogs 转换配置加载模块，负责读取 `config/convert_fflogs/`、校验 GCD 与坐标移动窗口参数、解析根目录 `.env` 的 `FFXIV_JOB_TAG` 和职业路由
│   │   │   └── constants.py                          共享常量与时间精度工具
│   │   ├── extraction/                               日志提取子包：动作与请求时刻提取、停手窗口检测和战斗载荷装帧
│   │   │   ├── __init__.py                           日志提取与战斗载荷装帧门面
│   │   │   ├── action_quality.py                     按原始事件索引将准入的评估建议关联到目标动作，拒绝身份、技能和时间错配
│   │   │   ├── downtime.py                           按本场全队玩家对敌方的施法事件推导 Boss 停手窗口；左界取最后 `cast`，右界取恢复段最早请求（优先配对 `begincast`，无有效配对时取 `cast`），不依赖伤害落地
│   │   │   ├── extraction.py                         FFLogs 事件提取、请求时刻解析、动作移动标注与 GCD 检测；硬读条请求取配对的 `begincast`，读条时长优先使用 `begincast.duration` 并结合技能表基础 GCD 校准，样本不足时回退间隔检测；无 begincast 的瞬发取 `cast`，首个硬读条按 schema 滑步窗口处理，读条来源与顺序校正量均保留供审计
│   │   │   ├── fight_payload.py                      战斗载荷装配：以最早请求为 0 同步平移动作与场景事实、保留 request/cast/effect 审计字段，并把 AoE 命中事件交给 scene builder 推导双目标辅助标记；不承载状态机目标倍率逻辑
│   │   │   ├── movement.py                           从玩家来源和目标坐标的贝塞尔连续速度曲线按阈值截取窗口；按实测 1 GCD 间隔合并，扣真实读条，再剔除短于 GCD 减滑步时间的片段
│   │   │   └── movement_curve.py                     转换模块内部的三次贝塞尔连续速度曲线与水平阈值交点；控制点限制相邻速度范围，避免负值与额外峰值
│   │   ├── scene/                                    scene window 子包：窗口构造、输入规范化与绝对时间查询
│   │   │   ├── __init__.py                           scene window 子包入口，重导出 scene build/query 的正式入口
│   │   │   ├── scene_builders.py                     scene window 构造、输入规范化和 token 生成；从实际双目标命中事件生成仅含 `target_count >= 2` 的辅助标记，按 2s 观测超时切段并在最后事件后保留 1.5s，不生成单目标/停机填充 token
│   │   │   ├── scene_context.py                      scene context 公共导出 facade，汇总 builders 与 queries 的正式入口，包含多目标辅助标记 builder/token
│   │   │   └── scene_queries.py                      绝对时间 scene window 查询：完整 scene_context 视图、人工排查 anchor 与停手总时长；移动与停手标量由 `scripts/common/scene_state.py` 承担
│   │   ├── source/                                   raw source 子包：raw JSON 只读解析与内存训练源读取
│   │   │   ├── __init__.py                           raw source 门面，导出 raw JSON 转换与内存训练源读取器
│   │   │   ├── raw_source.py                         raw JSON 只读解析入口，从批次共享引擎创建完整历史队列，成功或失败均释放队列；独立调用也使用同一转换路径，结果在内存中交给 cache 编译
│   │   │   ├── source_helpers.py                     raw source 的技能特征、数值 kind 校验及独立真实执行指标读取
│   │   │   └── source_reader.py                      读取内存 training source，生成 schema、冻结模型状态与效果历史增量；批量校验当前状态并按分片读取，执行指标独立读取，scene 从完整 fight_scene_context 取得
│   │   ├── training/                                 训练样本子包：决策回放、样本编码与历史 bank 构建
│   │   │   ├── __init__.py                           训练样本门面，导出决策回放、样本编码与历史 bank 入口
│   │   │   ├── history_bank.py                       完整历史 bank：校验技能、状态和执行统计前缀及增量游标，保存稳定逐行引用且不受模型读取窗口影响
│   │   │   ├── quality_supervision.py                compiled PT 编译时校验历史排名百分位与目录区间
│   │   │   ├── sample_builder.py                     把内存 training source 编码为最终 compiled cache 样本；历史只写完整 bank 的 end/length 引用，动作质量标签只存等级代码 1/2/3；编译侧按分片批量编码当前状态、固定动作监督与 scene，样本行互不重叠；只属于脚本转换层
│   │   │   └── training.py                           训练样本装配：公开接口逐步回放，编译侧借用 Rust 批量回放与增量冻结历史；按场景改写 player 向量并生成原有标签，严格沿日志请求时间、不补推进或重试，label leak 以真实效果历史前缀对齐断言
│   │   ├── __init__.py                               包导出层，暴露 raw JSON 转换和最终 compiled cache 入口
│   │   ├── cli.py                                    独立转换入口：支持显式输入或按 `training.max_files` 和 `training.validation_files` 同批选择模型配置目录中的 JSON 并写入职业 `.cache`；训练回放口径固定为 FFLogs `cast=动作生效时刻`
│   │   ├── pipeline.py                               报告→训练两级转换流水线；报告阶段从日志解析请求/读条，结合技能表与职业配置检测基础 GCD，再把整场时间线归一到最早请求为 0，训练阶段以 0 初始化后端并导出训练样本
│   │   └── utils.py                                  转换使用的静态职业配置与技能索引
│   ├── fflogs_scraper/                               FFLogs 下载工具包，按 API、配置、格式契约、下载编排和文件 I/O 分层；保存整场报告和事件供离线分析与训练转换
│   │   ├── api/                                      FFLogs V2 API 访问
│   │   │   ├── __init__.py                           API 子包入口
│   │   │   └── client.py                             OAuth、按地区绑定 GraphQL 端点、事件分页、玩家解析、API 默认/指定分区与角色历史排名查询
│   │   ├── config/                                   下载常量、环境变量与输入校验
│   │   │   ├── __init__.py                           配置子包入口
│   │   │   ├── batch.py                              加载并校验批量下载的验证集比例配置
│   │   │   ├── constants.py                          请求指纹、事件上限、格式版本与补丁日期常量
│   │   │   ├── encounters.py                         加载并校验下载目标 YAML；副本代号或单战斗 Zone ID 解析
│   │   │   ├── environment.py                        通过 python-dotenv 加载根目录 .env，环境变量优先
│   │   │   └── validation.py                         GraphQL 标量、排行指标和报告编号校验
│   │   ├── contracts/                                报告、战斗和事件格式契约
│   │   │   ├── __init__.py                           契约子包入口
│   │   │   ├── events.py                             嵌套技能适配、友好关系与事件完整性标记
│   │   │   ├── models.py                             FightInfo 和 ReportMeta 元数据容器
│   │   │   ├── rankings.py                           历史击杀记录、匿名筛选与 ranking 元数据契约；百分位分档复用脚本公共目录约定
│   │   │   └── report.py                             V2 报告适配、真实 fight ID 选择和训练索引包装
│   │   ├── download/                                 下载与目录查询编排
│   │   │   ├── __init__.py                           下载子包入口
│   │   │   ├── batch.py                              历史百分位十档配额下载、同档失败补位、本地有效库存核算及代号目标目录解析
│   │   │   ├── encounters.py                         副本区域与战斗目录展示
│   │   │   ├── sampling.py                           总量均分、API 默认/指定分区角色发现、匿名排除与历史记录去重；不负责文件下载
│   │   │   └── single.py                             单报告下载，支持整场事件与伤害表模式
│   │   ├── io/                                       下载链接解析与文件名构造
│   │   │   ├── __init__.py                           I/O 子包入口
│   │   │   ├── filenames.py                          单场与批量文件名构造及路径字符隔离
│   │   │   └── urls.py                               FFLogs URL 的 report、fight 和 source 解析
│   │   ├── __init__.py                               公开导出 API 客户端、元数据容器和 URL 解析器
│   │   ├── __main__.py                               python3 -m scripts.fflogs_scraper 模块入口
│   │   └── cli.py                                    命令参数、认证、中断处理和子命令分发
│   ├── model_analysis/                               科研模型分析 PNG 输出包，中心入口统一加载 checkpoint/raw JSON 对应的 compiled cache 与 YAML 模型精度；cache 缺失或过期时委托同包缓存编译入口补齐，各 outputs 子模块独立生成分析图
│   │   ├── job_labels/                               按职业路由科研分析标签；未知职业返回 `unknown`
│   │   │   ├── __init__.py                           分析标签职业路由入口
│   │   │   └── black_mage.py                         从 request_state 提取黑魔 AF/UI/MP 决策状态标签
│   │   ├── outputs/                                  分析图输出模块目录
│   │   │   ├── __init__.py                           输出模块导出入口
│   │   │   ├── attention.py                          基于显式最新状态 query 对上下文的注意力热图、逐层完整 query-key 矩阵、最后一层逐 head 矩阵和 scene/state/skill 类型块汇总；mask 区域灰显，矩阵热图使用线性色阶与逐 query 行最大有效权重归一化，类型块汇总保留原始权重语义
│   │   │   ├── hidden_statistics.py                  hidden 分布和按 token role 的范数统计图，历史状态与最新请求状态分别统计
│   │   │   ├── history_embedding.py                  正式输入编码后历史技能与状态 token 的 PCA 图，使用显式位置和有效历史过滤 padding，支持空历史与单条历史
│   │   │   ├── loss_landscape.py                     逐 Transformer 层 filter-normalized 正交参数方向的高精度交叉熵曲面/等高线、float64 数值网格与方法元数据导出
│   │   │   ├── pca_layers.py                         每层 hidden 的 2D/3D PCA 在原图中将历史状态与最新请求状态列为独立显示组，沿用普通 token 样式
│   │   │   └── skill_embedding.py                    原始技能 ID embedding PCA 图
│   │   ├── __init__.py                               分析包入口
│   │   ├── __main__.py                               `python3 -m scripts.model_analysis` 入口
│   │   ├── cache_compilation.py                      模型分析缺少 compiled cache 时，委托正式转换 API 补齐；保持源文件去重、并发和失败校验
│   │   ├── common.py                                 checkpoint/cache 加载、统一 YAML precision 的模型 dtype/autocast、loss 轻量上下文、缺失 cache 时委托同包编译入口、逐批释放 trace 的 hidden 提取、DataSpec 对齐、低内存 CPU PCA、分析角色显示标签与 magma 统一取色和统一图表布局
│   │   ├── main.py                                   分析中心编排入口；同一命令先输出基础图，释放重上下文后以相同 checkpoint/数据/精度追加 loss landscape
│   │   └── token_metadata.py                         按实际历史/当前状态位置构造 step、label/prediction 索引、标签技能基础威力与 GCD/oGCD 分类、AF/UI、MP 和 label logit 元数据
│   ├── onnx_export/                                  checkpoint 后处理 ONNX 导出脚本包，与训练循环解耦；纯 Tensor policy 复用已有模型权重并输出策略后处理前 logits
│   │   ├── io/                                       ONNX 部署产物公共 I/O
│   │   │   ├── __init__.py                           产物处理子包入口
│   │   │   └── artifact_io.py                        文件/确定性 JSON SHA-256、原子 JSON、确定性 golden NPZ 和 PyTorch 导出报告归一化
│   │   ├── config/                                   ONNX 导出与 parity 的根目录 `.env` 配置
│   │   │   ├── __init__.py                           配置子包入口
│   │   │   └── config.py                             checkpoint、部署包、Provider、scene 和导出门禁参数解析
│   │   ├── contracts/                                固定容量、部署契约和职业 profile
│   │   │   ├── __init__.py                           契约子包入口
│   │   │   ├── contract.py                           固定 batch 与场景、历史和当前状态共享的 token 总容量、padding 和 golden 输入契约
│   │   │   ├── deployment_contract.py                从 checkpoint、模型结构和职业 profile 派生完整部署契约并校验 manifest
│   │   │   └── deployment_profile.py                 读取固化的职业 vocab 与 scene 容量语料统计证据
│   │   ├── export/                                   ONNX 导出执行层，文件按阶段平铺
│   │   │   ├── __init__.py                           导出执行层公开入口，暴露导出 API 与 checkpoint 加载器
│   │   │   ├── api.py                                导出公开 API、临时目录生命周期和失败清理
│   │   │   ├── builder.py                            只负责串联环境、契约、图、运行时和文档阶段
│   │   │   ├── checkpoint.py                         checkpoint 加载、部署 profile 读取和部署契约装配
│   │   │   ├── context.py                            各导出阶段之间传递的契约、ONNX 产物和运行时验收结果
│   │   │   ├── documents.py                          golden、容量报告、导出报告和 manifest 落盘与自校验
│   │   │   ├── environment.py                        ONNX 依赖、精度、设备和 ORT provider 门禁
│   │   │   ├── graph.py                              ONNX 导出、metadata 盖章、checker/shape inference 和体量审计
│   │   │   ├── publish.py                            部署目录原子发布与 Windows 文件占用重试
│   │   │   └── runtime.py                            ORT 会话、运行时契约和 PyTorch/ORT 精度矩阵验收
│   │   ├── policy/                                   纯 Tensor ONNX policy adapter
│   │   │   ├── __init__.py                           policy 子包入口
│   │   │   └── policy.py                             复用因果 attention/FFN 和共享技能 embedding，输出宿主后处理前 raw_logits
│   │   ├── profiles/                                 首次语料分析后固化的职业部署画像目录
│   │   │   └── black_mage.json                       黑魔部署画像、技能 vocab 映射与 scene 容量统计证据
│   │   ├── release/                                  ONNX parity 证据与发布状态门禁
│   │   │   ├── __init__.py                           发布子包入口
│   │   │   ├── policy.py                             正式发布场景与动作预算策略
│   │   │   └── release.py                            parity 证据绑定、发布状态晋升和审计校验
│   │   ├── runtime/                                  ORT provider、精度、Tensor I/O 和矩阵验收
│   │   │   ├── __init__.py                           运行时子包入口
│   │   │   ├── ort_runtime.py                        ORT provider 与 session 策略
│   │   │   ├── precision.py                          PyTorch、ORT、TensorProto、manifest dtype 和数值门槛规格
│   │   │   ├── runtime_targets.py                    正式导出与运行库目标矩阵
│   │   │   ├── tensor_runtime.py                     PyTorch/ORT 共用的 Tensor 执行与 BF16 golden 编码
│   │   │   └── validation.py                         动态长度与固定容量的 CPU/CUDA/ORT padding 矩阵验收
│   │   ├── __init__.py                               ONNX 导出脚本包入口
│   │   ├── __main__.py                               `python3 -m scripts.onnx_export` 无参数导出入口
│   │   ├── manifest.schema.json                      ONNX 部署 manifest 的 JSON Schema
│   │   └── workflow.py                               根目录 Bash 菜单入口的 Python 工作流入口；empty/scene 验收共享模型与多队列引擎，普通 ONNX 回放复用批量入口
│   ├── __init__.py                                   脚本目录包入口，便于测试和其他模块复用转换辅助函数
│   ├── dataset_compression.py                        一次性迁移旧 raw/annotated 普通 JSON 到 `.json.br`；核对标注与原始战斗一致后移除旧 source SHA，默认保留原文件
│   ├── ffxiv_ccg.py                                  项目专属 Python 菜单模块：统一菜单编号、命令行 action 名称与交互分发
│   └── tensorboard_server.py                         仅加载所选 BC/GRPO 阶段配置后启动监控，使用根目录 `.env` 的模型选择与端口，支持后台就绪检查、训练入口退出时回收服务和打开本机 Web 视图
├── ffxiv_ccg.sh                                      Bash 薄入口，调用 Python 九项工具菜单和 `--action` 分发
├── main.py                                           命令行入口，提供 `validate`、`list-skills`、`list-actions`、`smoke` 等最小手动验证命令；`list-actions` 和 `smoke` 共用原生多队列接口，`--queues` 可并发运行多个独立自检
└── setup.sh                                          Linux 环境准备入口：复用 `.venv`、安装本地 Node 与锁定依赖、构建 Rust release 扩展并验证 CUDA provider，不覆盖 `.env`
```

## 回放评估与调度

训练的 `val_ppg` 仍从所选验证 source 的首个样本初始时刻运行到最后一个 Boss 可选中窗口结束；`context_capacity` 只限制模型输入，不截断战斗；回放记录保留完整历史以重建场景变化时的窗口边界。`none_ppg` 则按配置的 GCD 数运行。转换 CLI 和训练/VAL 补编译现已共用有界线程调度：每个编译批次只创建一个引擎，`CONVERT_FFLOGS_WORKERS` / `--workers` 同时决定线程数和队列容量，在途任务不超过该值；单文件失败后释放队列并继续处理后续文件，已有有效缓存不启动引擎。验证 PPG、GRPO、普通回放、历史消融和 ONNX 验收共用有界回放调度；模型推理由调用线程执行，PyTorch 可合批，固定 batch=1 的 ONNX 部署契约保持不变。

## 批量入口与资源归属

单条兼容入口已移除：`InProcessBackend` 必须接收已有引擎，`AutoregressiveReplaySession` 必须接收共享引擎和模型后端，`AutoregressiveReplay` 必须接收已有会话。队列/会话只关闭自身资源，不关闭宿主引擎。业务入口统一调用 `run_replays`、`run_rollout_parities` 或 `compile_raw_training_caches`；单条任务也以单元素序列提交。

## 回放与 parity 报告

回放与 parity 报告的动作、逐决策差异和 PPG 按轨迹记录；`backend_metrics`、`reference_metrics` 与 `compared_metrics` 则是生成报告时共享后端的累计快照（`scope=backend_lifetime`）。调用数和最大延迟覆盖后端生命周期，延迟分位数只取最近 4096 次；内存为进程工作集与 PyTorch 设备分配峰值，不能当作单条轨迹的独立用量。

# 状态机

路径相对仓库根目录。返回[项目目录](./index.md)。

本文件记录 Rust 状态机工程 `fight-engine/` 的目录和运行边界。Python 适配器见[脚本](./scripts.md)，测试工程见[tests](./tests.md)，模型输入与缓存契约见[公共](./common.md)。

## 目录与职责

### Rust 状态机

`fight-engine/` 是独立的 Cargo library crate，包名为 `fight-engine`，Rust 导入名为 `fight_engine`，使用 Rust 2024 edition。`jobs/` 下各职业分别是依赖核心的独立 crate，核心不引用它们，也不以 workspace 成员关系绑定它们。正式 Python 调用通过 `python/` 的 PyO3 扩展接入 Rust；旧 C# 工程已移除，仅保留冻结 JSON 基线中的来源记录与历史许可声明。

```text
fight-engine/
├── Cargo.toml                                       包元数据与依赖声明
├── Cargo.lock                                       Cargo 生成的依赖版本锁文件
├── src/
│   ├── lib.rs                                       核心库模块入口
│   ├── config.rs                                    配置装配、通用定义、技能索引、策略配置与输出 schema
│   ├── config/
│   │   └── yaml.rs                                  保留标量样式、别名和多文档校验的 YAML 读取
│   ├── contracts.rs                                 职业接口、惰性注册、职业状态装配、计时回调与资源投影
│   ├── engine.rs                                    从注册表装配规则，动作校验、排队判定、时序计划、效果与历史
│   ├── engine/
│   │   ├── simulator.rs                             单队列时钟、动作生命周期、外部批次、死亡取消及 snapshot/fork
│   │   ├── sessions.rs                              共享引擎、容量、独立队列锁、重置关闭和原子观测回执
│   │   └── replay.rs                                绝对动作序列、SplitMix64 固定种子随机回放与完整快照缓存
│   ├── outputs.rs                                   状态上下文、冻结历史、canonical 装配及按 Arc 身份复用缓存
│   ├── outputs/
│   │   ├── tokens.rs                                动态 schema 的状态与技能 token
│   │   └── format.rs                                秒/GCD 快照、canonical/tensor、场景与 schema 元数据
│   ├── policy.rs                                    策略等待、冻结决策、合并历史、完整及增量向量观测
│   ├── policy/
│   │   └── replay.rs                                绝对时间批量请求与同刻事实、等待及死亡标记回放
│   ├── state.rs                                     公共状态、职业状态容器、绝对截止时刻与不可变历史快照
│   ├── state/
│   │   ├── resources.rs                             职业资源注册、初始化、限值、归一化及前后转换
│   │   └── status.rs                                系统和职业各自复用的 Buff 定义、层数与删除前到期钩子
│   ├── system.rs                                    公共校验、动作时序、冷却、MP、DoT、外部事实及威力规则
│   └── system/
│       └── timeline.rs                              唯一时钟、稳定事件排序、到期待结算事实和 snapshot/fork
├── jobs/
│   ├── black-mage/                                  黑魔独立 crate：fight-engine-black-mage
│   │   ├── Cargo.toml / Cargo.lock                   只依赖上级核心；固定独立构建依赖
│   │   ├── src/lib.rs                               BlackMage 工厂、元素、瞬发、MP、通晓和死亡规则
│   │   └── tests/                                   本职业的真实 C# JSON 基线对比
│   └── machinist/                                   机工独立 crate：fight-engine-machinist
│       ├── Cargo.toml / Cargo.lock                   只依赖上级核心；固定独立构建依赖
│       ├── src/lib.rs                               Machinist 工厂、连击、整备、过热、野火和冷却规则
│       └── tests/                                   本职业的真实 C# JSON 基线对比
├── python/                                          独立 Python 原生扩展，装配核心与两个职业 crate
│   ├── Cargo.toml / Cargo.lock                       PyO3 与本地 Rust crate 依赖及锁文件
│   ├── pyproject.toml                               Maturin 构建入口，默认 release 模式
│   └── src/
│       ├── lib.rs                                   原生引擎/队列句柄、职业注册及基础值回执
│       └── packet.rs                                私有 pickle 2 基础值编码，不复用可变容器
└── tests/                                           纯 Rust 行为测试；不编译或执行 C#
    ├── foundation.rs                                配置、技能查询、倒计时和克隆隔离对比
    ├── system.rs                                    事件排序、同刻刷新、待结算事实、快照与错误路径对比
    ├── resources.rs                                 职业资源类型、限值、浮点精度和转换对比
    ├── shared_system.rs                             独立系统规则、外部事实、死亡复活及职业状态隔离对比
    ├── job_contracts.rs                             惰性注册、职业计时器、到期顺序及回调排程对比
    ├── engine.rs                                    通用职业探针的动作、外部事实、冻结历史和分支对比
    ├── observations.rs                              秒/GCD/canonical/tensor、动态特征、场景及打包边界
    ├── policy_sessions.rs                           策略等待、增量历史、批量回放和多队列生命周期
    ├── replays.rs                                   回放决策顺序、整段队列锁、快照副本与缓存分支对比
    ├── common.rs                                    离散值严格检查与浮点容差比较
    ├── common/jobs.rs                               职业 crate 共用的通用回放对照驱动，不引用具体职业
    ├── common/probe.rs                              通用测试职业和明确输入构造，复用引擎与输出测试
    └── fixtures/                                    原 C# 实现生成的 JSON 基线，记录来源提交
```

模块采用 `名称.rs` 配合 `名称/子模块.rs` 的布局，不使用 `mod.rs`；模块文件直接实现归属逻辑，不建立空转发文件，子模块仍按职责分类；模块使用 `snake_case` 命名。系统核心与职业实现必须分离：核心不得引用或写死具体职业，职业通过接口和注册机制接入；移除职业代码后，核心仍须能独立编译并执行系统逻辑。内部时间、计算及 canonical 输出保留 `f64`；通用 `format_tensors`／`pack_tensor_payload` 显式接收已从 `config/precision.yaml` 加载的 `PrecisionConfig`，在输出末端按 `float_dtype` 量化为 FP32／FP16／BF16，整数与缺失掩码保持原布局。模型输入与 raw bank 仍固定 FP32；事件序号和索引保留所需位宽。行为比较严格检查离散结果、事件与动作顺序，浮点按误差验证，不要求旧舍入结果逐字节一致。

`EngineRules::from_project` 先注册启用技能的目标 DoT，再经注入的 `JobRegistry` 构造职业；核心注册表没有内置职业。`Simulator` 为每个队列独立持有时间线，排队动作冻结请求时的校验和时序；Buff 的事件顺序按授予顺序，上下文输出顺序按定义注册顺序。职业计时回调的状态、取消和排程统一转换进该时间线，不反向依赖动作载荷。历史条目与模型 frame 通过不可变 `Arc` 共享，列表写入使用写时复制隔离 fork。

宿主显式 `registry.register::<BlackMage>()` 或 `registry.register::<Machinist>()` 接入职业。职业 YAML 继续使用根目录 `config/skills/`，野火时长与量谱上限的一致性校验归机工工厂；黑魔死亡清理职业状态，机工沿用原先不清理职业状态的行为。通用资源值类型转换由核心 `ResourceValue` 实现，注册器与职业复用相同截断语义。

`serde_json` 启用 `float_roundtrip`，避免解析绝对时间戳时额外舍入改变临界事件派发；不能用放宽时间线边界代替输入精度。职业对照同时覆盖定向边界和固定输入长序列，原输出四位小数的精度差只在测试投影中处理。

Rust `SimulationEngine` 注入一份共享 `EngineRules` 与 `PolicyActionRegistry`，容量为 1–100，按 `SessionOptions` 创建队列。基础 GCD、历史上限、战斗状态、策略历史和输出缓存均属于队列；`None` 保留完整历史，`Some(0)` 关闭记录，不限制累计战斗指标。执行只取得本队列锁，响应连同时间游标原子返回。注册和关闭采用注册锁到队列锁的顺序；关闭引擎会使旧队列句柄永久失效，重置先构造再替换。

策略等待只冻结当前观测并领取共享历史序号，不提前演算下一刻；实际技能生效后领取自己的序号。完整输出按序号合并，增量输出只向量化新冻结行，并要求真实和策略历史都完整保留。输出缓存持有源 `Arc` 并按身份匹配，恢复同一序号的新历史后不会沿用旧内容；动作布局切换时重建对应 token 缓存。

`engine::replay` 的绝对序列与随机回放都在同一队列锁内执行整段操作。序列只在最后推进到已接受动作的最晚效果时刻，返回真实动作词表的 canonical；随机回放复用既有合法动作排序，SplitMix64 使用溢出回绕算术，仅选择动作时推进随机数，没有合法动作时显式推进下一内部事件。`ReplayStateCache` 从独立 fork 开始，保存完整快照并返回独立副本；缓存不拥有第二套时钟或推进规则。

基线生成与实时 C# 差分工具只用于迁移，放在不入库的 `.tmp/rust-migration/`；Rust 源码和测试目录只包含 Rust 与 JSON 数据。构建产物位于 `fight-engine/target/`，由根目录 `.gitignore` 忽略。从仓库根目录验证工程：

```bash
cargo test --offline --locked --manifest-path fight-engine/Cargo.toml
cargo fmt --manifest-path fight-engine/Cargo.toml --check
cargo clippy --offline --locked --manifest-path fight-engine/Cargo.toml --all-targets -- -D warnings
```

职业独立验证时将 `--manifest-path` 分别换成 `fight-engine/jobs/black-mage/Cargo.toml` 和 `fight-engine/jobs/machinist/Cargo.toml`；原生绑定使用 `fight-engine/python/Cargo.toml`。各自的 `target/` 均被忽略；删除 `jobs/` 和绑定不影响核心清单解析、构建或系统测试。

## 多队列入口与生命周期

多队列入口是 `engine::sessions::SimulationEngine`，Python 对应 `InProcessEngine`。一个引擎装载一份职业规则、技能表和配置，多个队列分别持有基础 GCD、战斗状态、事件时间线、策略历史及输出缓存。执行请求只获取所在队列的锁；注册锁只管理创建和释放。同队列事件必须由调用方按时间顺序提交，不同队列可以由不同 Python 线程驱动。该共享关系位于同一进程中。

```python
from scripts.common.inprocess_backend import InProcessEngine

with InProcessEngine("black_mage", capacity=16) as engine:
    # 转换与回放保留完整记录；模型读取时按 context_capacity 选择输入窗口。
    with engine.create_backend(max_history=None, actual_base_gcd=2.46) as queue:
        queue.submit_action(0.0, "fire_iii")
        observation = queue.observe_at(4.0, format="vector", next_observation_timestamp=4.0)
        counts = queue.statistics()
```

引擎容量耗尽时立即报错，由宿主负责调度待执行任务，不在引擎内部积压请求。`queue.init()` 仅重置该队列，`queue.close()` 释放队列容量，`engine.close()` 关闭全部队列；关闭的句柄永久失效。请求结果在队列锁内连同时间游标一起生成，单队列错误不关闭其他队列。职业规则装配完成后只读，运行数据必须写入传入的 `CombatState`。

## 死亡、复活与历史标记

死亡重置通过 `queue.apply_external_events([{"timestamp": t, "event_kind": "player_death"}])` 提交，成功结果的 `token` 为 `<dead/>`，普通外部事实为 `None`。系统清 MP 和公共 Buff，黑魔清职业 Buff、量谱和计时截止时刻；机工尚无职业清理规则。重置取消尚未结算的旧动作，保留 DoT、GCD、冷却、已有充能和历史，冷却与充能继续倒计时。死亡期间停止自然回蓝；`player_revived` 表示普通复活，可携带 `mp` 指定复活时 MP，未指定则由系统按 `config/system.yaml` 的 `mp_recovery.revival_ratio` 乘最大 MP 后取整，默认比例 0.2；`player_revived_lb3` 表示治疗 LB3 复活，由系统恢复完整最大 MP，不接受 `mp` 覆盖。同刻不能同时提交两种复活事实。两者均重新开启自然回蓝，不恢复死亡已清掉的状态，也不产生额外历史 token。转换器以目标复活保护出现为复活时刻；只有对应死亡区间内存在已完成的治疗 LB3，且该保护时长为 6000ms 时，才提交 LB3 复活，其余保留普通路径。缺少保护时退回死亡后首次自身动作请求；普通复活只在复活同刻有可靠 MP 记录时传入覆盖值。取消未完成请求后，最新状态的前半段保留最后已落实动作的冻结快照，后半段为当前请求状态。宿主同刻先提交外部事件，再提交新动作。转换回放中，同刻且没有复活的死亡在该刻日志旧请求全部提交后执行，场景事实仍先于请求；同刻已有复活时先执行死亡和复活。死亡标记从下一条观测起可见，末条后的死亡只重置状态。转换器已将标记保存到 PT 独立的 `history_markers` 元数据；`build_death_token` 统一装配 token、时间和已落实历史位置，转换层补充首次可见的样本 `step`。训练按该步号读取可见标记前缀，在线回放与 GRPO 使用当前队列实际收到的死亡回执，队列初始化时清空标记。`<dead/>` 通过独立可训练的 FP32 embedding 和死亡角色进入模型，按已落实动作边界插入状态／技能对之间；不占用技能 ID，不改变状态维度或可选动作集合。历史消融快照保留当时可见的标记。

## 历史保留

队列的 `history_limit` 是统一的记录保留策略：`None` 保留完整记录，整数保留最近 N 条，0 关闭记录；它不控制模拟时长和累计威力。动作与策略记录分别受上限约束，输出时再合并并保留最新 N 条。完整转换仍保存完整 history bank，既不读取模型窗口，也不改变 cache signature。完整历史占用随记录数线性增长，不能承诺无限运行时固定内存；有限窗口模式的常驻记录量有界。

## Python 桥接

`fight-engine/python` 只装配职业并转换边界值，所有核心执行、锁等待和观测编码均在 PyO3 `detach` 中释放 GIL；独立队列可以并行准备数据。原生扩展返回 `bytes`，Python 通过标准库 `pickle.loads` 批量还原现有 dict/list 契约。编码器只产生基础值指令，输入仅来自当前扩展生成的观测，不读取外部 pickle 文件。Python 容器还原及后续 Python 逻辑仍受 GIL 约束，不能把多线程等同于整条转换流程完全并行。每次观测的可变容器保持独立，只有不可变字符串在单个包内复用。

原生扩展在编译时嵌入 `config/schema.yaml` 的运行时契约版本；Python 首次使用时惰性导入并核对该版本，拒绝旧构建产物。修改 Rust 或 schema 后运行 `.venv/bin/python3 -m pip install ./fight-engine/python` 并重启已加载扩展的进程。转换的 spawn worker 只导入适配器，通过父进程 RPC 使用共享引擎，不加载原生扩展。

Python 接入的真实差分覆盖双职业接口与黑魔 M5s／FRU 共 1,367 个样本的完整缓存。离散值、形状、dtype 和 metadata 一致；正式 FP32 状态张量因移除旧四位小数输出而有差异。仅在忽略目录副本恢复原舍入后，通过同一正式 CLI 重编的全部张量逐字节一致，生产实现不保留该舍入路径。`queue_entry_count` 统计 Rust 当前事件表大小，旧 C# 堆可能包含已取消的残留节点，因此该诊断值不要求一致。

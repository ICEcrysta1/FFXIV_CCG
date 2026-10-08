# 状态与技能可用性统一输入重构计划

记录日期：2026-10-08。
状态：代码、桥接和定向验收已完成；正式缓存、checkpoint、部署包与未执行的验证见第 16 节。
核查基线：`e4a2eeb07b184b4a9e4dcb2d6106ebf9caf923fa`，计划核查时分支为 `main`。实施分支为 `ice/codex/state-skill-availability`，实施基线为计划提交 `87080c9572c412bff5e44721c2055837d1b32c63`。
相关前置修改：`031c6a85c172ebb77fb37768bb30479e1fc3c1fc`；现有状态相对编码继续保留。

## 1. 已确认目标与重构范围

1. 原有状态保持当前 86 维及其 ABS 锚点、DELTA、null/reset 和时间重锚语义。
2. “上一次技能结束后”和“本次请求时”各有一份当时冻结的技能可用性状态；每个技能名对应一个绝对 `0` 或 `1`。
3. 当前黑魔完整动作空间为 25 项，含 `ogcd_wait`，最终状态为 `86 + 25 × 2 = 136` 维。
4. 技能可用性独立计算，与原状态统一装配，进入同一个 S-Emb 内容投影。
5. 利用这次迁移整理状态输入链路中已经存在的重复定义、解析、校验和调度规则。父级模块统一契约和装配，各子模块负责自己的原料与业务。
6. 迁移后删除被替代的实现。不能为照顾旧接口而再造一套字段、兼容 reader、额外队列或第二条拼接路径。
7. 主要行为变化集中在状态 token。其他 token 允许内部共享、缓存和性能优化，但重构前后的公共输出与模型输入必须保持既有行为，并验证离线、实时及部署入口的语义一致。

模型序列仍为：

```text
scene tokens, S1, A1, S2, A2, ..., SH, AH, S_current
```

只扩展每个 S 的内容，不增加技能可用性 token、角色 ID 或历史动作 token 数。

本次实施范围包含 C# 快照与输出装配、场景事实提交、Python 状态读取和字段布局、完整 raw cache、共享编码、模型输入、GRPO、分析、PPG 和 ONNX 消费边界。清理项必须能对应第 2 节的实际重复职责或本次发现的错误；不借此改动无关的输出头、优化器、训练算法或职业资源规则。

本轮仅修改本计划。后续实施保留用户的 `config/models/black_mage/artzip/training.yaml` 修改，不自动修改 CHANGELOG、发布版本、commit 或 push。需要新建分支时使用 `ice/codex/state-skill-availability`。

### 1.1 非状态 token 的外部行为约束

比较边界固定为两处：各 token 的 canonical/向量输出，以及经过既有 reader、窗口选择与 ContextEncoder 后交给模型/ONNX 的输入。同一份执行事实、场景原料、历史行、词表和读取参数下，两处都要分别与重构前基线对照；同时检查同一语义输入经离线与实时入口产生的模型输入一致。

| 对象 | 必须保持的外部行为 |
| --- | --- |
| 历史技能 token（含 policy wait） | 原字段、数值化 kind、is_legal、数值特征/归一化、技能到 vocab ID 的映射、行顺序、写入序号对应及 padding 语义；不追加可用性字段、不新增技能绝对时间 |
| scene token | 原类型 ID、字段/列顺序、canonical 数值与窗口记录、raw FP32 张量，以及现有锚点、裁剪、稳定排序、start/end DELTA、duration 编码和 mask 规则 |
| 非状态输入与相关元数据 | history_skill_ids/features、scene_vectors/types/mask、history_mask 的名称、dtype、维度和对齐规则；固定 action_keys/监督映射及真实执行统计仍按原契约独立保存，不因状态扩维进入 token |

明确允许的状态变化是两段技能表、原状态字段的事实/冻结修正及 ETA/停机剩余的跨入口一致性；原 86 维的编码算法继续保留。新增状态信息可能改变模型决策与后续轨迹，不能要求新旧模型 logits、选招或 PPG 完全相同来代替 token 契约验证。

场景/队列修复若改变真实执行事实，必须记录首个差异及其与本计划问题项的关联，再用相同已执行事实核验非状态 token 的构造行为。不能把轨迹变化当作随意改变技能/场景 token 语义的理由，也不能覆盖旧输出以掩盖差异。发现与状态重构无关的 token 语义变更，应移出本次实施范围。

内部可以优化对象分配、公共查询、缓存复用与重复校验；验收只要求唯一职责、外部等价和有依据的性能结果，不要求内部结构保持原样。基线输出仅保留为测试夹具，不在生产留下旧/新双实现。

## 2. 已核实的问题与本次处理方式

以下是基于核查基线源码确认的重复职责或实现风险，不代表已经修复。

| 编号 | 当前证据 | 本次处理 |
| --- | --- | --- |
| R1 | `JobSimulator.ApplyExternalEvent` 每提交一条事实就推进时间线；`SceneTemplateProvider.sync_state` 先提交 Boss，再提交移动等事实 | 同刻事实先整批校验、全部入队，再统一推进；动作效果不能看到半更新场景 |
| R2 | `CombatTimelineRuntime` 的 fork 构造复制处理器委托；`JobSimulator.Fork/ForkWithoutHistory` 沿用这些委托 | 给生效回调增加队列读取前，修正子会话处理器归属；不能读取父会话的 timeline |
| R3 | `TrainingSourceReader._build_state_matrix` 与 `LiveBatchBuilder._build_state_tensors/_extract_nullable_values` 各自解析状态，数值类型校验还不完全相同 | 迁入共享状态读取模块；两端使用相同类型、有限值、null、字段顺序和二值检查 |
| R4 | `context_fields.py` 维护名称、dtype、阶段，而部署 `tensor_inputs` 另有一份完整 shape 字典；GRPO 和模型又自行假定 mask 与 values 等宽 | 在现有字段声明中补齐逻辑维度；各入口从同一布局解析 shape，保留各自的容量和存储职责 |
| R5 | `OutputContextBuilder` 与 `StateHistoryContextBuilder` 各写四组 feature keys；`StateTokenBuilder` 又手写分组装配 | 父级输出装配统一生成分组元数据，当前/历史复用；子 builder 只产出所属组的值 |
| R6 | `scene_state.py` 与 `SceneTemplateProvider` 各自计算移动、目标状态及场景边界，且通过不同方式提交事实 | `scripts/common/scene_state.py` 统一场景查询、边界与事实流；转换和回放保留各自的数据适配与决策节奏 |
| R7 | `PolicyContextBuilder` 在真实 canonical 输出后排序动作、插入等待历史 | 改为在最终向量化前组合完整动作布局及冻结历史，再交统一输出装配；不增加“序列化后再补列”的分支 |
| R8 | 初稿把技能表放在原状态分组契约之外，并拟由总维度减去动作数反推基础宽度 | 技能表纳入同一状态分组声明，用明确的编码类型区分；所有维度从实际字段布局正向推导 |
| R9 | 部署 `DeploymentContract.from_dict` 用外层 state_layout 重新构造并覆盖 input_contract 的状态分组顺序 | 状态分组以有序序列保存；部署视图从父级生成，加载时仅核验一致性，不覆盖保存的输入契约 |
| R10 | 转换直接按 raw scene 的 double 时间生成事实，cache/回放再把场景量化为 FP32；共享窗口算法仍可能收到不同端点 | 场景父级在首次事实查询/提交前派生与现有 raw FP32 cache 一致的执行视图；统一派生边界，不改写原 scene token |
| R11 | `JobSimulator.ValidateActionAt/AvailableActionKeysAt` 在队列占用时拒绝所有动作，随后走立即执行校验；与 SubmitAction 和原始 mask 的接受能力不同 | 公开提交能力查询、冻结表和原始 mask 共用 EvaluateActionSubmission；内部执行校验保留自身职责 |
| R12 | `rewrite_scene_player_state` 只有转换端调用；实时 build/build_from_canonical 直接进入状态 reader，而 BossTargetableChanged 不提供停机 ETA/剩余 | 转换与实时在共享 reader 前使用同一状态场景字段合成入口，逐段按冻结时间计算，仅修改状态 token |

R1 的具体错误场景：在移动边界 t 上同时存在 ActionEffect，第一条 Boss 事实的提交就会排空 t 上的内部事件；移动事实尚未提交，动作后状态与技能表便按旧移动状态冻结。仅增加 `_FACT_ORDER` 或共享 `is_moving_at` 无法修复。

R2 的具体错误场景：动作请求后 fork，父会话后来占用排队槽，子会话独立推进至原动作生效；若子回调仍绑定父 `JobSimulator`，新增的 `HasQueuedAction()` 会读取父队列。不可变容器只能防止事后改写，不能纠正捕获时选错会话。

R10 的具体错误场景：raw 场景边界为 600.004 秒，FP32 保存值为 600.0040283203125 秒，约 28 微秒的偏差超过 scene_epsilon 的 1 微秒。若动作效果在原时刻生效，两端便可能按不同先后顺序冻结状态。R1 只能统一真正同刻的事件，不能修复输入端点已经不同的问题。

R11 的具体错误场景：现有 QueuedGcdDoesNotBlockImmediateOgcd 用例在 0 秒提交 GCD、2.1 秒排队下个 GCD，2.2 秒的 oGCD 仍可立即提交；公开 validate/available 却因队列占用返回不可用。重构只更新表和 mask 会留下同一会话内互相矛盾的答案。

R12 的具体错误场景：停机窗口为 [10,20) 时，t=5 的转换输入 ETA 为 5、实时为 0；t=15 的转换输入停机剩余为 5、实时为 0。共享查询函数但不接入实时状态装配，仍会使原 86 维在训练与回放间分叉。

本次修订明确替换初稿中“技能表单独旁路声明”“维度反推”“实时事实提交方式不变”“fork 只检查容器冻结”的安排。

## 3. 父级契约与模块职责

这里的父级指共同的契约与装配归属，不要求建立继承体系。优先复用现有模块；跨 C# 和 Python 共享声明与黄金测试，不强行共享语言实现。

| 权威内容 | 所属父级 | 子模块负责什么 |
| --- | --- | --- |
| canonical 分组、字段名、快照顺序、编码类型 | `config/schema.yaml`，由 C#/Python schema loader 读取 | 各组 builder 构造自己的值，不另写字段顺序 |
| 本会话真实技能接受能力、公开查询与快照时刻 | `JobSimulator` + 既有 `CombatStateMachine.EvaluateActionSubmission` | `SkillAvailabilityBuilder` 只批量提取接受结果，不持有队列；validate/available 不另判队列占用 |
| 模型真实/策略动作组合 | `PolicyContextBuilder` | SkillBook 管真实技能，PolicyActionRegistry 管策略动作，彼此不接管职责 |
| 当前/历史模型状态的最终输出 | `OutputContextBuilder` + `StateTokenBuilder` | 历史收集器提供冻结行，分组 builder 提供值，不独立生成另一份 schema |
| 保存后的字段语义、切片、原/合并宽度 | `common/policy/data/schema.py` 中的状态分组和 `StateFeatureLayout` | DataSpec、cache、编码、分析、模型与部署读取同一布局 |
| canonical 状态解析与张量拼接 | `common/policy/data/state_features.py` | 离线和实时入口提供 token；ContextEncoder 编码后调用唯一装配函数 |
| tensor 名称、阶段、dtype、逻辑 shape、padding | 现有 `common/policy/data/context_fields.py` | cache、collator、GRPO、ONNX 按声明处理，不再维护独立 shape 表 |
| 场景执行视图、查询、边界、事实流与状态场景字段合成 | 现有 `scripts/common/scene_state.py` | 转换与实时委托同一规则，只把 ETA/剩余写入状态副本；scene token 的输出与编码保留，DecisionScheduler 只决定何时推进 |
| 战斗时间、事件优先级和同刻结算 | 现有 `CombatTimelineRuntime`，由 session 加锁 | Python 提交事实，不建立第二套战斗事件队列 |

正式路径：

```text
C# 同一会话的真实状态与队列
  → 冻结 ModelStateFrame（原状态 + 技能名到可用性的映射）
  → policy 父级组合固定动作与等待记录
  → 统一 canonical 状态分组输出
  → Python 共享场景字段合成（只作用于状态副本中的 ETA/剩余）
  → 共享状态 reader → 原状态 raw bank + 绝对二值 bank
  → ContextEncoder：窗口 gather + 原状态编码
  → 统一 assembler：拼接 86 + 25 + 25
  → 同一个 S-Emb
```

新增生产文件原则上只有两个：C# 的 `SkillAvailabilityBuilder.cs` 和 Python 的 `state_features.py`。布局类型放入现有 `schema.py`，场景公共行为放入现有 `scene_state.py`。不建立可注册任意插件的编码框架，也不为每个消费者增加一层只转发调用的类。

## 4. 字段和维度契约

### 4.1 每个技能名对应一个值

字段使用稳定的技能 key，如 `fire`、`blizzard`、`ogcd_wait`，不使用可本地化的技能显示名。

一个快照在语义上就是：

```text
skill_availability = {
    fire: 0,
    blizzard: 1,
    ogcd_wait: 1,
    ...其余完整固定动作
}
```

两个快照分别保存，不能共用当前 mask：

```text
previous_action_after.skill_availability.fire = 0
request_state.skill_availability.fire = 1
```

每项都是一个标量，不是技能 ID、技能 embedding、候选对象或每个技能再展开 25 维的 one-hot。C# 冻结层可使用只读 `skill_key → bool` 字典；canonical 和模型中的对应值必须是数值 `0/1`。

canonical 沿用现有状态的 `feature_keys + values` 列式表达，不另存一份同义字典：

```text
skill_availability_feature_keys = [
    previous_action_after.a0, ..., previous_action_after.a24,
    request_state.a0,        ..., request_state.a24
]
skill_availability = [0, ..., 1, 1, ..., 0]
```

`a0...a24` 表示保存的 `action_keys` 中的实际技能 key，不是生产字段名。`feature_keys[i]` 与 `values[i]` 一一对应；技能名无需作为字符串输入模型。

### 4.2 原状态字段与最终布局

原有分组内部依次放两段状态，保持顺序：

| 分组 | 单个快照字段数 | 两段合计 | 原向量切片 |
| --- | ---: | ---: | --- |
| player_state | 9 | 18 | `[0:18]` |
| buff_state | 20 | 40 | `[18:58]` |
| target_buff_state | 7 | 14 | `[58:72]` |
| resource_state | 7 | 14 | `[72:86]` |

新增 `skill_availability` 为第五组，组内按“上一动作后全部动作、本次请求全部动作”排列：

```text
D_base = 原有 anchored_delta 分组的字段数之和
A = 保存的完整 action_keys 数量
D_availability = 2 × A
D_state = D_base + D_availability

当前黑魔：86 + 25 × 2 = 136
```

| 最终部分 | 当前切片 | 编码 |
| --- | --- | --- |
| 原状态 | `[0:86]` | 原 ABS 锚点和相对 DELTA |
| 上一次技能结束后技能表 | `[86:111]` | 绝对 0/1 |
| 本次请求技能表 | `[111:136]` | 绝对 0/1 |

生产实现从字段布局推导切片，禁止硬编码 86、25、136。相邻历史行的原状态两段分别求差，不能在一个 token 内让两段相减；可用性从 1 变 0 时输出 0，不输出 -1。

### 4.3 值与辅助 mask

| 字段 | 形状 |
| --- | --- |
| history_state_vectors | `[B,H,D_state]` |
| current_state_vectors | `[B,D_state]` |
| history_state_null_mask / history_state_reset_mask | `[B,H,D_base]` |
| current_state_null_mask / current_state_reset_mask | `[B,D_base]` |

技能表完整且已知，不增加 null/reset 列。S-Emb 为：

```text
Linear_D_state(combined_state)
+ Linear_D_base(null_mask, bias=False)
+ Linear_D_base(reset_mask, bias=False)
```

只有一个状态内容投影；不增加 availability embedding/projection。历史 padding 的完整合并值清零，辅助 mask 沿用已有语义，并由 history_mask 排除。

## 5. C# 接受规则、冻结与统一输出

### 5.1 技能可用性的定义

```text
availability(skill) = EvaluateActionSubmission(state, skill, queueOccupied).Accepted
```

- 复用现有 MP、资源、Buff、冷却、移动、读条/GCD 锁与排队窗口规则。
- `1` 表示当前允许提交，包含允许排队，不仅表示当前冷却为零。
- `queueOccupied` 由拥有此状态的 JobSimulator 读取一次；全部技能使用同一状态和队列上下文。
- 判定只读，不推进时间、不执行技能、不预演未来状态。
- 表不读取训练标签、该次日志动作的观测读条覆盖值或未来动作结果。
- 已占用排队槽不等于全部技能不可用；能立即接受的动作仍按正式规则判定。
- `ogcd_wait` 由 policy 层补入，沿用当前原始可用性恒为 1 的规则。以后若 policy 可用性依赖战斗状态，必须在对应快照时刻冻结。

顶层 `action_legal_mask` 继续服务选择约束。它在 C# 原始输出中从当前 request frame 的同一份表生成；Python 的 GCD/oGCD 阶段过滤只作用于选择 mask，不能改写冻结字段。

`SkillAvailabilityBuilder.Build(state, queueOccupied)` 只遍历本模拟器已确定的启用真实技能并调用接受规则。JobSimulator 初始化时确定一次稳定真实动作顺序，当前 mask、frame 与真实输出复用它；不在每个 builder 内重新排序。

公开查询也必须收敛到这份接受能力：

- JobSimulator 统一取得本会话的 state/queue 上下文，由现有 EvaluateActionSubmission 返回 Accepted、Queued、Reason 等结果；不新增另一套合法性规则或查询包装类。
- ValidateActionAt 在推进到请求时刻后返回同一接受结果及拒绝原因；AvailableActionKeysAt 按稳定真实动作顺序筛选 Accepted 项。删除两个入口“队列占用便全部拒绝”的分支，以及把立即执行校验当作提交能力的调用。
- 冻结表、未经过 Python 阶段过滤的原始 mask、单动作验证和可用动作列表，在同一 state/queue 下必须一致。SimulationSession、InProcessBackend.validate_at 和 RandomSequenceReplay 只消费该结果，不补一层自己的判断。
- At 入口保留既有推进到指定时刻的职责；取得上下文后的接受查询只读，不再次推进。真实 SubmitAction 仍使用同一规则；日志 observed cast override 仅属于正式提交的显式参数，不混入普通查询与技能表。
- 底层 ValidateActionCore 的即时执行检查继续服务于接受规则和动作生命周期，不能改成“允许排队即能立刻执行”。一致性比较限定真实技能；ogcd_wait 仍由 policy 层负责。

### 5.2 完整 frame 与捕获时刻

在现有 `Outputs/ModelStateSnapshot.cs` 内声明：

```csharp
public sealed record ModelStateFrame(
    StateContext State,
    IReadOnlyDictionary<string, bool> SkillAvailability);

public sealed record ModelStateSnapshot(
    ModelStateFrame PreviousActionAfter,
    ModelStateFrame RequestState);
```

`CombatState.LastDecisionAfter` 保存不可变 ModelStateFrame。Freeze 同时复制并只读化 StateContext 内的容器和技能表；只声明 IReadOnlyDictionary 而保留外部可修改的底层容器不算冻结。

JobSimulator 统一提供 `CaptureModelStateFrame(state)` 和 `CompleteModelDecision(decisionId, state)`。在 timeline mutation 内捕获时，必须使用传入的已变更 `state`，不能重新调用 GetState 读取尚未提交的旧状态。

| 时刻 | 行为 |
| --- | --- |
| 初始观测、缺前序状态 | 同一个完整 request frame 作为两段；不补零表、不虚构 S0 |
| 真实技能请求 | 在本次提交改变 timing/queue 前捕获 request frame |
| 真实技能效果生效 | 在 ApplyActionEffect 后按现有生效时刻捕获动作后 frame |
| policy wait 落实 | 经统一 RecordModelDecision 记录当时 frame；不预演等待终点 |
| 读条/排队中重复观测 | 只产生新的当前请求 frame，旧历史不回填 |
| 老请求晚生效 | 保留其历史记录；只有 LastDecisionId 匹配时更新 LastDecisionAfter |

将 LastDecisionAfter 写入从 `CombatStateMachine.RecordTimelineActionHistory` 迁到 JobSimulator 的完成入口。真实执行前后状态、直接/DoT 威力、历史写入序号仍由既有执行历史路径负责，不从模型输入反推。

### 5.3 在向量化前组合，统一生成当前和历史输出

1. SkillBook 和 PolicyActionRegistry 分别提供真实与策略动作；`PolicyContextBuilder` 作为组合父级，确定完整固定动作布局。
2. 真实历史与 wait 历史按既有 HistorySequence 合并，必须为正、唯一且单调；真实执行统计随所属行一起移动。
3. policy 层对当前/历史 ModelStateSnapshot 使用同一个纯组合方法，把策略动作值加入冻结映射；不修改原 frame。
4. `OutputContextBuilder` 接收最终动作顺序、当前快照与合并后的历史行，统一构造 current/history 的分组元数据和上下文。
5. `StateTokenBuilder` 按 schema 分组调用原状态 builder，并按固定动作顺序投影技能表。该过程只读取冻结值。
6. 当前顶层 action_keys、原始 legal mask、action_values 使用同一动作布局。技能价值仍走现有价值解析，不进入可用性字段。

真实技能输出入口与 policy 输出入口复用相同装配流程，区别只是传入的动作布局和历史来源。父级输出模块不依赖 PolicyActionRegistry，也不计算职业合法性。

`StateHistoryContextBuilder` 收敛为历史行/不可变 token 的收集与缓存，不再自行手写四组 feature keys。等待历史也不另造 schema。若需要内部统一行结构，放在现有输出装配文件，绑定 ModelState、技能信息和执行统计，不新建一套可持久化的兼容历史格式。

缓存必须区分动作布局；真实技能布局的 token 不能作为完整 policy 布局的 token 直接复用。重复观测、窗口裁剪和 fork 不得对缓存中的数组就地补列。

删除初稿拟定的“先得到 canonical 数组，再由 MergePolicySkillAvailability 补列/重排”的路径。分组元数据只在父级装配中生成；历史与当前调用同一个入口。

### 5.4 fork 的处理器必须属于子会话

这项修复必须和生效后技能表捕获一起完成，不能等到后续优化。

- JobSimulator 的初始创建和分支创建共用同一个 timeline 构造/处理器绑定入口。
- fork 复制运行状态、待处理事件、待结算事实、sequence 和冻结 frame；实例处理器必须重新绑定到子 JobSimulator。
- 不允许把父实例的 HandleActionEffect 委托原样装进子 timeline，也不能捕获父实例的队列查询闭包。
- Fork 与 ForkWithoutHistory 走同一路径；后者只省略历史前缀，保留待生效动作、请求关联和完整 frame。
- RestoreSnapshot 只恢复运行数据，不带入来源会话的处理器；恢复后仍由目标会话持有自己的 timeline。
- 调整现有 timeline 的运行数据复制/绑定边界，不新增第二套 fork 队列。避免反复深拷贝同一历史，保留不可变 frame 的安全共享。

验收必须包含：带待生效动作 fork 后让父子队列占用不同，分别推进效果，断言冻结表只反映各自队列；再覆盖无历史 fork、snapshot/restore 与子分支继续 fork。

## 6. 统一场景事实与同刻结算

### 6.1 同刻批量提交是正式入口

在现有 JobSimulator / SimulationSession / InProcessBackend 增加同刻外部事实批次入口，复用 ExternalCombatEvent 和现有 timeline：

```text
同一时刻 t 的全部场景事实
  → 在 session 锁内完成整个批次的校验
  → 全部以 ExternalScene 优先级入同一个 timeline
  → 仅在全部入队后 AdvanceClockTo(t)
  → 统一返回该次推进结果
```

要求：

1. 校验时间有限、非过去、批内同刻、事件种类和载荷均合法；任意一项失败，不得部分入队或推进。
2. 同刻同类冲突先由场景父级归并为最终事实，运输入口对仍存在的冲突明确拒绝，不用提交顺序猜测。
3. 稳定顺序由父级事实规则确定；ExternalScene 先于 ConfirmedActionEffect，沿用其余事件优先级。
4. 动作效果、DoT/周期结算和动作后 frame 捕获只能看见该批次全部场景事实生效后的状态。
5. 单事件公共 API 如保留，必须将一个事件委托给同一个批次实现；不得保留独立推进逻辑。正式场景生产路径必须按同刻分组后调用批次 API。
6. 不增加子进程、RPC 或另一个引擎。Python.NET 直接暴露同一个 session 能力；批次返回值复用既有结果类型或其集合，不另造第二套场景状态。

这是新增数据之前必须修复的 R1。调整 `_FACT_ORDER` 而仍逐条调用现有 ApplyExternalEvent 不算完成。

### 6.2 一个共享的场景查询和事实流

在现有 `scripts/common/scene_state.py` 内统一场景窗口规范化、查询、边界和事实构造：

- `SceneStateLookup` 回答有效移动、Boss 可选中、目标数、团辅及停手 ETA/剩余等场景查询。
- `SceneFactScheduler` 使用同一份规范化窗口生成事实流，提供 next-event、游标 reset/定位和按目标时间提交事实的能力。
- 转换保留既有 fight_scene_context 输出，同时从该原料委托父级派生第 6.3 节的执行视图，供查询与事实流共用；SceneTemplateProvider 从已有 schema/FP32 scene tensor 恢复相同执行视图。
- 适配器只转换容器和列位置，并委托父级精度规范化/校验，不复制量化、窗口判断、滑步计算、事实排序或同刻归并。
- DecisionScheduler 继续负责动作占用、GCD/oGCD 决策时间及结束条件；推进时先通过共享事实调度提交所有到期批次，再进行观测，不能直接跨越尚未提交的场景边界。

移动沿用 schema 中的 scene_epsilon 和 slidecast_window_seconds。每个窗口先求有效移动区间，再对区间求并集；重叠窗口只要任一仍有效就为移动，长度不超过滑步豁免的窗口无有效移动段。查询和事实生成必须共用相同端点表达，不分别用不同 epsilon 推算。

同刻多个窗口开始/结束先求最终场景状态，再产生一批事实。消除重复快照同步：不能每次移动变化都先发送无变化的 Boss 事实，导致提前排空内部事件。

从非零 t0 恢复时先定位共享事实游标，并在首次动作/推进前提交 t0 的有效场景状态；不把过去事实重新提交给状态机。轨迹 reset/fork 的场景游标必须独立，不复用上一轨迹的同步标志。

### 6.3 统一执行时间精度，保留 scene token 外部行为

沿用现有 raw scene 的 FP32 契约，**首次执行前的场景视图必须与已有 raw FP32 cache 的值一致**。canonical 原料和 raw FP32 张量仍是现有输出/读取阶段；执行视图只是从它们派生的内部查询数据，不增加落盘字段、格式、reader 或第二套事实队列。

1. 在 scene_state.py 的共同入口，从 canonical 场景原料按既有 raw tensor 的 FP32 规则投影端点：IEEE 754 binary32 最近可表示值（ties-to-even），再以 Python float / C# double 精确承载。首次查询/生成事实前就完成这一投影，dtype 继续由现有 context_fields 声明；禁止转换调度直接使用未投影的 double 端点。
2. 原 fight_scene_context 的字段、行、端点、duration 和其他输出值保持现有生成规则，不用内部执行视图回写或重建 scene token。source reader/sample builder 继续按既有规则生成 raw FP32 场景，验证其值与执行视图对应，不能要求 canonical 数值本身已是 FP32 精确表示。
3. 实时从保存的 FP32 端点恢复，投影对已为 FP32 的值必须幂等。滑步结束、epsilon 端点和事实时间等派生量，统一在提升后的 double 值上按共享公式计算；不只在某个入口再次量化派生边界，不分别保存派生事件时间。
4. 内部调度以投影后的 start/end 为权威，所需长度在查询时求差，不从 duration 字段倒推端点，也不把该长度写回原 duration。非有限值、FP32 溢出和非法 start > end 由父级校验；量化塌缩窗口在内部不产生事实，但不得因此删除或合并 canonical/raw scene token 行。模型仍按原 ContextEncoder 的裁剪、排序、duration 编码与 mask 处理。
5. 状态机的战斗时钟、动作请求/效果时间继续使用既有 double 语义，不能把整个 timeline 舍入到 FP32。不能用放大 epsilon、事后改写技能表或强行把邻近事件合为同刻来掩盖精度分叉。

例如 canonical 原端点继续输出 600.004 秒；旧/新 raw FP32 场景均为 600.0040283203125 秒，新转换与回放都用后者调度事实。其与动作效果的相对顺序可能相对旧转换修正，应记录为 R10 的状态/执行事实差异；场景 token 的公共输出和模型输入不得随之换一套表达。

### 6.4 转换与实时消费相同语义

- 转换的请求前和动作效果经过的所有场景边界，均使用共享事实流和同刻批次入口。
- 实时路径移除独立的场景边界构造及逐条快照提交循环；`sync_state` 如仍有必要，只能作为共享调度的薄入口，不能保留旧实现。
- `SceneTemplateProvider.is_moving_at/targetable_at/next_state_event_after` 等仍被调用的方法只委托共享规则；无调用的方法删除。
- `rewrite_scene_player_state` 按第 6.5 节接入转换与实时，只合成停机 ETA/剩余；不再覆盖已经由状态机事实确定的 is_moving。
- 真实日志执行保留 observed cast override；技能表不使用该覆盖值修正接受能力。

移动事实进入状态机可能改变旧转换中的接受结果。必须用固定日志片段对比动作接受、状态转移、历史游标、直接/DoT 威力和首个失败位置；报告证据后处理有依据的时序问题。禁止跳过动作、强制可用性为 1、用标签回填表或忽略失败。

原 86 维的编码一致性应在相同 raw 输入上比较；场景事实修复造成 raw 状态变化时单独记录。

### 6.5 状态场景字段的离线/实时统一装配

这是 R12 的正式修复，不能只保留一个无人调用的共享查询函数。

1. 在既有 scene_state.py 中收敛 `rewrite_scene_player_state` 为唯一状态场景字段合成入口，使用完整场景执行视图查询 next_untargetable_in_seconds 与 downtime_remaining_seconds。没有窗口时按既有约定输出 0；不依赖模型裁剪后的 scene token 窗口。
2. 转换 `_observe_canonical` 与实时 `LiveBatchBuilder` 的公共 canonical 装配路径都在状态 reader、raw DELTA 构造和历史行缓存校验之前调用它。build 与 build_from_canonical 共用这一入口，不只修复现场 observe 的分支；cache reader 直接消费转换时已经合成并保存的结果，不另算一遍。
3. current_state_context 与 state_history_context 中的 previous_action_after/request_state 分别按各自冻结的 time_seconds 查询；不能用本次观测时间统一刷新历史。无前序、wait、历史裁剪和重复观测沿用相同规则，幂等调用必须得到同值。
4. 只对需要修改的状态容器/向量制作副本，不就地改写共享 canonical、冻结 frame、历史缓存或回放快照；场景查询可以按场景实例和冻结时间缓存，避免每次观测深拷贝整个上下文。
5. 合成函数的允许写入集合只有这两个状态字段，保留原 null 规则；不改 is_moving、Boss 事实、技能表、技能历史 token、scene token、监督或执行统计。state_features.py 仍只做结构/数值读取，不能为了补这两个字段接管场景业务。

本次输入使用的完整场景由调用方既有契约提供；空场景与已有消融配置也通过同一入口处理，不新增隐式的场景启用规则。验收以 [10,20) 停机窗口在 t=5/t=15 的 5 秒值为最小用例，再比较真实历史两段及 build_from_canonical 复用路径。

## 7. 统一 schema、布局与 tensor 字段声明

### 7.1 技能表进入现有状态分组

在 `canonical_output.state_vector_groups` 中为全部分组显式声明 encoding，第五组为 skill_availability。快照顺序在同一个父级声明，C# builder 不各写一份前缀数组。

示意配置如下；原四组的其余字段定义和顺序保持：

```yaml
canonical_output:
  state_snapshots:
    - previous_action_after
    - request_state
  state_vector_groups:
    player_state:
      feature_keys_field: player_state_feature_keys
      context_key: player
      encoding: anchored_delta
    # buff_state、target_buff_state、resource_state 也显式为 anchored_delta
    skill_availability:
      feature_keys_field: skill_availability_feature_keys
      context_key: null
      feature_source: action_keys
      encoding: absolute_binary
```

技能表与原状态同属状态 schema；不同编码由字段声明决定。删除初稿独立的 `state_skill_availability` 配置旁路，不把二值字段交给差分/归一化后再通过特殊补丁恢复。

C# SchemaConfigLoader、OutputContextSchema 和 Python output_context_schema 读取同一声明，缺失/未知编码显式失败。current/history 均使用相同分组模板；每段技能字段必须精确等于对应输出的 action_keys 顺序。

### 7.2 保存有序分组，布局只正向推导

在现有 `common/policy/data/schema.py` 中定义小型不可变状态分组记录和 `StateFeatureLayout`：

- 分组记录保存 group_key、feature_keys_field、完整有序 feature_keys 和 encoding。
- `TrainingSchema` 用有序分组序列代替只有 key 列表的 `state_group_feature_keys` 字典；同时保存快照顺序。
- 分组集合、顺序、重复字段、编码类型、两段动作字段和 action_keys 全部严格校验。
- `StateFeatureLayout` 从这些记录推导原状态组、绝对二值组、分组切片、请求时间位置、base_state_dim、availability_dim、state_dim。
- layout 不读取本机职业 YAML；编译新数据时使用本次真实动作布局，恢复时只使用产物保存的布局。
- `state_vector_dim()` 这种含义不明确的旧 API 迁移为显式 layout 属性；所有调用者同时更新后删除，不保留“有时返回 86、有时返回 136”的兼容行为。

布局随 schema 和输入契约保存，不能依靠 JSON object 的插入顺序。以有序列表序列化后，`sort_keys=True` 也不能改变字段含义。

`DataSpec.state_dim` 保留为模型需要的合并宽度，同时明确保存 `base_state_dim` 作为模型辅助投影宽度；它们是从 layout 生成的规格快照，不是第二份配置。创建和恢复时由同一布局校验函数交叉校验，禁止用 `state_dim - 2*num_actions` 反推基础字段数。

`ModelInputContract` 是 checkpoint 恢复时的父级校验入口：校验 schema/layout、DataSpec、固定动作顺序和词表一致，再提供给各消费者。独立加载器不靠补默认值或当前 YAML 修正缺失信息。

### 7.3 在现有 context_fields 中补齐形状语义

扩展现有 TensorField 的逻辑轴/维度声明，统一表达 batch、scene、history、bank、base_state、availability、state 等轴。维度值来自 layout/DataSpec 和调用方容量。

- 缓存原状态的 ABS/DELTA/null/reset 末轴为 base_state。
- 冻结技能表的末轴为 availability。
- 正式模型状态值末轴为 state，辅助 mask 末轴为 base_state。
- sequence axis 和阶段视图由父级字段声明派生，不能在 shape 与 sequence_axis 两处维护相互矛盾的值。
- cache reader、dataset、collator、encoder、GRPO、模型校验和 ONNX tensor specs 使用同一个字段解析/校验入口。
- 删除 `startswith("state_")` 推断同宽的规则；字段集合根据显式语义/阶段筛选生成，不维护另一份手写清单。

共享规则只处理结构、dtype 和 padding。GRPO 负责轨迹元数据，部署负责固定容量与 provider 精度，模型负责自身容量，各自保留领域校验。GPU 热路径仅做无同步的结构检查；数值校验放在 CPU 原料/持久化入口，不逐步调用 `.item()` 回读 GPU。

## 8. 共享读取、完整缓存与唯一编码装配

### 8.1 离线与实时共用状态 reader

新增 `common/policy/data/state_features.py`，承载统一状态读取和装配函数；布局类型引用 schema.py，不再在此声明第二份字段结构。

拟定接口：

```text
read_state_tokens(state_context, layout)
  → base_values FP32 + base_null_mask bool + availability bool

assemble_state_inputs(encoded_base_values, availability, history_mask, layout)
  → combined_state FP32
```

reader 一次完成：

1. 校验完整分组与 feature keys；当前/历史及固定动作顺序必须匹配。
2. 原状态按统一数值/null 规则读取；拒绝字符串伪装数值、NaN/Inf 和转换到 FP32 后溢出的值。
3. request_state.time_seconds 必须为已知、有限数值。
4. 技能表必须存在、完整、无 null，canonical 元素必须为数值 0/1；转换为 bool 前校验值域，不能把 0.5 或字符串转成 true。
5. 为 empty history 构造 layout 规定的空张量，原状态与表宽度分别明确。

TrainingSourceReader 与 LiveBatchBuilder 都调用它，不分别增加一套技能表 reader。删除它们原有重复状态解析循环、nullable helper 和自行拼接 feature key 字段名的逻辑。

实时历史缓存可以保留行身份与 device tensor 缓存，但其原料和校验来自此 reader。行一致性必须包含冻结技能表，不能因原 86 维相同就复用另一份表。

### 8.2 raw/cache 字段

| 阶段 | 字段 | dtype | 形状 |
| --- | --- | --- | --- |
| 完整 bank | state_skill_availability | bool | `[N,2A]` |
| dataset bank | history_bank_state_skill_availability | bool | `[N,2A]` |
| dense raw history | history_state_skill_availability | bool | `[B,H,2A]` |
| 当前缓存样本 | current_state_skill_availability | bool | `[2A]` |
| 当前 raw batch | current_state_skill_availability | bool | `[B,2A]` |

这些名称是现有 tensor 阶段声明的映射，不是额外 canonical schema。原 ABS/DELTA/null/reset bank 保持 D_base 宽度；不增加技能表 DELTA/null/reset 字段。

`build_history_bank` 继续保存完整历史，技能、状态、技能表和执行统计按统一写入序号对齐，并验证旧前缀未变。第 0 行 sentinel 的表全 false，仅用于无效 gather。

样本继续保存 history_end/history_length 引用和当前 raw 数据；cache writer 复用字典流式写入。compiled reader 按共享声明校验 shape、dtype、完整布局和 sentinel，不自己写 86/50/136 分支。

以下参数仍仅属于读取侧，不能进入 cache signature、编译 worker 或 bank 构建参数：

- model.history_capacity
- model.history_reset_keep
- model.time_delta_scale
- 随机裁剪和读取侧历史消融设置

本次格式与场景语义升级需要一次性重编译；之后调整读取配置仍复用同一份完整 bank。

### 8.3 ContextEncoder 统一装配

ContextEncoder 接收已校验的 layout；训练、实时、分析三个生产构造点由各自父级保存契约提供 layout，不再分别传入零散的维度和动作顺序猜测布局。

顺序固定为：

1. 通过 context_fields 对原状态和技能表执行同一次窗口选择/gather。
2. 只从原状态 ABS 的 request time 计算现有时间锚点。
3. 只对 anchored_delta 原状态执行现有 ABS/DELTA 编码与 null/reset 处理。
4. 调用唯一 assemble_state_inputs，把对应 bool 表转为 FP32 0/1 后拼到末尾。
5. 对 history padding 清零完整合并向量；删除 raw 字段和 bank 引用。
6. 返回既有正式模型字段，不新增技能表独立模型输入。

Normalizer 的 register_schema/state_encoding_metadata 按 layout 只处理 anchored_delta 组；不能遍历全部五组再给技能表补特殊归一化规则。原状态和 scene 的归一化算法不变。

collator 只合批、gather/pad；模型 forward 与 ONNX runtime 不再拼接。StateInputAssembler 如用函数即可表达，不为了命名新增无状态包装类。

### 8.4 模型入口与初始化

- state_proj：D_state → d_model。
- state_null_proj、state_reset_proj：D_base → d_model；reset 保留原零初始化语义。
- history/current 仍共享 `_embed_state`。
- shape 校验来自共享 tensor 声明；删除完整 shape 必须相等的旧规则。
- `_embed_state` 当前的可选 null-mask 兜底若继续保留，必须按 D_base 构造；不能对 136 维 values 直接 zeros_like。正式生产输入仍必须携带辅助 mask。

本次状态投影扩宽会改变参数数量和默认初始化随机数消耗，不把新旧训练结果差异全部归因于技能表；效果对照另行固定公共参数与训练条件。

## 9. GRPO、分析、PPG 与部署共用父级布局

### 9.1 GRPO

`grpo/trainer.py` 创建 GrpoRolloutStore 时传入已恢复的 ModelInputContract；storage 使用父级布局与 tensor 字段声明检查 prepared batch，不再根据 values 宽度猜 mask。

manifest/trajectory 头部通过既有输入契约序列化保存和核验同一契约；不在每条 decision 内重复定义 schema。固定动作顺序以保存契约为准，现有 decision 中的 action_keys 只可作为同一顺序的校验值，不能成为另一套权威。

仅保存正式 encoded 输入和现有独立 history_cursor 元数据，拒绝 raw 技能表/history_bank 混入；校验有效状态行的表尾为二值。旧 rollout 格式明确拒绝。本次不改变 GRPO 更新算法或 checkpoint 热启动语义。

### 9.2 分析与 PPG

分析编码复用正式 ContextEncoder/layout；字段标签从完整布局读取。黑魔 AF/UI/MP 标签和 PPG 初始恢复仍读取原 raw ABS 的基础字段切片，不能把合并总宽度作为 raw ABS 宽度。

更新 `scripts/model_analysis/job_labels/black_mage.py` 与 `scripts/autoregressive_replay/ppg.py` 的 schema 调用，保持真实执行统计独立。val_ppg 运行完整验证 source 到最后一个可选中 Boss 窗口结束；失败返回全 0 且计入平均。none_ppg、Top-1/Top-3 的既有指标定义不变。

### 9.3 ONNX 与恢复

- 正式图仍使用现有 12 个 MODEL_INPUT_FIELDS，不增加独立技能表图输入。
- tensor_inputs 从 context_fields 的逻辑 shape 和 layout 生成，删除部署侧另一份 shape 字典。
- make_inputs 从共享字段规格创建输入；有效 state 行的技能表尾按二值生成，不能使用连续随机数。
- state_layout 展示/签名字段由保存的 schema 生成，包含分组顺序、完整字段名和编码类型。
- 部署恢复只核验外层 state_layout 与 input_contract 的一致性，删除当前用外层布局 replace 输入 schema 的路径。顺序已由有序序列保存，不需要覆写补救。
- manifest.schema.json、部署 Python 校验与保存输入契约同步更新；缺失字段、错序、同宽异义和不匹配签名均失败。
- 有效行表尾必须二值；无效 padding 仍按 mask 排除，保留脏 padding 对照测试。

checkpoint、训练续训、实时后端、分析和导出加载器均通过 ModelInputContract 校验。无需增加一个只为旧 DataSpec 或旧 schema 服务的兼容适配模块。

## 10. 生产修改和删除清单

### 10.1 新增文件

| 文件 | 职责 |
| --- | --- |
| `Combat.Sim/FightEngine/Facade/SkillAvailabilityBuilder.cs` | 真实技能只读接受能力构造；不拥有状态或队列 |
| `common/policy/data/state_features.py` | 各入口共用的状态 token 读取、入口校验和编码后拼接 |

### 10.2 修改已有父级和消费者

同目录内多个文件合并列出；所有调整均服务于第 2 节的问题与新状态契约。

| 文件/模块 | 必须完成的调整 |
| --- | --- |
| `config/schema.yaml` | 状态分组编码、快照顺序、第五组和运行时/canonical 版本 |
| `Combat.Sim/FightEngine/Config/SchemaConfigLoader.cs` | 严格读取完整分组声明，拒绝缺失/未知编码 |
| `Combat.Sim/FightEngine/Facade/JobSimulator.cs` | 捕获 frame、完成基准、统一 validate/available 与表/mask 的接受查询、动作布局复用、同刻批次、子会话处理器绑定 |
| `Combat.Sim/FightEngine/Facade/CombatStateMachine.cs` | 保留唯一接受规则和内部执行检查，移走模型动作后基准写入；旧 AvailableActions/AvailableActionKeys 无独立用途时随调用迁移删除 |
| `Combat.Sim/FightEngine/Facade/RandomSequenceReplay.cs` | 可用动作消费统一的提交能力；更新包含排队结果的回归预期，不另加即时合法性过滤 |
| `Combat.Sim/FightEngine/Models/Combat/CombatState.cs` | 保存完整不可变 LastDecisionAfter，clone 不共享可变容器 |
| `Combat.Sim/FightEngine/Outputs/ModelStateSnapshot.cs` | 完整 frame 和一致冻结 |
| `Combat.Sim/FightEngine/Outputs/OutputContextSchema.cs` | 统一五组元数据及编码描述 |
| `Combat.Sim/FightEngine/Outputs/OutputContextBuilder.cs`、`StateOutputRouter.cs` | 在完整布局与历史组合后统一生成输出 |
| `Combat.Sim/FightEngine/Outputs/TokenBuilders/StateTokenBuilder.cs` 及原分组 builder | 父级分派与两段字段顺序统一，子 builder 保留各组数值构造 |
| `Combat.Sim/FightEngine/Outputs/ContextBuilders/StateHistoryContextBuilder.cs` | 收集/缓存冻结历史行，删除独立 feature keys 装配和 StateBefore 兜底 |
| `Combat.Sim/FightEngine/Policy/PolicyContextBuilder.cs` | 在向量化前组合真实/策略动作和历史，删除序列化后补列方案 |
| `Combat.Sim/FightEngine/System/Timeline/CombatTimelineRuntime.cs` | 调整分支运行数据复制与实例处理器归属；复用既有同刻优先级 |
| `Combat.Sim/FightEngine/Sessions/SimulationSession.cs` | 在一次会话锁内执行事实批次并返回推进结果；公开验证转交统一接受查询 |
| `scripts/common/inprocess_backend.py`、`state_machine_types.py` | 暴露同一事实批次能力，复用现有运输/结果类型；validate_at 保留统一接受结果和原因 |
| `scripts/common/scene_state.py` | 唯一场景执行视图/校验、查询、边界、批次事实及只修改状态副本的 ETA/剩余合成；不回写 scene token |
| `scripts/convert_fflogs/training/training.py` | 保留原 fight_scene_context 输出；调度消费共享执行视图，reader 前调用共享状态字段合成，取消事后 is_moving 覆写 |
| `scripts/autoregressive_replay/context.py`、`scheduler.py` | scene 适配及决策节奏保留；build/build_from_canonical 共用状态字段合成后再交 reader，委托共享调度与 encoder |
| `common/output_context_schema.py` | 从同一 YAML 读取分组声明，输出完整有序元数据 |
| `common/policy/data/schema.py` | 保存有序状态分组，提供唯一 StateFeatureLayout 和布局校验 |
| `common/policy/data/spec.py`、`input_contract.py` | 从 layout 生成 DataSpec，恢复时统一校验，升级输入版本 |
| `common/policy/data/context_fields.py` | 唯一字段、逻辑 shape、阶段及 padding 声明 |
| `common/policy/data/normalizer.py` | 仅对 layout 的原状态组生成既有编码元数据 |
| `common/policy/data/context_encoding.py` | 同行 gather、原状态编码、唯一拼接、删除 raw 字段；技能和 scene 编码保持原语义并做基线对照 |
| `common/policy/data/compiled_cache.py` | 使用共享结构校验，保存完整两类 bank 并拒绝旧产物 |
| `scripts/convert_fflogs/source/source_reader.py` | schema 构造调用父级，状态解析委托共享 reader；保留 scene token 读取，验证 FP32 结果与执行视图对应 |
| `scripts/convert_fflogs/training/history_bank.py`、`sample_builder.py` | 保存完整表 bank 与当前表，保留行对齐、前缀稳定性及非状态 raw 数据；现有场景 tensor 表达不变 |
| `training/data/dataset.py`、`collator.py` | 数据阶段映射、维度与 padding 全部来自父级；无额外拼接 |
| `training/loop/training_loop.py`、`checkpoint.py` | 传入保存布局，训练/续训使用统一输入契约 |
| `common/policy/model/input_encoder.py` | 扩宽唯一值投影，辅助投影用基础宽度，共享结构校验 |
| `grpo/storage.py`、`trainer.py` | 传入/保存同一契约，校验 encoded 输入，升级 rollout 格式 |
| `scripts/autoregressive_replay/ppg.py`、`backends.py` | 初始恢复使用基础 raw 布局，后端使用保存的合并布局 |
| `scripts/model_analysis/common.py`、`job_labels/black_mage.py`、`token_metadata.py` | 共享 encoder/layout，基础字段标签不误读表尾 |
| `scripts/onnx_export/contracts/contract.py`、`deployment_contract.py` | 统一 tensor shape、二值 golden 输入、布局一致性校验 |
| `scripts/onnx_export/manifest.schema.json` | 有序分组、编码描述及版本 const 同步 |
| `scripts/onnx_export/export/checkpoint.py` 及 runtime/分析输出消费者 | 按保存契约恢复；残留调用只迁移到父级，不新增布局推断 |
| `AGENTS.md`、`docs/项目各文件说明.md` | 实施后同步冻结表、场景事实、格式版本和职责说明；结构目录按既有要求排序 |

`cache_compile.py/cache_load.py/cache_writer.py`、PolicyDecisionHistory、PythonBridge、ONNX graph 等复用既有业务流程，核验共享接口迁移即可。发现实际重复布局时移入已确定父级；不因为列入检查就无理由改写模块。

### 10.3 必须删除或收敛的旧实现

| 旧内容 | 唯一去向 |
| --- | --- |
| 独立 state_skill_availability schema 旁路 | 现有 state_vector_groups 中的第五组和 encoding |
| state_group_feature_keys 裸字典及含义不清的 state_vector_dim 调用 | 有序分组 + StateFeatureLayout 的显式视图 |
| TrainingSourceReader / LiveBatchBuilder 两套状态数值解析、nullable helper | state_features.py 的共享 reader |
| 多处写死 feature keys 字段名、当前/历史各一份元数据装配 | 输出 schema + OutputContextBuilder |
| 序列化后给 policy 技能表补列/重排 | policy 父级先组合冻结语义行，再统一向量化 |
| 所有 state_ 同宽、mask shape 必须等于 values 的假设 | context_fields 的显式逻辑维度 |
| 部署 tensor_inputs 独立 shape 表 | 现有 MODEL_INPUT_FIELDS 的完整形状声明 |
| 部署用 state_layout 覆盖输入 schema | 有序序列化和严格一致性核验 |
| `_resolve_is_moving` 与实时移动/窗口/边界的重复实现 | scene_state.py 的共享场景规则 |
| 转换直接按 double 原始端点执行、cache 生成时才量化的双精度分叉 | 首次事实查询/提交前共用与原 raw FP32 cache 一致的执行视图，保留 scene token 输出 |
| 只有转换调用 ETA/剩余合成、实时 canonical 直接进 reader 的分叉 | 离线/实时共用状态副本合成入口，覆盖 build_from_canonical 与历史两段 |
| 逐条场景事实提交即推进、实时快照反复发送所有事实 | 共享事实流 + 同刻批次入口 |
| ValidateActionAt/AvailableActionKeysAt 自行全局拒绝队列占用、用立即执行检查代替提交能力 | EvaluateActionSubmission 的同一接受结果，统一提供表/mask/validate/available |
| fork 原样复制父模拟器实例回调 | 父级统一初始化/绑定，子会话持有自己的处理器 |
| 缺 ModelState 时回退 StateBefore 的历史输入 | 新契约明确失败，不伪造历史技能表 |

删除前核对生产、测试、公共/动态调用及可选配置。多个入口复用父级不等于保留多个实现；薄入口只能负责参数和所属生命周期，不得复制规则。

## 11. 契约版本与产物迁移

以下以本次核查基线为准。实施前若其他工作已升级，继续从实际最新版本递增，不能覆盖或回退。

| 契约 | 当前 | 拟升级 |
| --- | --- | --- |
| sidecar_contract_version | 16 | 17 |
| canonical_output.schema_version | 14 | 15 |
| TRAINING_SAMPLE_SCHEMA_VERSION | 10 | 11 |
| INPUT_CONTRACT_VERSION | 21 | 22 |
| CACHE_FORMAT | raw_json_compiled_samples_v22_raw_state_deltas | raw_json_compiled_samples_v23_state_skill_availability |
| DEFAULT_CONVERSION_VERSION | raw_json_to_compiled_v24_raw_state_deltas | raw_json_to_compiled_v25_state_skill_availability |
| DEPLOYMENT_CONTRACT_VERSION | 24 | 25 |
| DEPLOYMENT_MANIFEST_VERSION | 15 | 16 |
| GRPO_ROLLOUT_FORMAT | 6 | 7 |

版本覆盖有序状态分组、冻结技能表、同刻事实、与 raw FP32 cache 一致的场景执行视图、移动语义、ETA/剩余统一装配、公开提交能力查询、mask 宽度、DataSpec 和部署结构的整体迁移。TOKEN_ENCODING_CONTRACT 保存两段表按动作顺序追加、表为绝对二值、辅助 mask 只作用于原状态、同一个 S-Emb，以及执行视图精度和两段状态场景合成语义。新 canonical/training source 必须已经按这些状态/执行规则生成，不能只给旧输出重写版本号。技能与 scene token 的输出字段、raw 表达和模型编码保持原契约；Normalizer 规则版本无需为本次状态装配单独升级。

| 旧产物 | 处理 |
| --- | --- |
| 原始 FFLogs 输入与完整 scene | 保留，复用已有文件，不要求重新下载 |
| canonical/training source、compiled cache | 旧版本或缺表直接拒绝；通过正式转换流程重编译，不补零兼容 |
| checkpoint | 输入契约拒绝；生成新结构训练产物，不迁移旧投影假装等价 |
| GRPO manifest/trajectory | 旧格式拒绝，重新生成 |
| ONNX/deployment package | 旧契约拒绝，由新 checkpoint 重新导出 |
| FightEngine/PythonBridge DLL | 同一工作树重建，程序集版本与 schema 不匹配则拒绝运行 |

兼容性版本不等于发布版本，不触发 CHANGELOG/release。实施验收使用新结构的小型模型/缓存完成闭环；正式训练、大规模重编译和正式部署包另记执行条件及结果，不能用“计划已写”或跳过的测试声称产物已经更新。

## 12. 回归验收

### 12.1 字段、顺序与单一权威

- 每个快照的每个动作恰好一个数值；当前黑魔为 25 + 25，合并宽度 136，辅助 mask 宽度 86。
- 使用不同动作数量/顺序的夹具验证自动推导，至少覆盖黑魔之外的布局。
- 缺字段、重复 key、同宽错序、错误编码、非二值和 null 表均失败；不允许 bool 强转掩盖错误值。
- 保存契约经过 JSON 字典键重排后，状态组与技能列顺序仍不变；部署旁份布局不一致必须失败。
- 相同 canonical 输入交给离线与实时 reader，结果完全一致；对相同非法输入给出一致的拒绝结果。
- ONNX shape、GRPO 校验、模型校验和 cache 结构均来自父级声明；修改一处布局不需要修改独立形状表。

### 12.2 冻结、时序和 fork

- builder 与正式接受规则在同一状态/队列下相同，且不改变时钟、状态、queue 或 history。
- R11：真实技能的冻结表、原始 mask、ValidateActionAt/validate_at、AvailableActionKeysAt 在同一状态与队列下逐项一致，拒绝原因来自同一接受结果；真实提交在独立 fork 上核验，避免比较过程中先改变原会话。
- 覆盖 0 秒 GCD → 2.1 秒排队 GCD → 2.2 秒立即 oGCD、空队列且进入排队窗口、满队列且仍需等待、资源不足四类情形。At 入口只按既有契约推进；内部执行校验不能因接受查询统一而提前执行排队动作。
- MP、资源、Buff、移动、冷却、读条锁、GCD 锁和排队窗口均有区分用例。
- 请求 frame 在本次 timing/queue 变更前冻结；无前序完整复制 request frame。
- 上一动作后与请求时表可以不同；冷却恢复、后续动作、窗口裁剪和重复输出不得回填旧表。
- 真实动作与 wait 的 sequence 对齐；当前监督动作不进入其请求前历史，技能表不使用标签信息。
- 老请求晚生效不能覆盖新 request/wait 的基准。
- R2：父子 queueOccupied 不同，子动作效果只读取子队列；覆盖 Fork、ForkWithoutHistory、restore、二次 fork。
- policy 组合不修改原 frame/缓存；同一记录在真实/完整动作布局下输出不会串列。
- vector/tensor 输出的字段与含义一致。

### 12.3 场景与批次

- R1：移动开始、有效移动结束分别与 ActionEffect 同刻，检查动作后 frame 与技能表。
- R10：以 600.004 秒等非二进制精确端点覆盖数百秒时域，实际经过转换 → canonical → compiled cache → 回放；比较规范化端点、事实批次顺序、状态和冻结表，不能只比较绕过落盘的两次共享 helper 调用。
- 分别让 ActionEffect 位于原始端点、规范化后的实际边界，以及其前后；覆盖开始/有效移动结束、向上/向下舍入。两端使用相同动作时间，事实优先级只在实际同刻时生效。
- 执行视图投影幂等，与已有 raw FP32 cache 的端点逐值一致；canonical 场景端点/duration 及记录行保持旧基线。零长度/量化塌缩、非有限值、FP32 溢出和非法端点顺序按父级规则处理，内部有效区间合并不回写 scene token；不改变动作时钟精度或扩大 epsilon。
- R12：同一场景与动作轨迹通过转换、live build、build_from_canonical 后，ETA/停机剩余在 reader 前与模型输入中一致；覆盖停机前、中、后、空场景、wait、历史两段时间不同及历史裁剪。
- 场景状态合成只改变允许的两个状态字段，不修改原 canonical、冻结 frame、缓存或回放快照；重复调用幂等，所有非状态 token 与调用前保持一致。
- 同刻 Boss/移动/目标数/团辅共同变化，动作效果及 DoT/周期结算看到完整场景状态；转换和实时结果一致。
- 批内出现一个非法事实时，时钟、待处理事件和状态全部保持原样。
- 单事实入口与单元素批次一致；批次稳定排序，不通过循环调用旧单事实实现。
- 单窗口、重叠、短窗口、相接窗口、滑步端点和 epsilon 附近查询与事实流一致。
- 非零初始时刻、场景 reset、多轨迹及并行 session 不共用事实游标。
- 固定真实日志片段比较新旧接受结果、状态、游标和直接/DoT 威力，对差异给出首个位置与原因。

### 12.4 编码、缓存与模型

- 固定同一份原 raw 输入，合并向量前 D_base 维与原算法逐项一致。
- ABS 首行、DELTA 后续行、字段恢复、empty history、窗口重置和随机裁剪都不改变表的绝对 0/1。
- skill、state、availability 共用 history gather 行；sentinel/padding 不能成为有效数据。
- 只存在一个扩宽 state_proj，表变化能影响 state embedding；阶段选择 mask 变化不反向改变 state vectors。
- 表尾 padding 噪声不影响有效输出；KV incremental 与 full forward、窗口/场景重建保持一致。
- cache 完整保存冻结表；容量、保留数、时间尺度和读取裁剪变化继续命中同一新格式 cache。
- Normalizer 不注册技能表差分规则；PPG/分析基础字段从原 ABS 布局恢复。

### 12.5 持久化、部署与维护性

- 训练/续训、实时、分析、GRPO 和部署均拒绝旧版本、错序动作和不一致布局。
- GRPO prepared 输入保持 FP32，无 raw 字段混入；保存/恢复后的字段和语义一致。
- ONNX 仍为 12 个图输入，make_inputs 有效表尾为二值；PyTorch/ONNX 在既定容差内一致，覆盖 CPU 与可用的目标精度/provider。
- 新结构真实转换 → cache → batch → 模型 → live replay/GRPO → export 至少完成一条可审计闭环，不能只以模型随机张量测试替代。
- val_ppg/none_ppg 使用原执行统计和完整运行规则；不因历史窗口或输入扩维缩短回放。
- 第 10.3 节旧实现完成调用迁移后删除；新增文件必须有独立职责，不能留下“为适配新结构而维护旧结构”的第二实现。

### 12.6 非状态 token 的基线与跨入口对照

- 在修改前固定基线夹具，覆盖真实技能/wait、空历史/窗口裁剪、移动/停机/团辅/目标数窗口，以及非二进制精确端点；分别保存 canonical 输出、raw cache 字段和正式模型输入。基线不因新实现失败而直接重录覆盖。
- 给定相同执行事实、场景原料和读侧参数，对照重构前后的全部技能 token 与 scene token，而非只检查 shape。离散 ID、字段/行顺序、mask、监督索引逐值相等；canonical 数值与原有 FP32 转换保持一致。
- 同一 raw 输入经离线 cache、实时 build/build_from_canonical、GRPO prepared 输入和部署宿主编码后，非状态模型字段保持对应一致。CPU 同路径要求一致；优化造成浮点运算次序变化或跨 provider 对照只能使用既有精度容差，不能放宽容差掩盖字段、时间、顺序或算法变化。
- 单独改变状态 token 的技能表或 ETA/剩余时，其他 token 的输出与输入必须不变；不要求状态扩维后的模型 logits 或自回归动作不变。
- 真实轨迹若因 R1/R10 等状态/事实修正而分叉，报告首个差异，再以相同事实夹具验证非状态编码，禁止为通过对照伪造执行统计。未能解释或来自非状态 token 规则变更的差异，不能通过本次验收。
- 内部优化记录相同工作量下的时间、内存/分配及缓存命中变化；不为追求性能放松上述输出/输入等价约束，也不把运行波动作为收益结论。

当前 A=25 时，状态值投影增加 `50 × d_model` 个权重，FP32 状态值增加 200 字节/行，bool bank 增加 50 字节/行；不含容器开销。记录规则判定次数、重复观测与 fork 的开销，确认未引入逐历史行重算合法性、GPU 热路径同步或重复深拷贝。结构验收不等于 PPG 提升。

## 13. 实施顺序与完成条件

### 阶段一：固定父级契约

- [x] 确认第五组、技能名到数值的一一对应及 86 + 25 × 2 布局。
- [x] 先冻结技能/scene token 的输出与模型输入基线，确认第 1.1 节行为边界。
- [x] 统一 YAML/schema 的分组编码与有序序列化。
- [x] 固定场景执行视图精度、状态 ETA/剩余装配与公开提交能力查询语义，保留非状态 token 表达，纳入同一次契约迁移。
- [x] 建立 StateFeatureLayout，补齐 context_fields 的逻辑 shape。
- [x] 让 DataSpec/ModelInputContract 从父级产生和校验规格。

完成条件：字段、顺序、编码与维度各有唯一权威；没有靠反推或旁份 schema 修补的路径。

### 阶段二：先关闭运行时捕获缺口

- [x] 同刻事实批次在单 session 锁内整批校验、入队、统一推进。
- [x] 修复 Fork/ForkWithoutHistory 的实例处理器归属。
- [x] 完成不可变 frame、SkillAvailabilityBuilder 和请求/生效/wait 捕获。
- [x] validate/available、原始 mask 和冻结表共用接受查询，迁移 Python 公开验证与随机回放消费者。
- [x] policy 先组合，父级统一输出 current/history。
- [x] R1/R2/R11 及 old request 保护通过 C# 回归。

完成条件：每份表来自正确时刻、正确状态、正确队列；不再生成随后无法补救的错误冻结数据。

### 阶段三：收敛场景与原料读取

- [x] 转换和实时都使用共享场景查询、事实流和批次入口。
- [x] R10：首次提交前的执行视图与原 raw FP32 场景统一，完成真实 cache 往返边界及 scene token 不变对照。
- [x] R12：转换、live build/build_from_canonical 在 reader 前统一 ETA/剩余合成，验证历史冻结时间和副本隔离。
- [x] 删除旧移动/窗口/边界重复实现和事后移动覆写。
- [x] 离线/实时状态解析迁入共享 reader。
- [x] 完整 raw bank 与当前样本保存表，compiled reader 按统一字段校验。
- [x] 完成真实日志片段差异审计和新 cache 的读取配置复用验证。

完成条件：状态和表消费同一精度、同一时间的事实；ETA/剩余没有离线/实时分叉，保存/恢复不改变执行视图，非状态 token 的输出表达保持。

### 阶段四：编码与消费者迁移

- [x] ContextEncoder 统一 gather、原编码和拼接。
- [x] 模型唯一值投影扩宽，辅助 mask 保持基础宽度。
- [x] dataset/collator、训练/续训、实时、分析和 PPG 迁移到同一 layout。
- [x] GRPO 父级传入保存契约，storage 删除等宽假设。
- [x] 部署从共享 shape 生成规格，删除 schema 覆写恢复逻辑。
- [x] 更新全部相关测试夹具，删除第 10.3 节旧实现。
- [x] 非状态 token 的基线与跨入口对照通过；内部优化只在等价约束内验收。

完成条件：任何消费者都不需要再定义自己的状态字段或拼接公式，原算法和窗口语义保持。

### 阶段五：版本、桥接与端到端验收

- [x] 按实施时最新基线同步升级所有契约常量和 manifest schema。
- [x] 同一工作树重建 PythonBridge，确认实际加载 DLL 版本。
- [x] 完成 Python/C# 回归和新结构真实链路、ONNX 对照。
- [x] 报告跳过项、环境缺项及尚未生成的正式产物，不冒充完成。
- [x] 实施后更新本计划、AGENTS.md 与模块说明，记录发布外的迁移要求。

完成条件：新契约全链路一致，旧产物明确拒绝；生产重复规则已经删除，用户训练配置保持。

## 14. 测试文件与后续验证命令

本节保留完整验收命令。2026-10-08 已使用项目 .venv 执行串行 CPU 定向回归；用户要求不训练，实际执行范围与尚未执行项见第 16 节，不能把下列完整命令视为均已运行。

新增：

- `Combat.Sim/FightEngine.Tests/Facade/SkillAvailabilityBuilderTests.cs`
- `tests/training/test_state_features.py`

主要更新范围：

- C#：Facade/JobSimulatorTests 及随机回放用例、Outputs/ModelStateSemanticsTests/OutputsHistoryTests/OutputsTokensTests/OutputsSchemaTests、Policy/PolicyContextBuilderTests、Sessions/SimulationEngineTests、System/CombatTimelineRuntimeTests/TimelineAuthorityTests、Config/SchemaConfigLoaderTests；必须覆盖公开接受查询的一致性。
- Python 数据/模型：`tests/training/_causal_fixtures.py`、`_common_fixtures.py`、`test_context_encoding.py`、`test_common_norm.py`、`test_common_pt.py`、`test_input_contract.py`、`test_causal_data_contract.py`、`test_independent_token_encoder.py`、`test_input_encoder_normalization.py`、`test_kv_cache.py` 及使用新 schema/DataSpec 的相关夹具。
- 场景/转换/实时：`tests/scripts/test_scene_state.py`、`test_inprocess_engine.py`、`test_context_packet.py`，以及 convert_fflogs、autoregressive_replay 中的 scene、timestamp、context、scheduler、cache、ppg、parity、parallel integration 测试；包含 validate_at 的真实桥接对照、实际 cache 往返的执行精度、ETA/剩余的双入口接入与纯合成边界，以及非状态 token 的输出/输入基线对照。
- 持久化/分析/部署：`tests/grpo/test_grpo.py`、`tests/scripts/test_model_analysis.py`、`tests/scripts/onnx_export/`。

```powershell
dotnet build Combat.Sim/PythonBridge/PythonBridge.csproj --configuration Debug
dotnet test Combat.Sim/FightEngine.Tests/FightEngine.Tests.csproj --configuration Debug
```

先验证父级布局、读取、编码和模型语义：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/training/test_state_features.py tests/training/test_context_encoding.py tests/training/test_common_norm.py tests/training/test_input_contract.py tests/training/test_causal_data_contract.py tests/training/test_independent_token_encoder.py tests/training/test_input_encoder_normalization.py tests/training/test_kv_cache.py
```

再验证共享场景和真实桥接：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/scripts/test_scene_state.py tests/scripts/test_inprocess_engine.py tests/scripts/test_context_packet.py tests/scripts/convert_fflogs tests/scripts/autoregressive_replay
```

完成分析、GRPO、导出边界：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/grpo tests/scripts/test_model_analysis.py tests/scripts/onnx_export
```

这次涉及公共 schema/DataSpec 和基础夹具迁移；合并前补一次完整相关测试域检查，以覆盖未在定向命令中运行的训练消费方。检查通过后不无理由重复运行昂贵用例。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/training
```

真实 checkpoint 的 ONNX 门禁必须使用新结构产物，并按现有测试要求启用实际导出；若文件/provider 不可用，明确记录未验证。固定日志片段转换、性能记录和正式数据/训练/部署产物更新分别保留执行条件与结果，不以单元测试代替。

## 15. 最终验收清单

- [x] 每个技能名对应一个值，两个快照各自冻结；当前黑魔确为 86 + 25 × 2。
- [x] 原状态前 86 维保持现有编码，表尾始终为绝对 0/1。
- [x] 技能/scene 等非状态 token 的公共输出与模型输入保持既有行为，内部优化有基线和跨入口等价证据。
- [x] 技能表作为同一 schema 的第五组，由 encoding 区分，不存在旁路字段契约。
- [x] 所有分组、列顺序、切片、辅助宽度与 tensor shape 从父级声明派生。
- [x] 状态读取和拼接各只有一份实现，模型使用同一个 S-Emb。
- [x] policy 在向量化前组合；当前/历史统一装配，无就地补列或缓存污染。
- [x] 同刻所有场景事实先入同一队列再结算，移动边界冻结结果正确。
- [x] 场景首次执行视图与原 raw FP32 cache 一致，保留原 scene token 输出/编码，实际 cache 往返不改变事实时间和结算顺序。
- [x] ETA/停机剩余在转换与实时两个 build 入口统一合成，各段使用自身冻结时间，原对象与非状态 token 不被修改。
- [x] 真实技能的冻结表、原始 mask、公开验证和可用列表使用同一接受规则，保留内部执行检查的职责。
- [x] fork/restore 使用所属会话处理器和队列，父子分歧测试通过。
- [x] 原状态 null/reset 与技能表分离，empty/padding/window/KV 行为一致。
- [x] 完整 raw cache 保存全部冻结表；读取配置变化复用同一 cache。
- [x] 真实执行统计和 PPG 不从模型请求状态推导，不改变完整回放时长。
- [x] 训练、转换、实时、GRPO、分析、部署使用同一保存布局，旧产物明确拒绝。
- [x] R1～R12 各有对应实现、删除项和回归证据；不留下新旧两套规则。
- [x] 用户 training.yaml 修改保留；仅在用户明确要求后本地提交，未发布或 push。


## 16. 实施记录（2026-10-08）

本轮完成代码、公共契约、相关夹具与文档迁移；未训练、未调用 GPU、未进行大规模缓存重编译或发布。原有 `training.yaml` 实验修改保持，实施前后 SHA256 均为 `1CFDDB387A59562B8F34F7D9E491277FC3566DAE56B69021E0DCB7DA407634CA`。后续按用户明确要求更新 CHANGELOG 并本地提交；用户已取消快速训练测试，未执行 push。

### 已完成的验证

- 当前工作树 PythonBridge 串行重建通过，0 警告、0 错误；FightEngine.Tests 全套 **309 项通过**。覆盖冻结表、同刻批次原子性、公开接受查询、真实职业、policy 布局与父子队列归属。
- Python 按数据/编码、缓存/转换、实时/PPG、GRPO 存储、分析、部署分批验证；报告过的失败均已修复并定向复验。全仓 **1982 项测试收集成功**，不等于完整测试执行。
- 部署契约定向 **120 项通过**；未训练的微型模型实际 ONNX 导出、checker 与 **CPUExecutionProvider** 对照通过。正式新 checkpoint 与生产部署包未生成。
- 同一真实 canonical 经离线 compact cache、live、GRPO 落盘恢复和 ONNX 宿主，12 项输入逐值一致；FP32/FP16/BF16 三种目标 dtype 的宿主转换均在 CPU 检验。状态内容 136 维、辅助 mask 86 维，图输入仍为 12 项。
- 新增 schema 持久化恢复、同宽错动作顺序、缺失/旧/未来 canonical 版本、二值尾部、padding/window/KV、完整 history bank 及读取参数复用检查。

### 非状态基线与真实日志

从固定提交 `87080c9572c412bff5e44721c2055837d1b32c63` 的隔离源码重建旧 PythonBridge，在独立进程采集旧实现，保存为 `tests/fixtures/state_refactor_non_state_baseline.json`，包含来源提交及五份源码 SHA256。该基线没有根据新实现的结果重录。两组固定轨迹覆盖真实技能与 wait、空场景/四种窗口、非二进制精确端点，以及 0/2/5 历史窗口；旧/新 canonical、raw bank 和六项非状态模型输入**逐值相同**。采集器 `_non_state_baseline.py` 拒绝覆盖已有文件，生产不保留旧实现。

额外审计一份真实日志的前 16 条动作（源 SHA256 `9acdbcdc7da3850652485f33c7b97cbd8e85aa7aa4272842503bad156e3aff28`），保留完整场景与原请求/读条时间。新旧均在 step=6、`fire_iv`、请求 `7.0584` 秒以 `gcd_locked` 拒绝；已执行前缀的提交结果、policy 决策和基础状态观测无差异。事实调用批次的分组形式改变，未造成此前缀的执行分叉。该片段未完整转换，不能把缺失的 raw/model 结果算作等价通过；完整非状态等价证据来自上述成功固定轨迹与跨入口测试。未修改日志以绕过拒绝。私有审计产物仅留于 `.tmp/state-availability-baseline-20261008/`。

### 独立审查后的修正

- GRPO 内层 `batch.action_keys`、外层 decision 顺序与文件头保存契约统一校验，避免重复惩罚错列。
- raw `0.1`/`1.7` 与 FP32 执行端点之间即使只差纳秒，也必须消费该未来事实；决策 epsilon 不再使调度器原地停滞。
- live 两入口严格检查 canonical 版本，与离线拒绝规则一致。
- 删除无调用的 replay 场景边界查询及仅可选中事件的备用调度路径，场景推进只使用公共事实流。

### 本轮边界

用户正在游戏，正式训练、优化器训练回归、GPU/provider 对照、长程压力/性能基准、大规模缓存重编译和正式 checkpoint/部署产物更新未执行。小型 CPU 前向与导出只验证结构/数值契约，不代表 PPG 提升或性能收益。基础维度之外每行新增 50 字节 bool bank 与 200 字节 FP32 模型状态值；状态投影新增 `50 × d_model` 权重，未新增 token 或独立投影。完整相关测试域及正式产物流程仍需在允许相应负载后执行。

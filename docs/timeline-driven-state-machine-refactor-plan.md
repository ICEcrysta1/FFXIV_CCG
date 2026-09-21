# 时间线驱动状态机重构计划

> 状态：阶段 1 时间线内核、阶段 2 公共时基资源、阶段 3 职业注册时间资源、阶段 4 统一动作生命周期、阶段 5 Sidecar 与 Python 调用方均已实施；阶段 6 尚未开始
> 记录日期：2026-09-13
> 范围：`Combat.Sim/FightEngine`、Sidecar、FFLogs 转换、自回归/PPG 回放及对应契约和测试

## 1. 背景与结论

当前 FightEngine 已经有 `CombatState.Time`、`AdvanceTime`、冷却/Buff/DoT/MP 等公共运行时，但它仍是“调用方手工拼装时间语义”的状态机，而不是单一时间基准的完整模拟器：

- `CombatStateMachine.StepInternal` 在一次调用里同时执行校验、动作时序和动作效果，普通路径不会把绝对时间推进到技能生效时刻。
- `SidecarHost.Program.Step` 为 FFLogs 回放单独实现“先挂时序、再推进到 logged effect time、最后应用效果”。
- `scripts/convert_fflogs` 还会执行 `advance_hidden`、直接同步 `state.time`、场景状态和历史模式，用来协调日志时间与状态机时间。
- 自回归、PPG、候选预演各自计算动作后的推进长度，容易让冷却、动画锁、GCD 和职业时间资源落在不同时间基准上。

目标是把状态机改为真正的时间线驱动模拟器：

1. 调用方必须为动作、外部事件和状态查询提供绝对时间戳。
2. 状态机独占逻辑时钟，只允许单调向前推进，拒绝回拨时间。
3. 状态机公共层统一维护 GCD、动画锁、冷却、MP tick、Buff、DoT 和职业注册的周期/倒计时资源。
4. 黑魔、机工及以后新增的职业只注册时间规则和事件处理器，不自行循环推进秒数。
5. 转换、自回归、游戏 Hook 和候选预演全部调用同一个时间线内核，不再各自实现动作结算顺序。
6. 状态快照必须包含待处理事件，确保候选预演、回放缓存和搜索分支可以确定性克隆。
7. 从状态机中删除虚拟动作 `ogcd_wait`，但把它保留为模型与调用方之间的策略控制 token。

## 2. 核心边界

### 2.1 调用方负责什么

- 提供动作请求时间：`SubmitAction(timestamp, actionKey)`。
- 提供游戏外部事实的时间：Boss 可选中变化、移动状态、目标数、团辅窗口等。
- 决定何时请求状态：`ObserveAt(timestamp)` 或先调用 `AdvanceTo(timestamp)`。
- FFLogs 转换器负责说明日志时间是 request/cast/effect 中的哪一种观测，不能把 effect time 假装成 request time。
- 游戏插件负责把 Hook 翻译为类型明确、带时间戳的外部事件。

### 2.2 状态机负责什么

- 将时钟推进到请求时间，并处理所有 `event.time <= timestamp` 的待处理事件。
- 在该时刻的真实状态上校验动作。
- 接受动作后安排读条完成、技能生效、动画锁结束、GCD 就绪、冷却回充等内部事件。
- 统一处理 MP tick、通晓 tick、DoT tick、状态过期和职业注册计时器。
- 生成动作历史和可查询快照。
- 保证同一初始快照与同一事件序列得到逐字段一致的结果。

### 2.3 禁止的边界穿透

生产调用路径不再允许：

- 直接写 `CombatState.Time`、`GcdRemaining`、`AnimationLockRemaining` 等时基字段。
- 用 `advance_hidden` 推进部分资源、再恢复另一些资源。
- 通过 `history_mode=replay|standard` 选择两套动作语义。
- 把 `ogcd_wait` 等策略控制 token 当作游戏技能提交给状态机。
- Sidecar、转换器或自回归脚本自行调用 `ApplyActionTiming` / `ApplyActionEffect`。
- 候选预演使用与真实执行不同的未来状态公式。

调试测试仍可通过专用测试夹具构造状态，但该能力不进入正式 Sidecar 协议。

## 3. 时间表示与确定性

### 3.1 使用绝对时间作为内部事实

内部不再以反复递减的 `remaining` 作为时间真相，改为保存绝对截止时刻：

- `GcdReadyAt`
- `AnimationUnlockAt`
- `FightEndsAt`
- `CooldownState.RechargeReadyAt[]`
- `StatusState.ExpiresAt`
- `DotState.ExpiresAt` / `NextTickAt`
- `NextNaturalMpTickAt`
- 职业注册计时器的 `NextTickAt` / `ExpiresAt`

输出层继续按 `max(0, deadline - currentTime)` 生成现有的 `*_remaining_seconds`，因此模型输入不需要仅因为内部表示变化而改变字段名。

### 3.2 时间推进不变量

- `CombatTimelineRuntime.AdvanceTo(t)` 是唯一有权移动逻辑时钟和排空到期事件的生产函数；其他模块不得再实现自己的 `AdvanceTime(seconds)` 或等价循环。
- `CombatState.Time`、事件队列以及各类绝对 deadline 只能由 `CombatTimelineRuntime` 写入；状态模型对外只暴露只读时间。
- `SubmitAction`、`ObserveAt`、`ApplyExternalEvent` 等带时间戳入口可以委托调用 `AdvanceTo`，但不得复制任何推进、排序或结算逻辑。
- `CooldownRuntime`、`StatusTimelineRuntime`、`DotTimelineRuntime`、`MpRecoveryRuntime` 和职业模块仍各自拥有领域规则；它们只注册规则、处理一个已经到期的事件，或返回声明式变更请求，不直接推进时钟、排空队列或自行重排下一事件。
- `AdvanceTo(t)` 中 `t < CurrentTime - epsilon` 必须报错。
- `AdvanceTo(t)` 必须先处理所有早于 `t` 的事件，再处理恰好位于 `t` 的事件，然后才能返回/接受该时刻的新命令。
- `AdvanceTo(10)` 与 `AdvanceTo(3); AdvanceTo(10)` 的最终状态和事件日志必须一致。
- 同一时间戳事件必须按稳定优先级和递增 sequence 排序，不能依赖字典或堆的偶然顺序。
- 浮点容差只用于比较，不用来偷偷提前恢复冷却或延后状态过期。

这里的“中心”只指时钟和事件编排的单一写入边界，不是把所有业务做成超大单体。`CombatTimelineRuntime` 负责时间校验、稳定排序、队列增删和事件派发；MP 数值、冷却规则、Buff/DoT 效果及职业资源变化仍留在各自的小模块中。处理器返回 `TimelineMutation`（状态变更及后续事件请求），由中心统一应用和调度。

### 3.3 同时间戳优先级

第一版明确建立 `TimelineEventPriority`，建议顺序如下，最终以游戏 Hook/回放验证结果锁定：

1. 外部场景事实变化（目标可选中、移动、目标数、团辅）。
2. 已确认的服务器动作生效事件。
3. 周期结算事件（MP、DoT、职业周期资源）。
4. 状态/计时器到期与冷却回充。
5. 决策边界通知。

不论最终优先级如何，必须写成公开契约并由同时间戳测试固定。

## 4. 事件与公开 API

### 4.1 核心事件

内部事件至少包括：

- `ActionAccepted`
- `CastCompleted`
- `ActionEffect`
- `GcdReady`
- `AnimationLockEnded`
- `CooldownChargeReady`
- `StatusExpired`
- `DotTick`
- `MpTick`
- `JobPeriodicTick`
- `JobTimerExpired`
- `SceneChanged`

每个事件具有：

- `Timestamp`
- `Priority`
- `Sequence`
- `EventKind`
- 可选 `ActionInstanceId`
- 可选 `OwnerKey` / `Payload`

`Sequence` 用于解决相同时间、相同优先级事件的稳定排序；`ActionInstanceId` 用于把 request、cast 和 effect 对应到同一次技能。

### 4.2 JobSimulator 最终公开接口

```csharp
public string JobTag { get; }
public double Time { get; }
public CombatState GetState();
public ActionSubmissionResult SubmitAction(double timestamp, string skillKey);
public ActionSubmissionResult SubmitAction(ActionRequest request);
public CombatState AdvanceTo(double timestamp);
public CombatState ObserveAt(double timestamp);
public ExternalEventResult ApplyExternalEvent(ExternalCombatEvent gameEvent);
public ValidationResult ValidateActionAt(double timestamp, string skillKey);
public IReadOnlyList<string> AvailableActionKeysAt(double timestamp);
public double? GetNextScheduledEventTime();
public SimulationSnapshot CreateSnapshot();
public void RestoreSnapshot(SimulationSnapshot snapshot);
public JobSimulator Fork();
public Dictionary<string, object?> FormatState(string mode = "seconds");
public Dictionary<string, object?> FormatVectorState();
public object? FormatTensorState();
```

所有带时间戳的方法都把时间移动委托给同一个 `AdvanceTo`。动作、校验和观测先推进到请求时刻；外部事实先完成载荷校验并排入中心时间线，再由 `AdvanceTo` 按同时间戳优先级派发，不能在门面复制推进或结算逻辑。`GetState` 和三个 `Format*` 入口只读取克隆快照或格式化输出，不改变时间和事件队列，也不属于旧协议兼容层。调用方不能通过先 `Validate`、过一段时间再 `Step` 绕过时序校验；动作校验与接受必须在 `SubmitAction` 内原子完成。`GetNextScheduledEventTime` 只暴露下一内部事件时间供调用方参考，不代表状态机决定下一次调用或自动推进。

### 4.3 动作提交语义

`SubmitAction(timestamp, actionKey)`：

1. 推进到 `timestamp` 并结算所有到期事件。
2. 在当前状态校验技能、GCD、动画锁、移动、目标、MP 和冷却。
3. 非法时不修改状态和事件队列，返回稳定 reason。
4. 合法时创建唯一 `ActionInstanceId`。
5. 在正确阶段消耗 GCD/冷却/职业时序资源。
6. 根据瞬发、读条和技能配置安排 `ActionEffect` 等事件。
7. 返回接受结果与下一关键时间；不自动跳到下一 GCD，也不替调用方决定下一次观测时间。

具体资源是在 request、cast complete 还是 effect 阶段消耗，必须由统一动作生命周期定义，不允许职业文件各自猜测。

游戏输入队列作为动作生命周期的一部分处理：队列容量为 1。动作请求进入队列时
立即执行通晓、MP、目标和其他消耗资源校验；资源不足时请求失败，不会等待资源
生成。若唯一阻塞原因是 GCD、动画锁或临界冷却，且仍在动作队列窗口内，则保留
一个待执行动作，并在最早满足对应 `ready_at` 的时间自动提交；队列已有动作时
返回稳定的队列占用 reason。队列动作、其预留资源和待执行事件必须进入
`SimulationSnapshot`，恢复或 fork 后不能重复执行。

## 5. 公共时间资源注册

### 5.1 注册模型

新增 `JobTimelineRegistry`，职业在 `Bind` 后只注册规则。运行时循环、事件调度以及周期事件的续排全部由 `CombatTimelineRuntime` 维护。

计划提供三类注册：

```csharp
RegisterPeriodicResource(...); // 固定周期累积，例如通晓
RegisterCountdown(...);        // 倒计时并在到期触发，例如连击/野火
RegisterMpTickModifier(...);   // 公共 MP tick 的职业修正
```

注册项保存稳定的 `Key`、周期/截止时间声明、激活条件、停用策略和事件处理器。注册表不提供 `Advance` 或直接队列写入 API；事件队列只保存注册 key 与时间，不序列化委托。恢复快照后由中心通过已绑定的注册表解析处理器，应用其返回的 `TimelineMutation`，并统一决定是否续排周期事件。

### 5.2 MP

MP tick 始终由系统层注册和安排：

- 公共配置提供 tick 周期和基础恢复量。
- `MpRecoveryRuntime` 只声明 MP tick 周期并处理单次 `MpTick` 的恢复量与 MP 钳制；下一次 tick 的安排由 `CombatTimelineRuntime` 完成。
- 黑魔只通过 `RegisterMpTickModifier` 声明 AF 下为 0、醒梦期间增加恢复量。
- 其他职业没有特殊规则时无需实现任何 MP 时间函数。

删除职业接口中的 `ResolveMpRecoveryAmount(...elapsedSeconds)`。状态机在精确 tick 时刻调用 modifier，因此无需再用 `elapsedSeconds` 反推 Buff 在推进窗口内是否仍有效。

### 5.3 黑魔通晓

黑魔在 `RegisterTimeline` 中注册：

- key：`black_mage.polyglot`
- 周期：`timing.polyglot_interval`
- 激活条件：AF 或 UI 生效，即 Enochian active。
- tick：`polyglot + 1`，受资源上限限制。
- 停用策略：清空/重置本周期进度。

删除 `BlackMageJobStateMachine.AdvanceTime` 内的 while 循环。是否在 AF/UI 切换时延续进度、在无元素态时重置，由注册规则显式表达并测试。

### 5.4 机工计时资源

将下列手工推进改为注册项：

- 连击超时：`combo_remaining` -> `JobTimerExpired`。
- 过热层数过期统计：状态到期事件。
- 野火到期结算：`JobTimerExpired`，在到期时读取已经由外部事件更新的 Boss 可选中状态。

删除 `MachinistJobStateMachine.AdvanceTimeBeforeSystem` 和空的 `AdvanceTime`。职业只保留到期事件处理逻辑。

## 6. 将 `ogcd_wait` 改为纯策略控制 token

`ogcd_wait` 仍然有必要，因为动作模型需要表达“当前 weave 机会不再使用 oGCD”。但它不是游戏技能，也不应进入 FightEngine 的 SkillBook、合法性校验、动作生命周期、冷却、动画锁或游戏动作历史。

最终边界：

- 模型候选和 SkillVocab 继续保留 `ogcd_wait`（raw id 0），黑魔候选数仍为 25。
- `ogcd_wait` 的定义从 `config/system.yaml` 移到独立的 policy action 配置；`ProjectConfigLoader` 和状态机不加载它。
- 模型输出真实技能时，调用方执行 `SubmitAction(timestamp, action)`。
- 模型输出 `ogcd_wait` 时，调用方不向状态机提交任何动作，只记录一条 policy decision，然后在调用方选定的后续时间再次请求状态机。
- `CombatState` 删除 `OgcdWaitBoundaryPending`，公共校验删除 `ogcd_after_wait`；是否在 wait 后立即再次调用完全是调用方自己的错误，状态机不为模型策略兜底。
- 删除只服务于 `end_weave_window` 的 `SystemSkillRuntime.cs`。爆发药仍是状态机中的真实系统动作并继续走公共 `grant_status`。

转换层继续根据真实动作时间间隔合成 `ogcd_wait` 监督样本，但只写入模型的 policy decision history，不调用 Sidecar `validate/step/submit_action`。自回归、PPG 和 GRPO 在选中它后同样只更新 policy history，并由外部调度器决定下一次调用时间。

候选特征和历史需要由状态机上方的 policy context 层合成：

- FightEngine 提供 24 个真实游戏动作的预演和战斗状态。
- `PolicyActionRegistry` 增补第 25 个 `ogcd_wait` 候选及其固定技能特征。
- FightEngine 游戏动作历史记录真实生效动作，独立 `PolicyDecisionHistory` 只记录 wait 等 policy 决策；`PolicyContextBuilder` 再按时间合并两者，供模型历史 token 使用。
- wait 候选的 `state_before` 取当前观测状态；`state_after` 由 policy context 层在 fork 上按调用方约定的后续观察时刻生成，不能让真实状态机游标自动跳转。

- 这样可以保持 25 候选（24 个真实技能和 `ogcd_wait`）的模型表达能力，同时彻底移除 wait 对游戏状态的副作用。由于时间线和历史所有权仍有变化，实施后需按新输入契约重新训练或验证 checkpoint。

## 7. 精确文件与函数修改计划

### 7.1 新增 C# 模型和公共时间线文件

| 文件 | 新增内容 |
| --- | --- |
| `Combat.Sim/FightEngine/Models/Timeline/TimelineEvent.cs` | `TimelineEventKind`、`TimelineEventPriority`、不可变 `TimelineEvent`，保存绝对时间、sequence、owner/action id 和 payload。 |
| `Combat.Sim/FightEngine/Models/Timeline/ActionRequest.cs` | 只携带 `Timestamp` 与 `SkillKey` 的动作请求；来源与观测标识不进入状态机协议。 |
| `Combat.Sim/FightEngine/Models/Timeline/ActionSubmissionResult.cs` | 接受/拒绝结果、reason、动作实例 id、请求/预计生效时间和下一关键时间。 |
| `Combat.Sim/FightEngine/Models/Timeline/ExternalCombatEvent.cs` | Boss、移动、目标数与团辅窗口的类型化输入；真实技能统一通过还原后的 request time 提交，不另建服务器确认技能外部事件。 |
| `Combat.Sim/FightEngine/Models/Timeline/SimulationSnapshot.cs` | `CombatState`、待处理事件、sequence 游标的完整可克隆快照。 |
| `Combat.Sim/FightEngine/Models/Timeline/TimelineMutation.cs` | 领域处理器返回的声明式状态变更、取消请求和后续事件请求；处理器本身不能直接写时钟或事件队列。 |
| `Combat.Sim/FightEngine/System/Timeline/CombatTimelineRuntime.cs` | 唯一实现 `AdvanceTo`；负责单调时间校验、稳定排序、队列增删/续排、事件派发、mutation 应用、快照/恢复和下一事件查询。只编排时间，不承载 MP、冷却、Buff、DoT 或职业效果规则。 |
| `Combat.Sim/FightEngine/System/Timeline/JobTimelineRegistry.cs` | 周期资源、倒计时和 MP modifier 的声明式注册、重复 key 校验与处理器查询；不提供推进或直接调度函数。 |
| `Combat.Sim/FightEngine/Models/Policy/PolicyActionDefinition.cs` | 定义不属于游戏技能的模型控制 token；第一项为 `ogcd_wait`（raw id 0），并保留模型所需的固定技能特征。 |
| `Combat.Sim/FightEngine/Policy/PolicyConfigLoader.cs` | 独立读取 `config/policy_actions.yaml`；不挂入 `ProjectConfigLoader`、`SystemStateMachine` 或 SkillBook。 |
| `Combat.Sim/FightEngine/Policy/PolicyActionRegistry.cs` | 从独立 policy 配置加载控制 token，校验它们不与 FightEngine SkillBook 的真实技能 key/id 冲突。 |
| `Combat.Sim/FightEngine/Policy/PolicyDecisionHistory.cs` | 保存带时间戳的模型决策历史，允许记录真实动作选择和不提交给状态机的 `ogcd_wait`。 |
| `Combat.Sim/FightEngine/Policy/PolicyContextBuilder.cs` | 合并 24 个真实技能预演、policy 控制候选和 policy history，输出模型使用的 25 候选 canonical context。 |
| `Combat.Sim/FightEngine.Tests/System/CombatTimelineRuntimeTests.cs` | 单调时钟、分段推进等价、同时间顺序、事件取消/重排和快照恢复测试。 |
| `Combat.Sim/FightEngine.Tests/System/TimelineAuthorityTests.cs` | 架构守卫：生产代码只有 `CombatTimelineRuntime` 实现时间推进；其他 Timeline/runtime/职业类型不得公开 `AdvanceTime`、写逻辑时钟或直接排空事件队列。 |
| `Combat.Sim/FightEngine.Tests/Policy/PolicyContextBuilderTests.cs` | 锁定 wait 仅存在于 policy 层、候选仍为 25、选择 wait 不改变 FightEngine 状态。 |

新增目录后同步更新 `AGENTS.md` 的项目结构，并保持目录/文件排序。

### 7.2 C# 状态模型

| 文件 | 修改函数/字段 | 计划 |
| --- | --- | --- |
| `Models/Combat/CombatState.cs` | `Time`、`GcdRemaining`、`AnimationLockRemaining`、`NaturalMpTickProgress`、`OgcdWaitBoundaryPending`、`Clone` | `Time` 改为无公共 setter，只有时间线内核可写；remaining 改为派生输出或兼容只读属性；删除 wait pending 字段；克隆覆盖只读时间快照。各领域 deadline 由时间线队列/快照统一持有，避免领域模块直接改时间。 |
| `Models/Combat/CooldownState.cs` | `RechargeTimers`、`Clone` | 改存 `RechargeReadyAt` 绝对时间列表，不再每次推进整体递减。 |
| `Models/Combat/StatusState.cs` | `Remaining`、`Clone` | 内部改存 `ExpiresAt`；对输出暴露按当前时间计算的 remaining。 |
| `Models/Combat/DotState.cs` | `Remaining`、`NextTickInSeconds`、`Clone` | 内部改存 `ExpiresAt`、`NextTickAt`。 |
| `Models/Combat/ActionHistoryEntry.cs` | `TimeSeconds` | 明确为 effect time，并增加 request/cast/effect 时间与 `ActionInstanceId`；是否进入模型 token 由输出契约单独决定。 |
| `Models/Combat/StepResult.cs` | 整体 | 消费者迁移期间只作临时适配；最终由 `ActionSubmissionResult` 和时间线事件结果替代并删除。 |

### 7.3 C# 系统层

| 文件 | 修改函数 | 计划 |
| --- | --- | --- |
| `System/SystemStateMachine.cs` | `AdvanceTime`、`SystemSkills`、`ValidateSpecialAction`、`ApplySpecialAction` | 删除按 seconds 顺序调用各 runtime 的总循环和 wait 专用转发；新增内置/职业规则注册与领域事件派发，但不在此处实现推进。 |
| `System/PlayerStateRuntime.cs` | `ValidateCommonAction`、`ApplyActionTiming`、`AdvanceTime`、`AdvanceDowntime` | 动作接受时返回 GCD/动画锁 deadline 的 `TimelineMutation`；删除 `ogcd_after_wait` 分支、pending 清理和递减逻辑；场景变化处理也只返回领域变更。 |
| `System/MpRecoveryRuntime.cs` | `AdvanceTime` | 改为 `Register` 与 `HandleMpTick`；只计算精确 tick 时刻的一次恢复并返回 mutation，不写当前时间、不直接安排下一 tick。 |
| `System/Timeline/CooldownRuntime.cs` | `AdvanceTime`、`ConsumeCooldown`、`ReduceCooldown`、`BuildCooldownSnapshot` | 删除全桶递减；消耗和主动减 CD 只返回 charge-ready 的新增/取消/重排请求，由中心写队列；快照接收当前时间并从 ready-at 派生。 |
| `System/Timeline/StatusTimelineRuntime.cs` | `AdvanceTime` | 授予/刷新状态只返回 `StatusExpired` 调度请求；单次到期处理返回状态 mutation，通过 generation/id 忽略旧事件，不直接操作队列。 |
| `System/Timeline/DotTimelineRuntime.cs` | `AdvanceTime`、`AdvanceDot` | 挂载 DoT 只返回 tick/expire 请求；单个 `DotTick` 只结算一次并返回后续请求，删除窗口 while 循环且不直接操作队列。 |
| `System/SystemSkillRuntime.cs` | 整体文件 | 当前仅实现 `end_weave_window`；删除 `ogcd_wait` 后整文件删除，系统共享 `grant_status` 继续由 `BuffStateRegistry` 承担。 |
| `System/HistoryRuntime.cs` | `RecordActionHistory` | 只在 `ActionEffect` 确认后记录游戏动作；接收 request/effect 时间及 action id。 |

### 7.4 C# 职业层

| 文件 | 修改函数 | 计划 |
| --- | --- | --- |
| `Jobs/IJobStateMachine.cs` | `AdvanceTime`、`AdvanceTimeBeforeSystem`、`ResolveMpRecoveryAmount` | 删除三个按秒推进接口；新增 `RegisterTimeline(JobTimelineRegistry)` 和必要的职业事件处理入口。 |
| `Jobs/Black.Mage/BlackMageJobStateMachine.cs` | `Bind`、`AdvanceTime`、`ResolveMpRecoveryAmount` | `Bind` 后注册通晓与 MP modifier；删除职业时间循环和 elapsedSeconds 推算。 |
| `Jobs/Black.Mage/BlackMageApplications.cs` | `GainPolyglot` | 作为注册的 `JobPeriodicTick` 处理器复用，仍由黑魔模块拥有具体规则。 |
| `Jobs/Machinist/MachinistJobStateMachine.cs` | `Bind`、`AdvanceTimeBeforeSystem`、`AdvanceTime` | 注册连击、过热、野火计时器；删除手工时间推进。 |
| `Jobs/Machinist/MachinistApplications.cs` | `AdvanceCombo`、`RecordExpiringOverheatedStacks`、`AdvanceWildfire` | 拆成明确的到期/事件处理函数，不再接收任意 seconds 窗口。 |
| `Jobs/JobMachineRegistry.cs` | 构造/Bind 路径 | 确保每个职业时间注册一次，重复 owner/key 直接报错。 |

测试用职业同步修改：

- `Combat.Sim/FightEngine.Tests/Facade/TestJobStateMachine.cs`
- `Combat.Sim/FightEngine.Tests/Jobs/RegistryTestJobs.cs`

### 7.5 C# 门面与输出

| 文件 | 修改函数 | 计划 |
| --- | --- | --- |
| `Facade/JobSimulator.cs` | `_state`、`Step`、`Advance`、`SetScene`、`PreviewCandidates` | 改为持有状态与 `CombatTimelineRuntime`；新增本计划第 4.2 节 API；移除无时间戳生产入口和直接场景写入。 |
| `Facade/CombatStateMachine.cs` | `Step*`、`AdvanceTime*`、`ApplyActionTiming`、`ApplyActionEffect`、`BuildOutputAfterState` | 从有游标模拟器收缩为规则/事件分派器；动作效果只由时间线事件调用；删除普通/回放双路径。 |
| `Facade/CandidatePreviewBuilder.cs` | `BuildEntries` / 单候选预演 | 只枚举和预演 FightEngine 的 24 个真实动作；从同一 `SimulationSnapshot` fork，在当前时间真正 `SubmitAction`，再推进 fork 到约定的 candidate-after 观测点。第 25 个 wait 候选由 `PolicyContextBuilder` 增补。 |
| `Facade/ReplayStateCache.cs` | `CurrentState`、`AdvanceTime`、`Step`、快照列表 | 缓存 `SimulationSnapshot`；改为 `AdvanceTo` / `SubmitAction`，确保 pending events 一起克隆。 |
| `Facade/SequenceRunner.cs` | `Run`、`ResolvePreAdvance` | 输入序列必须携带绝对时间；删除根据 reason 猜测推进秒数。 |
| `Facade/RandomSequenceReplay.cs` | 主循环 | 外部回放策略根据当前快照和可选的 `GetNextScheduledEventTime` 自行选择下一调用时间，再显式提交时间戳；状态机不自动跳到决策点。 |
| `Outputs/StateContextBuilder.cs` | player/status/dot/cooldown remaining 构造 | 从当前时间与绝对 deadline 派生现有 remaining 输出。 |
| `Outputs/ContextBuilders/SkillHistoryContextBuilder.cs` | history 时间字段 | 明确继续使用 effect time，或在契约升级后增加 request/effect 区分；禁止含混复用。 |

候选 `state_after` 第一阶段保持当前模型契约：合法性在当前时间判断，after 状态仍表示约定的下一决策视图；区别仅在于它由真实 fork 时间线生成，而不是专用公式生成。

### 7.6 Sidecar 与 Python 后端

| 文件 | 修改函数/命令 | 计划 |
| --- | --- | --- |
| `Combat.Sim/SidecarHost/SidecarSession.cs` | 会话协议 | 持有 `JobSimulator` 和独立 policy session；只暴露 `init`、`advance_to`、`submit_action`、`record_policy_action`、`apply_external_event`、`observe_at`、`close`。删除 `advance_hidden`、`inject_scene`、旧 `step` 和 `history_mode`；`submit_action` 明确拒绝 policy token。snapshot/fork 只保留为 FightEngine 进程内能力，不跨 JSON 暴露。 |
| `scripts/common/cs_backend.py` | `advance`、`advance_hidden`、`inject_scene`、`step`、`validate` | 改为带绝对时间的 `advance_to`、`submit_action`、`record_policy_action`、`apply_external_event`、`observe_at`、`validate_at`；Python 镜像只能读取，不能先改字段再回写。 |

Sidecar JSON Lines 请求示例：

```json
{"op":"init","seq":1,"job_tag":"black_mage","initial_timestamp":-5.0000}
{"op":"submit_action","seq":12,"timestamp":37.2500,"action":"transpose"}
{"op":"advance_to","seq":13,"timestamp":39.7500}
{"op":"record_policy_action","seq":14,"timestamp":39.7500,"action":"ogcd_wait","next_observation_timestamp":42.5000}
{"op":"apply_external_event","seq":15,"timestamp":40.0000,"event_kind":"boss_targetable_changed","value":false}
{"op":"observe_at","seq":16,"timestamp":40.0000,"next_observation_timestamp":42.5000,"format":"vector"}
```

`init`、`submit_action`、`advance_to`、`record_policy_action` 和 `apply_external_event` 都只返回各自的轻量元数据，
不附带状态或模型上下文。`observe_at(format="vector")` 才返回按 canonical 七个顶层块组织的完整模型输入；
`candidate_skill_context` 与 `candidate_state_context` 均包含 24 个真实技能和 1 个 policy 候选。

### 7.7 FFLogs 转换

| 文件 | 修改函数 | 计划 |
| --- | --- | --- |
| `scripts/convert_fflogs/replay_timing.py` | `advance_state_to_time`、`resolve_bootstrap_hidden_elapsed`、`commit_replay_action`、`_reconcile_timing_for_annotation` | 只负责把日志观测转换为带类型的 request/effect 外部事件；删除隐藏时间修补和按 reason 补推进。 |
| `scripts/convert_fflogs/training_sync.py` | `sync_state_for_decision` | 删除本地 state 字段改写与 `inject_scene`；把场景窗口转换为带时间戳的 `SceneChanged` 事件。 |
| `scripts/convert_fflogs/training.py` | `_reconcile_logged_timing`、`_append_training_sample`、`_build_injected_ogcd_wait_action` | 删除 reconcile；保留并重命名 wait 样本构造逻辑，使其只生成 policy 样本并调用 `record_policy_action`，绝不调用 FightEngine validate/submit；真实动作通过 `ObserveAt(decisionTime)` 和 `SubmitAction(requestTime, action)` 推进。 |
| `scripts/convert_fflogs/scene_builders.py` | scene window 构造 | 增加从窗口边界生成外部时间线事件的函数，继续保留模型需要的 scene token。 |
| `scripts/convert_fflogs/cache_compile.py` / `cache_writer.py` | manifest 元数据 | 写入新的转换和时间语义版本，并记录 action timestamp source。 |

日志适配必须区分：

- 明确 cast start 的动作。
- 只有 action effect 时间的动作。
- 瞬发动作。
- 首个 prepull 动作。
- Hook/日志缺失导致只能推断 request time 的动作。

推断值必须在样本元数据中标记，不能静默伪装为权威时间。

### 7.8 自回归、PPG 与搜索调用方

| 文件 | 修改函数 | 计划 |
| --- | --- | --- |
| `scripts/autoregressive_replay/replay.py` | `OGCD_WAIT_ACTION_KEY`, `ReplaySnapshot.forbid_ogcd_after_wait`, `_read_ogcd_wait_boundary_pending`, `_generate_full_trajectory`, `_advance_submitted_action`, `_advance_event_time`, scene 同步函数 | 保留 wait 作为 policy 输出分支; 从 vector 历史读取 wait 边界, 真实技能通过 `submit_action` 提交, 时间推进统一委托 `DecisionScheduler`. |
| `scripts/autoregressive_replay/scheduler.py` | `DecisionScheduler` | 自回归、PPG 和 GRPO 共用的绝对时间推进、动作占用计算与 scene 边界同步。 |
| `scripts/autoregressive_replay/ppg.py` | `_run_rollout`、`_run_rollout_until_time`、`_advance_after_action` | 与普通自回归复用同一个时间调度 helper；不再自行取 `max(next_gcd_window, occupancy)`。 |
| `scripts/autoregressive_replay/context.py` | 后端上下文获取 | 改用 `observe_at`；不直接依赖可变 Python state 镜像作为时间真相。 |
| `grpo/trainer.py` | `forbid_ogcd_after_wait` 及 rollout step | 删除状态机 pending/forbid 特判；保留 wait policy action，选中时只记录策略决策并交给 scheduler；每条轨迹恢复完整 simulation + policy snapshot。 |
| `scripts/autoregressive_replay/outputs/markdown.py` | `SKILL_NAMES["ogcd_wait"]` | 保留 `-` 显示别名，但明确它是 policy control，不是已执行游戏技能。 |

如果仓库中的 C# PUCT 代码在实施时重新引入或位于其他工作区，则其模拟器接口也必须改为克隆完整时间线快照；不能只克隆 `CombatState`。

### 7.9 契约、配置和文档

| 文件 | 修改项 | 计划 |
| --- | --- | --- |
| `config/schema.yaml` | `contracts.sidecar_contract_version` | 删除旧命令后提升版本，拒绝旧 Sidecar。 |
| `config/system.yaml` | `skills.ogcd_wait` | 从状态机配置删除虚拟技能；保留真实系统动作 `potion`。 |
| `config/policy_actions.yaml` | 新文件：`policy_actions.ogcd_wait` | 独立定义模型控制 token 的 raw id、名称、候选特征和 policy behavior；该文件不由 `ProjectConfigLoader` 加载，但随部署配置一起发布。 |
| `common/policy/data/compiled_cache.py` | `CACHE_FORMAT`、`DEFAULT_CONVERSION_VERSION` | 时间/历史语义变化后提升，强制重编 raw JSON。 |
| `common/policy/data/input_contract.py` | `INPUT_CONTRACT_VERSION` | 候选仍为 25，但 wait 的所有权和历史来源变化，提升版本并明确 policy action contract。 |
| `common/policy/data/policy_actions.py` | 新文件：policy YAML 加载、真实技能冲突校验 | 为转换、训练、回放和 ONNX 导出提供同一 policy action 定义，不复制 wait 常量，保持 YAML 为单一配置来源。 |
| `common/policy/data/skill_vocab.py` | `build_from_config` / 新 policy 合并入口 | 真实技能从项目配置读取，policy token 从独立配置读取；继续保证 raw id 0 映射到非 PAD vocab id。 |
| `common/policy/data/normalization.py` | 技能定义枚举 | 合并 policy action 的固定特征，使 wait token 的归一化与 checkpoint/部署一致。 |
| `training/data/oversampling.py` | `_DEFAULT_IGNORED_ACTIONS` | 保留 `ogcd_wait` 默认忽略语义，但定义来源改为 policy registry，避免字符串规则漂移。 |
| `common/policy/model/repetition.py` | `_candidate_penalty_indices`、`build_repetition_penalty_mask` | 保留 wait 不参与真实动作重复惩罚的语义，但通过 policy action 元数据识别，不再硬编码字符串。 |
| `config/models/black_mage/artzip/candidate_order.yaml` | `candidate_order` | 保持 1..25 和现有相对顺序；第 1 项 `ogcd_wait` 改由 policy registry 验证。 |
| `config/models/black_mage/artzip/training.yaml` | oversampling 注释/候选相关配置 | 更新 wait 为 policy control 的说明，其他用户正在试验的模型参数不得被覆盖。模型架构与精度位于同目录 `model.yaml`，GRPO 参数位于 `grpo.yaml`。 |
| `common/models.py` | `StateSnapshot.ogcd_wait_boundary_pending`、clone | 删除 Sidecar 镜像字段。 |
| `scripts/onnx_export/contracts/deployment_contract.py` | `DEPLOYMENT_CONTRACT_VERSION` | 同步新的历史/决策时间语义。 |
| `scripts/onnx_export/manifest.schema.json` | contract 常量/schema | 同步 deployment contract。 |
| `scripts/onnx_export/profiles/black_mage.json` | vocab/candidate/profile action kind | 重新生成并继续保留 raw skill id 0 和 25 候选，同时标记它来自 policy action 配置。 |
| `AGENTS.md` | 项目结构和职责说明 | 新目录/文件落地后按首字母顺序更新。 |
| `README.md` | 调用示例、vocab 和状态机说明 | 实施完成后删除旧 `Step + Advance`、`history_mode` 和 wait 状态机语义；明确 `ogcd_wait` 仅存在于模型/调用方 policy contract。 |

是否修改 `CHANGELOG.md` 由实际实施时的发布安排决定；本计划本身不修改发布说明。

## 8. 测试修改计划

### 8.1 新增核心不变量测试

1. 单次推进与任意分段推进逐字段相等。
2. 过去时间戳被拒绝且状态/队列不变。
3. 同时间戳事件顺序固定。
4. 快照恢复后事件 id、sequence 和最终状态一致。
5. 候选 fork 不污染主时间线。
6. 非法动作不安排任何事件。
7. 状态刷新后旧 expire event 不会提前删除新状态。
8. 冷却主动缩短后旧 ready event 不会重复回充。

### 8.2 必须锁定的业务回归

- `transpose` 第一次使用后，在 5 秒以前第二次请求必须始终 `cooldown_locked`，无论中间推进被拆成多少段。
- 瞬发 GCD、读条 GCD、oGCD 的 request/effect/lock/GCD 时间顺序。
- AF/UI 激活期间每 30 秒获得通晓；离开元素态按规则重置；跨多次小推进与一次大推进等价。
- AF 下自然 MP tick 为 0；UI/中立和醒梦在精确 tick 时刻结算；Buff 与 MP tick 同时发生时顺序固定。
- DoT 多 tick、状态刷新、冷却多充能。
- 机工连击、过热和野火到期。
- Boss 在野火/DoT 生效同一时间切换可选中状态时的确定结果。
- FightEngine SkillBook 和游戏动作历史中不存在 `ogcd_wait`，但 policy 候选、SkillVocab、模型历史和输出仍包含它。
- 模型选择 wait 时不调用 FightEngine `SubmitAction`，不改变战斗状态、不安排事件，也不产生动画锁/冷却。
- 调用方记录 wait 后，在自己选择的 `t2` 显式调用 `AdvanceTo/ObserveAt`；状态机不会自行跳到下一刻。
- policy context 仍稳定输出 25 个候选，raw id 0 仍映射到非 PAD vocab id。

### 8.3 现有测试迁移

重点修改：

- `Combat.Sim/FightEngine.Tests/System/TimelineTests.cs`
- `Combat.Sim/FightEngine.Tests/System/PlayerStateRuntimeTests.cs`
- `Combat.Sim/FightEngine.Tests/System/SystemStateMachineTests.cs`
- `Combat.Sim/FightEngine.Tests/Facade/CombatStateMachineTests.cs`
- `Combat.Sim/FightEngine.Tests/Facade/JobSimulatorTests.cs`
- `Combat.Sim/FightEngine.Tests/Outputs/OutputsHistoryTests.cs`
- `Combat.Sim/FightEngine.Tests/Outputs/OutputsTokensTests.cs`
- `Combat.Sim/FightEngine.Tests/Jobs/Black.Mage/BlackMageJobStateMachineTests.cs`
- `Combat.Sim/FightEngine.Tests/Jobs/Machinist/MachinistJobStateMachineTests.cs`
- `tests/scripts/convert_fflogs/test_cs_backend.py`
- `tests/scripts/convert_fflogs/test_replay_timing.py`
- `tests/scripts/convert_fflogs/test_training_builder_*.py`
- `tests/scripts/autoregressive_replay/test_replay.py`
- `tests/scripts/autoregressive_replay/test_ppg.py`
- `tests/scripts/autoregressive_replay/test_parity.py`
- `tests/training/test_common_pt.py`
- `tests/training/test_common_vocab.py`
- `tests/grpo/test_grpo.py`
- `tests/training/test_models.py`
- `tests/training/test_training_optimizations.py`

原 Python 状态机删除后，不再维护 C#/Python 状态机 golden 双跑、导出器或 fixture；固定 seed 随机回放保留为进程内确定性测试。

## 9. 实施阶段

### 阶段 0：冻结基线

- 记录当前 C# 与 Python 测试结果。
- 保留本次 M5S 全量转换基线：142/195，成功率 72.8%。
- 为星灵移位冷却、通晓、MP tick、野火建立当前期望行为测试。

### 阶段 1：建立时间线内核

- 新增事件、优先级、队列、snapshot/fork。
- 实现 `AdvanceTo` 的单调性和分段等价测试。
- 将逻辑时钟 setter、队列写入和事件排空能力限制在 `CombatTimelineRuntime` 内部，并新增 `TimelineAuthorityTests` 阻止其他模块重新创建推进入口。
- 暂不迁移职业效果。

### 阶段 2：迁移公共时基资源（已实施）

- 迁移玩家锁、冷却、Buff、DoT、MP。
- 输出层改为从 deadline 派生 remaining。
- 删除这些 runtime 的 seconds 递减循环。

实施边界：

- `SystemStateMachine.CreateTimeline` 装配公共事件；阶段 2 曾由 `AdvanceTime` 暂作相对时间适配，该入口已在阶段 4 删除。
- 公共资源保存绝对截止时刻，remaining 只作派生输出；冷却查询不再按容差提前恢复充能。场景窗口按边界事件切换，DoT 在 tick 当刻判断可选中状态。
- 资源刷新与冷却缩短后，中心根据领域声明取消旧事件并续排；snapshot/fork 保留 deadline、事件和 sequence。
- MP 回调在阶段 2 暂时保留，随后已在阶段 3 改为职业注册的单次 tick modifier。
- 阶段 2 曾保留的 `Time` setter、旧 Sidecar 所依赖的相对时间入口和 wait 系统技能均已在阶段 4 的 FightEngine 边界删除；Sidecar 协议迁移归阶段 5。
- 阶段 2 曾保留的 `AdvanceTime` 窗口结算契约已在阶段 4 删除，DoT 结算只由时间线事件累积。


### 阶段 3：迁移职业注册时间资源（已实施）

- 黑魔注册通晓与 MP modifier。
- 机工注册连击、过热和野火。
- 删除 `IJobStateMachine` 的三个旧时间推进接口。

实施边界：

- `JobTimelineRegistry` 统一声明职业周期资源、倒计时和 MP tick modifier；事件由 `CombatTimelineRuntime` 排程、推进并在快照/fork 中保持确定性。
- 黑魔通晓在精确周期事件中结算，MP modifier 读取 tick 时刻的职业状态；机工连击与野火到期通过 `JobTimerExpired` 处理，野火结算读取到期时刻的目标可选中事实；过热残余层数由 `StatusTimelineRuntime` 的状态过期钩子在移除状态前一次性记录，避免跨域同刻事件取消导致统计丢失。
- `IJobStateMachine` 不再暴露 `AdvanceTime`、`AdvanceTimeBeforeSystem`、`ResolveMpRecoveryAmount` 三个按秒推进接口，职业只提供注册和单事件处理逻辑。

### 阶段 4：统一动作生命周期（已完成）

`JobSimulator` 只保留 `SubmitAction(timestamp, skillKey)`、`AdvanceTo`、`ObserveAt`、`ValidateActionAt`、`AvailableActionKeysAt`、外部事实提交以及 snapshot/fork 等绝对时间协议。动作接受、读条完成和效果分别通过时间线事件处理，动作历史在 `ActionEffect` 时刻记录并保留 request/cast/effect 时间与 `ActionInstanceId`；同一模拟器实例持续持有完整时间线。

FightEngine 内的旧 Step/AdvanceTime/StepResult、直接场景写入和职业计时器反向兼容写入已经删除；CombatState.Time 无公共 setter。阶段 5 已把转换、自回归、PPG 与 GRPO 全部切换到 Sidecar 绝对时间协议，并由共用 DecisionScheduler 负责动作占用与 scene 边界推进。

### 阶段 5：迁移 Sidecar 和调用方（已完成）

- Sidecar 已切换为绝对时间协议并把契约提升到 v6，只保留 `init`、`submit_action`、`advance_to`、`record_policy_action`、`apply_external_event`、`observe_at` 与 `close`；snapshot/fork 不跨进程暴露。
- 契约提升到 v7：新增 `validate_at` 只读探测；`observe_at` 响应补 `next_scheduled_event_time`（宿主调度器需要它决定下一次调用时刻，此前只在动作/推进响应里）。
- Python backend、FFLogs 转换、自回归、PPG 和 GRPO 均使用绝对时间请求、observe_at 和 typed external events。
- 自回归、PPG 和 GRPO 共用外部 DecisionScheduler；状态机不决定下一次模型调用时间。
- 删除 `advance_hidden`、`inject_scene`、`history_mode` 和调用方时间补偿。

#### 阶段 5 实施中确认的语义事实（务必遵守，否则时序会系统性漂移）

1. **请求时刻来源**：完整硬读条直接使用日志 `begincast`；瞬发及瞬发化动作使用 `cast`。
   只有开怪首个硬读条因 `begincast` 被战斗窗口裁掉，才使用
   `request = cast − 实际读条时长 + slidecast_window` 恢复预读请求。随后整场动作与场景事实
   统一平移，使最早请求为 0。转换层不写入某份日志实测出的 0.543 等常量，也不为了让状态机
   效果事件精确贴合 `cast` 而反推正常硬读条的几十毫秒偏移；该偏差由容量一动作队列吸收。
2. **`begincast` 不携带 `packetID`**：无法与 `cast` 直接配对。改为按 `(sourceID, abilityGameID)`
   分组后时序邻近消费——每个 cast 取走其之前最后一个尚未配对的 begincast。
3. **读条时长来源必须标记**：`begincast_duration`（日志直读）、`instant_skill`（技能表瞬发）、
   `prepull_estimated`（开怪预读，按实测 GCD 缩放技能表读条时长，可为负时间）、
   `instant_cast_inferred`（三连/迅速瞬发化推断）。推断值不得当作日志权威事实。
4. **按调用路径区分 `movement_changed` 的语义**：FFLogs 转换链路的 `SceneFactScheduler` 仍只向状态机
   注入三类内部结算事实（Boss 可选中、目标数、团辅窗口），移动状态由输出层按场景上下文改写，且保留
   滑步豁免；自回归回放、PPG 与 GRPO 共用的 `SceneTemplateProvider` 则向状态机注入四类事实，包含经过
   滑步豁免的 `movement_changed`，使请求时移动合法性由状态机判断，同时不取消已经开始的读条。
   这一路径口径取代 `fd4aab71` / 工作项 #156 记录的“所有路径均不注入”旧结论：转换样本仍需保持输出层
   移动字段语义，在线回放则必须把移动事实送入状态机才能闭合实时合法性与结算链路。
5. **停手窗口是开区间，且两端参照时刻不同**：起点用最后一击的**生效时刻**，终点用恢复后第一击的
   **请求时刻**（合法性校验发生在请求时刻，读条技能的请求时刻天然早于生效时刻）。边界再内缩一个
   容差覆盖端点动作被状态机推迟执行的最坏情况。
6. **游戏动作队列窗口约为 0.5s；本项目默认 `action_queue_window_seconds` 为 0.6s**，用于覆盖严重网络延迟下的请求排队，
   不是临界冷却容差。该值由转换器与状态机共同消费，也是日志时序与状态机时序允许的偏差上限；取值过小会把本可被排队吸收的偏差退化成
   "拒绝 + 调用方补偿 + 累积漂移"。该键由旧的 `cooldown_ready_tolerance_seconds` 改名而来，默认值随后从 0.5s 调整为 0.6s。
   旧记录曾记载 2026-09-17 成功 188/195（96.4%）；按当前本地 `action_queue_window_seconds=0.6s`
   口径重跑的结果为成功 190/195（97.4%），失败 5 个（2 个 `not_enough_mp`、2 个 `requires_polyglot`、
   1 个请求顺序超过 0.6s 队列窗口）。后者取代前者作为当前验收数字，旧数字仅保留为历史基线。
7. **输出层改写只覆盖状态机推不出的量**：移动位、下次停手 ETA、停手剩余秒数及其 GCD 版本。
   `next_untargetable_in_*` / `downtime_remaining_*` 描述的是**未来窗口**，而状态机只应知道已发生的事实
   （`NextDowntimeStartsAt` / `_downtimeDuration` 至今没有外部写入口）。
8. **游标必须跟进动作生效时刻**：下一个动作的资源校验发生在请求时刻，而前一个动作的效果只在生效
   时刻写入；不跟进会让紧跟其后的动作读到过期资源。这不会引入累积漂移，因为日志节拍本身也反映
   读条占用。


### 阶段 6：重建数据和验收

- 重编全量缓存。
- 重跑 M5S 195 个文件，逐原因比较原 53 个失败项；不得仅通过跳过校验提高成功率。
- 重跑训练/自回归/PPG，并检查动作时间、冷却和资源事件日志。
- 重建 ONNX/deployment contract。
- 使用新的 25 候选 policy contract 验证或重新训练 checkpoint；即使输出形状相同，旧 checkpoint 也必须经过语义版本检查，不能只按 shape 放行。
- 全仓搜索并确认旧 API 无残留。

各阶段可以分别提交，但最终主线不能长期保留两套动作执行内核。临时适配器只能调用新内核，不能复制旧逻辑。

## 10. 完成标准

- 所有生产动作提交都携带绝对时间戳。
- `CombatState.Time` 不再被 Sidecar/Python/插件直接设置。
- 全部生产代码只有 `CombatTimelineRuntime.AdvanceTo` 能移动逻辑时间并排空事件；`System/Timeline` 的其他文件和职业模块不存在 `AdvanceTime(seconds)` 或等价推进循环。
- 领域处理器仅处理一个已到期事件并返回 `TimelineMutation`；事件的新增、取消、重排和周期续排均由中心时间线应用。
- 全仓不存在生产用途的 `advance_hidden` 和 `history_mode`。
- FightEngine 运行时、系统配置和 Sidecar 状态中不存在 `end_weave_window`、`OgcdWaitBoundaryPending` 或 `ogcd_after_wait`，并拒绝把 `ogcd_wait` 提交为游戏动作。
- policy 配置、候选、SkillVocab、模型历史和输出仍包含 `ogcd_wait`，候选数保持 25。
- 自回归、PPG 和 GRPO 选中 wait 时只记录 policy decision，下一调用时间由显式宿主调度器产生。
- MP、通晓、冷却、Buff、DoT 和职业计时器均由公共时间线推进。
- 黑魔/机工不存在接收任意 seconds 的时间循环。
- 候选预演、真实执行、转换和自回归使用同一个 `SubmitAction + AdvanceTo` 内核。
- 快照包含事件队列，分支和恢复确定一致。
- 时间分段不影响最终状态。
- 星灵移位 5 秒冷却回归、通晓/MP tick 回归、M5S 全量转换和自回归长序列均通过。
- Sidecar、compiled cache、checkpoint 和部署端能够明确拒绝旧时间语义产物。

## 11. 明确不做

- 不在时间线重构中用魔法秒数修补某个日志样本。
- 不把 FFLogs 的 effect time 无条件当成玩家 request time。
- 不把 `ogcd_wait` 或任何 policy token 重新送入 FightEngine 的技能执行路径。
- 不让领域/职业模块实现自己的事件队列、时间循环或逻辑时钟写入；但它们继续拥有各自的效果与资源规则，中心时间线不得膨胀成业务单体。
- 不为转换、自回归和游戏插件保留三套不同的动作结算函数。
- 不在缺少游戏证据时模拟网络延迟、服务器 tick 抖动或随机伤害；第一版保持确定性逻辑时间。

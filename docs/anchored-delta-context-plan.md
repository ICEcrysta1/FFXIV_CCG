# 状态差分与上下文重置实施计划

记录日期：2026-10-07。状态：代码替换、全部验收断言与四轮训练对照已完成。新格式不保留旧归一化缓存或平铺缓存兼容路径；实测结果与证据见本文末尾。

## 1. 目标与已确认规则

模型输入继续使用以下因果序列，RoPE 继续按 token 位置编号：

```text
scene tokens, S1, A1, S2, A2, ..., SH, AH, S_current
```

本次改变状态与场景数值的表达方式，以及超出历史容量后的窗口选择方式。已确认规则如下：

| 项目 | 规则 |
| --- | --- |
| 完整状态缓存 | 同时保存整场原始绝对值 ABS 与原始差分 DELTA；两份均为 FP32，不预先归一化、截断或取 log |
| 首个可见状态 | 使用真实绝对状态；以其 `request_state.time_seconds` 为时间原点，当前请求时间归零 |
| 后续状态 | 所有数值字段相对上一状态的同名字段求差，包括布尔标志的 −1/0/+1 |
| 当前状态 | 有历史时相对最后一个历史状态求差；无历史时自己作为 ABS 锚点 |
| 技能 token | 保持现有技能 ID、数值特征与动作语义，不对不同技能的内容求差 |
| 超限处理 | 历史容量为 300 条动作；超限后保留最近 8 对状态／技能，即 16 个历史 token，再追加当前状态 |
| 场景开始、结束 | 第一条相对状态锚点；后续分别对上一条场景的开始、结束求差 |
| 场景时长 | 保留该窗口自身时长，不对上一条求差；跨锚点裁剪后重算剩余时长 |
| 跨锚点场景 | 只保留锚点之后的部分；完全结束的场景删除 |
| 时间尺度 | 状态两段 `time_seconds` 与场景开始／结束／时长使用 120 秒尺度；它不限制战斗时长，也不表示上下文只能覆盖 120 秒 |
| 位置编码 | 保留现有 token 位置 RoPE，本阶段不引入时间 RoPE |

输入投影、role embedding、统一输入 RMSNorm、Transformer 主干、输出头和优化器沿用当前实现。新增的 ABS／DELTA 与字段重置标识属于本次输入语义，不能通过现有 role 标识代替。

## 2. 当前数据边界

每个状态 token 继续包含两份快照，维度仍为 86：

```text
X_i = [P_i, Q_i]
P_i = previous_action_after
Q_i = request_state
```

两段分别对各自上一状态的同名字段求差，不把“上一动作后→当前请求”的 token 内差分当作相邻状态差分。无法取得上一动作后快照时，继续使用 `[request_state, request_state]`，不虚构初始 S0。

时间和其他字段按保存的 feature keys 定位，不硬编码列索引；两份 bank、mask 和输入契约使用同一字段顺序。

替换前的 compiled cache 保存完整历史，但状态、当前状态和场景在写入前已经归一化。部分规则包含 clipping，累计威力包含 `log1p`；从旧缓存做逆变换不能无损获得全部原始值。因此新格式从转换过程中尚未归一化的数值构建，不把旧归一化 bank 重新命名为 raw bank。

主要边界：

| 边界 | 当前模块 |
| --- | --- |
| 历史 bank 构建 | `scripts/convert_fflogs/training/history_bank.py` |
| 当前状态与场景样本编码 | `scripts/convert_fflogs/training/sample_builder.py` |
| compiled cache 格式、签名与读取 | `common/policy/data/compiled_cache.py`、`scripts/convert_fflogs/cache/` |
| 训练窗口与随机裁剪 | `training/data/dataset.py`、`training/data/collator.py` |
| compact history 的 GPU gather 与数值编码 | `common/policy/data/context_encoding.py` |
| 在线历史与模型输入 | `scripts/autoregressive_replay/context.py` |
| KV-cache | `common/policy/model/kv_cache.py` |
| ONNX 宿主输入与部署契约 | `scripts/autoregressive_replay/backends.py`、`scripts/onnx_export/` |

## 3. 完整 ABS／DELTA 缓存

### 3.1 保存结构

每个 source 只保存一份完整历史 bank，不为每个训练样本复制不同长度的窗口。实现采用以下 raw 字段：

| 字段 | 内容 |
| --- | --- |
| `state_abs_values` | 全部历史状态的原始绝对数值，FP32，形状 `[N, 86]` |
| `state_delta_values` | 同一顺序下的原始相邻差分，FP32，形状 `[N, 86]` |
| `state_null_mask` | 当前字段是否缺失，与数值分开保存 |
| `state_delta_reset_mask` | 当前已知、前驱未知时，该字段必须使用 ABS 的标记 |
| 当前状态 raw 数据 | 保存原始 ABS，以及相对该样本最后实际历史状态的 DELTA／重置标记 |
| 场景 raw 数据 | 保存完整绝对开始、结束、时长和类型专属原始内容；窗口相关差分留在读取侧 |
| 历史引用 | 保留 source 身份、完整历史结束位置、状态／技能配对身份 |

普通已知字段满足 `DELTA_i = ABS_i - ABS_(i-1)`。差分不能跨 source、padding sentinel 或错误的动作身份。首条状态没有真实前驱，标为 ABS；sentinel 的零值不充当真实前驱。

缺失字段采用以下规则：

| 当前字段 | 前驱字段 | 编码规则 |
| --- | --- | --- |
| 已知 | 已知 | 保存原始差分 |
| 已知 | 未知或不存在 | 保存 ABS，并设置字段重置标记 |
| 未知 | 任意 | 保留 null mask；数值占位不参与差分 |

数值占位、null mask、字段重置 mask 是不同语义。布尔值变化产生的 −1 是有效数值，不能当作缺失标记。字段恢复后的 ABS 重置必须传入模型，使模型能够区别该值与普通 DELTA。

若恢复的字段是状态 `time_seconds`，该字段的 ABS 重置先减当前窗口锚点 B，再按 120 秒尺度归一化；不能直接使用全局绝对时间除以 120。其他字段的 ABS 重置使用真实绝对值及该字段的 ABS 归一化规则。

### 3.2 缓存复用契约

新 cache 格式上线时重编译一次。之后，同一 raw source 的以下调整必须复用同一份完整双 bank：

- `history_capacity`，例如 300→512。
- 超限后保留数量，例如 8→16。
- 训练随机历史裁剪策略。
- 模型时间差分尺度，例如 120→60。
- 模型读取侧的 ABS／DELTA 与场景编码策略。

这些参数由模型／训练 YAML 和 checkpoint 输入契约保存，不进入 raw cache signature、编译 worker、raw source 后端或 bank 构建的窗口参数。新 raw 状态／场景字段固定 FP32，不能因模型训练精度切换而改为 BF16。

source 内容、字段顺序、状态转换语义或缓存结构改变仍应按相应契约失效。技能、监督及执行统计现有的格式校验继续保留，不以窗口解耦为由放松其他缓存检查。

## 4. 窗口选择与状态重锚

### 4.1 分段窗口

容量单位仍是历史动作条数。设完整已生效历史行数为 `H`，容量 `C=300`，重置保留 `K=8`：

```text
H <= C：保留 H 对历史
H >  C：保留 K + ((H - C - 1) % (C - K + 1)) 对历史
```

边界示例：

| 完整历史数 H | 模型保留历史对数 | 行为 |
| --- | --- | --- |
| 0 | 0 | 当前状态自己作为 ABS |
| 300 | 300 | 满容量 |
| 301 | 8 | 第一次超限，保留最新 8 对 |
| 302 | 9 | 在新窗口中继续增长 |
| 593 | 300 | 第二个窗口满容量 |
| 594 | 8 | 再次重置 |

在训练中，以 source 内裁剪前的完整 history cursor 计算，不使用合并 bank 后带偏移的索引。在在线回放中，以真实累计历史行游标计算，不使用已裁剪列表长度、模型预测次数或 GCD 数。

核对 C# 内部 `LastHistorySequence` 与实际输出历史行是否严格对应；如存在过滤或序号间隙，提供与 canonical 历史行一致的只读计数。重复观测、没有新增历史、一次新增多行、会话恢复均须保持同一窗口阶段。若增加桥接公开元数据，同步升级相应契约并重建 PythonBridge。

### 4.2 编码顺序

1. 根据完整 cursor 选择分段窗口。
2. 训练时执行现有随机早期历史裁剪，仍只保留连续后缀。
3. 以最终首个有效状态确定锚点，禁止保留一个没有绝对锚点的孤立 DELTA。
4. 首状态读取 ABS；后续状态读取对应缓存 DELTA／字段重置信息。
5. 当前状态按最后实际历史状态编码；无历史时改用自身 ABS。
6. 按同一锚点生成场景 view，然后完成归一化和精度转换。

若最终首状态是 `X_k=[P_k,Q_k]`，时间原点为 `B=Q_k.time_seconds`：

```text
首状态 request 时间            = 0
首状态 previous_action_after 时间 = P_k.time_seconds - B
其余字段                       = X_k 的真实绝对值
```

上一动作后时间可能为负，不能把两份时间都强制设为零。后续时间分别为 `P_i.time-P_(i-1).time` 和 `Q_i.time-Q_(i-1).time`；并非各状态相对 B 的累计时间。

锚点 `request_state.time_seconds` 必须已知且有限；缺失或非有限值按入口错误语义处理，不能把数值占位当作真实时间原点。

示例：MP 为 `10000, 9200, 7600, 8000`，缓存差分为 `ABS, -800, -1600, +400`；窗口从第三条开始时，模型读取 `7600, +400`。

首状态的 ABS／DELTA 标识，以及后续字段级重置标识，需要显式进入输入编码。具体标识投影应保持模块职责清晰，并纳入 checkpoint、ONNX 和公共参数初始化一致性检查。

## 5. 场景裁剪与时间差分

场景完整绝对数据继续用于状态机事件、Boss 可选中判断、移动、Buff 和 PPG 结束条件；以下处理只生成模型可见 view。

对时间原点 B：

1. 删除 `end <= B` 的窗口。
2. 跨锚点窗口的开始裁为 B，结束保留；重算剩余时长。
3. 开始、结束都减去 B。
4. 按裁剪后的开始时间稳定排序，同开始时使用明确、可复现的结束／类型／原始行号顺序。
5. 首条场景的开始、结束使用相对 B 的 offset；后续开始、结束分别对排序后的上一条求差。
6. duration 保留自身裁剪后时长，不差分。类型与类型专属字段保留 ABS，不混减不同类型的同列值。

以 B=1200 秒为例，表内数值为归一化前秒数：

| 原始区间 | 裁剪并移动原点后 | 编码 start／end／duration |
| --- | --- | --- |
| 1198–1205 | 0–5 | 0／5／5 |
| 1203–1204 | 3–4 | +3／−1／1 |
| 1220–1240 | 20–40 | +17／+36／20 |

开始时间排序后，结束时间差仍可能为负。第一条若在锚点之后才开始，保留相应正 offset，不强制设零。远期窗口、长 duration 和场景差值可以超过 120 秒；除以 120 后允许超出 ±1，不因此丢弃场景或截断时间。场景 view 为空时使用真实空场景输入。

## 6. 归一化与 GPU 路径

### 6.1 数值处理

先取得 raw ABS／DELTA，在 FP32 中完成重锚、场景裁剪和必要的相邻相减，再归一化，最后进入 BF16／FP16 模型计算。状态两段 `time_seconds`、场景 start／end／duration 按保存的 120 秒尺度处理。GCD 长度、Buff／DoT 剩余时间、`polyglot_timer` 等其他字段仍对原始数值求差，但保留各自在现有归一化契约中的尺度，不统一改成除以 120；负 DELTA 不经过旧 clipping。

ABS 与 DELTA 的规则分别定义。DELTA 的线性缩放允许负值，布尔差分保持 −1/0/+1；不得把 DELTA 送入现有 `clamp_min(0)` 或非负 log 规则。累计威力先取得原始威力增量，再使用明确保存的 signed 变换，例如 `sign(x)*log1p(abs(x))`。这与 `log1p(ABS_i)-log1p(ABS_(i-1))` 不同；经过 signed log 后也不能直接累加编码值还原原始威力。

时间差分尺度属于模型编码配置，不通过修改原始缓存的旧 `fight_time_max` 来实现。所有编码规则及尺度随 checkpoint 保存，回放和部署从保存契约恢复。

### 6.2 批量处理

复用现有 compact history 路径：按 source bank ID 去重，由 GPU 批量 gather 选中窗口。后续行读取 DELTA；首状态和发生字段恢复的位置读取 ABS，通过 mask 选择。避免为每个 batch 对完整历史做 `cumsum`，也不通过 Python 逐 token／逐字段恢复状态。

场景裁剪、稳定排序、压实、首条选择和差分使用批量 Tensor 运算。训练、普通回放、分析、GRPO 与 ONNX 输入准备共享数值定义；CPU 和 GPU 实现必须对同一 fixture 输出一致结果。

GPU 编码只处理选中窗口和有限场景 view。完整双 bank 的磁盘体积、CPU 内存和传输成本增加，需要同时测量。优先避免传输未使用的 ABS 行；若完整 source bank 搬运成本明显，再比较 CPU 侧批量收集所需行和 GPU bank 复用方案，不复制第二套业务编码规则。

batch=20、历史=300、状态=86 时，单份 FP32 状态窗口约 2 MiB；这是输入张量体积，不代表训练总显存。性能验收统计完整训练步吞吐、数据等待、CPU→GPU 传输、输入编码耗时、峰值显存及 CPU bank 常驻量。使用固定 batch 的预热和重复测量，不把小规模算子的理论成本当作已验证提速。

## 7. 模块职责与部署一致性

| 模块 | 实现职责 |
| --- | --- |
| `common/policy/config.py`、模型／训练 YAML | 保存状态与场景编码、分段窗口、保留数量、时间尺度；校验 `0 < K <= C` 和有限正尺度 |
| `scripts/convert_fflogs/training/` | 在归一化前构建完整 raw 双 bank、当前状态数据和完整 raw 场景；不读取模型窗口 |
| `common/policy/data/compiled_cache.py` 与 cache 编排 | 新格式及 raw 字段校验、旧格式重编译、完整 bank 的共享读取 |
| `common/policy/data/history_window.py` | 纯窗口规划器：完整 cursor→起点／长度；不负责转换、模拟或 PPG |
| `training/data/dataset.py`、`collator.py` | 选窗、连续后缀随机裁剪、bank 去重和 batch；保留静态质量监督校验的现有职责 |
| `common/policy/data/context_encoding.py` | 在模型外完成 compact GPU gather、FP32 ABS／DELTA 选择与重锚、场景 view、有符号归一化和标识；训练、回放和部署宿主共用 |
| `common/policy/model/input_encoder.py` | 只消费已编码内容及 null／ABS reset 标识，最后转换到激活精度；不恢复 raw 或 compact 兼容分支 |
| `scripts/autoregressive_replay/context.py` | 维护绝对状态与完整场景事实；使用真实 cursor 选窗，随锚点生成模型 view |
| `common/policy/model/kv_cache.py` | 场景 view、窗口起点或首状态表达变化时，重建保留前缀 |
| ONNX 适配、部署契约与 parity | 宿主先 FP32 变换，再转目标精度；图内不重复做差，参考路径使用同一输入语义 |
| 分析、历史消融与 GRPO | 复用同一输入构建；保存重现窗口所需的 cursor／阶段和新编码契约 |

共享 helper 位于 `common/policy/data/`，不放到无业务归属的根目录公共函数中。模型输入 view 的裁剪不能修改模拟时钟、MP、CD、场景调度、伤害累计或 GCD 统计；`val_ppg` 继续运行完整验证 source。

重置后，保留的 token、场景前缀、位置编号和首状态内容发生变化，因此不能只删旧 K/V 再追加。KV-cache 必须按新的完整可见前缀重建。绝对状态／双 bank 数据缓存与 Transformer KV-cache 分属不同层次。

实施时升级 cache 格式／转换标识、checkpoint 输入、ONNX deployment／manifest 和受影响的 GRPO rollout 语义版本，明确拒绝不兼容产物。本计划不预编未来版本号。纯 raw 缓存与读取侧变化不改变 C# 状态机；若新增桥接 cursor 元数据或变更公开 schema，另按桥接契约要求同步版本并重建程序集。

## 8. 实施阶段与检查点

- [x] 阶段 1：建立字段清单、前驱身份和 raw 双 bank 格式；完成缓存回归。旧格式一次性重编译后，不同窗口配置复用同一缓存。
- [x] 阶段 2：实现纯窗口规划及 ABS／DELTA、缺失恢复、场景编码参考路径；先验证正确性，不启动训练。
- [x] 阶段 3：接入训练 compact GPU 路径，检查 FP32 保留、批量处理和公共参数初始化；完成固定长度性能测量，整轮吞吐待对照训练。
- [x] 阶段 4：接入回放、KV-cache、分析、GRPO 和 ONNX；升级相关契约并完成一致性验收。
- [x] 阶段 5：按本次实施授权，以冻结数据与种子完成四轮对照；根据实测评估整套方案。

最初落盘仅创建计划；随后按用户授权开始代码替换。保留用户的训练配置修改，不修改 CHANGELOG，不提交或推送。

## 9. 必须覆盖的验证

### 数据与缓存

- [x] 同一 source 的 ABS／DELTA、技能和前驱身份一一对应；不跨 source 或 sentinel 求差。
- [x] 已知字段在 FP32 容差内满足 ABS 与 DELTA 重建一致；MP 增减、资源、计时器、布尔 −1/0/+1、Buff 刷新／过期、累计威力都覆盖。
- [x] null→known 使用字段 ABS reset，恢复的 `time_seconds` 仍相对窗口 B；known→null 保持缺失，缺失占位不参加减法。
- [x] raw 数值写入前没有 log、clamp 或 BF16；未知格式、字段宽度或 dtype 自动拒绝。
- [x] 调整容量、保留数、随机裁剪和时间尺度不启动转换、不改变 raw cache signature；转换语义改变时仍正确失效。

### 窗口与场景

- [x] 0、1、299、300、301、302、593、594 条历史及连续多次重置。
- [x] 重复观测、零新增、多行新增、恢复会话和历史消融能重现同一窗口。
- [x] 超限保留 8 个 S 和 8 个 A，另有当前 S；随机裁剪后重新建立正确 ABS 锚点。
- [x] 两段状态分别差分；首请求时间为 0，上一动作后时间偏移可负。
- [x] 场景跨锚点裁剪、完全结束删除、未来第一条、负 end delta、长时间／长 duration、混合类型、同时间稳定排序与空 view。

### 训练、回放与部署

- [x] 同一原始 fixture 的训练／回放／ONNX 宿主编码一致；确认图内没有重复差分。
- [x] 使用会在 BF16 绝对时间中合并的小增量案例，验证 FP32 差分发生在精度转换之前。
- [x] 普通残差与 Full AttnRes 使用同一输入编码；重置后的 KV-cache 与完整 forward 一致。
- [x] 全副本 PPG 时长、累计威力和 GCD 分母不因窗口重置而归零或截短。
- [x] 现有质量监督、固定动作顺序、合法性 mask、标签和动作价值保持原语义。
- [x] CUDA 热路径没有逐样本 Tensor→Python 同步、CPU 回传或完整 bank 的每步前缀累加。

## 10. 对照实验计划

现有基线目录为 `artifacts/experiments/baseline_4x256x4x1024_20261007/`。比较使用其中已完整保存的第四轮 checkpoint；日志可能含停止前的第五轮未保存 batch，不能纳入四轮比较。

| 对照项 | 固定条件 |
| --- | --- |
| 模型 | 4 层、d_model=256、4 个 Q head、1 个 KV head、ff_dim=1024 |
| 数据 | 冻结的 100 份训练 source、30 份 VAL；按原始 source 身份和指纹配对，不因新 cache 格式重新随机选数据 |
| 随机性 | seed=42；对公共参数逐名称／形状校验初始化相同；不能用已训练基线权重作为实验初始化 |
| 训练 | batch=20、BF16、现有 Muon／AdamW 分工及其他 loss 配置保持 |
| 日程 | 比较完整前 4 轮；仍使用基线原始 8 轮 LR 总预算，不改成 4 轮调度 |
| 配置来源 | 基线 manifest／config snapshot；保存每组明确的差异及 source 指纹 |

建议分两组实施，另加性能验证：

1. **新缓存兼容基线**：从 raw ABS 生成旧输入编码，核对数值、监督、公共权重和前向输出；若偏差超出合理 FP32 变换容差，补跑同日程基线，不能直接归因于差分方案。
2. **完整已确认方案**：raw ABS／DELTA、全数值状态差分、场景时间差分、300→8 分段重置和 120 秒尺度一并启用。该组评估整套方案；若需判断某个组件的独立作用，再安排拆分对照。
3. **吞吐与缓存复用验证**：同 batch、同有效长度比较数据加载和训练耗时；额外用不同容量／保留数／尺度读取同一份 cache，确认没有重编译。

现有第四轮基线：Top-1 84.0606%，CE 0.514521，`val_ppg` 564.8883，`none_ppg` 559.8281。最终同时比较 CE／Top-1／Top-3、自回归 PPG、低质量循环或恢复行为、分层 std／norm 和性能；保留相同的分析样本与统计位置。

预期收益是更直接的变化信息和较好的小时间增量精度。所有资源和布尔状态改成 DELTA 后，模型需要从 ABS 锚点恢复当前信息；在长窗口中是否容易学习、是否改善隐藏层 std，均以对照结果为准，不作为方案已成立的前提。

## 11. 实施与验收记录

新实现使用唯一的模型外 `ContextEncoder`，神经模型不再支持 raw/compact 自动转换。缓存 v22／转换 v24 保存完整 raw 双 bank，拒绝旧平铺和归一化格式；checkpoint 输入 21、canonical 14、桥接 16、部署 24／manifest 15、GRPO rollout 5 已同步。新增 reset 投影为零初始化，不消耗随机数；4×256 普通残差和 Full AttnRes 的公共初始化均保持一致。

主要回归结果：训练测试整组 972 通过，相关回放／GRPO／ONNX／分析整组 407 通过、3 跳过，C# 292 通过；此后补充的资源／计时器／Buff、场景排序、三方输入与 CPU/CUDA 编码回归均通过。独立审计发现的合法零目标场景误拒绝与 compact 分析技能身份缺失均已修复并补测试。最终 PythonBridge Debug 构建成功，实际程序集内嵌契约为 16。

已验证容量、保留数量、时间尺度和随机裁剪组合复用同一实际缓存，文件字节、时间戳及 signature 不变，测试拦截转换与引擎创建入口以确认没有重编译。CPU/CUDA dense 与 compact 编码逐字段一致，CUDA 编码中禁止 Tensor 标量读取和 CPU 回传的测试通过；分段重锚及场景 view 变化后的 KV 与完整 forward 在两条残差路径一致。

固定长度测试使用 RTX 4070、batch20、历史300、86维状态、4×256、BF16，同样有效 token 数。训练步中位数（含 H2D，不含文件 I/O 和 PPG）为旧路径 74.602 ms、新路径 78.350 ms，约增加 5%；峰值已分配显存 900.17→901.90 MiB。完整 bank 额外搬运约 0.135 ms，当前不新增 GPU bank 缓存层。该结果不能代替整轮训练吞吐。详见 [性能报告](../artifacts/experiments/anchored_delta_validation_20261007/PERFORMANCE_REPORT.md)。

冻结首 source 的全部 854 个样本，在 v22 raw ABS 经隔离旧编码后，旧状态、技能、场景、监督及窗口字段逐值相同；同初始化旧 BF16 forward logits 误差为 0。冻结的 100 份训练 source 与 30 份 VAL 原文件 SHA256 均已核对一致；130 份新实验缓存全部编译成功，写入独立目录，原基线保留。所有旧实现仅在隔离验收 artifact 使用，不进入生产代码。

真实 86 维 schema 的独立 CPU ONNX 导出、checker、shape、固定容量 golden/padding 和 ORT parity 已通过：12 个输入，reset 投影设置非零验证其实际消费；真实编码样本 logits 最大绝对误差约 3.6e-7，Top-1/Top-3 一致。详见 [ONNX 结果](../artifacts/experiments/anchored_delta_onnx_cpu_20261007/result.json)。

同一真实 C# canonical 的六步动作和完整场景，经训练 compact 缓存、在线 LiveBatchBuilder 与正式 ONNX 宿主构建后，12 个神经输入逐张量一致；覆盖 C4/K2 的首次窗口重置。FP32、FP16、BF16 的宿主目标精度转换均已验证，不发生第二次差分。

四轮对照冻结源码、程序集、130 份语料与新缓存指纹，正式读取样本数为训练 56,824／VAL 17,165，每轮 2,896 个训练步，原八轮 LR 总预算 23,168 步；公共参数 FP32/BF16 及 Python／NumPy／Torch 初始 RNG 全部与基线匹配。实验保留原语料顺序、batch20、BF16 和 seed42，完成 11,584 次优化后在第五轮训练入口迭代 DataLoader 之前结束，第五轮 batch 数为 0。正式第四轮 checkpoint 字节复制为 final，不从已训练基线热启动。

### 四轮结果

| 第四轮指标 | 冻结基线 | 新方案 |
| --- | ---: | ---: |
| VAL CE | 0.514521 | 0.559994 |
| VAL Top-1 | 84.0606% | 82.9945% |
| VAL Top-3 | 97.7571% | 97.2560% |
| 完整 30 副本 `val_ppg` | 564.8883 | 561.1060 |
| `none_ppg` | 559.8281 | 562.9766 |

本次四轮没有显示整体质量收益：VAL CE 增加约 8.84%，Top-1 降低约 1.066 个百分点，完整副本 PPG 下降约 0.67%，空场景 PPG 增加约 0.56%。这是整套方案、单种子和四轮日程的结果，不能据此将变化归因于某个差分字段或窗口组件。

原全 token 图的末层 std 为 3.169→2.593，下降约 18.17%；但场景占比同时从 52.545% 变为 32%，20,000 token 上限也导致不同样本权重。额外直接读取冻结 v21 输入，对同一分析 source 的前 256 个当前状态、相同 BF16 精度与相同层输出位置进行配对：

| 层 | 基线 current-state std | 新方案 current-state std |
| --- | ---: | ---: |
| 1 | 1.3537 | 1.4282 |
| 2 | 1.7712 | 1.8357 |
| 3 | 2.1885 | 2.2143 |
| 4 | 2.5485 | 2.4570 |

同位置末层 std 仅下降约 3.59%，前三层略高，随深度增长的趋势没有反转。末层 L2 中位数为 40.775→38.178，但 P95 为 46.628→47.318、P99 为 52.377→54.931，尾部没有同步改善。前 256 个样本的 cursor 为 0–255，不覆盖超出 300 后的重置阶段；该阶段正确性由窗口／KV 回归验证，行为质量由完整副本指标覆盖。

含正式回放和原分析的运行总时长约 2,434 秒，进程峰值工作集约 2.50 GiB；这些时长受到模型动作、输入长度分布和运行环境影响，不替代固定长度微基准。实现和功能验收已完成，本次结果不支持宣称改善模型质量或解决隐藏层 std 增长。详见 [完整对照报告](../artifacts/experiments/anchored_delta_validation_20261007/comparison_report.md) 与 [同位置配对统计](../artifacts/experiments/anchored_delta_validation_20261007/paired_current_hidden.json)。

结束后只读统计的 130 份缓存、73,989 行 bank：静态 bank 张量总量为 38.03→69.85 MB（1.837 倍），完整缓存磁盘体积为 396.98→503.16 MB（1.267 倍）。这是数据量，不等于训练进程工作集；新增 ABS／DELTA 与字段标记的体积代价已实测。旧基线文件及全部旧缓存 manifest／shard 指纹保持原样，final 与第四轮 checkpoint SHA256 相同，冻结的 323 个源码／程序集和 22 份 YAML 均无漂移。详见 [完成核验](../artifacts/experiments/anchored_delta_validation_20261007/completion_verification.json) 与 [缓存存储统计](../artifacts/experiments/anchored_delta_validation_20261007/cache_storage_summary.json)。

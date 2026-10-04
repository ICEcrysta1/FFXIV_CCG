# 完全因果策略模型修改计划

记录日期：2026-10-04。状态：阶段 1、2 已完成；阶段 3 的状态语义已实现并通过单元与导出验证，但真实日志 compiled 历史输入验收发现错行，尚待修复；实施分支为 `ice/codex/causal-policy-stage1`；阶段 4～6 尚未实施。

本计划将现有候选评分模型改为只读取场景、历史状态和历史技能的因果策略模型。按用户最新决定，实施顺序固定为：彻底移除候选、拆分 embedding、调整状态语义、由用户确认技能和状态字段并完成字段删减、重建历史上下文并保留最多 300 组“状态＋技能”（600 个历史 token），再追加最新状态，非场景容量合计 601，最后由用户执行 100 份训练数据的小规模训练。

各实现阶段包含代码、测试与相匹配的产物契约升级，不自行启动训练实验、提交或推送，也不自动修改 CHANGELOG。阶段 3 已完成状态语义改造，技能字段与 384 条历史动作的容量单位保持阶段 2 基线；当前仍采用技能在前、状态在后的过渡布局，最终状态在前的布局与 601 个非场景 token 容量留在阶段 5。

## 目标契约

最终输入为一条连续的因果序列。这里的 `A` 表示历史技能或保留的 policy 控制动作，`S` 表示状态 token：

```text
scene tokens, S1, A1, S2, A2, ..., SH, AH, S_current
                                               |
                                 从 S_current 的 hidden 预测下一动作
```

每个状态 token 内包含两个状态快照：

| 快照 | 最终语义 |
| --- | --- |
| 上一步技能后状态 | 前一步动作已经发生、且在本次请求时可取得的动作后快照 |
| 当前请求时状态 | 本次请求选择动作前的真实状态 |

不能取得前一步动作后快照时，两段都使用当前请求状态，逐字段完全相同。无前序动作的首个状态适用这个规则。历史状态与末尾最新状态使用同一套字段、归一化、embedding 和状态类型标识。

技能输入和输出必须共享同一张可学习的技能语义 embedding 表 `E`。输入使用哪个技能向量，输出匹配该技能时就使用同一行、同一参数的向量；这是已确认的模型要求。

历史部分的“600”表示最多保留 300 组“状态＋技能”，每组占两个独立 token。末尾再追加最新状态 `S_current`，技能与状态的非场景总容量为 601，场景 token 另计。最终契约中的 `model.history_capacity: 601` 按包含最新状态的非场景 token 容量计量；当前实现以历史动作条数计量的旧含义必须显式迁移。

```text
历史状态与技能 token 容量 = 600 = 2 * 300
包含最新状态的非场景 token 容量 C = history_capacity = 601
历史动作数 H <= floor((C - 1) / 2) = 300
状态 token 数 = H + 1
技能 token 数 = H
有效技能与状态 token 数 = 2 * H + 1 <= C
总物理容量 = scene_capacity + history_capacity
```

完整交错序列始终以状态开始、以最新状态结束，长度为 `2H + 1`。满窗口保留 300 个历史状态与 300 个历史技能，再追加一个最新状态，共 601 个有效 token，无需为奇偶差额预留 padding。历史不足时按实际长度填充并屏蔽 padding。保留当前 `scene_capacity: 200` 时，最终最大物理容量为 `200 + 601 = 801`。有效场景数为 `s`、有效历史动作数为 `h` 时，有效序列长度为 `s + 2h + 1`。显存和耗时以最终实现的实测结果为准。

## 当前源码基线

以下记录阶段 1 实施前的源码基线，便于对照迁移原因。阶段 1 的实际交付另行记录，不能按此表描述新实现。

| 边界 | 实施前实现 | 主要位置 |
| --- | --- | --- |
| Transformer 输入 | `scene + history pair + candidate pair`；每条历史只有一个融合 token | [输入编码器](../common/policy/model/input_encoder.py) |
| embedding | 技能 ID embedding 与技能数值投影相加，再与状态投影经过 `pair_fusion`，最后进入共享 token 投影 | [输入编码器](../common/policy/model/input_encoder.py) |
| 注意力 | 场景和历史前缀因果；候选块可以读取整个前缀并在块内双向读取 | 原 `common/policy/model/split_encoder.py`（阶段 1 删除） |
| 输出 | 每个候选 hidden 经共享 scorer 输出一个分数 | 原 `common/policy/model/candidate_scorer.py`（阶段 1 删除） |
| 历史状态 | `before` 来自本技能请求快照，`after` 来自本技能生效后快照 | [真实动作历史记录](../Combat.Sim/FightEngine/Facade/CombatStateMachine.cs) |
| 历史记录时间 | 真实动作在 ActionEffect 时记录，历史技能 `time_seconds` 使用生效时刻；另有请求、读条结束、动作实例 ID | [动作历史定义](../Combat.Sim/FightEngine/Models/Combat/ActionHistoryEntry.cs) |
| 当前请求状态 | canonical 输出尚无独立最新状态 token，相关向量包含在候选状态的 before 中 | [canonical 输出](../Combat.Sim/FightEngine/Outputs/OutputContextBuilder.cs) |
| 技能字段发现 | 同时扫描候选技能和历史技能，推导数值字段名 | [技能字段提取](../scripts/convert_fflogs/source/source_helpers.py) |
| 验证 PPG 初始化 | 从首个样本的合法 GCD 候选技能特征恢复基础 GCD | [PPG](../scripts/autoregressive_replay/ppg.py) |
| policy wait | 不提交真实游戏技能；当前 policy 历史 after 通过分支推进到下一观测时刻获得 | [policy 历史](../Combat.Sim/FightEngine/Policy/PolicyDecisionHistory.cs) |
| 历史容量 | 当前 `history_capacity: 384` 以历史动作条数计量；compiled cache 固定保存完整历史，窗口只在读取侧生效 | [模型 YAML](../config/models/black_mage/artzip/model.yaml)、[compiled cache](../common/policy/data/compiled_cache.py) |

当前主干为 `d_model: 768`、12 层、12 个 Q head、1 个 KV head、`ff_dim: 3072`、Pre-LN、SwiGLU，Full AttnRes 关闭。本次模型迁移不同时调整这些参数或优化器算法。

## 阶段顺序与检查点

| 阶段 | 修改内容 | 结束条件 |
| --- | --- | --- |
| 1 | 完全移除候选 token 及其生产、缓存、模型和消费路径 | 所有正式入口都不再依赖候选；完成残留清单与测试 |
| 2 | 技能和状态分别 embedding，技能输入输出共享词向量 | 删除 pair fusion，共享参数与独立编码可验证，历史技能字段保持原有语义 |
| 3 | 状态改为上一动作后状态与本次请求状态 | 时间锚点、回退、policy wait 和跨步配对通过回归测试 |
| 4 | 向用户提交技能和状态字段对照，确认后完成字段删减 | 用户确认最终字段、维度、顺序、归一化和待生效动作的处理规则；技能 `time_seconds` 完全移除，相关消费者迁移完成 |
| 5 | 重建交错历史，历史状态与技能占 600 token，追加最新状态后共 601，RoPE 顺序编号 | 数据、模型、KV-cache、分析、ONNX 和回放遵守同一契约 |
| 6 | 100 份训练数据的小规模训练 | 用户本人运行并检查训练与自回归结果 |

每阶段只执行对应范围，不以候选空数组、伪候选、保留旧入口但不调用等方式宣布阶段 1 完成。阶段 3 先按已确认的两段状态语义实施，技能字段暂沿用阶段 2。阶段 4 是用户明确要求的字段确认点；字段确认和相应修改完成后才实施阶段 5。

## 阶段 1 完全移除候选

### 统一技能词表与下一技能预测

这一节采用 GPT-like 的离散技能 token 与下一技能预测方法：把每个真实技能和 policy 控制动作视为词表中的一个 token，由可学习的 embedding 表取得技能向量，因果 Transformer 处理上下文，再从最新状态 hidden 通过 LM Head 输出下一技能的词表概率。

“统一动作输出空间”描述的是输出技能词表和它的索引契约；它本身没有说明 embedding 或输出头结构。最终模型明确使用技能词表 embedding 和 LM Head，不再使用逐候选 Transformer hidden 的 scorer。

```text
历史技能 ID -> SkillVocab -> 查技能 embedding 表 E -> 技能 token
场景和状态 -> 各自数值投影                         -> 场景和状态 token
交错因果上下文 -> Transformer -> 最新状态 hidden h -> LM Head -> 下一技能概率
```

移除候选 token 后，建立稳定的输出技能词表契约：

- `action_keys` 定义所有启用的真实技能与注册的 policy 控制动作，按配置和明确的稳定规则构建。
- `num_actions` 定义 logits 宽度，`label_index` 映射到固定的动作输出索引。
- 技能输入词表与输出技能词表表达同一套技能身份；输出只包含允许输出的动作，使用显式 `action_to_vocab_id` 对齐输入 embedding 行与输出索引。输入的 padding、未注册或不可输出项不能成为输出动作。
- `ogcd_wait` 的原始 ID 为 0，但它在技能词表中有独立的非 padding ID；不能按原始 ID 直接删除。
- 动作顺序、输入 embedding 行映射和 `action_is_gcd` 类型标记固化在输入契约、checkpoint、回放和 ONNX 部署契约中，不再从候选 token 或候选顺序文件获取。离线恢复只读取保存的契约，不从本机当前 YAML 重建动作类型；回放保留 GCD/oGCD 阶段筛选，叠加在状态机提交合法性 mask 上。

输出头采用共享技能 embedding 的 LM Head，读取最新状态 hidden `h`，与各输出技能在 `E` 中的对应向量做点积，一次输出 `num_actions` 个 logits，再由 softmax 表达下一技能概率。这里的“每个技能一个 logit”就是词表预测，不需要把技能列表作为候选 token 输入 Transformer。交叉熵、Top-1、Top-3 和价值偏好损失同步改为输出技能词表索引，保持各指标现有的计算口径。

技能 ID embedding 与输入数值特征是两部分。技能 token 保留 `E[skill_vocab_id] + skill_feature_projection(features)` 的输入方式，之后独立归一化；不因改成 GPT-like 结构而丢弃现有技能字段。状态和场景继续使用数值投影形成 embedding，只有下一技能是离散词表预测。

输入输出共享权重是固定要求，按同一技能语义向量参与输入和输出匹配实现：

```text
输入技能语义向量 e_i = E[skill_vocab_id_i]
输出技能 i 的 logit = dot(h_last, e_i)
整张输出词表 logits = h_last @ E[action_to_vocab_id].T
```

`E` 是唯一的技能语义参数表。输出读取对应行时保留到 `E` 的梯度，不能创建独立 `W_out`、复制一份可训练权重或只让两份参数拥有相同初始值。技能数值特征仍是输入附加信息；共享的是基础技能语义向量 `e_i`，不要求输出重新构造数值特征、状态、位置或整个 Transformer 输入向量。[GPT-2 官方实现](https://github.com/openai/gpt-2/blob/master/src/model.py)使用同一个 `wte` 进行输入查表与输出词表投影，可作为共享方式的参考。

输入查表与输出匹配的梯度汇总到同一参数。优化器按参数身份只登记和更新一次 `E`，低精度训练也只维护一份对应的 FP32 主权重。沿用现有语义分组，技能 embedding 继续属于 AdamW，不因为同时承担输出投影而扩大 Muon 范围。checkpoint 恢复、设备或精度转换后都必须保留共享关系。

`action_legal_mask` 是执行和已有损失所需的独立动作合法性数据，不是 Transformer token，也不包含动作执行后的状态或技能预演字段。它由状态机在当前请求状态上校验获得；输出采样与执行继续使用它。

### 删除链路

| 所属层 | 必须删除或改造的内容 |
| --- | --- |
| C# 输出 | `candidate_skill_context`、`candidate_state_context`、候选 builder、`CandidateContextEntry` 及 schema 中相应顶层 key |
| C# 门面 | `CandidatePreview`、`CandidatePreviewBuilder` 及只为候选未来状态服务的分支；vector/tensor 观测改为直接生成当前状态与历史 |
| policy | 删除 policy 候选装配；`candidate_kind` 和 `CandidateKind` 改为普通动作 kind，数值 `kind` 的语义保持不变 |
| Python 桥接与客户端 | 清除旧候选响应、接口参数、类型定义及相关包装；保持进程内共享引擎与队列所有权 |
| 转换与缓存 | 删除候选技能 ID、特征、状态、null mask、非法原因、价值数组和候选索引；新增显式当前状态与固定动作 label |
| schema 与 DataSpec | 删除 `num_candidates`、`candidate_action_keys` 等候选维度，建立动作输出空间与当前状态输入契约 |
| dataset 与 collator | 删除候选读取、padding、重排、shuffle 和候选 batch 字段；增加当前状态字段 |
| 模型 | 删除 candidate token 位置、role、segment、双向候选 mask、候选执行路径、候选 scorer 和候选 KV-cache 分支 |
| 训练配置 | 删除 `candidate_order_file`、`candidate_shuffle` 和候选顺序文件；其他用户实验参数保持不变 |
| 损失与重复惩罚 | 价值和动作惩罚按固定 `action_keys` 映射，不再读取候选技能 ID 或候选特征 |
| 回放与 PPG | batch 构建、动作解码、无合法动作处理、基础 GCD 恢复和报告字段同步改造 |
| GRPO | rollout、存储、合法性 mask、action index 与策略更新消费新输入；不改变奖励、采样或 checkpoint 热启动策略 |
| ONNX | 删除候选输入、容量、顺序和 golden 数据，更新导出图、manifest、宿主动作映射及 parity |
| 模型分析 | 删除候选 query、候选角色块、候选元数据和候选 pair 图，围绕最新状态 query 与历史动作改造分析入口 |
| 测试与文档 | 替换正式测试夹具中的候选输入，修订使用说明和项目结构，增加旧契约拒绝测试 |

主要改动位置包括 `common/policy/`、`training/`、`grpo/`、`scripts/convert_fflogs/`、`scripts/autoregressive_replay/`、`scripts/common/`、`scripts/onnx_export/`、`scripts/model_analysis/`、`Combat.Sim/FightEngine/`、`Combat.Sim/PythonBridge/` 以及相关配置和测试。

移除候选未来状态预演后，真实动作提交、事件时间线、当前合法性校验、snapshot/fork 和公共时序查询继续由 FightEngine 负责；不删除真实执行和共享引擎所需的能力。

### 当前状态与必要的阶段过渡

阶段 1 必须先补齐显式当前状态，否则移除候选会同时丢失模型作出决策所需的当前输入。阶段 1 暂时用 `[当前请求状态，当前请求状态]` 装配这个状态 token，并复用现有状态投影与 token 投影，不为它构造伪技能。

这一阶段历史 pair fusion 暂时保留，输入过渡为 `scene + 历史 pair + 当前状态`，整个序列使用因果注意力。完整的历史拆分留到阶段 2，跨步状态语义留到阶段 3，最终序列组织留到阶段 5。

阶段 1 的技能 embedding 仍是旧融合维度，可能与最新状态 hidden 的 `d_model` 不同。此时在输出侧将 hidden 适配到旧 embedding 维度，再与同一张 `E` 做点积，技能输入编码保持原样。阶段 2 将 `E` 改为 `d_model` 维独立 embedding 后，删除这条维度适配，最终直接执行上述共享词表匹配。

### 删除候选时同步修复的消费者

技能数值字段改为由稳定的技能字段契约和职业资源定义推导。空历史、仅有一条历史、仅有 policy wait 的 source 也必须产生相同的字段宽度与顺序，不能依赖“找到第一个候选或历史样本”才确定 schema。

验证 PPG 改从首个样本的显式请求状态恢复初始基础 GCD，并使用 checkpoint 保存的 normalizer 反归一化。核对 `current_gcd_seconds` 是否处于无个人加速修正的初始状态；需要职业修正时通过职业规则恢复，不能直接把加速后的当前 GCD 当作基础 GCD，也不能回退到本机默认配置。增加与转换初始 GCD 一致的测试，不增加重复的 manifest 基础 GCD 或副本时长字段。

历史真实直接威力、累计 DoT 威力和执行 GCD 数继续保留。删除候选不能使 `val_ppg`、`none_ppg` 或训练初始化依赖失效。

### 无残留验收

实施时对当前 Git 跟踪的正式源码、配置、测试和文档进行扫描，再逐项检查命中的生产路径：

```powershell
rg -n "candidate|Candidate|num_candidates|candidate_index|candidate_kind" common/policy training grpo scripts/common scripts/convert_fflogs scripts/autoregressive_replay scripts/onnx_export scripts/model_analysis Combat.Sim/FightEngine Combat.Sim/PythonBridge config tests main.py ffxiv_ccg.ps1 .env.example README.md docs -g "*.py" -g "*.cs" -g "*.yaml" -g "*.json" -g "*.md" -g "*.ps1" -g "*.example" -g "!**/__pycache__/**" -g "!**/bin/**" -g "!**/obj/**"
```

验收要求：

- 正式输出、compiled sample、batch、模型输入、checkpoint 新契约和 ONNX 输入中均无候选字段。
- 已删除候选类、模块和配置没有 import、导出、默认值、fallback、CLI 参数或调用入口残留。
- 所有正式入口无需候选预演即可完成一次合法动作预测与执行。
- 旧 cache、旧 checkpoint、旧部署包明确拒绝，不能通过补空字段或忽略字段继续运行。
- 搜索剩余命中逐条记录：历史计划中的迁移说明、旧契约拒绝测试中的字段名，以及与模型候选无关的通用局部变量可以保留；不能保留实际候选能力或兼容读取分支。
- 扫描不改动 `old/`、第三方子模块、历史训练产物或普通文件查找中恰好名为 candidate 的无关变量。

阶段 1 的交付包含改动清单、残留分类结果、新动作输出契约和验证结果。未通过这一步，不开始阶段 2。

### 阶段 1 交付记录（2026-10-04）

本阶段在 `ice/codex/causal-policy-stage1` 完成。正式输入现在为 `scene + history pair + current state`，全部采用因果可见性。历史技能与状态的融合、历史 before/after 语义及 384 条动作的读取窗口保留；技能和状态独立 embedding、跨步状态语义与 601 token 容量尚属后续阶段。

- 删除 C# 候选预演/上下文 builder、Python 候选字段、候选顺序文件和 shuffle、分段双向注意力、候选 scorer；转换、缓存、BC、GRPO、回放、PPG、分析、ONNX 及根 CLI 均改用新契约。
- 固定动作空间按所有启用真实技能与注册 policy 动作的全局 ordinal 顺序保存；`action_keys`、`action_to_vocab_id`、`action_is_gcd` 随缓存/输入契约保存。当前合法性与动作价值只作为执行/监督元数据，回放额外保留原 GCD/oGCD 阶段筛选。
- 技能输入/输出直接共用 `input_encoder.skill_embed.weight`；阶段 1 用 `output_adapter` 将最新状态 hidden 适配到原 embedding 维度后点积，不维护独立输出词向量。输出索引允许非 identity 的 embedding 行映射，梯度和 AdamW/FP32 主权重只归属同一参数。
- 当前状态两段都是实际请求时快照；不构造伪技能或预测动作后状态。PPG 从当前状态与保存的归一化契约恢复基础 GCD，覆盖加速修正、缺失/null/零值拒绝，继续只统计实际执行历史。
- 新契约为桥接 `12`、canonical `11`、checkpoint 输入 `10`、样本 `8`、cache `raw_json_compiled_samples_v19_causal_policy`、转换 `raw_json_to_compiled_v20_causal_policy`、GRPO 轨迹 `2`、部署 `14` / manifest `8`。旧缓存重新转换，旧候选 checkpoint/部署包明确拒绝。
- 全 bank 与读取窗口隔离验证通过；调整新格式的历史读取窗口复用同一 cache。PythonBridge 已在当前工作树重建。
- 文档对照整理后，按用户授权清理七个文件中仍描述候选用途的中文注释与异常文字，统一为状态分支、历史与当前状态、合法动作等实际语义；仅文字修改，算法、字段与状态契约不变。

阶段 1 文字收尾前的既有验证：全仓 Python `1279 passed, 4 skipped`（98.63 秒），C# `277 passed`。四个条件跳过分别是旧真实 checkpoint、未开启慢速完整导出 gate、未提供外部完整 FFLogs 报告、Full AttnRes 不支持 Post-LN。独立小模型的 GELU/SwiGLU float32 CPU 和 BF16 CUDA 四套真实导出均通过 ONNX checker、ORT、padding/动态输入一致性、manifest 和包校验；BF16 采用严格 CUDA provider，禁止 CPU fallback。这些验证覆盖契约和执行行为，不替代用户后续的 100 份数据训练评估。

文字收尾后的本轮验证：PythonBridge 以 Debug、`--no-restore` 重建，`0 警告、0 错误`；既有 `FightEngine.Tests.Facade.RandomSequenceReplayTests` 以 `--no-build --no-restore` 定向运行，`2 passed、0 failed、0 skipped`（144 毫秒）。七个源文件的 UTF-8/LF、无 BOM 与 Python AST 检查通过，此前实现与测试文件的哈希未漂移，扩大残留扫描未发现陈旧候选职责说明或实际候选功能。本轮未重跑上述全仓 Python 或全部 C# 测试。

全仓集成命令：

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests -q --tb=short --basetemp=.tmp/stage1-final-integrated-temp-02 -o cache_dir=.tmp/stage1-final-integrated-cache --junitxml=.tmp/stage1-final-integrated-proof.xml
```

残留审计覆盖 455 个项目源文件：正式模型候选字段、类和执行入口为零；旧 context key 只出现在明确禁止它们存在的测试断言。宽 `candidate` 命中按文件/副本筛选、checkpoint 选项、事件队列、截止时间备选、历史重合长度、旧契约拒绝和版本/计划记录分类，不属于模型候选能力。删除了根 CLI 的旧读取路径和未使用的候选测试 helper。修改文件的 UTF-8/LF、Python AST 和 `git diff --check` 通过；未启动正式训练，未 commit/push，未改 CHANGELOG。

## 阶段 2 拆分技能与状态 embedding

删除 `pair_fusion` 全部参数、激活路径、辅助方法、旧分析接口和 `pair_embedding_dim` 配置。技能和状态在 Transformer 前不再融合。

阶段 2 采用以下独立编码结构。各线性投影直接输出 `d_model` 维，再经过各自的 `LayerNorm(d_model)`；归一化不改变维度，也不代替原有字段归一化：

```text
技能 token = LayerNorm_skill(E[skill_vocab_id] + Linear_skill(技能数值特征))
状态 token = LayerNorm_state(Linear_state(两段状态数值) + Linear_null(null_mask, bias=False))
场景 token = LayerNorm_scene(Linear_scene_type(场景数值))
三种 token 的输出宽度均为 d_model
```

技能 ID embedding 使用可学习的词向量查找表 `E`，形状为 `[vocab_size, d_model]`，每个技能 ID 对应一行向量。这是 GPT-like 的 token embedding 方法；技能名称不拆成文本子词。继续保留现有数值字段，不能把“拆分 embedding”解释为只保留技能 ID。新增或删除数值字段必须进入阶段 4 的对照与确认。

LM Head 直接读取这张 `E` 的输出技能行，用 `h_last @ E[action_to_vocab_id].T` 匹配。`E` 同时接受历史技能输入与下一技能监督的梯度，输入、输出之间不再维护第二套技能语义向量。

类型标识区分 `scene`、`state`、`skill`，历史状态和最新状态都使用 `state`。所有 token 进入同一 Transformer 和同一因果注意力路径，不建立第二套上下文或单独的当前状态编码器。role/segment 中重复表达旧候选分区的参数一并清理。

实际实现只保留一张类型 embedding，类型编号为 `scene=0`、`state=1`、`skill=2`，删除重复的 segment embedding。历史状态与最新状态共用同一组状态投影、null 投影、LayerNorm 和类型向量；最新状态由显式物理位置 metadata 识别，不使用特殊的当前状态类型。

按用户最终决定，保留 `is_legal`、`invalid_reason`、`action_legal_mask` 及现有字段级 null mask。`null_mask` 表达字段缺失，当前缺失值数值占位仍遵循原 `-1` 契约；它不表达动作合法性。历史不足的 padding 仍由 scene/history attention mask 排除。阶段 2 不删字段，黑魔技能数值特征保持 19 维，状态保持 86 维；状态时间语义在阶段 3 调整，字段删减在阶段 4 对照与确认。

checkpoint 输入契约升级为 `11`，保存严格的 `token_encoding` 描述，记录各路投影与 LayerNorm、状态共享、类型编号、共享技能输出头、历史容量单位和过渡序列。旧融合 checkpoint、缺少描述或描述不匹配的新版本输入契约均明确拒绝。共享的是基础技能语义参数表 `E`，不是加数值特征和归一化后的整个技能输入 token。

阶段 2 尚未改变旧历史状态语义，整模型使用过渡布局 `scene, A1, T1, A2, T2, ..., 当前状态`，其中 `Ti` 仍是第 i 个技能的旧 before/after。不能把包含 `Ai` 执行结果的 `Ti` 放在 `Ai` 前面并当作决策前状态。这个过渡布局不用于最终训练，阶段 5 必须改为最终的状态、技能交错顺序。

`history_capacity` 在本阶段仍以历史动作数计量：H 条历史输出 2H 个独立 token，总物理长度为 `scene_length + 2H + 1`。当前 384 条动作与 200 个场景槽对应最大物理容量 969；601 个非场景槽的容量单位迁移仍留在阶段 5。场景和每个有效技能、状态 token 均使用独立连续的 RoPE 位置，padding 不占逻辑位置；KV-cache、trace、分析和 ONNX 同步消费两 token 布局，输出头使用明确的最新状态位置。

compiled cache 的完整 bank、原始字段和状态机语义保持不变，本阶段复用 cache v19/转换 v20 和 C# 桥接 12，不因 embedding 参数或读取窗口变化重编译相同数据。ONNX 仍有 10 个输入，部署契约升级为 15、manifest 升级为 9，明确历史容量的动作单位、每条两 token、过渡顺序及长度公式。旧 pair 分析接口替换为独立历史技能和状态 embedding 视图。

验收包括：独立 token 数量和维度正确、类型标识正确、技能字段及归一化保持、所有投影有梯度、输入输出复用同一张 `E` 且两条梯度路径有效、优化器只更新共享参数一次、无 pair fusion 参数和旧配置残留，旧融合 checkpoint 明确拒绝。主干层数、MQA、激活、优化器、损失策略保持原有设定。

### 阶段 2 实施与验证记录

独立输入编码、共享技能输出头、读取侧长度统计、KV-cache、checkpoint 契约、模型分析和 ONNX 已同步完成。删除旧 pair 分析模块，分别生成历史技能和历史状态 PCA；决策位置从显式 metadata 读取。新增独立编码回归测试，补充共享参数梯度、字段隔离、因果顺序、padding 逻辑位置、缓存复用与失效、严格旧契约拒绝检查。旧融合配置或参数只保留在明确拒绝旧产物的校验和测试中，不保留可运行的兼容路径。

使用项目 `.venv` 完成全量 `pytest tests -q -x --tb=short`，为 pytest 分配独立系统临时目录：**1329 passed、4 skipped、0 failed、0 errors，75.31 秒**。报告保存在 [stage2-integrated-1329.xml](../.tmp/stage2-integrated-1329.xml)。四项跳过分别为旧融合真实 checkpoint 不兼容、未开启完整真实 checkpoint 导出、未提供完整 raw report 的外部集成输入，以及 Full AttnRes 不支持 Post-LN；未将这些边界计为通过。

GELU、SwiGLU 各完成 FP32 CPU 和 BF16 CUDA 的小模型真实导出，共四套。ONNX checker、shape inference、ORT 和每套六种空场景、空历史、有效长度及 padding 检查通过；动作 argmax 一致，FP32 最大 logit 绝对差为 `5.960e-7`，BF16 为 `0`，CUDA 禁止 CPU 算子 fallback。四套包均保存输入契约 11、部署契约 15、manifest 9；这些结果属于小模型图验证，真实训练 checkpoint 的完整导出与回放验收仍待后续匹配产物。

黑魔技能 19 维、状态 86 维及其 before/after、null masks 和合法性字段保持；raw source、完整 history bank、cache signature、转换逻辑、C# 与 schema 未改。UTF-8/LF、Python AST、JSON 和 `git diff --check` 通过。实现验收时未启动正式训练，未修改 `training.yaml` 或 CHANGELOG，未 commit/push；未提前实施跨步状态语义、字段删减或阶段 5 的 601 容量。随后按用户要求更新 Unreleased 变更说明并进行阶段 2 的本地提交。

## 阶段 3 调整状态语义

本阶段先实现已确认的“上一步技能后状态＋当前请求时状态”及统一回退，两段字段分别命名为 `previous_action_after.*`、`request_state.*`。底层状态字段和技能字段沿用阶段 2，技能字段删减与最终对照留在阶段 4。真实技能暂按现有规则在生效后提供 skill token，但对应的模型状态必须在请求时冻结，并保存前序请求关联；不提前扩展为待生效技能 token。最终交错顺序与 601 容量仍留在阶段 5。

### 时间锚点与跨步配对

定义 `Qi` 为请求选择第 i 步动作前、已处理该时刻到期事件和场景事实的真实状态；`Ei` 为第 i 步动作实际生效后的快照。最终状态的逻辑语义为：

```text
能取得前一步已发生的快照：Si = [E(i-1), Qi]
不能取得该快照：         Si = [Qi, Qi]
```

这两个快照均不依赖第 i 步将选择的动作。第一段保留上一动作生效时的时间和数值；第二段保留本次请求时的时间和数值。两个时刻之间发生的回蓝、DoT tick、Buff 过期、资源变化和场景事实由真实时间线体现，不能用第二段反推或重算第一段。

当前历史条目已有“本技能请求状态”和“本技能生效后状态”，可作为快照原料，但必须按动作实例和请求顺序重新关联。当前历史按效果事件记录，多个请求或效果可能同戳，不能仅把数组索引减一当作前一步，也不能用未来 bank 中后来获得的 after 填充早期请求。

旧历史的 `StateBefore`、`StateAfter` 若仍用于真实执行统计，应继续表达真实动作的请求和生效快照。新模型状态由输出或 policy 数据层跨步装配，不把 FightEngine 的动作历史记录直接改名换义。

### 回退与历史冻结

回退适用于无前序动作、前一步尚未生效、无法恢复动作关联、外部初始状态未提供前一步快照等情况。只能使用本次请求的真实状态作为两段。

历史窗口裁剪本身不必然触发回退：若完整 bank 或当时保存的状态 token 已携带窗口外前一步快照，就保留它；确实不可取得时才复制当前请求状态。随机历史截断遵循同一规则。

每个历史 `Si` 按当时请求可知的信息冻结。即使前一步后来生效，也不修改早期已生成状态、不回填其第一段；否则训练会读到未来，在线 KV-cache 也会复用错误前缀。

### policy wait 与场景处理

`ogcd_wait` 保留为 policy 控制动作，继续不提交 SkillBook 或真实游戏技能执行。它没有自己的游戏效果；作为前一步时，动作后基准使用该 policy 请求时的真实快照。等待期间的真实变化体现到下一请求状态。

删除“通过 preview 分支推进到未来观测时刻，生成 policy 模型 after”的做法。决定等待的输入与历史状态不能包含未来观测结果。本阶段沿用 policy wait 技能现有的 kind、零威力、资源消耗、词表映射和辅助训练用途，阶段 4 再进行最终字段对照与删减。

转换层继续可以在输出层合成移动字段；在线回放继续把移动和场景事实提交状态机。`scripts/common/scene_state.py` 按输出的 `player_state_feature_keys` 定位两段状态各自的 `time_seconds`，在归一化前使用原始秒数分别查询场景。两段不能统一改写为本次请求时间，也不能通过技能生效时刻减读条时间反推请求时刻；历史状态和末尾最新状态都遵守同一规则，为阶段 4 完全删除技能时间字段准备替代来源。

### 保持 PPG 统计独立

新状态第二段是请求状态，不再是本步动作生效后状态。不能继续从重新命名后的“第二段”提取本步执行结果或累计 DoT。

模型输入历史与真实动作效果统计分别管理：完整技能/状态 bank 记录模型所需输入和请求关联，真实执行威力与累计 DoT 由效果历史或状态机统计提供。必要的动作关联 metadata 不进入 Transformer。

`val_ppg`、`none_ppg` 继续使用真实执行直接威力、DoT 和执行 GCD 数。验证回放无合法动作时先由 DecisionScheduler 推进，无法推进时该副本全部返回零并仍计入副本平均；不改变场景终止条件、平均规则或 teacher-forced Top-1/Top-3 的角色。

### 阶段 3 实际交付与验收（缓存链路待修复）

阶段 3 的跨步状态装配已实现，compiled 历史输入整链路验收尚未通过。真实动作在请求被接受时创建不可变的 `ModelStateSnapshot`，以动作实例 ID 记录最近一次真实或 policy 请求。动作生效时只更新与当前请求身份相符的后状态；较早请求后来生效不会覆盖较新的请求。排队动作的输入冻结在原始请求时刻，不改用排队接受或生效时刻。

`LastDecisionAfter` 独立于历史保留窗口，并随 snapshot、fork 和 restore 保存。历史模型状态保持请求时的快照，真实历史 `StateBefore` / `StateAfter` 继续供技能威力、耗蓝及资源消耗计算。无法取得前一步快照时，两段逐字段复制本次请求状态，低层外部历史缺少模型快照也使用同一回退。

`ogcd_wait` 在落实决策时保存真实状态，不再 fork 并推进到未来生成 after。其历史和后续最新状态使用与真实技能相同的字段、状态 builder、归一化、embedding 和状态类型。等待过程的回蓝、DoT tick、Buff 时长与量谱变化只进入下一请求段。

场景改写按两段各自的原始 `time_seconds` 和输出 feature keys 查询，历史状态与最新状态采用同一规则；不再从技能时间减去读条时间，也不使用未来观测时刻。缺少字段、宽度不符或时间非有限时明确拒绝。

`state_history_context.execution_metrics` 保存与技能历史一一对应的真实累计直接/DoT 威力，合并 policy 历史和裁剪时同步处理，但不进入模型向量。完整 history bank 的累计 DoT 从该 metadata 编码，回放最终威力从真实最新 `request_state` 读取；基础 GCD、Ley Lines 与分析 AF/UI/MP 也从当前请求段读取。模型状态不再被误当成本步执行结果。

黑魔技能数值特征仍为 **19 维**，状态仍为 **86 维**；底层字段、归一化数值规则、null mask、合法性与输入输出共享技能 embedding 保持。技能 `time_seconds` 暂时保留，历史仍按实际效果时间与 policy 请求时间合并，未提前引入待生效技能 token。容量仍为 384 条动作，最多 `200 + 2×384 + 1 = 969` 个物理 token；最终请求顺序的状态/技能交错和 601 容量属于阶段 5。

同一工作树的 PythonBridge 已重建，构建为 0 warning、0 error。C# 全量测试 **286 passed、0 failed**，其中新增跨步语义回归覆盖首步回退、历史冻结、真实技能与 wait、未生效前序、排队、同戳动作、窗口外前序、fork/restore、回蓝及 DoT/Buff/量谱变化。

项目 `.venv` 的完整 Python 检查为 **1340 passed、4 skipped、0 failed、0 errors，96.01 秒**，报告见 [stage3-final-integrated.xml](../.tmp/stage3-final-integrated.xml)。四项跳过分别为旧融合真实 checkpoint 不兼容、未开启完整真实 checkpoint 导出、未提供完整 raw report 的外部集成输入，以及 Full AttnRes 不支持 Post-LN；这些边界未计为通过。随后移除无正式调用的旧场景 offset helper，并明确执行 metadata 的诊断名称；对应场景、source/bank 与 PPG 检查 **48 passed、0 failed**，报告见 [stage3-cleanup.xml](../.tmp/stage3-cleanup.xml)。UTF-8/LF、Python AST、manifest JSON 与 `git diff --check` 均通过。

GELU 与 SwiGLU 各通过 FP32 CPU 和 BF16 严格 CUDA 的小模型真实导出，共四套，每套覆盖六种 padding/有效长度检查，ONNX checker 与 ORT 通过，动作 argmax 全部一致。FP32 最大 logit 绝对差为 `5.960e-7`，BF16 为 `0`，CUDA 禁止 CPU 算子 fallback。该结果属于小模型图验证，真实训练 checkpoint 的完整导出与回放验收仍需后续匹配产物。当前输入契约为 12、部署契约为 16、manifest 为 10。

2026-10-04 按用户指定，将 `data/human/job/black_mage/raw/VAL/M5s/fflogs_2CHK3gRfrNJxhmwb_f1_Arcadia_Petralia.json.br` 通过正式入口转换到隔离审计目录，生成 491 个样本，其中真实技能 273 个、`ogcd_wait` 218 个、排队请求 46 个。原始历史状态与各自请求时最新状态共完成 120295 次逐字段对照，全部一致；218 次等待后的两段状态符合统一语义，109 个样本的两段累计 DoT 威力因时间推进而增加，首步回退与最新状态的同构字段也正确。

实际 compiled bank 验收发现 9 次历史前缀重排：真实技能使用输出 token 中已舍入到四位小数的时间排序，policy 等待使用内部原始浮点时间排序。同刻的 `10.160499999999999` 与 `10.1605` 可能改变已输出的技能/等待先后，bank 却仅按上一历史长度截取增量，最终遗漏 9 个等待并重复 9 个真实技能记录。样本 13～491 共 479 个样本的完整历史与正式 384 条动作窗口均与原始转换上下文不一致；491 个最新状态仍一致。因此此前“阶段 3 已完成验收”的结论更正为状态语义已实现、缓存链路待修复。

继续实施前需统一使用内部精确时间与稳定的动作先后关系合并历史，并在增量写入 bank 前校验既有前缀身份；补齐同刻真实技能/等待与本次真实日志回归后重新转换验收。本次记录问题，不包含该修复。审计明细见 [audit.json](../.tmp/context-audit-m5s-20261004/audit.json) 与 [bank_diff.json](../.tmp/context-audit-m5s-20261004/bank_diff.json)。

旧格式 cache 需重编译，旧 checkpoint 与 ONNX 包不可静默复用；本阶段没有批量重建用户的生产 cache、训练 checkpoint 或部署包。实现和验证使用独立临时目录，未修改 `training.yaml`，未启动训练或推送；阶段 3 的提交变更说明按用户要求写入 CHANGELOG 的 `Unreleased`。

## 阶段 4 技能与状态字段确认及删减

这一阶段在状态语义改造后向用户提交最终字段表，确认后完成字段删减。对照需覆盖字段名、顺序、类型、维度、取值来源、时间锚点、归一化和进入模型与否，不能只展示字段名称。

以下基线依据阶段 2 结束时的 C# builder、schema、Python 特征提取与黑魔资源注册核对；实施时须与阶段 3 的实际状态输出对照。技能 `time_seconds` 按用户要求计划完全删除，其余未明确要求变更的字段仍保留，并逐项向用户确认。

### 技能 token 基线

| 当前 canonical 字段 | 当前用途或来源 | 计划处理 |
| --- | --- | --- |
| `skill_id` | 原始技能或 policy ID，映射到技能词表 | 保留，仍走 ID embedding |
| `skill_key` | 动作身份、输出映射与诊断 | 保留，不作为浮点特征 |
| `skill_name` | 展示与诊断 | 保留，不作为浮点特征 |
| `potency` | 历史技能威力字段 | 保留现有口径，同时保留真实执行统计 |
| `value` | 训练价值偏好信息 | 保留其训练用途，不加入模型数值特征 |
| `kind` | 数值 1 为 GCD，0 为 oGCD | 保留，进入技能数值特征和历史 GCD 统计 |
| `actual_mp_cost` | 历史 MP before/after 的非负差 | 保留，不能顺手改为静态技能耗蓝 |
| `cast_time.seconds` | 历史实际读条时间 | 保留秒制与当前归一化 |
| `gcd_window.seconds` | 历史动作窗口 | 保留，删除候选不等于删除历史该字段 |
| `is_legal` | 历史合法性数值 | 保留；即使通常为真，也不在本阶段顺手裁剪 |
| `invalid_reason` | 合法性诊断文本 | 保留其诊断用途，不作为浮点特征 |
| `next_cooldown_seconds` | 当前历史记录中的技能冷却快照 | 保留取值时点与归一化 |
| `available_charges` | 历史技能充能快照 | 保留 |
| `max_charges` | 技能最大充能 | 保留 |
| `job_resources_consumed.*` | 历史动作资源差分 | 保留所有已注册资源维度，仍由技能 token 承载 |
| `time_seconds` | 真实技能历史为动作生效时刻，policy wait 为决策时刻 | 从 canonical 技能 token 和模型数值特征中完全删除；场景查询使用状态自身时间，历史排序保留内部事件时间 |

当前黑魔历史技能的数值特征共 19 维，按当前 Python 排序后的名称为：

```text
actual_mp_cost
available_charges
cast_time.seconds
gcd_window.seconds
is_legal
job_resources_consumed.astral_fire
job_resources_consumed.astral_soul
job_resources_consumed.firestarter_ready
job_resources_consumed.paradox_ready
job_resources_consumed.polyglot
job_resources_consumed.polyglot_timer
job_resources_consumed.thundercloud_ready
job_resources_consumed.umbral_hearts
job_resources_consumed.umbral_ice
kind
max_charges
next_cooldown_seconds
potency
time_seconds
```

19 维数值输入与技能 ID embedding 是两部分。`value`、文本字段和质量监督标签不进入这 19 维。删除 `time_seconds` 后，黑魔技能数值特征为 18 维，技能 token 的输出宽度仍为 `d_model`，共享语义 embedding 不变。新实现要用真实转换样本再验证这个基线；每职业动态资源按其职业注册推导，不把黑魔的列表硬编码进公共模型。

技能时间字段删除时，必须同步移除字段发现、特征顺序、归一化输入装配和完整 history bank 中对应的技能列，不能只删除 raw key 却保留一个默认填零的模型维度。历史合并排序改用内部真实动作与 policy 决策时间；回放前缀识别使用技能身份、状态时间和完整 token 对照，不再依赖技能时间字段。场景改写沿用阶段 3 建立的两段状态时间锚点，不使用技能时间减读条时间推算请求时刻。内部时间线继续保留执行所需的时间信息，不新增替代的技能时间特征。

### 状态 token 基线

阶段 2 的状态由 `player_state`、`buff_state`、`target_buff_state`、`resource_state` 四组组成。黑魔单个快照为 43 维；每组包含两段快照，整个状态 token 为 86 维。阶段 3 保留底层字段和四组装配顺序，重新定义两段快照的来源；本阶段核对实际字段和维度。

| 分组 | 单快照字段 | 黑魔单快照维度 |
| --- | --- | --- |
| player | `mp`、`max_mp`、`mp_ratio`、`time_seconds`、`current_gcd_seconds`、`boss_targetable`、`next_untargetable_in_seconds`、`downtime_remaining_seconds`、`is_moving` | 9 |
| buff | 系统 `burst_potion`、`raid_buff_window`，职业 `ley_lines`、`lucid_dreaming`、`swiftcast`、`triplecast` 各自的 `active`、`remaining_seconds`、`stacks`；另含 `resource.firestarter_ready`、`resource.thundercloud_ready` | 20 |
| target buff | `target.high_thunder` 的 `active`、`remaining_seconds`、`stacks`；以及 `target.cumulative_dot_potency`、`target.cumulative_potency`、`target.current_potency`、`target.current_gcd_dot_potency` | 7 |
| resource | `astral_fire`、`astral_soul`、`paradox_ready`、`polyglot`、`polyglot_timer`、`umbral_hearts`、`umbral_ice` | 7 |

Buff 顺序遵循现有 builder 的系统组、职业组与各组 ordinal 排序；资源顺序、DoT 顺序和四组顺序以实际 schema 为准。每组内部依次存放上一步动作后向量与当前请求向量，不为了本次迁移另行打乱底层字段。

阶段 3 将旧 `before.*`、`after.*` 改为明确的 `previous_action_after.*`、`request_state.*`；本阶段核对归一化器、特征索引、checkpoint 契约、状态分析与部署契约是否一致支持新名称，两段同名基础字段使用相同的归一化规则。不能保留旧字段名却静默改变其含义。

状态字段缺失与 padding 仍按数据契约处理；“缺少上一步快照”必须复制当前快照，不能通过 null、零向量、专用未知状态标识或截断后重新计算来替代。回退原因如需诊断，只记录为非模型 metadata。

### 需要用户确认的字段结论

- 技能 `time_seconds` 完全删除后的 canonical 字段、18 维数值特征、顺序和取值时点是否一致；其余字段如有必要变更，逐项说明原因。
- 状态底层 43 维字段是否全部保留，最终状态是否保持 86 维及上述两段顺序。
- 两段字段采用的正式名称，以及相同字段归一化规则是否一致。
- 是否扩展阶段 3 沿用的记录规则，让历史包含已经请求、但尚未生效的动作。当前规则只在生效时提供真实技能 token；若要求记录这些待生效动作，必须逐项确定历史技能字段当时是否可知，尤其是资源差分、耗蓝和实际执行结果。不能用未来结果回填早先请求的输入，也不能重新引入全动作预演来填充字段。
- 若保留当前“真实技能生效后才提供技能 token”的行为，明确其请求顺序映射、遗漏的待生效动作和末尾状态前一步的识别规则；不能将“最新效果历史”自动等同于“上一次请求”。

最后两项会影响历史 bank 的记录时机。阶段 3 先沿用现有的真实技能生效后提供 token 的规则，并冻结请求时的模型状态；阶段 4 随字段表复核。若确认结果要求改变记录范围或关联规则，在阶段 5 重建最终历史上下文前补齐状态装配与相应契约修改。

## 阶段 5 重建历史上下文与 RoPE

### 完整历史 bank 与读取窗口

沿用阶段 3 建立、阶段 4 确认的状态语义与记录规则，重建可对齐的模型历史 bank，保存稳定的动作身份、请求锚点、技能字段和冻结状态。模型的最近 H 条历史必须形成一一对应的 H 个技能 token、H 个历史状态 token，再追加一个本次请求状态。

`model.history_capacity` 改为 601，并显式改为技能与状态 token 的合计容量，包含末尾当前状态。读取侧先为当前状态预留一个槽位，再按 `floor((history_capacity - 1) / 2)` 选择完整的最近历史状态与技能对；容量为 601 时最多读取 300 条历史动作，对应 600 个历史 token。不能将 601 或 600 直接作为历史动作条数传入旧的 `max_history`、gather 或回放裁剪接口，也不能截断一对 token 后留下孤立技能。compiled cache 编译阶段保存完整 bank；不得将 601 或其他读取窗口加入 cache signature、编译 worker、raw source 后端或 bank 构建参数。

容量单位变更同步写入模型配置说明、checkpoint 输入契约与部署契约，旧的动作条数配置和产物不得按新单位静默读取。固定容量输入按 601 分配非场景槽位；满窗口全部 601 个 token 有效，历史不足时的 padding 使用同一套有效长度和 mask。

同一 source 的新格式 cache 建成后，将读取侧 token 容量从 601 改为其他值必须复用它。格式或时间语义变化引发的一次性重编译，与历史窗口调整导致的重编译严格区分。

更新 dataset、collator、GPU gather、历史裁剪、随机截断、回放输入、GRPO 输入、运行时容量统计、分析 token 元数据以及 ONNX 固定容量。保留 source 级 bank 去重和多队列生命周期约束，避免为每个样本重复保存整段历史。

### 单一因果序列

最终 token 顺序必须是 `scene, S1, A1, S2, A2, ..., SH, AH, S_current`。输出头准确定位最后一个有效状态 token，不通过物理数组最后一列猜测，避免 padding 改变预测位置。

整个有效序列使用下三角注意力，token 可以读取自身和之前的有效 token。技能、状态和场景都不建立双向块、pair 内特许可见性或第二套上下文。场景仍为既有完整 scene token，未来时间的已知场景窗口不等于未来动作或结果标签。

BC 继续以请求样本的目标动作进行监督。本次不额外扩展为对每个历史位置计算语言模型损失，不引入新的训练目标或辅助标签。

### RoPE 顺序编号

有效场景 token 从 0 开始连续编号。设有效场景数为 `s`：

```text
position(Si)        = s + 2 * (i - 1)
position(Ai)        = s + 2 * (i - 1) + 1
position(S_current) = s + 2 * H
```

状态与相邻技能相差一个位置。`S1` 与 `A1` 不能共享 position ID，两段状态快照也不会各自获得一个 Transformer 位置，因为它们仍在同一个状态 token 内。

编号按每个样本的有效长度生成，scene/history padding 不占用逻辑位置，且被注意力 mask 排除。物理位置可以为 padding 预留槽位，但下一有效 token 的 position ID 不跳号。

保留统一的 `rope_theta` 和 Q/K 上的 RoPE，不增加第二种位置编码。RoPE 频率与角度计算保持已有 FP32 保护，不能因模型 BF16 权重转换导致频率先量化。

### KV-cache 与分析部署

因果 KV-cache 改为缓存场景和交错历史，不再有候选 suffix 分支。末尾状态在选择并记录动作后转为普通历史状态；若值和逻辑位置一致，可复用其缓存，然后追加技能与下一状态。

裁剪窗口、随机截断、场景输入变化、position ID 重编号或输入数值变化时，按实际有效前缀失效或重建缓存。不能在窗口移动后直接保留旧 RoPE 位置的 K/V。完整前向与缓存推理必须对同一个最终窗口比较，不能把完整全历史输出作为裁剪窗口的等价参照。

trace、hidden/PCA、技能 embedding 和 attention 图按 `scene/state/skill` 重建；最新状态 query 的注意力能准确对应它之前的状态与技能。旧 pair 图与候选分析入口彻底移除。

ONNX 的输入、容量公式、padding、输出动作顺序、manifest、golden 和 parity 统一更新到最终序列。非场景容量固定为 601，总物理容量为 `scene_capacity + 601`；保留场景容量 200 时为 801。部署 batch=1 的现有边界保持，当前状态和历史状态使用相同编码参数。旧部署包必须拒绝新输入。

## 契约升级与产物管理

以下记录阶段 1～3 的实际版本。契约按各自负责的兼容边界升级，不按阶段号统一递增；已升级且当前阶段没有变化的契约保留原值。

| 契约 | 阶段 1 | 阶段 2 | 阶段 3 | 阶段 3 处理原因 |
| --- | --- | --- | --- | --- |
| canonical 输出 schema | 11 | 11 | 12 | 状态字段改为跨步语义，并增加独立执行 metadata |
| `sidecar_contract_version` | 12 | 12 | 13 | 真实/policy 请求冻结与 Python 观测语义改变 |
| `INPUT_CONTRACT_VERSION` | 10 | 11 | 12 | 同宽状态含义改变，明确快照类型与请求冻结规则 |
| `CACHE_FORMAT` | v19 causal_policy | v19 causal_policy | `raw_json_compiled_samples_v20_causal_state` | 旧状态 bank 不可复用，执行统计独立于模型状态 |
| `DEFAULT_CONVERSION_VERSION` | v20 causal_policy | v20 causal_policy | `raw_json_to_compiled_v21_causal_state` | 快照来源、wait 和场景查询时刻改变 |
| normalizer 契约 | 2 | 2 | 保持 2 | 新前缀仍映射同一底层字段，归一化数值规则未变 |
| `DEPLOYMENT_CONTRACT_VERSION` | 14 | 15 | 16 | 相同张量宽度下输入状态语义已改变 |
| manifest | 8 | 9 | 10 | 部署包绑定新的输入与状态语义契约 |

训练样本 schema 从 8 升级为 9，GRPO rollout 格式从 2 升级为 3，拒绝同宽但语义不同的旧样本与轨迹。

阶段 3 的跨步状态语义已按上述边界升级；阶段 4 删除技能时间字段、阶段 5 将历史容量从动作条数改为技能与状态 token 合计容量时，再按实际不兼容边界升级。纯 embedding 参数变化不重编译相同数据；在新格式完整 bank 已建立后，单独调整读取侧 token 容量到 601 或其他值不升级 cache 身份。即使 ONNX 外部张量名称和 shape 不变，也不能将内部融合编码和独立编码声明为同一个模型、部署契约。

每次修改 FightEngine、PythonBridge 或 `config/schema.yaml` 后，在同一工作树重建：

```powershell
dotnet build Combat.Sim/PythonBridge/PythonBridge.csproj --configuration Debug
```

最终 DLL、schema、输入契约、cache、checkpoint 和 ONNX 部署包必须对应同一套语义。旧融合/候选 checkpoint 不用于连续续训或静默热启动；新模型从新权重开始。BC 后续仍支持同契约连续续训，GRPO 仍按既有权重热启动规则处理。

新版 cache 和验证用产物写到独立目录，保留旧训练与部署结果，不自动清理用户产物。实现迁移需要准备匹配的新 cache 和部署验证产物，但不授权远程发布或开始用户的训练实验。

## 实现验证矩阵

以下矩阵覆盖全部阶段，已执行的检查与结果见各阶段实际交付记录；尚未实施的阶段不视为通过。

| 验证边界 | 必须覆盖的情形 |
| --- | --- |
| 候选移除 | canonical、compiled sample、collator、forward、回放、GRPO、ONNX 均无候选字段；旧输入明确拒绝 |
| 技能词表与预测头 | 技能 embedding 查表、共享 LM Head 输出宽度及 `action_to_vocab_id` 一致；启用技能、policy wait、padding 和未注册 ID 正确区分；检查共享参数身份、输入输出两条梯度路径、单次优化器更新和 checkpoint 往返 |
| 技能字段 | 新旧历史技能字段、顺序、数值和归一化对照；空历史与不同技能出现顺序不能改变 schema |
| 状态回退 | 开场、缺前一步、前一步未生效时两段逐字段相等；空历史只有一个真实当前状态 |
| 状态时间 | 前一步效果到本次请求之间有 MP tick、DoT tick、Buff 过期、场景与职业资源变化，两个快照独立正确 |
| 动作关联 | 排队 GCD、同戳多个请求、不同生效顺序和动作实例 ID，不错误关联前一步 |
| policy wait | 不提交游戏技能、不读取未来 preview after；下一请求状态反映真实时间推进 |
| 因果性 | 改变后续动作或结果不能改变更早状态输入与 hidden；历史冻结，不能事后回填 |
| 窗口 | 容量 601 时 H 为 0、1、299、300，H 为 301 的输入裁剪为最近 300 对；满窗口为 600 个历史 token 加一个最新状态，共 601 个有效 token；历史不足时的 padding、长 source 裁剪、已知窗口外前一步、随机截断及完整 bank 复用 |
| RoPE | 状态与技能位置严格递增；不同有效 scene/history 长度、padding 布局和 BF16 频率保护 |
| KV-cache | 同窗口完整前向与 prefill/增量推理一致；窗口移动和场景变化正确失效 |
| 训练损失 | label 映射、交叉熵、质量权重、价值偏好和重复惩罚仍消费正确动作索引 |
| PPG | 基础 GCD 恢复一致；真实执行威力和 GCD 数一致；无合法动作且无法推进时返回全零且参与平均 |
| 多职业与多队列 | 黑魔主路径及机工公共边界回归；共享引擎队列隔离、reset/close 和有限历史保留不改变 |
| ONNX | 空场景、真实场景、空历史、满窗口、padding 和容量拒绝；raw logits 数值与动作选择 parity |
| 分析与入口 | CLI、TensorBoard 容量统计、trace、attention、PCA 不再引用候选或融合 pair |

实现与相应测试在同一阶段同步修改。优先运行变化直接影响的测试，阶段 5 完成后再进行整链路验证，不把完整训练当作字段或契约测试的替代品。

使用项目 `.venv` 运行 Python 检查；C# 输出和时序测试跟随桥接程序集重建。测试夹具按新契约更新，不保留可进入生产路径的旧候选兼容实现。

## 阶段 6 用户执行 100 份训练数据测试

这一阶段由用户本人执行。实现方在阶段 5 后交付字段确认结论、测试结果、匹配的产物路径和训练前检查说明，不自行启动训练、改实验参数或代替用户选择 checkpoint。

“100 训练集”按现有 `training.max_files: 100` 解释，为 100 份训练 source 文件，不是 100 个样本或 100 个训练步。验证集仍从独立 VAL 中按既有副本配额选择，不从训练集拆出或替补。

训练前检查：

- 当前模型为最终无候选、独立技能与状态 embedding、技能输入输出共享语义向量、跨步状态语义、300 组历史状态与技能加最新状态（共 601 token）和顺序 RoPE 的组合。
- 阶段 4 字段表已经用户确认；所有实现与部署契约一致，使用新模型权重与匹配的新格式 cache。
- 用户将训练文件上限设为 100；副本、百分位和质量标签选择继续遵循现有训练规则。
- 模型与算法参数由 YAML 管理，本机路径、设备和硬件相关并发参数按项目 `.env` 约定管理。
- 旧实验结果保留，新实验明确记录配置、数据 source、随机种子、契约版本和输出目录。
- 非场景容量为 601，场景容量保留 200 时总物理容量为 801；用户依据实测显存调整 batch 等实验设置，并记录变化。

用户观察训练 loss、Top-1、Top-3，以及同一轮 checkpoint 的 `val_ppg` 与 `none_ppg`。同时观察回放是否有低质量循环、无进展等待、非法动作和恢复困难。历史容量、数据、主干或验证集合改变的实验不能直接归因于某一个架构因素。

100 份数据的训练用于检查训练稳定性、字段与执行链路是否正常、模型是否出现初步可学习行为，不据此宣布大规模效果优于旧架构。是否扩大数据、调参或进一步改变字段由用户根据结果决定。

## 最终完成清单

- [x] 阶段 1 正式链路完全移除候选，残留扫描和旧契约拒绝检查完成。
- [x] 阶段 2 独立技能与状态 embedding 及技能输入输出共享参数完成，pair fusion 和旧配置删除。
- [x] 阶段 3 状态跨步语义、统一回退、历史冻结与 policy wait 处理完成。
- [ ] 阶段 3 同刻历史排序与 compiled bank 前缀一致性问题修复，真实日志整链路验收通过。
- [ ] 阶段 4 用户确认技能、状态字段和待生效动作处理规则；技能 `time_seconds` 完全删除并完成消费者迁移。
- [ ] 阶段 5 单一交错因果上下文、600 个历史状态与技能 token 加一个最新状态（共 601 token）和逐 token 顺序 RoPE 完成。
- [ ] 阶段 5 最终契约下的 Python、C#、PPG、GRPO 输入、KV-cache、分析和 ONNX 验证通过。
- [ ] 新旧产物明确隔离，旧契约不能静默复用，未进行未授权提交或远程发布。
- [ ] 用户本人执行 100 份训练数据的小规模训练并记录结果。

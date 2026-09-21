# GRPO 算法详解与实现指南

> Group Relative Policy Optimization（组相对策略优化）
> 本文档内容全部摘自三篇论文 PDF 原文（公式、超参均照抄原文，见文末来源）
> 2026-09-09 整理

---

## 0. 一句话概括

GRPO 是 PPO 的一个变体：**去掉价值网络（critic/value model）**，改为对同一问题采样一组输出，用**组内奖励的均值/标准差**来归一化奖励，得到每个输出的相对优势（advantage）。原先由 critic 估计的基线，现在由"组内其他采样的实际奖励"充当。

- 出处：DeepSeekMath 论文（arXiv:2402.03300，2024-02）第 4.1 节，作者 Zhihong Shao 等（DeepSeek）
- 成名：DeepSeek-R1 系列（arXiv:2501.12948）用 GRPO 做 RL 阶段，训练出 R1-Zero / R1
- 演进：DAPO（arXiv:2503.14476，2025-03）在 GRPO 基础上做四项改进，用 Qwen2.5-32B 基座达到 AIME24 avg@32=50（超过 DeepSeek-R1-Zero-Qwen-32B 的 47，且只用一半训练步数）

---

## 1. 为什么需要 GRPO：PPO 的问题

PPO（PPO-Clip）在 LLM 的 RL 阶段的目标式（DeepSeekMath Eq. 1）：

```
J_PPO(θ) = E[q~P(Q), o~π_θold(O|q)] · Σ_t min( r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t )
```

其中重要性采样比 `r_t(θ) = π_θ(o_t|q,o_<t) / π_θold(o_t|q,o_<t)`，Â_t 由 **GAE**（Generalized Advantage Estimation）计算——这需要 **critic（value model）** 拟合状态价值。

问题：

| PPO | GRPO |
|---|---|
| 需要 value model（与策略同规模，显存翻倍） | 不需要值函数，直接删除 |
| GAE 要算 λ 折扣的 TD 误差，值函数难学准 | 组内 reward 归一化即得优势，无学习目标 |
| 调节系数多（γ、λ、值函数损失权重等） | 只需 ε、β、组大小 G |
| 值函数误差会引入训练噪声 | 基线来自真实采样的组内平均，统计上更稳 |

**核心直觉**：奖励模型（或规则奖励）本身就是**相对比较**性质的（"哪个输出更好"），GRPO 用组内相对奖励直接对齐这一点——同一问题下，比组平均水平好的输出被加重，差的被抑制。

**另一个关键细节**（DeepSeekMath 原文强调）：GRPO 把 **KL 惩罚直接加进损失**，而不是像 PPO 那样塞进奖励。否则组内相对优势的计算会被 KL 项"污染"（每个 token 的 KL 值不同，归一化就用不了简单的组平均）。

---

## 2. 算法定义（DeepSeekMath 原文 Eq. 3）

对每个问题 q，从旧策略 π_θold 采样一组 G 个输出 {o₁,…,o_G}，最大化：

```
J_GRPO(θ) = E[q~P(Q), {o_i}~π_θold(O|q)]
            [ 1/G Σ_i ( 1/|o_i| Σ_t min( r_i,t(θ)·Â_i,t, clip(r_i,t(θ), 1-ε, 1+ε)·Â_i,t) )
              − β · D_KL[π_θ || π_ref] ]
```

- `r_i,t(θ) = π_θ(o_i,t | q, o_i,<t) / π_θold(o_i,t | q, o_i,<t)`：重要性采样比（逐 token）
- `Â_i,t`：组内相对优势（见下）
- `ε`：裁剪范围（PPO 同款，默认 0.2）
- `β`：KL 惩罚系数
- `D_KL[π_θ||π_ref]`：**无偏估计**（Schulman 2020 的 K1 估计器，DeepSeekMath Eq. 4，保证为正）：

```
D_KL[π_θ||π_ref] = π_ref(o)/π_θ(o) − log( π_ref(o)/π_θ(o) ) − 1
```

### 2.1 优势估计（组内归一化）

**结果监督（outcome supervision）**（DeepSeekMath §4.1.2）：

```
Â_i,t = ẽr_i = ( r_i − mean({r₁,…,r_G}) ) / std({r₁,…,r_G})
```

即：组内奖励减去均值、除以标准差；该输出的所有 token 共享同一个归一化优势值。

**过程监督（process supervision）**（§4.1.3）：每步奖励也做组内归一化 ẽr_i^index(j)，token 的优势 = 后续所有步骤的归一化奖励之和：

```
Â_i,t = Σ_{j: index(j) ≥ t} ẽr_i^index(j)
```

### 2.2 迭代式 GRPO（Algorithm 1）

```
输入：初始策略 π_θinit；奖励模型 r_φ；任务 prompts D；超参 ε, β, μ
1:  π_θ ← π_θinit
2:  for iteration = 1..I do
3:      π_ref ← π_θ                 # 参考策略 = 当前策略
4:      for step = 1..M do
5:          采样 batch D_b ~ D
6:          π_θold ← π_θ             # 冻结旧策略
7:          每个问题 q ∈ D_b 采样 G 个输出 {o_i} ~ π_θold(·|q)
8:          用 r_φ 给每个输出打分，得 {r_i}
9:          组内相对优势估计 Â_i,t
10:         for GRPO iteration = 1..μ do
11:             用 GRPO 目标更新 π_θ
12:         r_φ 用回放机制持续训练（含 10% 历史数据）
13: 输出 π_θ
```

关键点：
- **π_ref 每迭代刷新**为当前策略（不是固定初始模型）
- **奖励模型也持续训练**（replay 含 10% 历史数据），解决"旧奖励模型监督不了新策略"的问题
- `μ` = 一次采样后内层更新的轮数（DeepSeekMath-RL 里 μ=1，即采样一次只更新一步）

---

## 3. 实现步骤（工程伪代码）

```python
# 一次 GRPO 训练 step（以结果监督、批量梯度为例）

for batch_q in dataloader:
    # 1. 冻结旧策略 + 采样
    with torch.no_grad(), policy.old_mode():
        outputs = [policy.generate(q, num_return_sequences=G) for q in batch_q]

    # 2. 奖励（规则或 reward model）
    rewards = reward_fn(prompts, outputs)          # shape: [B, G]

    # 3. 组内优势归一化
    mean = rewards.mean(dim=1, keepdim=True)
    std  = rewards.std(dim=1, keepdim=True)
    advantages = (rewards - mean) / (std + 1e-4)   # 每组的 relative advantage

    # 4. 计算重要性比率与损失（每个 token）
    logprobs_new = policy.logprobs(prompt_tokens, output_tokens)
    logprobs_old = policy.old_logprobs(prompt_tokens, output_tokens)
    ratio = torch.exp(logprobs_new - logprobs_old)
    # clip 的目标：min(ratio*adv, clip(ratio, 1-eps, 1+eps)*adv)
    pg_loss = -torch.min(
        ratio * advantages_masked,
        torch.clamp(ratio, 1-eps, 1+eps) * advantages_masked,
    ).mean()

    # 5. KL 惩罚（无偏 K1 估计，加进损失而非奖励）
    ref_logprobs_logp = policy.ref_logprobs(...)
    delta = ref_logprobs_logp - logprobs_new  # log(π_ref / π_θ)
    kl = torch.expm1(delta) - delta
    # 等价写法: exp(delta) - delta - 1
    loss = pg_loss + beta * kl.mean()

    loss.backward()
    optimizer.step()
```

**落地注意（推导自三篇论文）**：
1. **sample-level vs token-level loss**：GRPO 原本是"每个样本先按 token 求平均，再跨样本平均"（`1/G Σ_i 1/|o_i| Σ_t`）——长样本权重被稀释；DAPO 改成"所有 token 加总后除总 token 数"（`1/Σ|o_i| Σ_i Σ_t`）。长 CoT 训练建议直接上 token-level。
2. **KL 估计**：第一项 `π_ref/π_θ − 1` 是对数形式的无偏估计，配合 `log(π_ref/π_θ)` 不对称惩罚项；数值上注意 `expm1` 防溢出。
3. **掩码**: padding、截断部分要 mask；DAPO 对超长截断样本默认直接 mask 掉 loss（Overlong Filtering）。
4. **group 归一化只在组内做**（同 prompt 的 G 个输出），跨 prompt 不比较——这是"组相对"的含义。

---

## 4. DeepSeek-R1 怎么用 GRPO（2501.12948v2 原文）

### 4.1 目标与奖励

- 用 GRPO 代替 PPO（附 A.3 有专门对比：PPO 需 value model + GAE；在 MATH 任务上 PPO 明显差，λ=1.0 调优后才接近 GRPO）
- **奖励全部为规则奖励**（rule-based，无 reward model）：
  - Accuracy reward：答案判等（数学用严格判等、代码用可执行测试）等
  - Format reward：模型被要求输出 `<think>...</think><answer>...</answer>` 结构
  - 语言一致性奖励（第一 RL 阶段）：`Reward_lang = Num(word_target)/Num(words)`，抑制 CoT 多语混用
- R1-Zero 训练模板：系统提示要求先思考再作答，这是"无监督式长 CoT 涌现"的关键触因

### 4.2 训练超参（R1-Zero，论文原文）

| 超参 | 值 |
|---|---|
| 学习率 | 3e-6 |
| KL 系数 β | 0.001 |
| GRPO clip ε | **10**（论文原文如此；注意多数开源复现如 veRL/TRL 默认 0.2，见 DAPO 对该差异的讨论） |
| 采样温度 | 1.0 |
| 每问采样数 G | 16 |
| 最大长度 | 32,768 tokens（8.2k step 后 65,536） |
| 每步问题数 / batch | 32 问 / 512 |
| 参考模型刷新频率 | 每 400 步 π_ref ← π_θ |
| Rollout 效率 | 每 step 生成 8,192 条输出 → 随机拆 16 个 minibatch → 单 inner epoch |
| 训练规模 | 共 10,400 步 ≈ 1.6 epoch |

### 4.3 关键的训练观察

- 最大长度在 8.2k 步时从 32K 升到 64K，性能与响应长度都出现显著跳升——说明**长度余量本身是长 CoT 涌现的推手**
- 每个 rollout 生成 8192 输出比"32×16=512"多得多，是**动态划分 minibatch**（内存受限时多生成、分批训）
- clip ratio 的角色：值低会导致大量 token 梯度被截断（性能下降），值高会导致训练不稳定（§3.2.1 原文）

---

## 5. DAPO：GRPO 的四个改进（2503.14476v2）

DAPO（Decouple Clip and Dynamic sAmpling Policy Optimization）基于 verl 框架，Qwen2.5-32B base 训练，AIME 2024 avg@32：naive GRPO 30 → DAPO **50**（DeepSeek-R1-Zero-Qwen-32B 是 47，且 DAPO 只需其一半训练步数）。

目标式（去掉 KL 项；注意是 token-level 归一化）：

```
J_DAPO(θ) = E[..., {o_i}~π_θold] [ 1/Σ_i|o_i| Σ_i Σ_t min( r_i,t·Â_i,t, clip(r_i,t, 1-εlow, 1+εhigh)·Â_i,t ) ]
           s.t. 0 < #{o_i 正确} < G      # 动态采样的约束
```

| 改进 | 问题 | 做法 |
|---|---|---|
| **Clip-Higher** | ε=0.2 时低概率"探索 token"被上界卡死（π_θ ≤ π_θold×1.2），熵塌缩 | 解耦上下界：维持 εlow=0.2，上调 εhigh=**0.28**（论文实验值） |
| **Dynamic Sampling** | 全对（acc=1）的组优势为 0 → 零梯度，有效 prompt 越来越少 | 过采样并**丢弃 acc=1 和 acc=0** 的组，填满 batch 为止 |
| **Token-level Loss** | sample-level 平均使长样本的 token 贡献被稀释 | `1/Σ_i|o_i|` 归一化，每个 token 权重均等 |
| **Overlong Reward Shaping** | 截断样本给惩罚 → 奖励噪声，好推理只因超长被罚 | ① Overlong Filtering：mask 截断样本 loss；② Soft Overlong Punishment：长度感知线性惩罚（设 L_max=16,384，soft cache=4,096，生成上限 20,480） |

**Soft Overlong Punishment 公式（Eq. 13）**：

```
R_length(y) = 0                                  (|y| ≤ L_max − L_cache)
            = (L_max − L_cache − |y|)/L_cache    (L_max − L_cache < |y| ≤ L_max)
            = −1                                  (|y| > L_max)
```

**DAPO 训练超参**：AdamW，lr=1e-6（linear warmup 20 rollout steps）；prompt batch 512，每问 16 响应；minibatch 512（每 rollout step 16 次梯度更新）；生成上限 20,480 tokens。

**数据集**：DAPO-Math-17K（17K 数学题，答案为整数——统一答案格式使规则判等可行）。

---

## 6. 三篇论文超参速查表

| 参数 | DeepSeekMath-RL (7B) | DeepSeek-R1-Zero | DAPO (Qwen-32B) |
|---|---|---|---|
| Policy 学习率 | 1e-6 | 3e-6 | 1e-6 |
| KL 系数 β | 0.04 | 0.001 | 移除 |
| Clip ε | 0.2（默认语义） | **10**（原文如此） | εlow=0.2 / εhigh=0.28 |
| 组大小 G | 64 | 16 | 16 |
| 采样温度 | — | 1.0 | 1.0（评估 topp=0.7） |
| 最大长度 | 1,024 | 32,768 → 65,536 | 16,384+4,096 cache=20,480 |
| Batch | 1,024 | 512（32 问×16） | 512（512 问×16 拆分） |
| 奖励来源 | Reward model（可迭代训练） | 纯规则（acc+format+lang） | 纯规则（acc=1/-1） |

---

## 7. 参考实现与复现入口

- **verl**（DAPO 官方框架，字节）：Volcano Engine Reinforcement Learning，支持 GRPO/DAPO
- **TRL**（HuggingFace）：`GRPOTrainer`（`trl.trainer.grpo_trainer`），低成本复现 R1
- **OpenRLHF**：Ray 分布式实现 PPO/GRPO/DPO
- **DeepSeekMath 官方**：github.com/deepseek-ai/DeepSeekMath（含 GRPO 算法实现）
- 论文算法伪代码：DeepSeekMath Algorithm 1（上文 §2.2）、DAPO Algorithm 1

---

## 8. 作者与出处（本文所有公式/超参的原始来源）

| 论文 | arXiv | 版本 | 角色 |
|---|---|---|---|
| DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | 2402.03300 | v3 (2024-04-27) | **GRPO 首次提出**（§4.1，Eq.3/4/21，Algorithm 1） |
| DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | 2501.12948 | v2 (2026-01-04) | GRPO 大规模应用（§2.1，§3.2，附录 A.3 PPO 对比） |
| DAPO: An Open-Source LLM Reinforcement Learning System at Scale | 2503.14476 | v2 (2025-05-20) | GRPO 四大改进（§2-3，Algorithm 1） |

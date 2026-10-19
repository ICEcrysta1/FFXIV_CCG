"""BC 与 GRPO 共用的 TensorBoard 中文展示标签，不改动训练指标键。"""

from __future__ import annotations

_PREFIX_LABELS = {
    "train/step": "训练/记录区间",
    "train/epoch": "训练/每轮",
    "grpo/update": "GRPO/优化更新",
    "grpo/iteration": "GRPO/每轮迭代",
    "grpo/rollout": "GRPO/回放",
    "grpo/model": "GRPO/模型输出",
    "train": "训练",
    "validation": "验证",
    "rollout": "回放",
    "diagnostics": "诊断",
    "run": "运行信息",
    "embeddings": "嵌入向量",
    "grpo": "GRPO",
}

_METRIC_LABELS = {
    "loss": "总损失",
    "cross_entropy_loss": "交叉熵损失",
    "value_preference_loss": "动作价值偏好损失",
    "top1_accuracy": "Top-1准确率",
    "top3_accuracy": "Top-3准确率",
    "learning_rate": "本次更新学习率",
    "next_learning_rate": "下次更新学习率",
    "gradient_norm": "裁剪前梯度范数",
    "grad_norm": "裁剪前梯度范数",
    "val_ppg": "真实副本每GCD平均威力（val_ppg）",
    "none_ppg": "空场景每GCD平均威力（none_ppg）",
    "val_ppg_normalized": "归一化真实副本每GCD平均威力（val_ppg_normalized）",
    "none_ppg_normalized": "归一化空场景每GCD平均威力（none_ppg_normalized）",
    "top1_val_ppg_average": "Top-1与归一化副本PPG均值",
    "logits": "动作原始得分",
    "probabilities": "动作预测概率",
    "predicted_action": "预测动作编号",
    "label_action": "标签动作编号",
    "input_skills": "输入技能表",
    "output_actions": "输出动作头",
    "confusion_matrix": "动作混淆矩阵（按真实动作归一化）",
    "confusion_counts": "动作混淆计数",
    "support": "样本数",
    "recall": "召回率",
    "precision": "精确率",
    "diagnostic_samples": "有效诊断样本数",
    "diagnostic_rejected_samples": "拒绝诊断样本数",
    "pr_sample_count": "PR采样数",
    "nonfinite_samples": "非有限值采样数",
    "graph_error": "模型结构图写入错误",
    "configuration": "配置",
    "model": "模型结构",
    "metric_semantics": "指标说明",
    "iteration": "迭代轮次",
    "policy": "策略",
    "policy_loss": "策略损失",
    "kl": "KL散度",
    "entropy": "策略熵",
    "clip_fraction": "概率比裁剪占比",
    "ratio": "新旧策略概率比",
    "mean_ratio": "平均概率比",
    "legal_logits": "合法动作原始得分",
    "legal_probabilities": "合法动作概率",
    "selected_probability": "已选动作概率",
    "greedy_ppg_before": "更新前贪心PPG",
    "greedy_ppg_after": "更新后贪心PPG",
    "greedy_ppg_after_update": "更新后贪心PPG（回滚前）",
    "best_greedy_ppg": "最佳贪心PPG",
    "sample_ppg": "采样PPG",
    "sample_ppg_mean": "采样PPG均值",
    "reward_delta": "相对基准奖励差",
    "reward_delta_mean": "相对基准奖励差均值",
    "reward_delta_std": "相对基准奖励差标准差",
    "trajectory_advantage": "轨迹优势值",
    "advantages": "优势值",
    "advantage_mean": "优势值均值",
    "advantage_std": "优势值标准差",
    "advantage_abs_max": "优势值绝对值最大值",
    "old_logprob": "旧策略已选动作对数概率",
    "action_index": "已选动作编号",
    "actions": "总决策数",
    "decision_count": "轨迹决策数",
    "rollout_max_duration_seconds": "回放时长上限（秒）",
    "greedy_guard_rollback": "贪心表现保护回滚标记",
}

_PARAMETER_LABELS = {
    "encoder": "编码器",
    "input_encoder": "输入编码器",
    "skill_embed": "技能嵌入",
    "skill_feat_proj": "技能数值特征投影",
    "state_proj": "状态投影",
    "state_null_proj": "状态空值掩码投影",
    "state_reset_proj": "状态重置掩码投影",
    "scene_proj": "场景投影",
    "role_embed": "Token角色嵌入",
    "output_head": "输出动作头",
    "self_attn": "自注意力",
    "gate_proj": "前馈门控投影",
    "linear1": "前馈输入投影",
    "linear2": "前馈输出投影",
    "q_proj": "查询投影Q",
    "k_proj": "键投影K",
    "v_proj": "值投影V",
    "out_proj": "输出投影",
    "in_proj_weight": "查询键值联合投影权重",
    "in_proj_bias": "查询键值联合投影偏置",
    "norm": "归一化",
    "norm1": "注意力归一化",
    "norm2": "前馈归一化",
    "residual_mix": "残差混合",
    "r": "残差尺度r",
    "a": "初始Token注入系数a",
    "attention_residual": "注意力残差",
    "pseudo_queries": "深度查询参数",
    "key_norms": "键归一化",
    "weight": "权重",
    "bias": "偏置",
}


def _parameter_label(name: str) -> str:
    parts = name.split(".")
    labels = []
    # 层号和其他 ModuleList 索引均来自实际参数路径，不依赖配置中的层数。
    if len(parts) >= 3 and parts[:2] == ["encoder", "layers"] and parts[2].isdigit():
        labels.append(f"编码器第{int(parts[2]) + 1}层")
        parts = parts[3:]
    labels.extend(
        f"第{int(part) + 1}项" if part.isdigit() else _PARAMETER_LABELS.get(part, part)
        for part in parts
    )
    return "/".join(labels) + f"（{name}）"


def tensorboard_tag(tag: str) -> str:
    """只在事件写入边界翻译名称；动态动作标识和未知字段原样保留。"""
    if tag == "epoch/index":
        return "训练/轮次"
    for prefix, label in (("gradients/", "梯度"), ("parameters/", "参数")):
        if tag.startswith(prefix):
            return f"{label}/{_parameter_label(tag[len(prefix):])}"
    if tag.startswith("diagnostics/nonfinite_samples/"):
        source = tag.removeprefix("diagnostics/nonfinite_samples/")
        return f"诊断/非有限值采样数/{tensorboard_tag(source)}"
    if tag.startswith("validation/actions/"):
        action, separator, metric = tag.removeprefix("validation/actions/").rpartition("/")
        if not separator:
            return f"验证/动作/{metric}/精确率与召回率曲线"
        return f"验证/动作/{action}/{_METRIC_LABELS.get(metric, metric)}"
    for prefix, label in _PREFIX_LABELS.items():
        if tag.startswith(prefix + "/"):
            suffix = tag[len(prefix) + 1:]
            return label + "/" + "/".join(
                _PREFIX_LABELS.get(part, _METRIC_LABELS.get(part, part))
                for part in suffix.split("/")
            )
    return _METRIC_LABELS.get(tag, tag)

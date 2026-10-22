# 训练

路径相对仓库根目录。返回[项目目录](./index.md)。

`training/` 负责 BC 预训练的数据读取、损失、训练循环与 checkpoint 生命周期。跨 BC、GRPO、回放和部署共用的策略模型、输入编码、优化器及监控组件见[公共](./common.md)。

BC 续训恢复 checkpoint 的学习率、优化器状态和 scheduler 步号，权重衰减使用当前训练配置的 `weight_decay`，使 `optimizer.yaml` 中的调整在续训时生效。

## 目录与职责

```text
FFXIV_CCG/
└── training/                                         预训练专属数据管线、训练循环与训练配置；不再承载 GRPO/回放/ONNX 共用的策略模型和输入数据组件
    ├── config/                                       预训练运行配置、价值偏好配置和训练解析器
    │   ├── __init__.py                               训练运行配置、价值偏好与模型配置加载入口
    │   └── action_quality.py                         动作质量损失曲线与职业严重度权重的配置解析
    ├── data/                                         预训练数据管线目录，只消费最终 compiled cache，不负责 raw 转换或 cache 构建
    │   ├── __init__.py                               预训练数据管线导出入口，仅暴露 dataset / collator / sampler / oversampling
    │   ├── collator.py                               训练 collator，按显式 `history_bank_id` 合并 compact history 引用并完成 历史 / scene 的 batch padding，stack 当前状态和固定动作监督
    │   ├── dataset.py                                训练数据集，负责 cache schema 校验、skill vocab 映射、shard 读取、历史窗口裁剪和 structured tensor 输出
    │   ├── oversampling.py                           可选职业 YAML 变长动作序列过采样规则加载器与样本权重构建器
    │   └── sampler.py                                shard-aware batch sampler，按已编译缓存 shard 组织训练/验证 batch
    ├── loop/                                         数据加载、训练循环和 checkpoint 生命周期
    │   ├── losses/                                   行为克隆各类别损失实现，供统一装配入口调用
    │   │   ├── __init__.py                           损失实现子包入口
    │   │   ├── action_quality.py                     按样本错误最大配置权重与精确排名计算主/辅助损失共用权重
    │   │   ├── primary.py                            主交叉熵损失，支持逐样本动作质量权重
    │   │   └── value_preference.py                   技能价值辅助排序损失，对同一目标应用动作质量权重
    │   ├── __init__.py                               预训练训练流程导出入口
    │   ├── checkpoint.py                             checkpoint 保存、续训契约校验、最佳指标、scheduler/RNG 恢复和安全加载
    │   ├── dataloaders.py                            独立训练/验证缓存读取、职业动作价值定义校验、normalizer/collator、过采样、shard sampler 和 DataLoader 构造
    │   ├── epochs.py                                 单轮训练与验证，负责批次编码、损失、梯度/参数更新及样本加权指标，保留训练诊断与 TensorBoard 批次监控
    │   ├── loss.py                                   统一装配主损失和可选辅助损失，提供权重、开关与分项指标
    │   ├── progress.py                               BC 训练/验证进度显示：累计指标表头与数值列、显存和批次，按最近 5 秒估算速度与剩余时间，验证附加指标沿用同宽表格
    │   └── training_loop.py                          BC 训练编排：模型/optimizer/scheduler 初始化、续训恢复、CUDA 编译前后向预热、逐轮调用、PPG 回调、监控与 checkpoint 生命周期；续训步号沿用 scheduler
    ├── __init__.py                                   训练包导出入口，仅暴露预训练 dataset / collator / sampler / oversampling
    ├── runtime.py                                    batch、输入编码、Attention/FFN、反向和优化器阶段的显存/耗时诊断
    └── train.py                                      通用行为克隆训练入口，只读选择已编译数据；支持 `python3 training/train.py` 和 `python3 -m training.train`
```

## 相关入口

- 模型变体配置、compiled cache 数据目录与输入契约见[公共](./common.md)。
- cache 编译、PPG 评估、训练菜单与 TensorBoard 启动见[脚本](./scripts.md)。
- 后训练见[GRPO](./grpo.md)，回归覆盖见[tests](./tests.md)。

# GRPO

路径相对仓库根目录。返回[项目目录](./项目目录.md)。

`grpo/` 负责强化学习后训练，包含 rollout 采样、轨迹落盘、策略更新与 checkpoint 调度，与 BC 预训练循环分离。

## 目录与职责

```text
FFXIV_CCG/
└── grpo/                                             独立 GRPO 强化学习后训练包，与预训练训练循环和模型配置实现分离
    ├── __init__.py                                   GRPO 配置导出入口
    ├── __main__.py                                   `python3 -m grpo` 模块入口
    ├── checkpoint.py                                 GRPO checkpoint 保存、CPU 参数与优化器快照、贪心保护回滚；保持原断点格式和恢复边界
    ├── config.py                                     GRPO 参数定义、校验与 YAML 配置读取
    ├── constants.py                                  GRPO 优势归一化与对数概率计算使用的固定数值容差
    ├── grpo.py                                       GRPO 后训练命令行编排，只读选择已编译场景，读取 checkpoint 与 GRPO 专属配置
    ├── rollout.py                                    单场景采样、决策 tensor 收集与脱离推理模式、PPG 统计；采样阶段不构建梯度图
    ├── storage.py                                    GRPO rollout 的原子磁盘写入、manifest 与按 minibatch 流式读取；保存死亡输入并校验可见动作边界及顺序
    ├── trainer.py                                    GRPO 训练编排：模型与资源初始化、组内采样与奖励标注、贪心保护、迭代指标及 checkpoint 调度；轨迹 tensor 由 `storage.py` 持久化
    └── update.py                                     组内相对优势、决策 batch 装配、裁剪策略损失/KL 与参数更新；统一从磁盘 Store 流式读取并保留采样顺序
```

## 相关入口

- 共享策略模型、输入契约、优化器、监控及模型变体配置见[公共](./公共.md)。
- 多队列引擎与死亡、复活规则见[状态机](./状态机.md)，共享回放调度与 PPG 见[脚本](./脚本.md)。
- BC 预训练见[训练](./训练.md)，GRPO 回归覆盖见[tests](./tests.md)。

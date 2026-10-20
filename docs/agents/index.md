# 项目目录

`docs/agents/` 下的文档供智能体按任务查阅。本文件仅提供分类索引，具体目录、文件职责和运行边界由对应分类维护；路径均相对仓库根目录。

| 分类 | 文档 | 范围 |
| --- | --- | --- |
| 状态机 | [状态机](./fight-engine.md) | `fight-engine/` 核心、独立职业 crate 与 Python 原生扩展，引擎与队列生命周期、死亡复活、历史保留 |
| 训练 | [训练](./training.md) | `training/`，BC 数据管线、损失、训练循环与 checkpoint |
| GRPO | [GRPO](./grpo.md) | `grpo/`，rollout、轨迹存储、策略更新与后训练编排 |
| 脚本 | [脚本](./scripts.md) | `scripts/`、`ffxiv_ccg.sh`、`main.py`、`setup.sh`，下载、标注、转换、回放、分析与导出 |
| 公共 | [公共](./common.md) | `common/`、`config/`、共享数据与产物目录、根目录公共文件，策略模型与输入契约 |
| tests | [tests](./tests.md) | `tests/`、`fight-engine/` 核心、职业和原生绑定 Rust 测试、pytest 配置与测试产物 |

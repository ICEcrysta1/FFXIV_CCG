## 准则

- 保持项目结构的清晰和无多余代码，不要让 A 模块干 B 模块的事情，模块应该专职。
- 根目录 `common/` 只放跨顶层模块复用、无具体业务归属的公共函数；模块内部共享 helper 留在所属模块，不因为“看起来通用”就上移。
- 根目录 `.env` 统一通过 `python-dotenv` 加载，禁止各模块维护自定义 `.env` 解析器；系统配置已有的运行时参数必须以 YAML 为权威来源，不在脚本常量中重复维护。
- `README.md` 顶部保留项目展示区域；`preview/` 只承载 README 和项目展示资源，不承载业务代码，目录内文件无需逐个写入架构说明。
- 使用中文思考和回答，思考过程中发现什么问题先报告，再继续推进。
- 你是一个用中文思考的助手。在分析任何问题时，先用中文建立思维框架，再执行具体步骤。
- 使用中文注释
- 每次需要更新变更说明时，统一修改 [CHANGELOG.md](./CHANGELOG.md)；如果没有历史 tag，就先维护 `Unreleased`。
- 修改 changelog 前，先看 git 历史里的最新 tag。
- 如果最新 tag 还是上一版，就不要乱编新版本号；先以最新 tag 为基准决定下一个版本，或者继续写在 `Unreleased`。
- 默认不允许顺手 `git commit`，`git push`，除非我要求了。
- commit的时候保持提交信息用fix: docs: feat:等清晰的前缀，然后commit内容使用中文。
- 不允许commit到main，除了文档更新，其余代码变更严禁推送到main，commit之前需要检查，如果检查会推送到main需要停止commit并且告知用户，如果用户执意推送到main，你需要拒绝。

### 例外

- 只有当我在当前对话里明确要求“push / 推送 / 发布到远程仓库”，并且模型已经确认当前分支、远程仓库和待推送 commit 后，才可以执行 `git push`。
- main上打tag不受影响，可以执行修改CHANGELOG之后commit并且tag，其他时候不可以随便commit到main，tag必须和commit一起发布才行

#### 执行 push 前必须先展示

1. 当前分支
2. 远程仓库
3. 将要推送的 commit
4. 要执行的完整命令

- 除非我明确说“直接推送”，否则先等待我确认。
- 测试文件需要跟随代码修改一起更新；如果改了实现，就要同步检查相关测试是否仍然对得上。

## 环境要求

- 文件编码：UTF-8
- 换行：LF
- 不要依赖 PowerShell 或编辑器的默认编码
- 使用项目.venv作为py正式环境

## 发布文件约定

- 默认只修改代码和测试，不要因为修一个普通问题就自动改发布说明或版本号。
- 只有当用户明确要求 `update changelog`、`bump version`、`prepare release`、`create release notes` 这类发布动作时，才修改发布相关文件。
- 如果只是普通修复，优先在回复里说明“可选：需要时再补 changelog”，不要替用户直接决定发布节奏。


## 项目架构

- 详细看 [项目各文件说明.md](./docs/项目各文件说明.md) 每次新会话时请自动阅读了解项目结构
- 修改 [项目各文件说明.md](./docs/项目各文件说明.md) 时内部项目结构的时候保持首字母排序，顶部文件夹底部文件的方式修改

## 训练指标与缓存契约

- compiled cache 编译阶段固定保存完整 history bank；`model.history_capacity` 只属于模型/训练分析读取窗口或在线回放输入，不得进入 cache signature、编译 worker、raw source 后端或 history bank 构建参数。
- 修改 `model.history_capacity` 只应改变读取侧的 history window；旧格式 cache 因格式版本升级可以一次性重建，但同一 raw source 的后续窗口调整必须复用同一份 compiled cache。

- 技能 token 不保留原始 `kind` 字符串：状态机内部仍使用 `gcd` / `ogcd` 语义，输出 token 只写数值维度，`gcd=1`、`ogcd=0`。该维度同时进入模型 `skill_features`，并由验证 PPG 统计历史 GCD 数。
- `val_ppg` 是模型在真实验证副本上的自回归表现：每个验证 source 从首个缓存样本的初始状态和完整 scene token 开始，由模型 Top-1 自循环到最后一个 `targetable=true` Boss 窗口结束；按该次回放的 `(历史 GCD 直接威力 + 历史 oGCD 直接威力 + 历史累计 DoT 威力) / 执行 GCD 数` 得到单副本 PPG，最后对副本平均。候选技能伤害、候选预览、label 和 teacher-forced logits 不参与该指标。回放中候选全非法时先由 `DecisionScheduler` 推进到下一可决策事件；若已无法推进，则该副本按正式失败语义返回全 0（包括此前已执行的伤害），仍计入副本平均。“跳过本副本并记 0”只表示返回零结果，不表示从平均值排除。
- `none_ppg` 是模型相关指标：训练验证完成后，使用当前 checkpoint 在空场景中由 C# 状态机驱动自回归 Top-1 回放，再按累计直接/DoT 威力除以执行 GCD 数计算。`top1` / `top3` 仍单独使用人类前上下文评估。
- compiled cache 的 v9 / 转换版本 v8 继续携带历史 bank 中已有的技能直接威力、累计 DoT 威力和数值化 `kind` 维度，并固定保存完整 history bank；验证 PPG 从现有 cache reader 和 scene 自行恢复回放所需信息，不为副本时长或基础 GCD 增加重复 manifest 字段。状态机语义变化导致历史状态转移结果不可复用时，必须提升转换版本并让旧 cache 自动回到重编译路径。

## Combat.Sim 部署契约

转换与自回归回放通过 Python.NET 在当前 Python 进程内调用 C# 状态机。修改
`FightEngine`、`PythonBridge` 或 `config/schema.yaml` 后，必须在同一工作树重建桥接程序集：

```powershell
dotnet build Combat.Sim/PythonBridge/PythonBridge.csproj --configuration Debug
```

Python 客户端会将程序集嵌入的契约版本与 `config/schema.yaml`（当前 `sidecar_contract_version: 8`）比较，拒绝旧 DLL；新版 DLL 使用缺少
`contracts.scene_epsilon` 或 `contracts.sidecar_contract_version` 的旧 schema 则会在 C# 配置加载时失败。
FightEngine DLL、`config/schema.yaml` 与输出 token 契约必须作为同一版本构建和使用。自回归回放把移动事实直接提交给状态机，训练样本转换仍可在输出层合成移动字段。

职业状态机的合法性或状态转移语义发生变化时，必须检查所有受影响的边界并同步升级契约：`sidecar_contract_version`、checkpoint 输入契约（`INPUT_CONTRACT_VERSION`）、compiled cache 转换版本（`CACHE_FORMAT` / `DEFAULT_CONVERSION_VERSION`）和 ONNX deployment contract（`DEPLOYMENT_CONTRACT_VERSION`，以及 `manifest.schema.json` 里对应的 `const`）；同时补充 Python/C# 状态机回归测试，并重建 PythonBridge、缓存、checkpoint 和部署包，禁止旧产物静默复用。

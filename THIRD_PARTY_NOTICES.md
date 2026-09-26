# 第三方软件声明

本文件记录第三方组件的来源、使用范围及许可声明。项目自有代码的许可见
[LICENSE](./LICENSE)；第三方组件保留各自的版权和许可。

## xivanalysis

- 上游：<https://github.com/xivanalysis/xivanalysis>
- 位置：`third_party/xivanalysis/`（Git 子模块）
- 版本：由主仓库记录的子模块 commit 固定，可通过 `git submodule status` 查看。
- 使用范围：由 `scripts/action_quality/` 的离线桥接调用，分析 raw 日志并导出结构化 JSON；训练和部署不直接导入该组件。
- 许可证：MIT，原始文件见 [third_party/xivanalysis/LICENSE](./third_party/xivanalysis/LICENSE)。

以下保留上游完整版权声明与许可文本。分发包含该组件代码的产物时，须一并保留该声明。
组件所使用的其他第三方依赖仍遵循各自的许可证。

```text
MIT License

Copyright (c) 2018 Saxon Landers & contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

#!/usr/bin/env bash
# 项目入口只定位正式 Python 环境，业务编排交由 Python 模块。
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python3"
cd "$PROJECT_ROOT"
export PYTHONUTF8=1
# 帮助只使用标准库，允许尚未安装项目环境时查看入口参数。
for argument in "$@"; do
    if [[ "$argument" == --help || "$argument" == -h ]]; then
        exec python3 -m scripts.ffxiv_ccg "$@"
    fi
done
if [[ ! -x "$PROJECT_PYTHON" ]]; then
    printf '找不到项目 Python：%s，请先运行 ./setup.sh。\n' "$PROJECT_PYTHON" >&2
    exit 1
fi
exec "$PROJECT_PYTHON" -m scripts.ffxiv_ccg "$@"

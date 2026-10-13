#!/usr/bin/env bash
# 准备 Linux／WSL2 环境；根目录 .env 始终由 Python 的 python-dotenv 加载。
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for argument in "$@"; do
    case "$argument" in
        --help|-h)
            printf '用法：./setup.sh [--help]\n准备 Python 3.12+、项目 Node 22/pnpm 10、.NET 10 桥接与既有 GPU 依赖。\n'
            exit 0 ;;
        *) printf '未知参数：%s\n' "$argument" >&2; exit 2 ;;
    esac
done
cd "$PROJECT_ROOT"
export PYTHONUTF8=1
[[ "$(uname -s)" == Linux ]] || { printf '开发与运行环境要求 Linux／WSL2。\n' >&2; exit 1; }
for executable in python3 dotnet git curl tar xz sha256sum; do
    command -v "$executable" >/dev/null || { printf '缺少命令：%s，请先安装后重新运行。\n' "$executable" >&2; exit 1; }
done
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "需要 Python 3.12 或更新版本")'
dotnet --list-sdks | awk '$1 ~ /^10\./ { found=1 } END { exit !found }' || {
    printf '需要 .NET 10 SDK，请先安装后重新运行。\n' >&2
    exit 1
}
if [[ ! -e .venv ]]; then
    python3 -m venv .venv
fi
PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python3"
[[ -x "$PROJECT_PYTHON" ]] || { printf '.venv 不是有效的 Linux 虚拟环境，请移走后重新运行。\n' >&2; exit 1; }
"$PROJECT_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else ".venv 需要 Python 3.12 或更新版本")'
"$PROJECT_PYTHON" -m pip install --upgrade pip
"$PROJECT_PYTHON" -m pip install -r requirements.txt
if "$PROJECT_PYTHON" -m pip show onnxruntime >/dev/null 2>&1; then
    "$PROJECT_PYTHON" -m pip uninstall -y onnxruntime
fi
"$PROJECT_PYTHON" -m pip install -r requirements-onnx-gpu.txt
NODE_HOME="$PROJECT_ROOT/.node"
NODE_RUNTIME="$NODE_HOME/runtime"
[[ "$(tr -d '\r\n' < .node-version)" == 22 ]] || { printf '.node-version 当前要求为 22。\n' >&2; exit 1; }
case "$(uname -m)" in
    x86_64) NODE_ARCH=x64 ;;
    aarch64|arm64) NODE_ARCH=arm64 ;;
    *) printf '不支持的 Node 架构：%s\n' "$(uname -m)" >&2; exit 1 ;;
esac
if [[ ! -d "$NODE_RUNTIME" ]]; then
    mkdir -p "$NODE_HOME"
    NODE_STAGE="$(mktemp -d "$NODE_HOME/stage.XXXXXX")"
    trap 'rm -rf -- "$NODE_STAGE"' EXIT
    NODE_BASE_URL=https://nodejs.org/dist/latest-v22.x
    curl --fail --location --retry 3 "$NODE_BASE_URL/SHASUMS256.txt" -o "$NODE_STAGE/SHASUMS256.txt"
    NODE_ARCHIVE="$(awk -v arch="$NODE_ARCH" '$2 ~ ("^node-v22\\.[0-9]+\\.[0-9]+-linux-" arch "\\.tar\\.xz$") { print $2; exit }' "$NODE_STAGE/SHASUMS256.txt")"
    [[ -n "$NODE_ARCHIVE" ]] || { printf '官方校验清单中找不到 Linux Node 22 归档。\n' >&2; exit 1; }
    curl --fail --location --retry 3 "$NODE_BASE_URL/$NODE_ARCHIVE" -o "$NODE_STAGE/$NODE_ARCHIVE"
    awk -v archive="$NODE_ARCHIVE" '$2 == archive' "$NODE_STAGE/SHASUMS256.txt" > "$NODE_STAGE/selected.sha256"
    (cd "$NODE_STAGE" && sha256sum --check selected.sha256)
    tar -xJf "$NODE_STAGE/$NODE_ARCHIVE" -C "$NODE_STAGE"
    mv -- "$NODE_STAGE/${NODE_ARCHIVE%.tar.xz}" "$NODE_RUNTIME"
    rm -rf -- "$NODE_STAGE"
    trap - EXIT
fi
NODE="$NODE_RUNTIME/bin/node"
[[ -x "$NODE" ]] || { printf '项目 Node 安装不完整：%s\n' "$NODE_RUNTIME" >&2; exit 1; }
[[ "$("$NODE" --version)" == v22.* ]] || { printf '项目 Node 版本必须为 22.x。\n' >&2; exit 1; }
export PATH="$NODE_RUNTIME/bin:$PATH"
export npm_config_cache="$NODE_HOME/npm-cache"
PNPM_VERSION=10.0.0
PNPM_HOME="$NODE_HOME/pnpm/$PNPM_VERSION"
PNPM_CLI="$PNPM_HOME/lib/node_modules/pnpm/bin/pnpm.cjs"
if [[ ! -f "$PNPM_CLI" ]]; then
    "$NODE" "$NODE_RUNTIME/lib/node_modules/npm/bin/npm-cli.js" install --global --prefix "$PNPM_HOME" "pnpm@$PNPM_VERSION" --ignore-scripts --no-audit --no-fund
fi
[[ "$("$NODE" "$PNPM_CLI" --version)" == "$PNPM_VERSION" ]] || { printf '项目 pnpm 版本不匹配。\n' >&2; exit 1; }
if [[ ! -f third_party/xivanalysis/package.json ]]; then
    git submodule update --init -- third_party/xivanalysis
fi
"$NODE" "$NODE_RUNTIME/lib/node_modules/npm/bin/npm-cli.js" ci --ignore-scripts --no-audit --no-fund
"$NODE" "$PNPM_CLI" --dir "$PROJECT_ROOT/third_party/xivanalysis" install --frozen-lockfile --store-dir "$NODE_HOME/pnpm-store"
dotnet build Combat.Sim/PythonBridge/PythonBridge.csproj --configuration Debug
"$PROJECT_PYTHON" -c 'import torch, onnxruntime; from pythonnet import load; load("coreclr"); print("torch:", torch.__version__, "CUDA:", torch.cuda.is_available()); print("ONNX providers:", onnxruntime.get_available_providers())'
if [[ ! -f .env ]]; then
    printf '请参考 .env.example 创建根目录 .env：cp .env.example .env\n'
fi
printf '环境准备完成，运行 ./ffxiv_ccg.sh 打开操作菜单。\n'

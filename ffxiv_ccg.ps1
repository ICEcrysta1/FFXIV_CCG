# FFXIV_CCG：项目常用工具统一入口（全部通过项目 .venv 的 Python 执行）
#
# 交互菜单：
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\ffxiv_ccg.ps1
#
# 无交互调用：
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\ffxiv_ccg.ps1 -Action train
#
# 脚本本身不保存任何业务参数：模型路径、精度、opset、Provider、回放解码和门禁配置
# 全部来自 config/ 下的 YAML 与根目录 .env。

[CmdletBinding()]
param(
    [ValidateSet(
        "menu",
        "train",
        "grpo",
        "export",
        "analysis",
        "replay",
        "fflogs"
    )]
    [string]$Action = "menu"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ScraperScript = Join-Path $ProjectRoot "scripts\fflogs_scraper.py"

function Show-Menu {
    Write-Host ""
    Write-Host "FFXIV_CCG 常用工具菜单" -ForegroundColor Cyan
    Write-Host "所有命令都使用项目 .venv，参数读取 config/ 与根目录 .env。"
    Write-Host ""
    Write-Host "  1. 训练（BC 预训练）"
    Write-Host "  2. GRPO 后训练"
    Write-Host "  3. ONNX 导出（导出 + PT/ORT parity 门禁 + 发布校验）"
    Write-Host "  4. 模型分析图生成（不含损失地形图）"
    Write-Host "  5. 模型自回归回放"
    Write-Host "  6. FFLogs 数据下载"
    Write-Host "  0. 退出"
    Write-Host ""
    $choice = Read-Host "请输入选项编号"
    $resolvedChoice = switch ($choice) {
        "1" { "train" }
        "2" { "grpo" }
        "3" { "export" }
        "4" { "analysis" }
        "5" { "replay" }
        "6" { "fflogs" }
        "0" { "exit" }
        default { throw "未知选项：$choice" }
    }
    return $resolvedChoice
}

function Invoke-FFLogsDownload {
    Write-Host ""
    Write-Host "可粘贴 FFLogs 报告 URL（含 ?fight=..&source=.. 时效果最完整），或只填报告码。"
    $target = (Read-Host "请输入 FFLogs 报告 URL 或报告码（直接回车取消）").Trim()
    if ([string]::IsNullOrWhiteSpace($target)) {
        Write-Host "已取消 FFLogs 数据下载。"
        return 0
    }
    if ($target -match "^https?://") {
        & $ProjectPython $ScraperScript $target
    }
    else {
        & $ProjectPython $ScraperScript single --report $target
    }
    return $LASTEXITCODE
}

function Invoke-Tool {
    param(
        [Parameter(Mandatory)]
        [string]$ResolvedAction
    )

    $exitCode = 0
    switch ($ResolvedAction) {
        "train" {
            & $ProjectPython -m training.train
            $exitCode = $LASTEXITCODE
        }
        "grpo" {
            & $ProjectPython -m grpo
            $exitCode = $LASTEXITCODE
        }
        "export" {
            & $ProjectPython -m scripts.onnx_export.workflow all
            $exitCode = $LASTEXITCODE
            if ($exitCode -eq 2) {
                throw "ONNX 图已导出，但 PT/ORT parity 门禁未通过；请查看 parity JSON 与部署包中的 release_report.json。"
            }
        }
        "analysis" {
            & $ProjectPython -m scripts.model_analysis
            $exitCode = $LASTEXITCODE
        }
        "replay" {
            & $ProjectPython -m scripts.autoregressive_replay
            $exitCode = $LASTEXITCODE
        }
        "fflogs" {
            $exitCode = Invoke-FFLogsDownload
        }
        default {
            throw "未知操作：$ResolvedAction"
        }
    }
    if ($exitCode -ne 0) {
        throw "工具执行失败，Python 退出码：$exitCode"
    }
}

if (-not (Test-Path -LiteralPath $ProjectPython -PathType Leaf)) {
    throw "找不到项目 Python：$ProjectPython。请先创建并安装 .venv。"
}

Push-Location $ProjectRoot
try {
    $resolvedAction = if ($Action -eq "menu") { Show-Menu } else { $Action }
    if ($resolvedAction -eq "exit") {
        Write-Host "已退出。"
        exit 0
    }
    Invoke-Tool -ResolvedAction $resolvedAction
}
catch {
    Write-Host ""
    Write-Host "执行失败：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}

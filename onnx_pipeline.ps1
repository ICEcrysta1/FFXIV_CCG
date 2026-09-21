# FFXIV_CCG：由根目录 .env 驱动的 PT → ONNX → 发布工作流
#
# 直接运行显示中文菜单：
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\onnx_pipeline.ps1
#
# 非交互调用示例：
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\onnx_pipeline.ps1 -Action all
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\onnx_pipeline.ps1 -Action export

[CmdletBinding()]
param(
    [ValidateSet(
        "menu",
        "all",
        "env",
        "export",
        "empty-parity",
        "scene-parity",
        "verify",
        "run"
    )]
    [string]$Action = "menu"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Show-Menu {
    Write-Host ""
    Write-Host "FFXIV_CCG：PT → ONNX → 发布流程" -ForegroundColor Cyan
    Write-Host "所有模型路径、精度、opset、Provider 和门禁参数均读取根目录 .env。"
    Write-Host ""
    Write-Host "  1. 一键完整流程（环境检查、导出、两项 parity、发布校验）"
    Write-Host "  2. 只检查 CUDA、BF16 和 ONNX Runtime 环境"
    Write-Host "  3. 只从 .env 指定的 checkpoint 导出 ONNX"
    Write-Host "  4. 只运行空 scene parity"
    Write-Host "  5. 只运行真实 scene parity"
    Write-Host "  6. 只校验并显示当前部署包发布状态"
    Write-Host "  7. 使用当前 ONNX 包执行普通回放"
    Write-Host "  0. 退出"
    Write-Host ""
    $choice = Read-Host "请输入选项编号"
    $resolvedChoice = switch ($choice) {
        "1" { "all" }
        "2" { "env" }
        "3" { "export" }
        "4" { "empty-parity" }
        "5" { "scene-parity" }
        "6" { "verify" }
        "7" { "run" }
        "0" { "exit" }
        default { throw "未知选项：$choice" }
    }
    return $resolvedChoice
}

function Invoke-PythonAction {
    param(
        [Parameter(Mandatory)]
        [string]$ResolvedAction
    )

    if ($ResolvedAction -eq "export") {
        & $ProjectPython -m scripts.onnx_export
    }
    else {
        & $ProjectPython -m scripts.onnx_export.workflow $ResolvedAction
    }
    if ($LASTEXITCODE -eq 2) {
        throw "ONNX 图已生成，但 PT/ORT 发布门禁未通过；请查看 parity JSON。"
    }
    if ($LASTEXITCODE -ne 0) {
        throw "ONNX 工作流失败，Python 退出码：$LASTEXITCODE"
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
    Invoke-PythonAction -ResolvedAction $resolvedAction
}
catch {
    Write-Host ""
    Write-Host "执行失败：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}

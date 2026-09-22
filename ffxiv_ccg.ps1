# FFXIV_CCG：项目常用工具统一入口（全部通过项目 .venv 的 Python 执行）
#
# 交互菜单：
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\ffxiv_ccg.ps1
#
# 无交互调用：
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\ffxiv_ccg.ps1 -Action train
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\ffxiv_ccg.ps1 -Action resume -ForceResumeDataMismatch
#
# 脚本本身不保存任何业务参数：模型路径、精度、opset、Provider、回放解码和门禁配置
# 全部来自 config/ 下的 YAML 与根目录 .env。

[CmdletBinding()]
param(
    [ValidateSet(
        "menu",
        "train",
        "resume",
        "grpo",
        "export",
        "analysis",
        "replay",
        "fflogs"
    )]
    [string]$Action = "menu",
    [switch]$ForceResumeDataMismatch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ScraperScript = Join-Path $ProjectRoot "scripts\fflogs_scraper.py"
# 由项目自己的 .env / YAML 解析器给出当前职业、模型变体和产物目录，脚本不自行解析配置。
$CheckpointProbe = @'
import json

from common.policy.config import (
    resolve_policy_model_config_path,
    resolve_policy_model_job_tag,
    resolve_policy_model_variant,
)
from training.config import load_run_config
from training.loop.checkpoint import collect_checkpoint_candidates

config_path = resolve_policy_model_config_path()
run_config = load_run_config(config_path)
resumable, rejected = collect_checkpoint_candidates(
    run_config.output_dir,
    max_epochs=run_config.max_epochs,
)
print(
    json.dumps(
        {
            "job_tag": resolve_policy_model_job_tag(config_path),
            "model_variant": resolve_policy_model_variant(config_path),
            "config_path": str(config_path),
            "output_dir": str(run_config.output_dir),
            "max_files": run_config.max_files,
            "max_epochs": run_config.max_epochs,
            "resumable": [
                {
                    "path": str(item.path),
                    "name": item.path.name,
                    "epoch": item.epoch,
                    "max_files": item.max_files,
                }
                for item in resumable
            ],
            "rejected": [
                {
                    "path": str(item.path),
                    "name": item.path.name,
                    "epoch": item.epoch,
                    "max_files": item.max_files,
                }
                for item in rejected
            ],
        }
    )
)
'@

function Show-Menu {
    Write-Host ""
    Write-Host "FFXIV_CCG 常用工具菜单" -ForegroundColor Cyan
    Write-Host "所有命令都使用项目 .venv，参数读取 config/ 与根目录 .env。"
    Write-Host ""
    Write-Host "  1. 训练（BC 预训练）"
    Write-Host "  2. 恢复训练（选择已有 checkpoint）"
    Write-Host "  3. GRPO 后训练"
    Write-Host "  4. ONNX 导出（导出 + PT/ORT parity 门禁 + 发布校验）"
    Write-Host "  5. 模型分析图生成（不含损失地形图）"
    Write-Host "  6. 模型自回归回放"
    Write-Host "  7. FFLogs 数据下载"
    Write-Host "  0. 退出"
    Write-Host ""
    $choice = Read-Host "请输入选项编号"
    $resolvedChoice = switch ($choice) {
        "1" { "train" }
        "2" { "resume" }
        "3" { "grpo" }
        "4" { "export" }
        "5" { "analysis" }
        "6" { "replay" }
        "7" { "fflogs" }
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

function Get-ResumeTarget {
    $raw = $CheckpointProbe | & $ProjectPython -
    if ($LASTEXITCODE -ne 0) {
        throw "读取模型配置失败，请检查根目录 .env 的 FFXIV_JOB_TAG 与 FFXIV_MODEL_VARIANT。"
    }
    try {
        return ($raw | Out-String | ConvertFrom-Json)
    }
    catch {
        throw "无法解析模型配置输出：$raw"
    }
}

function Invoke-ResumeTraining {
    param(
        [switch]$ForceDataMismatch
    )

    $target = Get-ResumeTarget
    $directory = [string]$target.output_dir
    Write-Host ""
    Write-Host ("当前 .env 配置：FFXIV_JOB_TAG={0}，FFXIV_MODEL_VARIANT={1}" -f $target.job_tag, $target.model_variant)
    Write-Host ("模型配置：{0}" -f $target.config_path)
    Write-Host ("checkpoint 目录：{0}" -f $directory)
    $maxFiles = if ($null -eq $target.max_files) { "全部有效文件" } else { [string]$target.max_files }
    Write-Host ("训练数据上限：{0}（来自模型 YAML 的 training.max_files）" -f $maxFiles)

    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        throw "找不到 checkpoint 目录：$directory（请先执行一次训练）"
    }
    # 可续训与否由 Python 读取 checkpoint 载荷里的真实 epoch 判定，不按文件名猜测。
    $resumable = @($target.resumable)
    $rejected = @($target.rejected)
    if ($resumable.Count -eq 0 -and $rejected.Count -eq 0) {
        throw "checkpoint 目录中没有 .pt 文件：$directory"
    }
    if ($resumable.Count -eq 0) {
        throw "没有可续训的 checkpoint：目录内 $($rejected.Count) 个 .pt 的 epoch 都已达到 training.max_epochs=$($target.max_epochs)，请先提高该值。"
    }
    # 中间 checkpoint 按名称升序在前，best.pt 排后。
    $ordered = @()
    $ordered += @($resumable | Where-Object { $_.name -ne "best.pt" } | Sort-Object -Property name)
    $ordered += @($resumable | Where-Object { $_.name -eq "best.pt" })

    Write-Host ""
    for ($index = 0; $index -lt $ordered.Count; $index++) {
        Write-Host ("  {0,2}. {1}（epoch {2}）" -f ($index + 1), $ordered[$index].name, $ordered[$index].epoch)
    }
    if ($rejected.Count -gt 0) {
        Write-Host ""
        Write-Host ("已跳过 {0} 个不可续训的 checkpoint（epoch 已达 training.max_epochs={1}，需先提高该值）：" -f $rejected.Count, $target.max_epochs)
        foreach ($item in ($rejected | Sort-Object -Property epoch, name)) {
            Write-Host ("  - {0}（epoch {1}）" -f $item.name, $item.epoch)
        }
    }
    Write-Host ""
    $choice = (Read-Host "请输入要恢复训练的 checkpoint 编号（直接回车取消）").Trim()
    if ([string]::IsNullOrWhiteSpace($choice)) {
        Write-Host "已取消恢复训练。"
        return 0
    }
    if ($choice -notmatch "^\d+$") {
        throw "无效的编号：$choice"
    }
    $selected = [int]$choice
    if ($selected -lt 1 -or $selected -gt $ordered.Count) {
        throw "编号超出范围：$choice（可选 1 ~ $($ordered.Count)）"
    }
    $selectedItem = $ordered[$selected - 1]
    $checkpoint = $selectedItem.path
    $forceSelectedDataMismatch = $ForceDataMismatch
    if ($selectedItem.max_files -ne $target.max_files -and -not $ForceDataMismatch) {
        $checkpointMaxFiles = if ($null -eq $selectedItem.max_files) { "全部有效文件（旧 checkpoint）" } else { [string]$selectedItem.max_files }
        $currentMaxFiles = if ($null -eq $target.max_files) { "全部有效文件" } else { [string]$target.max_files }
        Write-Warning ("checkpoint {0} 保存时的 training.max_files={1}，当前 YAML/CLI 为 {2}。两者会改变训练/验证数据集。" -f $selectedItem.name, $checkpointMaxFiles, $currentMaxFiles)
        $confirmation = (Read-Host "仍要强制继续吗？输入 Y 确认，直接回车或输入 N 取消").Trim()
        if ($confirmation -notmatch "^y$") {
            Write-Host "已取消恢复训练。"
            return 0
        }
        $forceSelectedDataMismatch = $true
    }
    Write-Host ""
    Write-Host ("恢复训练：{0}" -f $checkpoint)
    $trainingArguments = @("-m", "training.train", "--resume", $checkpoint)
    if ($forceSelectedDataMismatch) {
        $trainingArguments += "--force-resume-data-mismatch"
        Write-Host "已启用强制续训：允许 checkpoint 与当前 training.max_files 不一致。" -ForegroundColor Yellow
    }
    & $ProjectPython @trainingArguments
    return $LASTEXITCODE
}

function Invoke-Tool {
    param(
        [Parameter(Mandatory)]
        [string]$ResolvedAction,
        [switch]$ForceDataMismatch
    )

    $exitCode = 0
    switch ($ResolvedAction) {
        "train" {
            & $ProjectPython -m training.train
            $exitCode = $LASTEXITCODE
        }
        "resume" {
            $exitCode = Invoke-ResumeTraining -ForceDataMismatch:$ForceDataMismatch
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
    Invoke-Tool -ResolvedAction $resolvedAction -ForceDataMismatch:$ForceResumeDataMismatch
}
catch {
    Write-Host ""
    Write-Host "执行失败：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}

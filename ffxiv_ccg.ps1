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
        "resume",
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
# 由项目自己的 .env / YAML 解析器给出当前职业、模型变体和产物目录，脚本不自行解析配置。
$CheckpointProbe = @'
import json

from common.policy.config import (
    resolve_policy_model_config_path,
    resolve_policy_model_job_tag,
    resolve_policy_model_variant,
)
from training.config import load_run_config

config_path = resolve_policy_model_config_path()
run_config = load_run_config(config_path)
print(
    json.dumps(
        {
            "job_tag": resolve_policy_model_job_tag(config_path),
            "model_variant": resolve_policy_model_variant(config_path),
            "config_path": str(config_path),
            "output_dir": str(run_config.output_dir),
            "max_files": run_config.max_files,
            "max_epochs": run_config.max_epochs,
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
    $checkpoints = @(Get-ChildItem -LiteralPath $directory -Filter "*.pt" -File)
    if ($checkpoints.Count -eq 0) {
        throw "checkpoint 目录中没有 .pt 文件：$directory"
    }
    # final.pt 的 epoch 固定等于当前 max_epochs，epoch 达到上限的中间 checkpoint 也会被
    # 训练的续训校验拒绝，先过滤掉，避免准备完 cache 才失败。
    $maxEpochs = [int]$target.max_epochs
    $resumable = @()
    $skipped = @()
    foreach ($item in $checkpoints) {
        $epochMatch = [regex]::Match($item.Name, "^epoch_(\d+)")
        if ($item.Name -eq "final.pt") {
            $skipped += $item
        }
        elseif ($epochMatch.Success -and [int]$epochMatch.Groups[1].Value -ge $maxEpochs) {
            $skipped += $item
        }
        else {
            $resumable += $item
        }
    }
    if ($resumable.Count -eq 0) {
        throw "没有可续训的 checkpoint：目录内 $($checkpoints.Count) 个 .pt 都已达到 training.max_epochs=$maxEpochs，请先提高该值。"
    }
    # 中间 checkpoint 按名称升序在前，best.pt 排后。
    $ordered = @()
    $ordered += @($resumable | Where-Object { $_.Name -ne "best.pt" } | Sort-Object -Property Name)
    $ordered += @($resumable | Where-Object { $_.Name -eq "best.pt" })

    Write-Host ""
    for ($index = 0; $index -lt $ordered.Count; $index++) {
        Write-Host ("  {0,2}. {1}" -f ($index + 1), $ordered[$index].Name)
    }
    if ($skipped.Count -gt 0) {
        Write-Host ""
        Write-Host ("已跳过 {0} 个不可续训的 checkpoint（epoch 已达 training.max_epochs={1}，需先提高该值）：" -f $skipped.Count, $maxEpochs)
        foreach ($item in ($skipped | Sort-Object -Property Name)) {
            Write-Host ("  - {0}" -f $item.Name)
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
    $checkpoint = $ordered[$selected - 1].FullName
    Write-Host ""
    Write-Host ("恢复训练：{0}" -f $checkpoint)
    & $ProjectPython -m training.train --resume $checkpoint
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
        "resume" {
            $exitCode = Invoke-ResumeTraining
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

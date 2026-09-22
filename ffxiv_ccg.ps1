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
    [string]$Action = "menu",
    [switch]$ForceResumeDataMismatch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ScraperScript = Join-Path $ProjectRoot "scripts\fflogs_scraper.py"
$MenuModulePath = Join-Path $ProjectRoot "scripts\ffxiv_ccg_menu.psm1"
if (-not (Test-Path -LiteralPath $MenuModulePath -PathType Leaf)) {
    throw "找不到公共菜单模块：$MenuModulePath"
}
Import-Module -Name $MenuModulePath -Force

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

function Get-CheckpointTarget {
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

function Show-CheckpointTargetSummary {
    param(
        [Parameter(Mandatory)]
        [object]$Target
    )

    Write-Host ""
    Write-Host ("当前 .env 配置：FFXIV_JOB_TAG={0}，FFXIV_MODEL_VARIANT={1}" -f $Target.job_tag, $Target.model_variant)
    Write-Host ("模型配置：{0}" -f $Target.config_path)
    Write-Host ("checkpoint 目录：{0}" -f $Target.output_dir)
    $maxFiles = if ($null -eq $Target.max_files) { "全部有效文件" } else { [string]$Target.max_files }
    Write-Host ("训练数据上限：{0}（来自模型 YAML 的 training.max_files）" -f $maxFiles)
}

function Select-ToolCheckpoint {
    param(
        [Parameter(Mandatory)]
        [object]$Target,
        [switch]$ResumableOnly
    )

    $directory = [string]$Target.output_dir

    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        throw "找不到 checkpoint 目录：$directory（请先执行一次训练）"
    }
    $resumable = @($Target.resumable)
    $rejected = @($Target.rejected)
    if ($resumable.Count -eq 0 -and $rejected.Count -eq 0) {
        throw "checkpoint 目录中没有 .pt 文件：$directory"
    }
    if ($ResumableOnly -and $resumable.Count -eq 0) {
        throw "没有可续训的 checkpoint：目录内 $($rejected.Count) 个 .pt 的 epoch 都已达到 training.max_epochs=$($Target.max_epochs)，请先提高该值。"
    }

    # 续训只允许当前 max_epochs 尚未完成的文件；其余动作需要能选择目录中的全部 checkpoint。
    $candidates = if ($ResumableOnly) { $resumable } else { @($resumable) + @($rejected) }
    if ($ResumableOnly -and $rejected.Count -gt 0) {
        Write-Host ""
        Write-Host ("已跳过 {0} 个不可续训的 checkpoint（epoch 已达 training.max_epochs={1}，需先提高该值）：" -f $rejected.Count, $Target.max_epochs)
        foreach ($item in ($rejected | Sort-Object -Property epoch, name)) {
            Write-Host ("  - {0}（epoch {1}）" -f $item.name, $item.epoch)
        }
    }
    return Select-FfxivCcgCheckpoint -Candidates $candidates
}

function Invoke-Tool {
    param(
        [Parameter(Mandatory)]
        [string]$ResolvedAction,
        [switch]$ForceDataMismatch
    )

    $selectedCheckpoint = $null
    $checkpointTarget = $null
    if ($ResolvedAction -in @("resume", "grpo", "export", "analysis", "replay")) {
        $checkpointTarget = Get-CheckpointTarget
        Show-CheckpointTargetSummary -Target $checkpointTarget
        $selectedCheckpoint = Select-ToolCheckpoint -Target $checkpointTarget -ResumableOnly:($ResolvedAction -eq "resume")
        if ($null -eq $selectedCheckpoint) {
            Write-Host "已取消操作。"
            return 0
        }
    }

    $exitCode = 0
    switch ($ResolvedAction) {
        "train" {
            & $ProjectPython -m training.train
            $exitCode = $LASTEXITCODE
        }
        "resume" {
            $forceSelectedDataMismatch = $ForceDataMismatch
            if ($selectedCheckpoint.max_files -ne $checkpointTarget.max_files -and -not $ForceDataMismatch) {
                $checkpointMaxFiles = if ($null -eq $selectedCheckpoint.max_files) { "全部有效文件（旧 checkpoint）" } else { [string]$selectedCheckpoint.max_files }
                $currentMaxFiles = if ($null -eq $checkpointTarget.max_files) { "全部有效文件" } else { [string]$checkpointTarget.max_files }
                Write-Warning ("checkpoint {0} 保存时的 training.max_files={1}，当前 YAML/CLI 为 {2}。两者会改变训练/验证数据集。" -f $selectedCheckpoint.name, $checkpointMaxFiles, $currentMaxFiles)
                $confirmation = (Read-Host "仍要强制继续吗？输入 Y 确认，直接回车或输入 N 取消").Trim()
                if ($confirmation -notmatch "^y$") {
                    Write-Host "已取消恢复训练。"
                    return 0
                }
                $forceSelectedDataMismatch = $true
            }
            Write-Host ""
            Write-Host ("恢复训练：{0}" -f $selectedCheckpoint.path)
            $trainingArguments = @("-m", "training.train", "--resume", $selectedCheckpoint.path)
            if ($forceSelectedDataMismatch) {
                $trainingArguments += "--force-resume-data-mismatch"
                Write-Host "已启用强制续训：允许 checkpoint 与当前 training.max_files 不一致。" -ForegroundColor Yellow
            }
            & $ProjectPython @trainingArguments
            $exitCode = $LASTEXITCODE
        }
        "grpo" {
            & $ProjectPython -m grpo --checkpoint $selectedCheckpoint.path
            $exitCode = $LASTEXITCODE
        }
        "export" {
            & $ProjectPython -m scripts.onnx_export.workflow all --checkpoint $selectedCheckpoint.path
            $exitCode = $LASTEXITCODE
            if ($exitCode -eq 2) {
                throw "ONNX 图已导出，但 PT/ORT parity 门禁未通过；请查看 parity JSON 与部署包中的 release_report.json。"
            }
        }
        "analysis" {
            & $ProjectPython -m scripts.model_analysis --checkpoint $selectedCheckpoint.path
            $exitCode = $LASTEXITCODE
        }
        "replay" {
            & $ProjectPython -m scripts.autoregressive_replay --checkpoint $selectedCheckpoint.path
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
    $resolvedAction = if ($Action.Trim().ToLowerInvariant() -eq "menu") {
        Show-FfxivCcgMenu
    }
    else {
        Resolve-FfxivCcgAction -RequestedAction $Action
    }
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

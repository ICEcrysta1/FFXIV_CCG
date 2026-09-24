# FFXIV_CCG 公共菜单模块：统一交互菜单、菜单编号和命令行 action。

Set-StrictMode -Version Latest

$script:ToolMenuEntries = @(
    [PSCustomObject]@{ Number = "1"; Action = "train"; Label = "训练（BC 预训练）" }
    [PSCustomObject]@{ Number = "2"; Action = "resume"; Label = "恢复训练（选择 BC checkpoint）" }
    [PSCustomObject]@{ Number = "3"; Action = "grpo"; Label = "GRPO 后训练" }
    [PSCustomObject]@{ Number = "4"; Action = "export"; Label = "ONNX 导出（导出 + PT/ORT parity 门禁 + 发布校验）" }
    [PSCustomObject]@{ Number = "5"; Action = "analysis"; Label = "模型分析图生成（不含损失地形图）" }
    [PSCustomObject]@{ Number = "6"; Action = "replay"; Label = "模型自回归回放" }
    [PSCustomObject]@{ Number = "7"; Action = "fflogs"; Label = "FFLogs 数据下载" }
    [PSCustomObject]@{ Number = "8"; Action = "monitor"; Label = "TensorBoard Web 监控" }
)

function Get-FfxivCcgMenuEntries {
    # 返回公共菜单动作目录。
    return @($script:ToolMenuEntries)
}

function Resolve-FfxivCcgAction {
    param(
        [Parameter(Mandatory)]
        [string]$RequestedAction,
        [switch]$AllowMenuNumber
    )

    $normalizedAction = $RequestedAction.Trim().ToLowerInvariant()
    if ($normalizedAction -eq "menu") {
        return "menu"
    }
    if ($AllowMenuNumber -and $normalizedAction -eq "0") {
        return "exit"
    }
    $entry = @(
        $script:ToolMenuEntries | Where-Object {
            $_.Action -eq $normalizedAction -or ($AllowMenuNumber -and $_.Number -eq $normalizedAction)
        }
    ) | Select-Object -First 1
    if ($null -eq $entry) {
        $available = ($script:ToolMenuEntries | ForEach-Object { "$($_.Number)/$($_.Action)" }) -join ", "
        throw "未知操作：$RequestedAction；可用值：$available，或 menu"
    }
    return [string]$entry.Action
}

function Show-FfxivCcgMenu {
    Write-Host ""
    Write-Host "FFXIV_CCG 常用工具菜单" -ForegroundColor Cyan
    Write-Host "所有命令都使用项目 .venv，参数读取 config/ 与根目录 .env。"
    Write-Host ""
    foreach ($entry in $script:ToolMenuEntries) {
        Write-Host ("  {0}. {1}" -f $entry.Number, $entry.Label)
    }
    Write-Host "  0. 退出"
    Write-Host ""
    $choice = Read-Host "请输入选项编号"
    return Resolve-FfxivCcgAction -RequestedAction $choice -AllowMenuNumber
}

function Select-FfxivCcgCheckpoint {
    param(
        [Parameter(Mandatory)]
        [object[]]$Candidates,
        [string]$Prompt = "请输入 checkpoint 编号（如 1 或 a1，直接回车取消）"
    )

    $bcCandidates = @($Candidates | Where-Object { $_.source -ne "grpo" })
    $grpoCandidates = @($Candidates | Where-Object { $_.source -eq "grpo" })
    if ($bcCandidates.Count -eq 0 -and $grpoCandidates.Count -eq 0) {
        throw "checkpoint 目录中没有可选择的 .pt 文件"
    }

    $orderedBc = @($bcCandidates | Where-Object { $_.name -ne "best.pt" } | Sort-Object -Property name)
    $orderedBc += @($bcCandidates | Where-Object { $_.name -eq "best.pt" } | Sort-Object -Property name)
    $orderedGrpo = @($grpoCandidates | Where-Object { $_.name -ne "best.pt" } | Sort-Object -Property name)
    $orderedGrpo += @($grpoCandidates | Where-Object { $_.name -eq "best.pt" } | Sort-Object -Property name)

    $entries = @()
    for ($index = 0; $index -lt $orderedBc.Count; $index++) {
        $entries += [PSCustomObject]@{
            Key = [string]($index + 1)
            Candidate = $orderedBc[$index]
            SourceLabel = "BC"
            ProgressLabel = "epoch"
        }
    }
    for ($index = 0; $index -lt $orderedGrpo.Count; $index++) {
        $entries += [PSCustomObject]@{
            Key = "a$($index + 1)"
            Candidate = $orderedGrpo[$index]
            SourceLabel = "GRPO"
            ProgressLabel = "iteration"
        }
    }

    Write-Host ""
    Write-Host "可用 checkpoint："
    foreach ($entry in $entries) {
        $directoryLabel = [IO.Path]::GetFileName([string]$entry.Candidate.output_dir)
        Write-Host ("  {0,3}. {1}（{2}，{3} {4}，目录 {5}）" -f $entry.Key, $entry.Candidate.name, $entry.SourceLabel, $entry.ProgressLabel, $entry.Candidate.epoch, $directoryLabel)
    }
    Write-Host ""
    $choice = (Read-Host $Prompt).Trim()
    if ([string]::IsNullOrWhiteSpace($choice)) {
        return $null
    }
    $normalizedChoice = $choice.ToLowerInvariant()
    $selectedEntry = $entries | Where-Object { $_.Key -eq $normalizedChoice } | Select-Object -First 1
    if ($null -eq $selectedEntry) {
        throw "无效的 checkpoint 编号：$choice（BC 使用 1、2、3；GRPO 使用 a1、a2、a3）"
    }
    return $selectedEntry.Candidate
}

Export-ModuleMember -Function Get-FfxivCcgMenuEntries, Resolve-FfxivCcgAction, Show-FfxivCcgMenu, Select-FfxivCcgCheckpoint

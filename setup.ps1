# FFXIV_CCG：一键准备项目 Python GPU 环境
#
# 从项目根目录执行：
#   .\setup.ps1
# 如果执行策略拦截脚本，再改用当前进程临时放行：
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup.ps1
#
# 脚本只负责创建/复用根目录 .venv 和安装依赖，不会覆盖本地 .env。

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$VenvPath = Join-Path $ProjectRoot ".venv"
$ProjectPython = Join-Path $VenvPath "Scripts\python.exe"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
$GpuRequirementsPath = Join-Path $ProjectRoot "requirements-onnx-gpu.txt"
$EnvExamplePath = Join-Path $ProjectRoot ".env.example"
$EnvPath = Join-Path $ProjectRoot ".env"
$MinimumPythonVersion = [version]"3.12.0"

$PythonVersionProbe = "import sys; print(sys.version_info.major, sys.version_info.minor, sys.version_info.micro, sep='.')"
$GpuDependencyProbe = @'
import onnxruntime as ort
import torch

print('torch_version=' + torch.__version__)
print('torch_cuda_build=' + str(torch.version.cuda))
print('torch_cuda_available=' + str(torch.cuda.is_available()))
print('onnxruntime_version=' + ort.__version__)
print('onnxruntime_providers=' + ','.join(ort.get_available_providers()))
'@

function Invoke-CapturedCommand {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,
        [Parameter(Mandatory)]
        [AllowEmptyCollection()]
        [string[]]$Arguments
    )

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # 探测命令可能用非零退出码表示“未安装”，不能让全局 Stop 策略提前中断。
        $ErrorActionPreference = "Continue"
        $capturedOutput = & $FilePath @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    $output = (($capturedOutput | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine).Trim()
    return [PSCustomObject]@{
        ExitCode = $exitCode
        Output   = $output
    }
}

function Invoke-RequiredCommand {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,
        [Parameter(Mandatory)]
        [string[]]$Arguments,
        [Parameter(Mandatory)]
        [string]$Description
    )

    Write-Host "`n>>> $Description" -ForegroundColor Cyan
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "命令执行失败（退出码 $LASTEXITCODE）：$Description"
    }
}

function Test-PipPackageInstalled {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,
        [Parameter(Mandatory)]
        [string]$PackageName
    )

    $probe = Invoke-CapturedCommand -FilePath $FilePath -Arguments @("-m", "pip", "show", $PackageName)
    return $probe.ExitCode -eq 0
}

function Get-PythonVersion {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,
        [AllowEmptyCollection()]
        [string[]]$Arguments
    )

    $probe = Invoke-CapturedCommand -FilePath $FilePath -Arguments ($Arguments + @("-c", $PythonVersionProbe))
    if ($probe.ExitCode -ne 0) {
        return $null
    }
    $match = [regex]::Match($probe.Output, "(?m)^\s*(\d+\.\d+(?:\.\d+)?)\s*$")
    if (-not $match.Success) {
        return $null
    }
    try {
        return [version]$match.Groups[1].Value
    }
    catch {
        return $null
    }
}

function Select-SystemPython {
    $candidates = @(
        [PSCustomObject]@{ Name = "python"; CommandName = "python"; Arguments = [string[]]@() },
        [PSCustomObject]@{ Name = "py -3"; CommandName = "py"; Arguments = [string[]]@("-3") }
    )

    foreach ($candidate in $candidates) {
        $command = Get-Command -Name $candidate.CommandName -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $command) {
            continue
        }

        $version = Get-PythonVersion -FilePath $command.Path -Arguments $candidate.Arguments
        if ($null -eq $version) {
            continue
        }
        if ($version -lt $MinimumPythonVersion) {
            Write-Host ("忽略 {0}：检测到 Python {1}，要求 Python {2} 或更高版本。" -f $candidate.Name, $version, $MinimumPythonVersion) -ForegroundColor Yellow
            continue
        }

        return [PSCustomObject]@{
            Name      = $candidate.Name
            Path      = $command.Path
            Arguments = $candidate.Arguments
            Version   = $version
        }
    }

    throw "找不到可用的系统 Python。请安装 Python $MinimumPythonVersion 或更高版本，并确认 python 已加入 PATH。"
}

try {
    foreach ($requiredPath in @($RequirementsPath, $GpuRequirementsPath, $EnvExamplePath)) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "项目文件不存在：$requiredPath"
        }
    }

    $systemPython = Select-SystemPython
    Write-Host ("使用 {0}：{1}（Python {2}）" -f $systemPython.Name, $systemPython.Path, $systemPython.Version) -ForegroundColor Green

    if (-not (Test-Path -LiteralPath $ProjectPython -PathType Leaf)) {
        Invoke-RequiredCommand `
            -FilePath $systemPython.Path `
            -Arguments ($systemPython.Arguments + @("-m", "venv", $VenvPath)) `
            -Description "创建项目虚拟环境 .venv"
    }
    else {
        Write-Host "已找到项目虚拟环境，复用 .venv。" -ForegroundColor Green
    }

    $venvVersion = Get-PythonVersion -FilePath $ProjectPython -Arguments @()
    if ($null -eq $venvVersion) {
        throw "无法执行项目 Python：$ProjectPython"
    }
    if ($venvVersion -lt $MinimumPythonVersion) {
        throw "现有 .venv 使用 Python $venvVersion，要求 Python $MinimumPythonVersion 或更高版本。请删除 .venv 后重新运行 setup.ps1。"
    }
    Write-Host ("项目 Python 就绪：Python {0}" -f $venvVersion) -ForegroundColor Green

    Invoke-RequiredCommand `
        -FilePath $ProjectPython `
        -Arguments @("-m", "pip", "install", "--upgrade", "pip") `
        -Description "升级虚拟环境中的 pip"
    Invoke-RequiredCommand `
        -FilePath $ProjectPython `
        -Arguments @("-m", "pip", "install", "-r", $RequirementsPath) `
        -Description "安装项目核心依赖与 CUDA 版 PyTorch"
    if (Test-PipPackageInstalled -FilePath $ProjectPython -PackageName "onnxruntime") {
        Invoke-RequiredCommand `
            -FilePath $ProjectPython `
            -Arguments @("-m", "pip", "uninstall", "-y", "onnxruntime") `
            -Description "清理冲突的 CPU 版 ONNX Runtime"
    }
    else {
        Write-Host "未检测到 CPU 版 ONNX Runtime，跳过卸载。" -ForegroundColor Green
    }
    Invoke-RequiredCommand `
        -FilePath $ProjectPython `
        -Arguments @("-m", "pip", "install", "-r", $GpuRequirementsPath) `
        -Description "安装 ONNX Runtime GPU 与导出依赖"

    $dependencyProbe = Invoke-CapturedCommand -FilePath $ProjectPython -Arguments @("-c", $GpuDependencyProbe)
    if ($dependencyProbe.ExitCode -ne 0) {
        throw "GPU 依赖导入检查失败：`n$($dependencyProbe.Output)"
    }
    Write-Host "`nGPU 依赖检查结果：" -ForegroundColor Cyan
    Write-Host $dependencyProbe.Output

    if ($dependencyProbe.Output -notmatch "onnxruntime_providers=.*CUDAExecutionProvider") {
        Write-Warning "当前 ONNX Runtime 没有报告 CUDAExecutionProvider；请检查 NVIDIA 驱动和 CUDA 运行库。"
    }
    if ($dependencyProbe.Output -match "torch_cuda_available=False") {
        Write-Warning "当前 PyTorch 未检测到可用 CUDA 设备；项目训练、导出和自回归不提供 CPU 支持。"
    }

    Write-Host "`n环境准备完成。" -ForegroundColor Green
    if (Test-Path -LiteralPath $EnvPath -PathType Leaf) {
        Write-Host "已检测到根目录 .env，setup.ps1 未覆盖它；请仍按 .env.example 检查本机配置。"
    }
    else {
        Write-Host "请参考 .env.example 的说明创建并配置根目录 .env，例如：" -ForegroundColor Yellow
        Write-Host "  Copy-Item .env.example .env"
    }
    Write-Host "`n后续可选步骤："
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  .\ffxiv_ccg.ps1"
}
catch {
    Write-Host "`n环境准备失败：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

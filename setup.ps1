# FFXIV_CCG：一键准备项目 Python GPU 环境
# 同时在项目 .node 中安装 Node 22、pnpm 10 与离线分析依赖。
#
# 从项目根目录执行：
#   .\setup.ps1
# 如果执行策略拦截脚本，再改用当前进程临时放行：
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup.ps1
#
# 脚本创建/复用根目录 .venv、.node 并安装依赖，不会覆盖本地 .env。

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$VenvPath = Join-Path $ProjectRoot ".venv"
$ProjectPython = Join-Path $VenvPath "Scripts\python.exe"
$PythonRuntimeAssembly = Join-Path $VenvPath "Lib\site-packages\pythonnet\runtime\Python.Runtime.dll"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
$GpuRequirementsPath = Join-Path $ProjectRoot "requirements-onnx-gpu.txt"
$EnvExamplePath = Join-Path $ProjectRoot ".env.example"
$EnvPath = Join-Path $ProjectRoot ".env"
$NodeVersionPath = Join-Path $ProjectRoot ".node-version"
$NodeHome = Join-Path $ProjectRoot ".node"
$NpmCachePath = Join-Path $NodeHome "npm-cache"
$PnpmStorePath = Join-Path $NodeHome "pnpm-store"
$AnalyzerRoot = Join-Path $ProjectRoot "third_party\xivanalysis"
$RootPackageLock = Join-Path $ProjectRoot "package-lock.json"
$PnpmVersion = "10.0.0"
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
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 会把原生 stderr 转成 PowerShell 错误记录；
        # 执行期间使用 Continue，命令成败只按原生进程退出码判断。
        $ErrorActionPreference = "Continue"
        & $FilePath @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($exitCode -ne 0) {
        throw "命令执行失败（退出码 $exitCode）：$Description"
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

function Install-ProjectNode {
    $nodeVersion = (Get-Content -LiteralPath $NodeVersionPath -Raw -Encoding UTF8).Trim()
    if ($nodeVersion -ne "22") {
        throw ".node-version 当前只支持 Node 22：$nodeVersion"
    }
    $architecture = switch ($env:PROCESSOR_ARCHITECTURE) {
        "AMD64" { "x64" }
        "ARM64" { "arm64" }
        default { throw "不支持的 Windows 架构：$env:PROCESSOR_ARCHITECTURE" }
    }
    $nodeDir = Join-Path $NodeHome "runtime"
    $nodeExe = Join-Path $nodeDir "node.exe"
    $npmCli = Join-Path $nodeDir "node_modules\npm\bin\npm-cli.js"
    if (Test-Path -LiteralPath $nodeDir) {
        if (-not (Test-Path -LiteralPath $nodeExe -PathType Leaf) -or -not (Test-Path -LiteralPath $npmCli -PathType Leaf)) {
            throw "项目 Node 安装不完整：$nodeDir；请检查该目录后手动清理再运行。"
        }
    }
    else {
        New-Item -ItemType Directory -Path $NodeHome -Force | Out-Null
        $baseUrl = "https://nodejs.org/dist/latest-v22.x"
        Write-Host "`n>>> 下载项目专用 Node 22.x" -ForegroundColor Cyan
        $checksums = Invoke-WebRequest -Uri "$baseUrl/SHASUMS256.txt" -UseBasicParsing
        $archivePattern = "node-v(22\.\d+\.\d+)-win-$architecture\.zip"
        $checksumLine = ($checksums.Content -split "`n" | Where-Object { $_ -match "^[0-9a-fA-F]{64}\s+$archivePattern\s*$" } | Select-Object -First 1)
        if (-not $checksumLine) {
            throw "Node 官方校验清单中找不到 Windows $architecture 的 Node 22 压缩包"
        }
        $nodeName = ([regex]::Match($checksumLine, "node-v22\.\d+\.\d+-win-$architecture")).Value
        $archiveName = "$nodeName.zip"
        $archivePath = Join-Path $NodeHome $archiveName
        $stagePath = Join-Path $NodeHome "stage-$PID"
        $expectedHash = ($checksumLine -split '\s+')[0]
        try {
            Invoke-WebRequest -Uri "$baseUrl/$archiveName" -OutFile $archivePath -UseBasicParsing
            $actualHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
            if ($actualHash -ne $expectedHash) {
                throw "Node 压缩包 SHA-256 校验失败：$archiveName"
            }
            if (Test-Path -LiteralPath $stagePath) {
                throw "临时安装目录已存在：$stagePath"
            }
            Expand-Archive -LiteralPath $archivePath -DestinationPath $stagePath
            $extracted = Join-Path $stagePath $nodeName
            if (-not (Test-Path -LiteralPath (Join-Path $extracted "node.exe") -PathType Leaf)) {
                throw "Node 压缩包缺少 node.exe：$archiveName"
            }
            # 两条路径均由项目根目录和官方归档名拼出，不跨工作区移动。
            Move-Item -LiteralPath $extracted -Destination $nodeDir
        }
        finally {
            if (Test-Path -LiteralPath $archivePath -PathType Leaf) {
                Remove-Item -LiteralPath $archivePath
            }
            if (Test-Path -LiteralPath $stagePath -PathType Container) {
                # 解压失败时保留临时目录供人工检查，成功时只删除空目录。
                if (-not (Get-ChildItem -LiteralPath $stagePath -Force | Select-Object -First 1)) {
                    Remove-Item -LiteralPath $stagePath
                }
            }
        }
    }
    $installedVersion = Invoke-CapturedCommand -FilePath $nodeExe -Arguments @("--version")
    if ($installedVersion.ExitCode -ne 0 -or $installedVersion.Output -notmatch '^v22\.\d+\.\d+$') {
        throw "项目 Node 版本不匹配：预期 Node 22.x，实际 $($installedVersion.Output)"
    }
    return [PSCustomObject]@{ Directory = $nodeDir; Executable = $nodeExe; NpmCli = $npmCli }
}

function Install-ProjectPnpm {
    param([Parameter(Mandatory)]$Node)
    $pnpmHome = Join-Path $NodeHome "pnpm\$PnpmVersion"
    $pnpmCli = Join-Path $pnpmHome "node_modules\pnpm\bin\pnpm.cjs"
    if (-not (Test-Path -LiteralPath $pnpmCli -PathType Leaf)) {
        Invoke-RequiredCommand -FilePath $Node.Executable `
            -Arguments @($Node.NpmCli, "install", "--global", "--prefix", $pnpmHome, "--cache", $NpmCachePath, "pnpm@$PnpmVersion", "--ignore-scripts", "--no-audit", "--no-fund") `
            -Description "在项目 .node 中安装 pnpm $PnpmVersion"
    }
    $installed = Invoke-CapturedCommand -FilePath $Node.Executable -Arguments @($pnpmCli, "--version")
    if ($installed.ExitCode -ne 0 -or $installed.Output -ne $PnpmVersion) {
        throw "项目 pnpm 版本不匹配：预期 $PnpmVersion，实际 $($installed.Output)"
    }
    return $pnpmCli
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
    foreach ($requiredPath in @($RequirementsPath, $GpuRequirementsPath, $EnvExamplePath, $NodeVersionPath, $RootPackageLock)) {
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
    if (-not (Test-Path -LiteralPath $PythonRuntimeAssembly -PathType Leaf)) {
        Invoke-RequiredCommand `
            -FilePath $ProjectPython `
            -Arguments @("-m", "pip", "install", "--ignore-installed", "pythonnet>=3.0.5,<4.0") `
            -Description "确保 Python.NET 安装在项目 .venv 中"
    }
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

    $node = Install-ProjectNode
    $previousPath = $env:PATH
    $previousNpmCache = [Environment]::GetEnvironmentVariable("npm_config_cache", "Process")
    try {
        # pnpm 安装脚本及其子进程只从项目目录查找 Node，不依赖系统 Node。
        New-Item -ItemType Directory -Path $NpmCachePath, $PnpmStorePath -Force | Out-Null
        $env:PATH = "$($node.Directory);$previousPath"
        $env:npm_config_cache = $NpmCachePath
        $pnpmCli = Install-ProjectPnpm -Node $node
        if (-not (Test-Path -LiteralPath (Join-Path $AnalyzerRoot "package.json") -PathType Leaf)) {
            Invoke-RequiredCommand -FilePath "git" -Arguments @("-C", $ProjectRoot, "submodule", "update", "--init", "--", "third_party/xivanalysis") -Description "初始化分析器子模块"
        }
        Invoke-RequiredCommand -FilePath $node.Executable `
            -Arguments @($node.NpmCli, "ci", "--ignore-scripts", "--prefix", $ProjectRoot, "--cache", $NpmCachePath, "--no-audit", "--no-fund") `
            -Description "按 package-lock.json 安装桥接 DOM 依赖"
        Invoke-RequiredCommand -FilePath $node.Executable `
            -Arguments @($pnpmCli, "--dir", $AnalyzerRoot, "install", "--frozen-lockfile", "--store-dir", $PnpmStorePath) `
            -Description "按子模块锁文件安装分析器依赖"
    }
    finally {
        $env:PATH = $previousPath
        if ($null -eq $previousNpmCache) {
            Remove-Item Env:npm_config_cache -ErrorAction SilentlyContinue
        }
        else {
            $env:npm_config_cache = $previousNpmCache
        }
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
    Write-Host "  项目 Node：$($node.Executable)"
    Write-Host "  .\ffxiv_ccg.ps1"
}
catch {
    Write-Host "`n环境准备失败：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

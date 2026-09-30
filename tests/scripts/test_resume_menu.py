"""交互续训菜单对训练/验证数据份数变化的确认测试。"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("powershell.exe") is None, reason="需要 Windows PowerShell")
@pytest.mark.parametrize("saved_validation_files", ["missing", "15"])
def test_resume_menu_prompts_for_validation_count_mismatch(
    tmp_path: Path, saved_validation_files: str
) -> None:
    """旧 checkpoint 无记录或数量变化时，在启动训练前要求确认。"""
    project_root = Path(__file__).resolve().parents[2]
    harness = tmp_path / "resume_menu.ps1"
    harness.write_text(
        """
$ErrorActionPreference = "Stop"
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:FFXIV_TEST_MENU_SCRIPT, [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) { throw "菜单脚本有语法错误" }
$function = $ast.Find({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq "Invoke-Tool"
}, $true)
if ($null -eq $function) { throw "找不到 Invoke-Tool" }
Invoke-Expression $function.Extent.Text

$savedValidation = if ($env:FFXIV_TEST_SAVED_VALIDATION -eq "missing") {
    $null
} else {
    [int]$env:FFXIV_TEST_SAVED_VALIDATION
}
$script:PromptCount = 0
function Get-CheckpointTarget {
    [pscustomobject]@{ max_files = 1280; validation_files = 20 }
}
function Show-CheckpointTargetSummary { param($Target) }
function Select-ToolCheckpoint {
    param($Target, [switch]$ResumableOnly)
    [pscustomobject]@{
        name = "epoch_002.pt"
        path = "unused.pt"
        source = "bc"
        max_files = 1280
        validation_files = $savedValidation
    }
}
function Read-Host {
    param($Prompt)
    $script:PromptCount++
    return "N"
}
$result = Invoke-Tool -ResolvedAction resume -InteractiveCheckpointSelection
if ($result -ne 0 -or $script:PromptCount -ne 1) {
    throw "验证份数变化未在训练启动前要求确认"
}
""".lstrip(),
        encoding="utf-8-sig",
        newline="\n",
    )
    env = os.environ.copy()
    env["FFXIV_TEST_MENU_SCRIPT"] = str(project_root / "ffxiv_ccg.ps1")
    env["FFXIV_TEST_SAVED_VALIDATION"] = saved_validation_files
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-NonInteractive",
            "-File",
            str(harness),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "training.validation_files" in result.stdout

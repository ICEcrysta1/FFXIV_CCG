"""Linux 操作入口的菜单选择、续训确认和监控生命周期回归测试。"""

from __future__ import annotations

import pickle
import subprocess
from pathlib import Path

import pytest

from scripts import ffxiv_ccg as menu


def selected_checkpoint(**changes):
    result = dict(name="epoch_002.pt", path="unused.pt", source="bc", output_dir="checkpoints",
                  epoch=2, max_files=1280, validation_files=20)
    result.update(changes)
    return result


def prepare_resume(monkeypatch, *, validation_files=20):
    selected = selected_checkpoint(validation_files=validation_files)
    target = dict(max_files=1280, validation_files=20)
    monkeypatch.setattr(menu, "checkpoint_target", lambda: target)
    monkeypatch.setattr(menu, "show_checkpoint_target", lambda _target: None)
    monkeypatch.setattr(menu, "interactive_checkpoint", lambda *_args, **_kwargs: selected)
    return selected, target


@pytest.mark.parametrize("saved_validation_files", [None, 15])
def test_resume_menu_prompts_for_validation_count_mismatch(monkeypatch, capsys, saved_validation_files):
    prepare_resume(monkeypatch, validation_files=saved_validation_files)
    monkeypatch.setattr("builtins.input", lambda _prompt: "N")
    monkeypatch.setattr(menu, "run_module", lambda *_args, **_kwargs: pytest.fail("取消后不得启动训练或监控"))
    with pytest.raises(menu.OperationCancelled):
        menu.invoke_tool("resume", interactive=True)
    assert "training.validation_files" in capsys.readouterr().out


def test_resume_confirm_passes_force_and_reaps_monitor_on_training_error(monkeypatch):
    prepare_resume(monkeypatch, validation_files=15)
    monkeypatch.setattr("builtins.input", lambda _prompt: "Y")
    calls = []

    def run(module, *arguments, **kwargs):
        calls.append((module, arguments, kwargs))
        if module == "training.train":
            raise RuntimeError("训练失败")
        return 0

    monkeypatch.setattr(menu, "run_module", run)
    with pytest.raises(RuntimeError, match="训练失败"):
        menu.invoke_tool("resume", interactive=True)
    assert "--force-resume-data-mismatch" in calls[1][1]
    assert calls[0][1][:3] == ("--background", "--open-browser", "--if-enabled")
    assert calls[-1][1][0] == "--stop-owned"
    assert calls[0][1][-1] == calls[-1][1][-1]


def test_resume_unchanged_data_does_not_prompt_or_force(monkeypatch):
    prepare_resume(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("份数一致时不应询问"))
    calls = []
    monkeypatch.setattr(menu, "run_module", lambda module, *args, **kwargs: calls.append((module, args)) or 0)
    assert menu.invoke_tool("resume", interactive=True) == 0
    assert calls[1] == ("training.train", ("--resume", "unused.pt"))


def test_noninteractive_resume_never_reads_stdin(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("非交互 action 禁止读取 stdin"))
    monkeypatch.setattr(menu, "configured_checkpoint", lambda _path: selected_checkpoint())
    calls = []
    monkeypatch.setattr(menu, "run_module", lambda module, *args, **kwargs: calls.append((module, args, kwargs)) or 0)
    assert menu.main(["--action", "resume", "--checkpoint", "saved.pt", "--force-resume-data-mismatch"]) == 0
    assert "--force-resume-data-mismatch" in calls[1][1]


@pytest.mark.parametrize("action, message", [("annotate", "工具执行失败，Python 退出码：2"),
                                            ("export", "ONNX 图已导出，但 PT/ORT parity 门禁未通过")])
def test_tool_failure_returns_one_and_reports_error(monkeypatch, capsys, action, message):
    monkeypatch.setattr(menu, "configured_checkpoint", lambda _path: selected_checkpoint())
    monkeypatch.setattr(menu, "run_module", lambda *_args: 2)
    assert menu.main(["--action", action]) == 1
    assert f"执行失败：{message}" in capsys.readouterr().err


def test_checkpoint_unpickling_error_uses_unified_error_message(monkeypatch, capsys):
    def load(_path):
        raise pickle.UnpicklingError("损坏的 checkpoint")

    monkeypatch.setattr(menu, "configured_checkpoint", load)
    assert menu.main(["--action", "replay"]) == 1
    assert capsys.readouterr().err == "执行失败：损坏的 checkpoint\n"


def test_checkpoint_selection_sorts_best_last_and_keeps_grpo_namespace(monkeypatch, capsys):
    candidates = [selected_checkpoint(name="best.pt"), selected_checkpoint(),
                  selected_checkpoint(name="grpo.pt", source="grpo")]
    monkeypatch.setattr("builtins.input", lambda _prompt: "A1")
    assert menu.select_checkpoint(candidates)["source"] == "grpo"
    output = capsys.readouterr().out
    assert output.index("epoch_002.pt") < output.index("best.pt") < output.index("grpo.pt")
    assert "a1." in output


def test_grpo_checkpoint_is_rejected_for_bc_resume(monkeypatch):
    monkeypatch.setattr(menu, "configured_checkpoint", lambda _path: selected_checkpoint(source="grpo"))
    monkeypatch.setattr(menu, "run_module", lambda *_args, **_kwargs: pytest.fail("检查来源应早于监控和训练"))
    with pytest.raises(ValueError, match="仅支持 BC"):
        menu.invoke_tool("resume")


def test_linux_shell_entrypoints_syntax_help_and_permissions(tmp_path):
    root = Path(__file__).resolve().parents[2]
    # 仅复制帮助所需的标准库模块，证明未创建 .venv 时也能查看帮助。
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/ffxiv_ccg.py").write_bytes((root / "scripts/ffxiv_ccg.py").read_bytes())
    for name in ("setup.sh", "ffxiv_ccg.sh"):
        path = root / name
        assert path.stat().st_mode & 0o111
        assert path.read_bytes().startswith(b"#!/usr/bin/env bash\n")
        subprocess.run(["bash", "-n", str(path)], check=True)
        isolated = tmp_path / name
        isolated.write_bytes(path.read_bytes())
        isolated.chmod(path.stat().st_mode)
        result = subprocess.run([str(isolated), "--help"], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert ("./setup.sh [--help]" if name == "setup.sh" else "--action") in result.stdout

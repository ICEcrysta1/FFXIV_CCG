"""ONNX 根目录 `.env` 配置与 checkpoint 路径映射测试。"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.onnx_export.config.config import (
    derive_onnx_output_dir,
    load_export_config,
    load_parity_config,
)
from scripts.onnx_export import __main__ as export_main
from scripts.onnx_export import workflow
from scripts.onnx_export.release.policy import minimum_empty_action_budget
from scripts.onnx_export.runtime.runtime_targets import (
    BF16_TARGET_ONNX_VERSION,
    BF16_TARGET_ONNXSCRIPT_VERSION,
    BF16_TARGET_ORT_VERSION,
    BF16_TARGET_TORCH_VERSION,
    validate_bf16_export_environment_versions,
)


def test_checkpoint_path_derives_matching_export_directory(tmp_path):
    checkpoint = (
        tmp_path
        / "artifacts"
        / "checkpoints"
        / "black_mage"
        / "artzip_bc"
        / "best.pt"
    )

    output = derive_onnx_output_dir(checkpoint, project_root=tmp_path)

    assert output == (
        tmp_path / "artifacts" / "exports" / "black_mage" / "artzip_bc"
    ).resolve()


def test_checkpoint_outside_artifacts_requires_explicit_package(tmp_path):
    with pytest.raises(ValueError, match="AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE"):
        derive_onnx_output_dir(tmp_path / "best.pt", project_root=tmp_path)


def test_export_config_reads_shared_and_export_env(monkeypatch, tmp_path):
    checkpoint = tmp_path / "best.pt"
    output = tmp_path / "deployment"
    profile = tmp_path / "profile.json"
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE", str(output))
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_ORT_PROVIDER", "CPUExecutionProvider")
    monkeypatch.setenv("ONNX_EXPORT_DEPLOYMENT_PROFILE", str(profile))
    monkeypatch.setenv("ONNX_EXPORT_OPSET", "19")
    monkeypatch.setenv("ONNX_EXPORT_PRECISION", "float32")
    monkeypatch.setenv("ONNX_EXPORT_VALIDATION_DEVICES", "cpu,cuda")
    monkeypatch.setenv("ONNX_EXPORT_OVERWRITE", "true")

    config = load_export_config()

    assert config.checkpoint_path == checkpoint.resolve()
    assert config.output_dir == output.resolve()
    assert config.deployment_profile_path == profile.resolve()
    assert config.opset == 19
    assert config.precision == "float32"
    assert config.ort_provider == "CPUExecutionProvider"
    assert config.validation_devices == ("cpu", "cuda")
    assert config.overwrite is True
    assert not hasattr(config, "history_capacity")


def test_parity_config_reads_dedicated_env(monkeypatch, tmp_path):
    empty_report = tmp_path / "empty.json"
    scene_report = tmp_path / "scene.json"
    empty_max_gcds = 129
    empty_max_steps = minimum_empty_action_budget(empty_max_gcds)
    monkeypatch.setenv("ONNX_PARITY_EMPTY_MAX_STEPS", str(empty_max_steps))
    monkeypatch.setenv("ONNX_PARITY_EMPTY_MAX_GCDS", str(empty_max_gcds))
    monkeypatch.setenv("ONNX_PARITY_EMPTY_REPORT", str(empty_report))
    monkeypatch.setenv("ONNX_PARITY_SCENE_MAX_STEPS", "120")
    monkeypatch.setenv("ONNX_PARITY_SCENE_REPORT", str(scene_report))
    monkeypatch.setenv("ONNX_PARITY_TOLERANCE", "0.125")

    config = load_parity_config()

    assert config.empty_max_steps == empty_max_steps
    assert config.empty_max_gcds == empty_max_gcds
    assert config.empty_report_path == empty_report.resolve()
    assert config.scene_max_steps == 120
    assert config.scene_report_path == scene_report.resolve()
    assert config.tolerance == 0.125


def test_parity_config_derives_default_empty_action_budget(monkeypatch):
    monkeypatch.setattr(
        "scripts.onnx_export.config.config.load_root_dotenv",
        lambda _root: None,
    )
    for name in (
        "ONNX_PARITY_EMPTY_MAX_STEPS",
        "ONNX_PARITY_EMPTY_MAX_GCDS",
        "ONNX_PARITY_TOLERANCE",
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_parity_config()

    assert config.empty_max_gcds == 128
    assert config.empty_max_steps == minimum_empty_action_budget(128) == 544


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_parity_tolerance_rejects_non_finite_or_negative(monkeypatch, value):
    monkeypatch.setenv("ONNX_PARITY_TOLERANCE", value)

    with pytest.raises(ValueError, match="finite and >= 0"):
        load_parity_config()


def test_export_cli_accepts_no_arguments_and_uses_env_config(monkeypatch, tmp_path):
    checkpoint = tmp_path / "best.pt"
    output = tmp_path / "deployment"
    config = SimpleNamespace(
        checkpoint_path=checkpoint,
        output_dir=output,
        deployment_profile_path=None,
        opset=18,
        precision="bf16",
        ort_provider="CUDAExecutionProvider",
        validation_devices=("cuda",),
        overwrite=True,
    )
    received = []
    monkeypatch.setattr(export_main, "load_export_config", lambda **_kwargs: config)
    monkeypatch.setattr(
        export_main,
        "export_from_config",
        lambda value: received.append(value) or output,
    )
    monkeypatch.setattr(sys, "argv", ["scripts.onnx_export"])

    assert export_main.main() == 0
    assert received == [config]


def test_full_workflow_returns_gate_exit_without_raising(monkeypatch, capsys):
    monkeypatch.setattr(workflow, "check_environment", lambda: None)
    monkeypatch.setattr(workflow, "run_export", lambda: None)
    monkeypatch.setattr(workflow, "print_release_status", lambda: None)

    def fail_parity(scenario):
        raise AssertionError(f"{scenario} mismatch")

    monkeypatch.setattr(workflow, "run_parity", fail_parity)

    assert workflow._run_all() == 2
    output = capsys.readouterr().out
    assert "部署包状态为 parity_failed" in output
    assert "empty mismatch" in output
    assert "scene mismatch" in output


def test_full_workflow_reports_export_failure_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(workflow, "check_environment", lambda: None)
    monkeypatch.setattr(
        workflow,
        "run_export",
        lambda: (_ for _ in ()).throw(FileExistsError("deployment exists")),
    )
    monkeypatch.setattr(
        workflow,
        "run_parity",
        lambda _scenario: pytest.fail("parity must not run after export failure"),
    )

    assert workflow._run_all() == 1
    assert "ONNX 导出失败，已停止发布流程：deployment exists" in (
        capsys.readouterr().out
    )


def test_full_workflow_reports_environment_failure_without_traceback(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        workflow,
        "check_environment",
        lambda: (_ for _ in ()).throw(RuntimeError("onnxscript mismatch")),
    )
    monkeypatch.setattr(
        workflow,
        "run_export",
        lambda: pytest.fail("export must not run after environment failure"),
    )

    assert workflow._run_all() == 1
    assert "ONNX 环境检查失败，已停止发布流程：onnxscript mismatch" in (
        capsys.readouterr().out
    )


def test_bf16_export_environment_versions_are_strictly_pinned():
    validate_bf16_export_environment_versions(
        torch_version=BF16_TARGET_TORCH_VERSION,
        onnx_version=BF16_TARGET_ONNX_VERSION,
        onnxscript_version=BF16_TARGET_ONNXSCRIPT_VERSION,
        ort_version=BF16_TARGET_ORT_VERSION,
    )

    with pytest.raises(RuntimeError, match="onnxscript=0.8.0"):
        validate_bf16_export_environment_versions(
            torch_version=BF16_TARGET_TORCH_VERSION,
            onnx_version=BF16_TARGET_ONNX_VERSION,
            onnxscript_version="0.8.0",
            ort_version=BF16_TARGET_ORT_VERSION,
        )


def test_formal_workflow_uses_fixed_precision_tolerance(monkeypatch, tmp_path):
    export_config = SimpleNamespace(
        checkpoint_path=tmp_path / "best.pt",
        output_dir=tmp_path / "deployment",
        precision="bf16",
        ort_provider="CUDAExecutionProvider",
    )
    parity_config = SimpleNamespace(
        empty_max_steps=minimum_empty_action_budget(128),
        empty_max_gcds=128,
        empty_report_path=tmp_path / "empty.json",
        scene_max_steps=100,
        scene_report_path=tmp_path / "scene.json",
        tolerance=None,
    )
    replay_config = SimpleNamespace()
    received = {}
    monkeypatch.setattr(workflow, "load_export_config", lambda: export_config)
    monkeypatch.setattr(workflow, "load_parity_config", lambda: parity_config)
    monkeypatch.setattr(
        workflow.DeploymentManifest,
        "load",
        lambda *_args, **_kwargs: SimpleNamespace(
            contract=SimpleNamespace(precision="bf16")
        ),
    )
    monkeypatch.setattr(
        workflow,
        "load_replay_config",
        lambda **_kwargs: replay_config,
    )

    def run_parity(_config, **kwargs):
        received.update(kwargs)
        return tmp_path / "empty.json"

    monkeypatch.setattr(workflow, "run_rollout_parity", run_parity)

    workflow.run_parity("empty")

    assert received["tolerance"] == pytest.approx(0.25)
    assert received["release_gate"] is True

    parity_config.empty_max_steps = minimum_empty_action_budget(128) - 1
    with pytest.raises(ValueError, match="action budget is too small"):
        workflow.run_parity("empty")
    parity_config.empty_max_steps = minimum_empty_action_budget(128)

    parity_config.tolerance = 1.0
    with pytest.raises(ValueError, match="fixed by manifest precision"):
        workflow.run_parity("empty")

    parity_config.tolerance = None
    export_config.precision = "float32"
    with pytest.raises(ValueError, match="differs from the existing deployment package"):
        workflow.run_parity("empty")

    export_config.precision = "bf16"
    parity_config.scene_max_steps = 99
    with pytest.raises(ValueError, match="max_steps >= 100"):
        workflow.run_parity("scene")

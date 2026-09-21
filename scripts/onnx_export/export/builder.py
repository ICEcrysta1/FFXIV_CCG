"""按阶段编排 ONNX 导出，不承载具体门禁实现。"""

from __future__ import annotations

from pathlib import Path

from .checkpoint import load_policy_contracts
from .documents import write_package_documents
from .environment import validate_export_environment
from .graph import export_and_stamp
from .runtime import verify_runtime


def build_package(
    *,
    checkpoint_path: Path,
    package_dir: Path,
    deployment_profile_path: Path | None,
    opset: int,
    precision: str,
    ort_provider: str,
    validation_devices: tuple[str, ...],
) -> None:
    """执行完整导出阶段；具体环境、图、运行时和文档逻辑由专职模块负责。"""
    onnx, ort, onnxscript = validate_export_environment(
        precision=precision,
        ort_provider=ort_provider,
        validation_devices=validation_devices,
    )
    contracts = load_policy_contracts(
        checkpoint_path=checkpoint_path,
        deployment_profile_path=deployment_profile_path,
        precision=precision,
    )
    artifacts = export_and_stamp(
        onnx=onnx,
        contracts=contracts,
        package_dir=package_dir,
        opset=opset,
        precision=precision,
    )
    validation = verify_runtime(
        ort=ort,
        contracts=contracts,
        artifacts=artifacts,
        ort_provider=ort_provider,
        validation_devices=validation_devices,
    )
    write_package_documents(
        checkpoint_path=checkpoint_path,
        package_dir=package_dir,
        opset=opset,
        precision=precision,
        ort_provider=ort_provider,
        onnx=onnx,
        ort=ort,
        onnxscript=onnxscript,
        contracts=contracts,
        artifacts=artifacts,
        validation=validation,
    )

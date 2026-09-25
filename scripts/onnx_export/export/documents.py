"""golden、容量报告、导出报告和部署 manifest 的落盘与自校验。"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..contracts.contract import OUTPUT_NAMES, TENSOR_INPUT_NAMES
from ..contracts.deployment_contract import (
    DEPLOYMENT_MANIFEST_VERSION,
    MANIFEST_SCHEMA_FILENAME,
    DeploymentManifest,
)
from ..io.artifact_io import file_sha256, write_deterministic_npz
from ..runtime.ort_runtime import is_cpu_fallback_disabled
from ..runtime.runtime_targets import runtime_targets
from ..runtime.tensor_runtime import (
    GOLDEN_FORMAT,
    golden_encoding,
    tensor_to_golden_array,
)
from .context import ExportArtifacts, ExportContracts, RuntimeValidation


def write_package_documents(
    *,
    checkpoint_path: Path,
    package_dir: Path,
    opset: int,
    precision: str,
    ort_provider: str,
    onnx,
    ort,
    onnxscript,
    contracts: ExportContracts,
    artifacts: ExportArtifacts,
    validation: RuntimeValidation,
) -> None:
    """落盘 golden、报告和 manifest，并执行最终 manifest 自校验。"""
    input_arrays = {
        name: tensor_to_golden_array(tensor)
        for name, tensor in zip(
            TENSOR_INPUT_NAMES,
            validation.runtime_golden_inputs,
            strict=True,
        )
    }
    output_arrays = {
        OUTPUT_NAMES[0]: tensor_to_golden_array(validation.runtime_golden_outputs)
    }
    write_deterministic_npz(package_dir / "golden_inputs.npz", input_arrays)
    write_deterministic_npz(package_dir / "golden_outputs.npz", output_arrays)
    _write_json(package_dir / "capacity_report.json", contracts.capacity_report)
    schema_source = Path(__file__).resolve().parents[1] / MANIFEST_SCHEMA_FILENAME
    schema_target = package_dir / MANIFEST_SCHEMA_FILENAME
    shutil.copyfile(schema_source, schema_target)

    checkpoint_hash_after = file_sha256(checkpoint_path)
    if contracts.checkpoint_hash != checkpoint_hash_after:
        raise RuntimeError("checkpoint changed during export")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "graph_validated",
        "release_gate": "requires_rollout_parity",
        "checkpoint_sha256_before": contracts.checkpoint_hash,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "deployment_contract_sha256": contracts.contract_sha256,
        "capacity_report_sha256": contracts.capacity_report["semantic_sha256"],
        "deployment_profile": str(contracts.resolved_profile_path),
        "onnx_checker": "passed",
        "shape_inference": "passed",
        "ort_session": "passed",
        "ort_provider": validation.active_ort_providers[0],
        "ort_provider_chain": list(validation.active_ort_providers),
        "ort_requested_provider_chain": list(validation.resolved_ort_providers),
        "ort_cpu_fallback_disabled": is_cpu_fallback_disabled(ort_provider),
        "runtime_targets": runtime_targets(
            precision=precision,
            ort_version=str(ort.__version__),
            provider=validation.active_ort_providers[0],
        ),
        "precision": precision,
        "golden_format": GOLDEN_FORMAT,
        "golden_float_encoding": golden_encoding(precision),
        "model_alignment": artifacts.alignment,
        "pytorch_padding_matrix": validation.pytorch_matrix,
        "ort_padding_matrix": validation.ort_matrix,
    }
    _write_json(package_dir / "export_report.json", report)

    manifest = build_manifest(
        checkpoint_path=checkpoint_path,
        package_dir=package_dir,
        opset=opset,
        precision=precision,
        onnx=onnx,
        ort=ort,
        onnxscript=onnxscript,
        contracts=contracts,
        artifacts=artifacts,
        schema_target=schema_target,
    )
    _write_json(package_dir / "manifest.json", manifest)
    DeploymentManifest.load(package_dir / "manifest.json", verify_files=True)


def build_manifest(
    *,
    checkpoint_path: Path,
    package_dir: Path,
    opset: int,
    precision: str,
    onnx,
    ort,
    onnxscript,
    contracts: ExportContracts,
    artifacts: ExportArtifacts,
    schema_target: Path,
) -> dict[str, object]:
    """装配部署 manifest；文件哈希由最终落盘内容计算。"""
    return {
        "$schema": MANIFEST_SCHEMA_FILENAME,
        "manifest_version": DEPLOYMENT_MANIFEST_VERSION,
        "format": "onnx",
        "opset": opset,
        "contract": contracts.contract_payload,
        "checkpoint": {
            "filename": checkpoint_path.name,
            "sha256": contracts.checkpoint_hash,
        },
        "model": {
            "filename": artifacts.model_path.name,
            "model_variant": contracts.model_variant,
            "compute_precision": (
                "float32"
                if precision == "bf16" and contracts.policy.compute_model is not None
                else precision
            ),
            "sha256": file_sha256(artifacts.model_path),
            "size_bytes": artifacts.model_path.stat().st_size,
            "external_data": bool(artifacts.external_files),
            "external_files": artifacts.external_artifacts,
            **artifacts.alignment,
            "contract_sha256": contracts.contract_sha256,
            "checkpoint_sha256": contracts.checkpoint_hash,
        },
        "capacity_report": {
            "filename": "capacity_report.json",
            "sha256": file_sha256(package_dir / "capacity_report.json"),
        },
        "golden": {
            "case": "scene_valid=0,history_valid=0,random_padding",
            "format": GOLDEN_FORMAT,
            "float_encoding": golden_encoding(precision),
            "inputs": {
                "filename": "golden_inputs.npz",
                "sha256": file_sha256(package_dir / "golden_inputs.npz"),
            },
            "outputs": {
                "filename": "golden_outputs.npz",
                "sha256": file_sha256(package_dir / "golden_outputs.npz"),
            },
        },
        "exporter": {
            "path": "torch.onnx.export",
            "dynamo": True,
            "torch": str(torch.__version__),
            "onnx": str(onnx.__version__),
            "onnxscript": str(onnxscript.__version__),
            "onnxruntime": str(ort.__version__),
            "manifest_schema_sha256": file_sha256(schema_target),
        },
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

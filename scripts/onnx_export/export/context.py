"""ONNX 导出各阶段之间传递的不可变上下文与结果对象。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from common.policy.data import DataSpec

from ..contracts.contract import CapacityContract
from ..contracts.deployment_contract import DeploymentContract
from ..policy.policy import OnnxPolicy


@dataclass(frozen=True)
class ExportContracts:
    """导出期间复用的 checkpoint 与部署契约上下文。"""

    checkpoint_hash: str
    policy: OnnxPolicy
    data_spec: DataSpec
    vocab_size: int
    dtype: torch.dtype
    model_variant: str
    resolved_profile_path: Path
    capacity_report: Mapping[str, object]
    contract: CapacityContract
    deployment_contract: DeploymentContract
    contract_payload: Mapping[str, object]
    contract_sha256: object


@dataclass(frozen=True)
class ExportArtifacts:
    """已导出并完成 ONNX 图验收的产物。"""

    model_path: Path
    external_files: list[str]
    external_artifacts: list[dict[str, object]]
    alignment: dict[str, object]
    golden_inputs: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class RuntimeValidation:
    """ORT 会话、golden 运行和精度矩阵的验收结果。"""

    resolved_ort_providers: tuple[str, ...]
    active_ort_providers: tuple[str, ...]
    runtime_golden_inputs: tuple[torch.Tensor, ...]
    runtime_golden_outputs: torch.Tensor
    pytorch_matrix: dict[str, object]
    ort_matrix: dict[str, object]

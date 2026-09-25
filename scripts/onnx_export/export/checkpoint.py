"""加载 checkpoint 并装配 ONNX 导出所需的部署契约。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch

from common.policy.config import ModelConfig
from common.policy.data import DataSpec, ModelInputContract
from common.policy.model import (
    CandidateTransformerModel,
    repetition_config_from_checkpoint,
)
from common.torch_serialization import safe_torch_load

from ..contracts.contract import CapacityContract
from ..contracts.deployment_contract import DeploymentContract
from ..contracts.deployment_profile import DeploymentProfile
from ..io.artifact_io import file_sha256
from ..policy.policy import OnnxPolicy
from ..runtime.precision import PRECISION_BF16, precision_torch_dtype
from .context import ExportContracts


def load_policy(
    checkpoint_path: Path,
    *,
    precision: str,
) -> tuple[OnnxPolicy, DataSpec, int, torch.dtype, Mapping[str, object]]:
    """只从 checkpoint 读取部署所需字段，不恢复训练状态。"""
    checkpoint = safe_torch_load(checkpoint_path, mmap=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")  # noqa: TRY004
    data_spec_payload = checkpoint.get("data_spec")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(data_spec_payload, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError(  # noqa: TRY004
            "checkpoint missing data_spec or model_state_dict"
        )
    data_spec = DataSpec.from_dict(dict(data_spec_payload))
    input_contract = ModelInputContract.from_checkpoint(checkpoint)
    input_contract.assert_matches_data_spec(data_spec)
    embedding = state_dict.get("input_encoder.skill_embed.weight")
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
        raise ValueError("checkpoint missing skill embedding weight")
    vocab_size = int(embedding.shape[0])
    if vocab_size < 2:
        raise ValueError("checkpoint vocab_size must be >= 2")
    model = CandidateTransformerModel(
        data_spec,
        CandidateTransformerModel.checkpoint_model_config(dict(checkpoint)),
        vocab_size=vocab_size,
    )
    model.load_state_dict(state_dict, strict=True)
    dtype = _precision_dtype(precision)
    model.to(device="cpu", dtype=dtype)
    model.eval()
    return OnnxPolicy(model, bf16_float_compute=precision == PRECISION_BF16), data_spec, vocab_size, dtype, checkpoint


def load_policy_contracts(
    *,
    checkpoint_path: Path,
    deployment_profile_path: Path | None,
    precision: str,
) -> ExportContracts:
    """加载 checkpoint、部署 profile 和部署契约。"""
    checkpoint_hash = file_sha256(checkpoint_path)
    policy, data_spec, vocab_size, dtype, checkpoint = load_policy(
        checkpoint_path,
        precision=precision,
    )
    input_contract = ModelInputContract.from_checkpoint(checkpoint)
    model_config = _required_mapping(checkpoint, "model_config")
    model_variant = _required_text(checkpoint, "model_variant")
    checkpoint_history_capacity = int(model_config.get("history_capacity", -1))
    if checkpoint_history_capacity < 1:
        raise ValueError("checkpoint.model_config.history_capacity must be >= 1")
    resolved_profile_path = (
        DeploymentProfile.default_path(data_spec.job_tag)
        if deployment_profile_path is None
        else Path(deployment_profile_path).resolve()
    )
    profile = DeploymentProfile.load(resolved_profile_path)
    if profile.job_tag != data_spec.job_tag:
        raise ValueError("deployment profile job differs from checkpoint DataSpec")
    # scene 容量以职业模型配置为权威来源（随 checkpoint 的 model_config
    # 进入部署契约），profile 只保留 vocab 与语料统计证据佐证该容量；
    # 旧 checkpoint 缺失该字段时与训练路径的恢复逻辑一致，回退默认值。
    scene_capacity = int(
        model_config.get("scene_capacity", ModelConfig.scene_capacity)
    )
    if scene_capacity < 1:
        raise ValueError("checkpoint.model_config.scene_capacity must be >= 1")
    measured_scene_max = int(profile.evidence.get("scene_length_max", 0))
    if measured_scene_max > scene_capacity:
        raise ValueError(
            "model_config.scene_capacity is below the measured corpus maximum: "
            f"{scene_capacity} < {measured_scene_max}"
        )
    capacity_report = profile.to_capacity_report(
        scene_capacity=scene_capacity,
        history_capacity=checkpoint_history_capacity,
    )
    contract = CapacityContract(
        scene_capacity=scene_capacity,
        history_capacity=checkpoint_history_capacity,
        candidate_count=data_spec.num_candidates,
    )
    deployment_contract = DeploymentContract.create(
        precision=precision,
        capacity=contract,
        data_spec=data_spec,
        input_contract=input_contract,
        model_config=model_config,
        repetition_config=asdict(repetition_config_from_checkpoint(checkpoint)),
        vocab_entries=profile.vocab_entries,
        capacity_report=capacity_report,
        embedding_vocab_size=vocab_size,
    )
    contract_payload = deployment_contract.to_dict()
    contract_sha256 = contract_payload["signatures"]["deployment_contract_sha256"]
    return ExportContracts(
        checkpoint_hash=checkpoint_hash,
        policy=policy,
        data_spec=data_spec,
        vocab_size=vocab_size,
        dtype=dtype,
        model_variant=model_variant,
        resolved_profile_path=resolved_profile_path,
        capacity_report=capacity_report,
        contract=contract,
        deployment_contract=deployment_contract,
        contract_payload=contract_payload,
        contract_sha256=contract_sha256,
    )


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(  # noqa: TRY004
            f"checkpoint missing {key}"
        )
    return value


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"checkpoint missing {key}")
    return value.strip()


def _precision_dtype(precision: str) -> torch.dtype:
    return precision_torch_dtype(precision)

"""真实 46.9M 参数 checkpoint 的固定容量 padding 回归。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from common.policy.config import ModelConfig
from common.policy.data.input_contract import INPUT_CONTRACT_VERSION
from common.torch_serialization import safe_torch_load
from scripts.onnx_export import CapacityContract, DeploymentManifest
from scripts.onnx_export.contracts.deployment_profile import DeploymentProfile
from scripts.onnx_export.export import export_package, load_policy
from scripts.onnx_export.runtime.validation import validate_pytorch_matrix

CHECKPOINT_PATH = Path("artifacts/checkpoints/black_mage/artzip_bc/best.pt")


def _load_current_checkpoint() -> dict[str, object]:
    """仅跳过实际启用旧 raw candidate scorer 的 checkpoint。"""
    checkpoint = safe_torch_load(CHECKPOINT_PATH)
    model_config = checkpoint.get("model_config")
    if (
        isinstance(model_config, dict)
        and model_config.get("scorer_use_raw_projection") is True
    ):
        pytest.skip(
            "real checkpoint enables removed scorer_use_raw_projection; "
            "retrain it with Transformer-only candidate scoring"
        )
    input_contract = checkpoint.get("input_contract")
    try:
        input_contract_version = int(
            input_contract.get("version", -1)
        ) if isinstance(input_contract, dict) else -1
    except (TypeError, ValueError):
        input_contract_version = -1
    if input_contract_version != INPUT_CONTRACT_VERSION:
        pytest.skip(
            "real checkpoint uses an older model input contract; "
            "retrain it with the current candidate set and architecture"
        )
    return checkpoint


@pytest.mark.skipif(not CHECKPOINT_PATH.is_file(), reason="real checkpoint artifact is absent")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_bf16_checkpoint_padding_matrix_cuda():
    _load_current_checkpoint()
    policy, data_spec, vocab_size, dtype, checkpoint = load_policy(
        CHECKPOINT_PATH,
        precision="bf16",
    )
    profile = DeploymentProfile.load(DeploymentProfile.default_path(data_spec.job_tag))
    # scene 容量以 checkpoint 的职业模型配置为准（旧 checkpoint 缺失时回退
    # ModelConfig 默认值），与导出侧读取逻辑保持一致，不硬编码具体数值。
    scene_capacity = int(
        checkpoint["model_config"].get(
            "scene_capacity",
            ModelConfig.scene_capacity,
        )
    )
    contract = CapacityContract(
        scene_capacity=scene_capacity,
        history_capacity=int(checkpoint["model_config"]["history_capacity"]),
        candidate_count=data_spec.num_candidates,
    )
    assert vocab_size == len(profile.vocab_entries) + 1
    required_positions = (
        contract.scene_capacity
        + contract.history_capacity
        + contract.candidate_token_count
        + 1
    )
    assert contract.total_token_count == required_positions
    contract.validate()
    assert torch.cuda.is_bf16_supported()
    results = validate_pytorch_matrix(
        policy,
        data_spec,
        contract,
        vocab_size=vocab_size,
        dtype=dtype,
        device=torch.device("cuda"),
    )
    assert len(results) == 6
    assert all(row["argmax_match"] for row in results)
    assert max(row["max_logit_abs_diff"] for row in results) <= 2.5e-1


@pytest.mark.skipif(not CHECKPOINT_PATH.is_file(), reason="real checkpoint artifact is absent")
@pytest.mark.skipif(
    os.environ.get("RUN_REAL_ONNX_EXPORT") != "1",
    reason="set RUN_REAL_ONNX_EXPORT=1 for the slow full export gate",
)
def test_real_checkpoint_full_export_profile_and_ort(tmp_path):
    """显式慢速门禁：真实 profile、契约、导出、checker 和 ORT 必须全链路通过。"""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        pytest.skip("ORT CUDAExecutionProvider is unavailable")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("native CUDA BF16 is unavailable")
    checkpoint = _load_current_checkpoint()
    profile_path = DeploymentProfile.default_path(
        str(checkpoint["data_spec"]["job_tag"])
    )
    profile = DeploymentProfile.load(profile_path)
    history_capacity = int(checkpoint["model_config"]["history_capacity"])
    candidate_count = int(checkpoint["data_spec"]["num_candidates"])
    scene_capacity = int(
        checkpoint["model_config"].get(
            "scene_capacity",
            ModelConfig.scene_capacity,
        )
    )
    required_positions = scene_capacity + history_capacity + candidate_count + 1
    embedding = checkpoint["model_state_dict"]["input_encoder.skill_embed.weight"]

    output = export_package(
        checkpoint_path=CHECKPOINT_PATH,
        output_dir=tmp_path / "deployment",
        deployment_profile_path=profile_path,
        opset=18,
        precision="bf16",
        ort_provider="CUDAExecutionProvider",
        validation_devices=("cuda",),
        overwrite=False,
    )

    assert (output / "model.onnx").is_file()
    assert (output / "manifest.json").is_file()
    assert int(embedding.shape[0]) == len(profile.vocab_entries) + 1 == 26

    manifest = DeploymentManifest.load(output / "manifest.json", verify_files=True)
    contract = manifest.contract
    assert manifest.payload["opset"] == 18
    assert contract.precision == "bf16"
    assert contract.data_spec.job_tag == "black_mage"
    assert contract.capacity.scene_capacity == int(
        checkpoint["model_config"].get(
            "scene_capacity",
            ModelConfig.scene_capacity,
        )
    )
    assert contract.capacity.history_capacity == history_capacity
    assert contract.capacity.candidate_count == contract.data_spec.num_candidates == 25
    assert contract.vocab_entries == profile.vocab_entries
    assert contract.capacity.total_token_count == required_positions

    report = json.loads(
        (output / "export_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "graph_validated"
    assert report["onnx_checker"] == "passed"
    assert report["shape_inference"] == "passed"
    assert report["ort_session"] == "passed"
    assert report["ort_provider"] == "CUDAExecutionProvider"
    assert report["precision"] == "bf16"

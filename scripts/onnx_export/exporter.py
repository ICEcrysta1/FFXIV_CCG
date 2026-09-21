"""checkpoint 到原子、可校验 ONNX 部署目录的导出编排。"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from common.policy.config import ModelConfig
from common.policy.data import DataSpec, ModelInputContract
from common.policy.model import (
    CandidateTransformerModel,
    repetition_config_from_checkpoint,
)
from common.torch_serialization import safe_torch_load

from .artifact_io import (
    file_sha256,
    normalize_torch_reports,
    write_deterministic_npz,
)
from .config import OnnxExportConfig
from .contract import OUTPUT_NAMES, TENSOR_INPUT_NAMES, CapacityContract, make_inputs
from .deployment_contract import (
    DEPLOYMENT_MANIFEST_VERSION,
    MANIFEST_SCHEMA_FILENAME,
    DeploymentContract,
    DeploymentManifest,
)
from .deployment_profile import DeploymentProfile
from .ort_runtime import (
    ORT_PROVIDER_CUDA,
    create_ort_session,
    is_cpu_fallback_disabled,
    resolve_ort_providers,
)
from .policy import OnnxPolicy
from .precision import (
    PRECISION_BF16,
    PRECISION_FLOAT16,
    PRECISION_FLOAT32,
    precision_initializer_dtype_name,
    precision_onnx_data_type,
    precision_tolerances,
    precision_torch_dtype,
)
from .runtime_targets import (
    runtime_targets,
    validate_bf16_export_environment_versions,
)
from .tensor_runtime import (
    GOLDEN_FORMAT,
    golden_encoding,
    run_ort_tensors,
    tensor_to_golden_array,
)
from .validation import validate_ort_matrix, validate_pytorch_matrix

PUBLISH_RENAME_ATTEMPTS = 5
PUBLISH_RENAME_DELAY_SECONDS = 0.5


@dataclass(frozen=True)
class _ExportContracts:
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
class _ExportArtifacts:
    """已导出并完成 ONNX 图验收的产物。"""

    model_path: Path
    external_files: list[str]
    external_artifacts: list[dict[str, object]]
    alignment: dict[str, object]
    golden_inputs: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _RuntimeValidation:
    """ORT 会话、golden 运行和精度矩阵的验收结果。"""

    resolved_ort_providers: tuple[str, ...]
    active_ort_providers: tuple[str, ...]
    runtime_golden_inputs: tuple[torch.Tensor, ...]
    runtime_golden_outputs: torch.Tensor
    pytorch_matrix: dict[str, object]
    ort_matrix: dict[str, object]


def export_from_config(config: OnnxExportConfig) -> Path:
    """执行已完成 `.env` / CLI 合并和校验的导出配置。"""
    return export_package(
        checkpoint_path=config.checkpoint_path,
        output_dir=config.output_dir,
        deployment_profile_path=config.deployment_profile_path,
        opset=config.opset,
        precision=config.precision,
        ort_provider=config.ort_provider,
        validation_devices=config.validation_devices,
        overwrite=config.overwrite,
    )


def export_package(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    opset: int,
    precision: str,
    deployment_profile_path: Path | None = None,
    ort_provider: str,
    validation_devices: tuple[str, ...],
    overwrite: bool,
) -> Path:
    """完整生成并验证部署包；验证失败时保持既有目录不变。"""
    checkpoint_path = checkpoint_path.resolve()
    output_dir = output_dir.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.export-", dir=output_dir.parent)
    )
    try:
        _build_package(
            checkpoint_path=checkpoint_path,
            package_dir=temp_dir,
            deployment_profile_path=deployment_profile_path,
            opset=opset,
            precision=precision,
            ort_provider=ort_provider,
            validation_devices=validation_devices,
        )
        _publish_directory(temp_dir, output_dir, overwrite=overwrite)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return output_dir


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
    return OnnxPolicy(model), data_spec, vocab_size, dtype, checkpoint


def _build_package(
    *,
    checkpoint_path: Path,
    package_dir: Path,
    deployment_profile_path: Path | None,
    opset: int,
    precision: str,
    ort_provider: str,
    validation_devices: tuple[str, ...],
) -> None:
    onnx, ort, onnxscript = _validate_export_environment(
        precision=precision,
        ort_provider=ort_provider,
        validation_devices=validation_devices,
    )
    contracts = _load_policy_contracts(
        checkpoint_path=checkpoint_path,
        deployment_profile_path=deployment_profile_path,
        precision=precision,
    )
    artifacts = _export_and_stamp(
        onnx=onnx,
        contracts=contracts,
        package_dir=package_dir,
        opset=opset,
        precision=precision,
    )
    validation = _verify_runtime(
        ort=ort,
        contracts=contracts,
        artifacts=artifacts,
        ort_provider=ort_provider,
        validation_devices=validation_devices,
    )
    _write_package_documents(
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


def _validate_export_environment(
    *,
    precision: str,
    ort_provider: str,
    validation_devices: tuple[str, ...],
):
    """导入导出依赖并执行精度、设备与 ORT provider 门禁。"""
    onnx, ort, onnxscript = _import_onnx_dependencies()
    if precision == PRECISION_BF16:
        if ort_provider != ORT_PROVIDER_CUDA:
            raise ValueError(
                "bf16 ONNX export requires explicit CUDAExecutionProvider; "
                "auto/CPU fallback is not a valid BF16 release gate"
            )
        if validation_devices != ("cuda",):
            raise ValueError(
                "bf16 ONNX export requires --validation-devices cuda"
            )
        validate_bf16_export_environment_versions(
            torch_version=torch.__version__,
            onnx_version=onnx.__version__,
            onnxscript_version=onnxscript.__version__,
            ort_version=ort.__version__,
        )
    resolved_ort_providers = resolve_ort_providers(ort, ort_provider)
    if precision == PRECISION_BF16:
        if not torch.cuda.is_available():
            raise RuntimeError("bf16 ONNX export validation requires PyTorch CUDA")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("current CUDA device does not support native BF16")
    if (
        precision == PRECISION_FLOAT16
        and resolved_ort_providers[0] == "CPUExecutionProvider"
    ):
        raise ValueError(
            "float16 ONNX export validation requires a non-CPU ORT provider; "
            "use CUDAExecutionProvider or export float32"
        )
    return onnx, ort, onnxscript


def _load_policy_contracts(
    *,
    checkpoint_path: Path,
    deployment_profile_path: Path | None,
    precision: str,
) -> _ExportContracts:
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
    return _ExportContracts(
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


def _export_and_stamp(
    *,
    onnx,
    contracts: _ExportContracts,
    package_dir: Path,
    opset: int,
    precision: str,
) -> _ExportArtifacts:
    """生成 golden 输入、导出 ONNX，并完成 metadata、图和参数精度验收。"""
    golden_inputs = make_inputs(
        contracts.data_spec,
        contracts.contract,
        vocab_size=contracts.vocab_size,
        scene_valid=0,
        history_valid=0,
        dtype=contracts.dtype,
        seed=20260812,
    )
    contracts.deployment_contract.validate_tensor_inputs(golden_inputs)
    with torch.no_grad():
        golden_outputs = contracts.policy(*golden_inputs)
    if not torch.isfinite(golden_outputs).all():
        raise AssertionError("PyTorch golden output contains NaN/Inf")

    model_path = package_dir / "model.onnx"
    torch.onnx.export(
        contracts.policy,
        golden_inputs,
        model_path,
        input_names=TENSOR_INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=opset,
        dynamo=True,
        external_data=False,
        report=True,
        verbose=False,
        artifacts_dir=package_dir,
    )
    normalize_torch_reports(package_dir)

    model_proto = onnx.load(model_path, load_external_data=True)
    _set_onnx_metadata(
        model_proto,
        {
            "ffxiv.checkpoint_sha256": contracts.checkpoint_hash,
            "ffxiv.deployment_contract_sha256": str(contracts.contract_sha256),
            "ffxiv.job_tag": contracts.data_spec.job_tag,
            "ffxiv.history_capacity": str(contracts.contract.history_capacity),
            "ffxiv.scene_capacity": str(contracts.contract.scene_capacity),
            "ffxiv.candidate_count": str(contracts.data_spec.num_candidates),
            "ffxiv.state_dim": str(contracts.data_spec.state_dim),
            "ffxiv.skill_feature_dim": str(contracts.data_spec.skill_feature_dim),
            "ffxiv.scene_dim": str(contracts.data_spec.scene_dim),
            "ffxiv.num_scene_types": str(contracts.data_spec.num_scene_types),
            "ffxiv.precision": precision,
        },
    )
    onnx.save(model_proto, model_path)
    model_proto = onnx.load(model_path, load_external_data=True)
    onnx.checker.check_model(model_proto)
    inferred = onnx.shape_inference.infer_shapes(model_proto)
    onnx.checker.check_model(inferred)
    _assert_onnx_metadata(
        model_proto,
        checkpoint_sha256=contracts.checkpoint_hash,
        contract_sha256=str(contracts.contract_sha256),
        precision=precision,
    )
    external_files = _external_data_files(model_proto)
    if model_path.stat().st_size < 2 * 1024**3 and external_files:
        raise AssertionError("sub-2GiB model unexpectedly uses external data")
    external_artifacts = []
    for relative_path in external_files:
        external_path = package_dir / relative_path
        if not external_path.is_file():
            raise FileNotFoundError(f"ONNX external data file missing: {relative_path}")
        external_artifacts.append(
            {
                "filename": relative_path,
                "sha256": file_sha256(external_path),
                "size_bytes": external_path.stat().st_size,
            }
        )

    alignment = _model_alignment(
        contracts.policy.model,
        model_proto,
        precision=precision,
        tensor_proto=onnx.TensorProto,
    )
    return _ExportArtifacts(
        model_path=model_path,
        external_files=external_files,
        external_artifacts=external_artifacts,
        alignment=alignment,
        golden_inputs=golden_inputs,
    )


def _verify_runtime(
    *,
    ort,
    contracts: _ExportContracts,
    artifacts: _ExportArtifacts,
    ort_provider: str,
    validation_devices: tuple[str, ...],
) -> _RuntimeValidation:
    """创建 ORT 会话并执行运行时契约与 PyTorch/ORT 精度矩阵门禁。"""
    session, resolved_ort_providers, active_ort_providers = create_ort_session(
        ort,
        artifacts.model_path,
        ort_provider,
    )
    golden_device = (
        torch.device("cuda")
        if active_ort_providers[0] == ORT_PROVIDER_CUDA
        else torch.device("cpu")
    )
    contracts.policy.to(device=golden_device, dtype=contracts.dtype)
    runtime_golden_inputs = tuple(
        tensor.to(device=golden_device) for tensor in artifacts.golden_inputs
    )
    with torch.no_grad():
        runtime_golden_outputs = contracts.policy(*runtime_golden_inputs)
    if not torch.isfinite(runtime_golden_outputs).all():
        raise AssertionError("PyTorch runtime golden output contains NaN/Inf")
    _assert_runtime_contract(
        session,
        runtime_golden_inputs,
        runtime_golden_outputs,
        deployment_contract=contracts.deployment_contract,
    )

    pytorch_matrix: dict[str, object] = {}
    for device_name in validation_devices:
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA padding validation requested but CUDA is unavailable")
        pytorch_matrix[device_name] = validate_pytorch_matrix(
            contracts.policy,
            contracts.data_spec,
            contracts.contract,
            vocab_size=contracts.vocab_size,
            dtype=contracts.dtype,
            device=torch.device(device_name),
        )
    ort_matrix = validate_ort_matrix(
        session,
        contracts.policy,
        contracts.data_spec,
        contracts.contract,
        vocab_size=contracts.vocab_size,
        dtype=contracts.dtype,
        reference_device=golden_device,
    )
    return _RuntimeValidation(
        resolved_ort_providers=tuple(resolved_ort_providers),
        active_ort_providers=tuple(active_ort_providers),
        runtime_golden_inputs=runtime_golden_inputs,
        runtime_golden_outputs=runtime_golden_outputs,
        pytorch_matrix=pytorch_matrix,
        ort_matrix=ort_matrix,
    )


def _write_package_documents(
    *,
    checkpoint_path: Path,
    package_dir: Path,
    opset: int,
    precision: str,
    ort_provider: str,
    onnx,
    ort,
    onnxscript,
    contracts: _ExportContracts,
    artifacts: _ExportArtifacts,
    validation: _RuntimeValidation,
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
    schema_source = Path(__file__).with_name(MANIFEST_SCHEMA_FILENAME)
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

    manifest = _build_manifest(
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


def _build_manifest(
    *,
    checkpoint_path: Path,
    package_dir: Path,
    opset: int,
    precision: str,
    onnx,
    ort,
    onnxscript,
    contracts: _ExportContracts,
    artifacts: _ExportArtifacts,
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


def _assert_runtime_contract(
    session,
    inputs,
    expected_output,
    *,
    deployment_contract: DeploymentContract,
) -> None:
    runtime_inputs = session.get_inputs()
    runtime_outputs = session.get_outputs()
    expected_inputs = deployment_contract.tensor_inputs()
    expected_outputs = deployment_contract.tensor_outputs()
    if [value.name for value in runtime_inputs] != [spec.name for spec in expected_inputs]:
        raise AssertionError("ORT input names differ from deployment contract")
    if [value.name for value in runtime_outputs] != [spec.name for spec in expected_outputs]:
        raise AssertionError("ORT output names differ from deployment contract")
    for runtime, spec in zip(runtime_inputs, expected_inputs, strict=True):
        if list(runtime.shape) != list(spec.shape):
            raise AssertionError(f"ORT shape mismatch for {runtime.name}")
        if runtime.type != spec.dtype:
            raise AssertionError(f"ORT dtype mismatch for {runtime.name}")
    for runtime, spec in zip(runtime_outputs, expected_outputs, strict=True):
        if list(runtime.shape) != list(spec.shape):
            raise AssertionError(f"ORT shape mismatch for {runtime.name}")
        if runtime.type != spec.dtype:
            raise AssertionError(f"ORT dtype mismatch for {runtime.name}")
    actual = run_ort_tensors(
        session,
        dict(zip(TENSOR_INPUT_NAMES, inputs, strict=True)),
        OUTPUT_NAMES,
    )[0]
    expected = expected_output.detach()
    if not torch.isfinite(actual).all():
        raise AssertionError("ORT golden output contains NaN/Inf")
    rtol, atol = precision_tolerances(expected_output.dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    if not torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1)):
        raise AssertionError("ORT/PyTorch golden argmax mismatch")


def _model_alignment(
    model: torch.nn.Module,
    model_proto,
    *,
    precision: str,
    tensor_proto,
) -> dict[str, object]:
    parameters = tuple(model.parameters())
    state_tensors = tuple(
        value for value in model.state_dict().values() if torch.is_tensor(value)
    )
    pt_parameter_count = sum(parameter.numel() for parameter in parameters)
    pt_state_bytes = sum(value.numel() * value.element_size() for value in state_tensors)
    float_types = {
        tensor_proto.FLOAT: (
            precision_initializer_dtype_name(PRECISION_FLOAT32),
            torch.float32,
        ),
        tensor_proto.FLOAT16: (
            precision_initializer_dtype_name(PRECISION_FLOAT16),
            torch.float16,
        ),
        tensor_proto.DOUBLE: ("float64", torch.float64),
        tensor_proto.BFLOAT16: (
            precision_initializer_dtype_name(PRECISION_BF16),
            torch.bfloat16,
        ),
    }
    float_sizes = {
        data_type: torch.empty((), dtype=torch_dtype).element_size()
        for data_type, (_type_name, torch_dtype) in float_types.items()
    }
    float_initializers = [
        initializer
        for initializer in model_proto.graph.initializer
        if initializer.data_type in float_types
    ]
    onnx_float_initializer_count = sum(
        math.prod(int(dimension) for dimension in initializer.dims)
        for initializer in float_initializers
    )
    onnx_float_initializer_bytes = sum(
        math.prod(int(dimension) for dimension in initializer.dims)
        * float_sizes[initializer.data_type]
        for initializer in float_initializers
    )
    dtype_counts = {
        type_name: sum(
            math.prod(int(dimension) for dimension in initializer.dims)
            for initializer in float_initializers
            if initializer.data_type == data_type
        )
        for data_type, (type_name, _torch_dtype) in float_types.items()
    }
    expected_data_type = precision_onnx_data_type(precision, tensor_proto)
    expected_initializer_count = dtype_counts[
        precision_initializer_dtype_name(precision)
    ]
    other_float_initializer_count = (
        onnx_float_initializer_count - expected_initializer_count
    )
    other_float_initializers = [
        initializer
        for initializer in float_initializers
        if initializer.data_type != expected_data_type
        and not _is_rope_frequency_initializer(
            initializer,
            model,
            float_data_type=tensor_proto.FLOAT,
        )
    ]
    other_float_initializer_max_elements = max(
        (
            math.prod(int(dimension) for dimension in initializer.dims)
            for initializer in other_float_initializers
        ),
        default=0,
    )
    if other_float_initializer_max_elements > 1:
        raise AssertionError(
            "ONNX contains a non-scalar floating initializer outside requested precision: "
            f"precision={precision}, dtype_counts={dtype_counts}"
        )
    return {
        "pt_parameter_count": int(pt_parameter_count),
        "pt_state_tensor_count": len(state_tensors),
        "pt_state_bytes": int(pt_state_bytes),
        "onnx_float_initializer_count": int(onnx_float_initializer_count),
        "onnx_float_initializer_bytes": int(onnx_float_initializer_bytes),
        "onnx_float_initializer_dtype_counts": dtype_counts,
        "onnx_requested_precision_initializer_count": int(
            expected_initializer_count
        ),
        "onnx_other_float_initializer_count": int(
            other_float_initializer_count
        ),
        "onnx_other_float_initializer_max_elements": int(
            other_float_initializer_max_elements
        ),
        "retained_float_initializer_ratio": (
            float(onnx_float_initializer_count) / float(pt_parameter_count)
        ),
    }


def _is_rope_frequency_initializer(
    initializer,
    model: torch.nn.Module,
    *,
    float_data_type: int,
) -> bool:
    """识别 RoPE 为三角函数保留的 FP32 频率常量。

    BF16/FP16 的 ONNX Runtime 不接受 BF16/FP16 作为 Cos/Sin 输入，
    因此这一个非训练参数常量必须保持 FP32。导出器可能把它保留为
    ``inv_freq``，也可能把它 reshape 成匿名 initializer；按值匹配可以
    避免依赖具体 exporter 命名。
    """
    if initializer.data_type != float_data_type:
        return False
    from common.policy.model.position_encoding import (
        RotaryPositionEncoding,
    )

    rotary = next(
        (
            module
            for module in model.modules()
            if isinstance(module, RotaryPositionEncoding)
        ),
        None,
    )
    if rotary is None or not hasattr(rotary, "inv_freq"):
        return False
    try:
        from onnx import numpy_helper

        actual = torch.from_numpy(numpy_helper.to_array(initializer).copy()).float()
    except (ImportError, TypeError, ValueError):
        return False
    expected = rotary.inv_freq.detach().to(device="cpu", dtype=torch.float32)
    if actual.numel() != expected.numel():
        return False
    return torch.equal(actual.reshape(-1), expected.reshape(-1))


def _set_onnx_metadata(model_proto, values: Mapping[str, str]) -> None:
    existing = {item.key: item for item in model_proto.metadata_props}
    for key, value in values.items():
        item = existing.get(key)
        if item is None:
            item = model_proto.metadata_props.add()
            item.key = key
        item.value = value


def _assert_onnx_metadata(
    model_proto,
    *,
    checkpoint_sha256: str,
    contract_sha256: str,
    precision: str,
) -> None:
    metadata = {item.key: item.value for item in model_proto.metadata_props}
    if metadata.get("ffxiv.checkpoint_sha256") != checkpoint_sha256:
        raise AssertionError("ONNX checkpoint signature metadata mismatch")
    if metadata.get("ffxiv.deployment_contract_sha256") != contract_sha256:
        raise AssertionError("ONNX deployment contract metadata mismatch")
    if metadata.get("ffxiv.precision") != precision:
        raise AssertionError("ONNX deployment precision metadata mismatch")


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


def _publish_directory(temp_dir: Path, output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        temp_dir.rename(output_dir)
        return
    if not overwrite:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    backup = output_dir.parent / f".{output_dir.name}.backup-{uuid.uuid4().hex}"
    _rename_with_retry(
        output_dir,
        backup,
        operation="move existing deployment package to backup",
    )
    try:
        _rename_with_retry(
            temp_dir,
            output_dir,
            operation="publish new deployment package",
        )
    except BaseException:
        _rename_with_retry(
            backup,
            output_dir,
            operation="restore previous deployment package",
        )
        raise
    shutil.rmtree(backup)


def _rename_with_retry(source: Path, destination: Path, *, operation: str) -> None:
    """重试 Windows 导出目录的短暂拒绝访问，最终失败时保留原始异常。"""
    for attempt in range(1, PUBLISH_RENAME_ATTEMPTS + 1):
        try:
            source.rename(destination)
            return
        except PermissionError as exc:
            if attempt == PUBLISH_RENAME_ATTEMPTS:
                raise PermissionError(
                    f"{operation} failed after {PUBLISH_RENAME_ATTEMPTS} attempts: "
                    f"{source} -> {destination}; close processes or file viewers "
                    "that use the deployment package and retry"
                ) from exc
            time.sleep(PUBLISH_RENAME_DELAY_SECONDS)


def _import_onnx_dependencies():
    try:
        import onnx
        import onnxruntime as ort
        import onnxscript
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ONNX export dependencies are missing; run "
            "python -m pip install -r requirements-onnx.txt"
        ) from exc
    return onnx, ort, onnxscript


def _external_data_files(model_proto) -> list[str]:
    files: set[str] = set()
    for initializer in model_proto.graph.initializer:
        for item in initializer.external_data:
            if item.key == "location" and item.value:
                files.add(item.value)
    return sorted(files)


def _precision_dtype(precision: str) -> torch.dtype:
    return precision_torch_dtype(precision)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

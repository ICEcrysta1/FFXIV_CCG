"""ONNX 图导出、metadata 盖章、checker 和参数精度审计。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import torch

from ..contracts.contract import OUTPUT_NAMES, TENSOR_INPUT_NAMES, make_inputs
from ..io.artifact_io import file_sha256, normalize_torch_reports
from ..runtime.precision import (
    PRECISION_BF16,
    PRECISION_FLOAT16,
    PRECISION_FLOAT32,
    precision_initializer_dtype_name,
    precision_onnx_data_type,
)
from .context import ExportArtifacts, ExportContracts


def export_and_stamp(
    *,
    onnx,
    contracts: ExportContracts,
    package_dir: Path,
    opset: int,
    precision: str,
) -> ExportArtifacts:
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
    if precision == PRECISION_BF16 and contracts.policy.compute_model is not None:
        _store_float_compute_weights_as_bf16(model_proto, onnx)
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
            "ffxiv.compute_precision": (
                "float32"
                if precision == PRECISION_BF16 and contracts.policy.compute_model is not None
                else precision
            ),
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
    external_files = external_data_files(model_proto)
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

    alignment = model_alignment(
        contracts.policy.model,
        model_proto,
        precision=precision,
        tensor_proto=onnx.TensorProto,
    )
    return ExportArtifacts(
        model_path=model_path,
        external_files=external_files,
        external_artifacts=external_artifacts,
        alignment=alignment,
        golden_inputs=golden_inputs,
    )


def _store_float_compute_weights_as_bf16(model_proto, onnx) -> None:
    """将量化后 FP32 计算图的常量还原为 BF16 存储，并显式恢复计算 dtype。"""
    from onnx import numpy_helper

    casts = []
    for initializer in model_proto.graph.initializer:
        if initializer.data_type != onnx.TensorProto.FLOAT or math.prod(initializer.dims) <= 1:
            continue
        values = torch.from_numpy(numpy_helper.to_array(initializer).copy())
        quantized = values.to(torch.bfloat16)
        if not torch.equal(values, quantized.float()):
            raise AssertionError(
                f"float compute initializer cannot be stored exactly as BF16: {initializer.name}"
            )
        compute_name = initializer.name
        initializer.name = f"{compute_name}_bf16_stored"
        initializer.data_type = onnx.TensorProto.BFLOAT16
        initializer.ClearField("float_data")
        initializer.raw_data = quantized.view(torch.uint16).numpy().tobytes()
        casts.append(
            onnx.helper.make_node(
                "Cast",
                [initializer.name],
                [compute_name],
                to=onnx.TensorProto.FLOAT,
            )
        )
    if not casts:
        raise AssertionError("BF16 storage conversion found no model initializers")
    for cast in reversed(casts):
        model_proto.graph.node.insert(0, cast)


def model_alignment(
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


def external_data_files(model_proto) -> list[str]:
    files: set[str] = set()
    for initializer in model_proto.graph.initializer:
        for item in initializer.external_data:
            if item.key == "location" and item.value:
                files.add(item.value)
    return sorted(files)

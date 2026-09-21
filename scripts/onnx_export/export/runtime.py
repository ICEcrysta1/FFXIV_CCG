"""ORT 会话、运行时契约和 PyTorch/ORT 精度矩阵验收。"""

from __future__ import annotations

import torch

from ..contracts.contract import OUTPUT_NAMES, TENSOR_INPUT_NAMES
from ..runtime.ort_runtime import ORT_PROVIDER_CUDA, create_ort_session
from ..runtime.precision import precision_tolerances
from ..runtime.tensor_runtime import run_ort_tensors
from ..runtime.validation import validate_ort_matrix, validate_pytorch_matrix
from .context import ExportArtifacts, ExportContracts, RuntimeValidation


def verify_runtime(
    *,
    ort,
    contracts: ExportContracts,
    artifacts: ExportArtifacts,
    ort_provider: str,
    validation_devices: tuple[str, ...],
) -> RuntimeValidation:
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
    assert_runtime_contract(
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
    return RuntimeValidation(
        resolved_ort_providers=tuple(resolved_ort_providers),
        active_ort_providers=tuple(active_ort_providers),
        runtime_golden_inputs=runtime_golden_inputs,
        runtime_golden_outputs=runtime_golden_outputs,
        pytorch_matrix=pytorch_matrix,
        ort_matrix=ort_matrix,
    )


def assert_runtime_contract(
    session,
    inputs,
    expected_output,
    *,
    deployment_contract,
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

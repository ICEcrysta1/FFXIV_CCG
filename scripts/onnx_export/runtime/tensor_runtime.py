"""PyTorch Tensor 与 ONNX Runtime 之间的无损部署 I/O。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


GOLDEN_FORMAT = "deterministic-npz-v2"
GOLDEN_NATIVE_ENCODING = "native-numpy"
GOLDEN_BF16_ENCODING = "uint16-little-endian-bfloat16-bits"


def run_ort_tensors(
    session,
    inputs: Mapping[str, torch.Tensor],
    output_names: Sequence[str] | None = None,
) -> tuple[torch.Tensor, ...]:
    """通过 DLPack 和 I/O Binding 执行 ORT，完整保留 BF16 dtype。"""
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ONNX Runtime tensor execution requires requirements-onnx.txt"
        ) from exc

    device = _session_torch_device(session)
    bound_tensors: list[torch.Tensor] = []
    bound_arrays: list[np.ndarray] = []
    bound_values = []
    binding = session.io_binding()
    device_id = 0 if device.index is None else device.index
    for name, tensor in inputs.items():
        value = tensor.detach().to(device=device).contiguous()
        if value.dtype == torch.bool:
            # 部分 ORT DLPack importer 仍拒绝 bool code；
            # bool 不涉及 BF16 精度，使用 ORT 自身的显式设备拷贝保持类型正确。
            array = value.cpu().numpy()
            bound_arrays.append(array)
            ort_value = ort.OrtValue.ortvalue_from_numpy(
                array,
                device_type=device.type,
                device_id=device_id,
            )
        else:
            ort_value = ort.OrtValue.from_dlpack(value)
        bound_tensors.append(value)
        bound_values.append(ort_value)
        binding.bind_ortvalue_input(name, ort_value)

    names = (
        tuple(item.name for item in session.get_outputs())
        if output_names is None
        else tuple(output_names)
    )
    for name in names:
        binding.bind_output(name, device.type, device_id)

    session.run_with_iobinding(binding)
    ort_outputs = binding.get_outputs()
    if len(ort_outputs) != len(names):
        raise RuntimeError(
            f"ORT returned {len(ort_outputs)} outputs, expected {len(names)}"
        )
    # torch.from_dlpack 会接管 OrtValue 暴露的 capsule 生命周期。
    return tuple(torch.from_dlpack(value) for value in ort_outputs)


def tensor_to_golden_array(tensor: torch.Tensor) -> np.ndarray:
    """把 Tensor 编码为确定性 NPZ 数组；BF16 以原始 16-bit 位模式保存。"""
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy()
    return value.numpy()


def golden_encoding(precision: str) -> str:
    """返回 manifest 中声明的浮点 Tensor 存储编码。"""
    return GOLDEN_BF16_ENCODING if precision == "bf16" else GOLDEN_NATIVE_ENCODING


def _session_torch_device(session) -> torch.device:
    providers = tuple(session.get_providers())
    if not providers:
        raise RuntimeError("ORT session has no active Execution Provider")
    primary = providers[0]
    if primary == "CUDAExecutionProvider":
        if not torch.cuda.is_available():
            raise RuntimeError("ORT CUDA session requires PyTorch CUDA for DLPack I/O")
        return torch.device("cuda")
    if primary == "CPUExecutionProvider":
        return torch.device("cpu")
    raise RuntimeError(
        "ORT Tensor I/O only supports CUDAExecutionProvider or CPUExecutionProvider, "
        f"got {primary!r}"
    )

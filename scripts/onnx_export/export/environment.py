"""ONNX 依赖、精度、设备和 ORT provider 的导出前门禁。"""

from __future__ import annotations

import torch

from ..runtime.ort_runtime import ORT_PROVIDER_CUDA, resolve_ort_providers
from ..runtime.precision import PRECISION_BF16, PRECISION_FLOAT16
from ..runtime.runtime_targets import validate_bf16_export_environment_versions


def validate_export_environment(
    *,
    precision: str,
    ort_provider: str,
    validation_devices: tuple[str, ...],
):
    """导入导出依赖并执行精度、设备与 ORT provider 门禁。"""
    onnx, ort, onnxscript = import_onnx_dependencies()
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


def import_onnx_dependencies():
    """延迟导入可选 ONNX 依赖，保持训练/回放基础导入轻量。"""
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

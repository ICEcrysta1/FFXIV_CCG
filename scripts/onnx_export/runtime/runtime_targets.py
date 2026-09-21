"""ONNX 正式部署运行库目标版本。"""

from __future__ import annotations

from collections.abc import Mapping

from .ort_runtime import ORT_PROVIDER_CUDA
from .precision import PRECISION_BF16


BF16_TARGET_ORT_VERSION = "1.27.0"
BF16_TARGET_TORCH_VERSION = "2.12.0+cu132"
BF16_TARGET_ONNX_VERSION = "1.22.0"
BF16_TARGET_ONNXSCRIPT_VERSION = "0.7.1"
BF16_PYTHON_ORT_PACKAGE = "onnxruntime-gpu"
BF16_DOTNET_ORT_PACKAGE = "Microsoft.ML.OnnxRuntime.Gpu"


def validate_bf16_export_environment_versions(
    *,
    torch_version: str,
    onnx_version: str,
    onnxscript_version: str,
    ort_version: str,
) -> None:
    """拒绝未经项目验证的 BF16 导出器与运行库组合。"""
    actual = {
        "torch": torch_version,
        "onnx": onnx_version,
        "onnxscript": onnxscript_version,
        "onnxruntime-gpu": ort_version,
    }
    expected = {
        "torch": BF16_TARGET_TORCH_VERSION,
        "onnx": BF16_TARGET_ONNX_VERSION,
        "onnxscript": BF16_TARGET_ONNXSCRIPT_VERSION,
        "onnxruntime-gpu": BF16_TARGET_ORT_VERSION,
    }
    mismatches = [
        f"{name}={actual[name]} (required {version})"
        for name, version in expected.items()
        if actual[name] != version
    ]
    if mismatches:
        raise RuntimeError(
            "formal BF16 export environment version mismatch: "
            + ", ".join(mismatches)
        )


def runtime_targets(
    *,
    precision: str,
    ort_version: str,
    provider: str,
) -> dict[str, object]:
    """生成语言无关的已验证运行库目标。"""
    if precision == PRECISION_BF16:
        if ort_version != BF16_TARGET_ORT_VERSION:
            raise RuntimeError(
                "BF16 export requires the pinned ONNX Runtime version: "
                f"{ort_version} != {BF16_TARGET_ORT_VERSION}"
            )
        if provider != ORT_PROVIDER_CUDA:
            raise RuntimeError("BF16 runtime target requires CUDAExecutionProvider")
        return {
            "execution_provider": ORT_PROVIDER_CUDA,
            "python": {
                "package": BF16_PYTHON_ORT_PACKAGE,
                "version": BF16_TARGET_ORT_VERSION,
            },
            "dotnet": {
                "package": BF16_DOTNET_ORT_PACKAGE,
                "version": BF16_TARGET_ORT_VERSION,
            },
        }
    return {
        "execution_provider": provider,
        "python": {
            "package": "onnxruntime",
            "version": ort_version,
        },
        "dotnet": {
            "package": "Microsoft.ML.OnnxRuntime",
            "version": ort_version,
        },
    }


def validate_runtime_targets(
    payload: Mapping[str, object],
    *,
    precision: str,
) -> None:
    """拒绝与当前正式 BF16 运行库矩阵不一致的 manifest。"""
    if precision != PRECISION_BF16:
        return
    expected = runtime_targets(
        precision=precision,
        ort_version=BF16_TARGET_ORT_VERSION,
        provider=ORT_PROVIDER_CUDA,
    )
    if dict(payload) != expected:
        raise ValueError("BF16 deployment runtime targets are unsupported")

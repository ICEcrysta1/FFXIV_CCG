"""ONNX 部署精度、Tensor dtype 和数值门槛的单一权威。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


PRECISION_BF16 = "bf16"
PRECISION_FLOAT16 = "float16"
PRECISION_FLOAT32 = "float32"
ONNX_TENSOR_BFLOAT16 = "tensor(bfloat16)"
ONNX_TENSOR_BOOL = "tensor(bool)"
ONNX_TENSOR_FLOAT = "tensor(float)"
ONNX_TENSOR_FLOAT16 = "tensor(float16)"
ONNX_TENSOR_INT64 = "tensor(int64)"


@dataclass(frozen=True)
class PrecisionSpec:
    """一种部署精度在 PyTorch、ORT 和 ONNX 图中的统一定义。"""

    torch_dtype: torch.dtype
    onnx_dtype: str
    tensor_proto_name: str
    initializer_dtype_name: str
    rtol: float
    atol: float


_PRECISION_SPECS = {
    PRECISION_BF16: PrecisionSpec(
        torch_dtype=torch.bfloat16,
        onnx_dtype=ONNX_TENSOR_BFLOAT16,
        tensor_proto_name="BFLOAT16",
        initializer_dtype_name="bfloat16",
        rtol=2.5e-1,
        atol=2.5e-1,
    ),
    PRECISION_FLOAT32: PrecisionSpec(
        torch_dtype=torch.float32,
        onnx_dtype=ONNX_TENSOR_FLOAT,
        tensor_proto_name="FLOAT",
        initializer_dtype_name="float32",
        rtol=1e-5,
        atol=1e-5,
    ),
    PRECISION_FLOAT16: PrecisionSpec(
        torch_dtype=torch.float16,
        onnx_dtype=ONNX_TENSOR_FLOAT16,
        tensor_proto_name="FLOAT16",
        initializer_dtype_name="float16",
        rtol=5e-3,
        atol=5e-3,
    ),
}
SUPPORTED_PRECISIONS = tuple(_PRECISION_SPECS)
_TORCH_PRECISION_SPECS = {
    spec.torch_dtype: spec for spec in _PRECISION_SPECS.values()
}
_ONNX_TORCH_DTYPES = {
    **{spec.onnx_dtype: spec.torch_dtype for spec in _PRECISION_SPECS.values()},
    ONNX_TENSOR_BOOL: torch.bool,
    ONNX_TENSOR_INT64: torch.int64,
}


def _precision_spec(precision: str) -> PrecisionSpec:
    try:
        return _PRECISION_SPECS[precision]
    except KeyError as exc:
        raise ValueError(
            f"unsupported ONNX deployment precision: {precision!r}; "
            f"expected one of {SUPPORTED_PRECISIONS}"
        ) from exc


def precision_torch_dtype(precision: str) -> torch.dtype:
    """把部署精度映射为 PyTorch dtype。"""
    return _precision_spec(precision).torch_dtype


def precision_onnx_dtype(precision: str) -> str:
    """返回 ORT session 元数据使用的 Tensor 类型名。"""
    return _precision_spec(precision).onnx_dtype


def precision_onnx_data_type(precision: str, tensor_proto) -> int:
    """使用调用方加载的 ``onnx.TensorProto`` 返回正式枚举值。"""
    spec = _precision_spec(precision)
    return int(getattr(tensor_proto, spec.tensor_proto_name))


def precision_initializer_dtype_name(precision: str) -> str:
    """返回 manifest initializer 审计使用的稳定 dtype 名。"""
    return _precision_spec(precision).initializer_dtype_name


def onnx_torch_dtype(onnx_dtype: str) -> torch.dtype:
    """把 manifest/ORT Tensor 类型映射回 PyTorch dtype。"""
    try:
        return _ONNX_TORCH_DTYPES[onnx_dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported ONNX tensor dtype: {onnx_dtype!r}") from exc


def precision_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """返回逐层/最终 logits 的相对和绝对一致性门槛。"""
    try:
        spec = _TORCH_PRECISION_SPECS[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported deployment dtype: {dtype}") from exc
    # BF16 容差覆盖真实 768-history 的后端级舍入差异；Top-1、Top-3 和
    # 最终动作仍由独立硬门禁约束。
    return spec.rtol, spec.atol


def parity_max_abs_tolerance(precision: str) -> float:
    """返回回放 raw logits 的最大绝对误差门槛。"""
    return precision_tolerances(precision_torch_dtype(precision))[1]

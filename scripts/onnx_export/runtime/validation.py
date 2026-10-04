"""完全因果布局下固定容量与动态输入的 PyTorch/ORT 验收。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from ..contracts.contract import (
    TENSOR_INPUT_NAMES,
    CapacityContract,
    fill_padding_values,
    make_inputs,
    slice_dynamic_inputs,
)
from ..policy.policy import _build_batch, _raw_logits as _policy_raw_logits
from .precision import precision_tolerances
from .tensor_runtime import run_ort_tensors


@dataclass(frozen=True)
class MatrixCaseResult:
    scene_valid: int
    history_valid: int
    max_logit_abs_diff: float
    max_dynamic_baseline_abs_diff: float
    argmax_match: bool
    padding_value_invariant: bool


def padding_matrix(contract: CapacityContract) -> tuple[tuple[int, int], ...]:
    """返回 Issue 3 约定的六类边界组合，并去除小容量下的重复项。"""
    cases = (
        (0, 0),
        (0, 1),
        (0, contract.history_capacity),
        (1, 0),
        (contract.scene_capacity - 1, contract.history_capacity - 1),
        (contract.scene_capacity, contract.history_capacity),
    )
    return tuple(dict.fromkeys(cases))


@torch.no_grad()
def validate_pytorch_matrix(
    policy,
    data_spec,
    contract: CapacityContract,
    *,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[dict[str, object]]:
    """校验 trace 有限性、padding 屏蔽、padding 值不变性和同输入一致性。

    逻辑位置排除 padding；同一有效上下文的动态输入和固定容量输入
    必须产生相同动作输出，同时验证不同 padding 载荷不影响输出。
    """
    policy.to(device=device, dtype=dtype)
    policy.eval()
    results: list[dict[str, object]] = []
    rtol, atol = _tolerances(dtype)
    for case_index, (scene_valid, history_valid) in enumerate(padding_matrix(contract)):
        fixed_cpu = make_inputs(
            data_spec,
            contract,
            vocab_size=vocab_size,
            scene_valid=scene_valid,
            history_valid=history_valid,
            dtype=dtype,
            seed=1701 + case_index,
        )
        zero_padding_cpu = fill_padding_values(
            fixed_cpu,
            scene_valid=scene_valid,
            history_valid=history_valid,
            value=0.0,
        )
        dynamic_cpu = slice_dynamic_inputs(
            fixed_cpu,
            scene_valid=scene_valid,
            history_valid=history_valid,
        )
        fixed = _to_device(fixed_cpu, device)
        zero_padding = _to_device(zero_padding_cpu, device)
        dynamic = _to_device(dynamic_cpu, device)

        dynamic_trace = policy.trace(*dynamic)
        fixed_trace = policy.trace(*fixed)
        zero_padding_trace = policy.trace(*zero_padding)
        raw_dynamic_trace = policy.model.trace(_build_batch(*dynamic))
        raw_dynamic_logits = _raw_logits(policy, raw_dynamic_trace)
        fixed_logits = policy(*fixed)
        zero_padding_logits = policy(*zero_padding)
        _assert_trace_finite(dynamic_trace)
        _assert_trace_finite(fixed_trace)
        _assert_trace_finite(zero_padding_trace)
        _assert_padding_keys_blocked(
            fixed_trace,
            scene_valid=scene_valid,
            history_valid=history_valid,
            contract=contract,
        )
        _assert_raw_trace_finite(raw_dynamic_trace, raw_dynamic_logits)
        _assert_same_input_trace_close(
            dynamic_trace,
            raw_dynamic_trace,
            raw_dynamic_logits,
            rtol=rtol,
            atol=atol,
        )
        torch.testing.assert_close(fixed_trace.logits, dynamic_trace.logits, rtol=rtol, atol=atol)
        dynamic_logits = policy(*dynamic)
        torch.testing.assert_close(fixed_logits, dynamic_logits, rtol=rtol, atol=atol)
        # trace 会为了保留逐层 attention 而走手工 attention；正式 forward
        # 会走 SDPA/导出路径。两条路径在 BF16 下允许有微小数值差异，
        # 因此 padding 不变性必须在各自的同一条路径内比较。
        torch.testing.assert_close(
            fixed_trace.logits,
            zero_padding_trace.logits,
            rtol=rtol,
            atol=atol,
        )
        torch.testing.assert_close(
            fixed_logits,
            zero_padding_logits,
            rtol=rtol,
            atol=atol,
        )
        trace_argmax_match = bool(
            torch.equal(
                fixed_trace.logits.argmax(dim=-1),
                zero_padding_trace.logits.argmax(dim=-1),
            )
        )
        if not trace_argmax_match:
            raise AssertionError(
                "fixed/zero-padding trace argmax mismatch: "
                f"scene_valid={scene_valid}, history_valid={history_valid}"
            )
        forward_argmax_match = bool(
            torch.equal(
                fixed_logits.argmax(dim=-1),
                zero_padding_logits.argmax(dim=-1),
            )
        )
        if not forward_argmax_match:
            raise AssertionError(
                "fixed/zero-padding forward argmax mismatch: "
                f"scene_valid={scene_valid}, history_valid={history_valid}"
            )
        dynamic_argmax_match = bool(torch.equal(fixed_logits.argmax(-1), dynamic_logits.argmax(-1)))
        if not dynamic_argmax_match:
            raise AssertionError("fixed/dynamic valid-context argmax mismatch")
        argmax_match = trace_argmax_match and forward_argmax_match and dynamic_argmax_match
        results.append(
            asdict(
                MatrixCaseResult(
                    scene_valid=scene_valid,
                    history_valid=history_valid,
                    max_logit_abs_diff=float(
                        max(
                            (fixed_trace.logits - zero_padding_trace.logits)
                            .abs()
                            .max()
                            .item(),
                            (fixed_logits - zero_padding_logits).abs().max().item(),
                        )
                    ),
                    max_dynamic_baseline_abs_diff=float(
                        max(
                            (dynamic_trace.logits - raw_dynamic_logits).abs().max().item(),
                            (fixed_logits - dynamic_logits).abs().max().item(),
                        )
                    ),
                    argmax_match=argmax_match,
                    padding_value_invariant=True,
                )
            )
        )
    return results


def validate_ort_matrix(
    session,
    policy,
    data_spec,
    contract: CapacityContract,
    *,
    vocab_size: int,
    dtype: torch.dtype,
    reference_device: torch.device,
) -> list[dict[str, object]]:
    """在目标 ORT EP 上执行同一固定容量边界矩阵。"""
    policy.to(device=reference_device, dtype=dtype)
    policy.eval()
    rtol, atol = _tolerances(dtype)
    results: list[dict[str, object]] = []
    for case_index, (scene_valid, history_valid) in enumerate(padding_matrix(contract)):
        fixed_cpu = make_inputs(
            data_spec,
            contract,
            vocab_size=vocab_size,
            scene_valid=scene_valid,
            history_valid=history_valid,
            dtype=dtype,
            seed=1701 + case_index,
        )
        zero_padding_cpu = fill_padding_values(
            fixed_cpu,
            scene_valid=scene_valid,
            history_valid=history_valid,
            value=0.0,
        )
        fixed = _to_device(fixed_cpu, reference_device)
        with torch.no_grad():
            expected = policy(*fixed)
        actual = run_ort_tensors(
            session,
            dict(zip(TENSOR_INPUT_NAMES, fixed_cpu, strict=True)),
        )[0]
        zero_padding_actual = run_ort_tensors(
            session,
            dict(zip(TENSOR_INPUT_NAMES, zero_padding_cpu, strict=True)),
        )[0]
        if not torch.isfinite(actual).all():
            raise AssertionError("ORT logits contain NaN/Inf")
        if not torch.isfinite(zero_padding_actual).all():
            raise AssertionError("ORT zero-padding logits contain NaN/Inf")
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        torch.testing.assert_close(
            actual,
            zero_padding_actual,
            rtol=rtol,
            atol=atol,
        )
        argmax_match = bool(
            torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))
        )
        if not argmax_match:
            pt_top = expected.float().topk(3, dim=-1)
            ort_top = actual.float().topk(3, dim=-1)
            raise AssertionError(
                "ORT/PyTorch argmax mismatch: "
                f"scene_valid={scene_valid}, history_valid={history_valid}, "
                f"pt_top={list(zip(pt_top.indices[0].tolist(), pt_top.values[0].tolist()))}, "
                f"ort_top={list(zip(ort_top.indices[0].tolist(), ort_top.values[0].tolist()))}, "
                f"max_abs_diff={(actual.float() - expected.float()).abs().max().item():.6g}"
            )
        results.append(
            asdict(
                MatrixCaseResult(
                    scene_valid=scene_valid,
                    history_valid=history_valid,
                    max_logit_abs_diff=float((actual - expected).abs().max().item()),
                    max_dynamic_baseline_abs_diff=0.0,
                    argmax_match=True,
                    padding_value_invariant=True,
                )
            )
        )
    return results


def _assert_trace_finite(trace) -> None:
    tensors = (*trace.layer_hidden, trace.hidden, *trace.attentions, trace.logits)
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
        raise AssertionError("policy trace contains NaN/Inf")


def _assert_padding_keys_blocked(
    trace,
    *,
    scene_valid: int,
    history_valid: int,
    contract: CapacityContract,
) -> None:
    padding_keys = list(range(scene_valid, contract.scene_capacity))
    padding_keys.extend(
        range(
            contract.scene_capacity + 2 * history_valid,
            contract.scene_capacity + 2 * contract.history_capacity,
        )
    )
    if not padding_keys:
        return
    indices = torch.tensor(
        padding_keys,
        device=trace.hidden.device,
        dtype=torch.long,
    )
    for attention in trace.attentions:
        padding_weights = attention.index_select(-1, indices)
        if not torch.equal(padding_weights, torch.zeros_like(padding_weights)):
            raise AssertionError("valid tokens can read padding attention keys")


def _assert_raw_trace_finite(trace, logits: torch.Tensor) -> None:
    tensors = (*trace.layer_hidden, trace.hidden, *trace.attentions, logits)
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
        raise AssertionError("dynamic PyTorch baseline contains NaN/Inf")


def _raw_logits(policy, trace) -> torch.Tensor:
    """用与 policy trace 相同的方式从 raw model trace 计算 logits。"""
    return _policy_raw_logits(policy.model, trace.encoded, trace.hidden)


def _assert_same_input_trace_close(
    stable,
    raw,
    raw_logits: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    """断言 policy trace 与 raw model trace 在相同输入下逐 token 一致。"""
    trace_rtol = max(rtol, 1e-4)
    trace_atol = max(atol, 1e-5)
    for stable_hidden, raw_hidden in zip(
        (*stable.layer_hidden, stable.hidden),
        (*raw.layer_hidden, raw.hidden),
        strict=True,
    ):
        torch.testing.assert_close(
            stable_hidden,
            raw_hidden,
            rtol=trace_rtol,
            atol=trace_atol,
        )
    for stable_attention, raw_attention in zip(
        stable.attentions,
        raw.attentions,
        strict=True,
    ):
        torch.testing.assert_close(
            stable_attention,
            raw_attention,
            rtol=trace_rtol,
            atol=trace_atol,
        )
    torch.testing.assert_close(stable.logits, raw_logits, rtol=rtol, atol=atol)


def _to_device(inputs: tuple[torch.Tensor, ...], device: torch.device):
    return tuple(tensor.to(device=device) for tensor in inputs)


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    return precision_tolerances(dtype)

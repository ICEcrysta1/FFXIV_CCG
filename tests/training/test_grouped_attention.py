"""Grouped-query attention 的投影、SDPA 与 trace 回归测试。"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from common.policy.model.grouped_attention import (
    GroupedQueryAttention,
    scaled_dot_product_attention,
)
from common.policy.model.position_encoding import RotaryPositionEncoding
from common.policy.model.split_encoder import (
    project_qkv,
    run_split_encoder,
)
from common.policy.model.trace import TraceableTransformerEncoderLayer


def _make_encoder(
    *,
    num_heads: int = 4,
    num_kv_heads: int = 1,
    batch_first: bool = True,
):
    encoder = nn.TransformerEncoder(
        TraceableTransformerEncoderLayer(
            d_model=8,
            nhead=num_heads,
            num_kv_heads=num_kv_heads,
            dim_feedforward=16,
            dropout=0.0,
            batch_first=batch_first,
            norm_first=True,
        ),
        1,
        norm=nn.LayerNorm(8),
    ).eval()
    encoder.rotary_position_encoding = RotaryPositionEncoding(8 // num_heads)
    return encoder


def _make_encoded(tokens: torch.Tensor) -> dict[str, torch.Tensor | int]:
    return {
        "tokens": tokens,
        "prefix_length": 2,
        "candidate_count": 2,
        "prefix_valid": torch.ones((tokens.shape[0], 2), dtype=torch.bool),
        "candidate_valid": torch.ones((tokens.shape[0], 2), dtype=torch.bool),
        "cls_valid": torch.ones((tokens.shape[0], 1), dtype=torch.bool),
    }


def test_mqa_uses_small_kv_projections_and_runs_layer_forward():
    torch.manual_seed(7)
    encoder = _make_encoder()
    layer = encoder.layers[0]
    values = torch.randn(2, 5, 8, requires_grad=True)

    query, key, value = project_qkv(layer.self_attn, values)

    assert layer.self_attn.num_heads == 4
    assert layer.self_attn.num_kv_heads == 1
    assert query.shape == (2, 5, 8)
    assert key.shape == (2, 5, 2)
    assert value.shape == (2, 5, 2)

    output = layer(values)
    output.square().mean().backward()

    assert output.shape == values.shape
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_non_batch_first_keeps_attention_weights_batch_first():
    encoder = _make_encoder(batch_first=False)
    attention = encoder.layers[0].self_attn
    values = torch.randn(5, 2, 8)

    output, weights = attention(values, values, values)
    _, per_head_weights = attention(
        values,
        values,
        values,
        average_attn_weights=False,
    )

    assert output.shape == values.shape
    assert weights is not None
    assert weights.shape == (2, 5, 5)
    assert per_head_weights is not None
    assert per_head_weights.shape == (2, 4, 5, 5)


@pytest.mark.parametrize("num_kv_heads", (1, 2))
def test_grouped_training_preserves_per_head_attention_mask(num_kv_heads):
    encoder = _make_encoder(num_kv_heads=num_kv_heads).train()
    attention = encoder.layers[0].self_attn
    values = torch.randn(2, 5, 8, requires_grad=True)
    # 标准 MHA 允许 [batch * heads, query, key] 形式的按头 mask。
    mask = torch.zeros((2 * attention.num_heads, 5, 5))
    mask[1::2, :, 0] = float("-inf")

    output, weights = attention(
        values,
        values,
        values,
        need_weights=False,
        attn_mask=mask,
    )
    output.square().mean().backward()

    assert output.shape == values.shape
    assert weights is None
    assert values.grad is not None
    # 显式权重路径独立计算所有头，校验按组切分后每个头仍使用自己的 mask。
    expected, _ = attention(values, values, values, attn_mask=mask)
    torch.testing.assert_close(output, expected)


def test_grouped_attention_accepts_configured_divisor_kv_head_counts():
    for num_kv_heads in (1, 2, 4):
        encoder = _make_encoder(num_kv_heads=num_kv_heads)
        values = torch.randn(1, 5, 8)
        query, key, value = project_qkv(encoder.layers[0].self_attn, values)

        assert query.shape[-1] == 8
        assert key.shape[-1] == value.shape[-1] == 2 * num_kv_heads
        prefix_output, candidate_output, cls_output, _, _ = run_split_encoder(
            encoder,
            _make_encoded(values),
        )
        assert prefix_output.shape == (1, 2, 8)
        assert candidate_output.shape == (1, 2, 8)
        assert cls_output.shape == (1, 1, 8)


@pytest.mark.parametrize("training", (False, True))
@pytest.mark.parametrize("split", (False, True))
@pytest.mark.parametrize("num_kv_heads", (1, 2, 4))
def test_sdpa_batches_query_heads_with_compressed_kv_storage(
    monkeypatch, training, split, num_kv_heads,
):
    encoder = _make_encoder(num_kv_heads=num_kv_heads).train(training)
    attention = GroupedQueryAttention(
        8, 4, num_kv_heads, batch_first=True,
    ).train(training)
    values = torch.randn(2, 5, 8, requires_grad=training)
    original = torch.nn.functional.scaled_dot_product_attention
    call_count = 0
    key_storages = set()
    value_storages = set()

    def capture(query, key, value, **kwargs):
        nonlocal call_count
        call_count += 1
        group_size = 4 // num_kv_heads if num_kv_heads < 4 else 4
        assert query.shape[1] == key.shape[1] == value.shape[1] == group_size
        assert "enable_gqa" not in kwargs
        key_storages.add(key.untyped_storage().data_ptr())
        value_storages.add(value.untyped_storage().data_ptr())
        if num_kv_heads < 4:
            for tensor in (key, value):
                assert tensor.stride(1) == 0
                # 分段是整条序列投影的视图，底层仍只有 num_kv_heads 份 K/V。
                assert tensor.untyped_storage().nbytes() == (
                    tensor.shape[0] * num_kv_heads * values.shape[1]
                    * tensor.shape[3] * tensor.element_size()
                )
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", capture)
    with torch.set_grad_enabled(training):
        if split:
            outputs = run_split_encoder(encoder, _make_encoded(values))[:3]
        else:
            output, weights = attention(
                values, values, values, need_weights=False,
            )
            assert output.shape == values.shape
            assert weights is None
            outputs = (output,)
        if training:
            sum(output.square().mean() for output in outputs).backward()
            assert values.grad is not None
            assert torch.isfinite(values.grad).all()

    # MQA 和 MHA 每个区域一次调用；GQA 每个 KV 组一次调用。
    assert call_count == (3 if split else 1) * (num_kv_heads if num_kv_heads < 4 else 1)
    assert len(key_storages) == len(value_storages) == 1


@pytest.mark.parametrize("num_kv_heads", (1, 2, 4))
@pytest.mark.parametrize("mask_kind", ("none", "causal", "shared_bool", "head_bool", "head_bias"))
@pytest.mark.parametrize(
    "device,dtype,rtol,atol",
    (
        ("cpu", torch.float64, 1e-9, 1e-9),
        pytest.param(
            "cuda", torch.bfloat16, 3e-2, 2e-2,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA"),
        ),
    ),
)
def test_grouped_sdpa_matches_explicit_kv_forward_and_gradients(
    num_kv_heads, mask_kind, device, dtype, rtol, atol,
):
    torch.manual_seed(42)
    # 非连续 Q/K/V 与投影后的布局一致；Q/K 长度不同，覆盖候选及 cache 使用场景。
    query = torch.randn(2, 5, 4, 16, device=device, dtype=dtype).transpose(1, 2).requires_grad_()
    key = torch.randn(2, 7, num_kv_heads, 16, device=device, dtype=dtype).transpose(1, 2).requires_grad_()
    value = torch.randn_like(key).requires_grad_()
    mask = None
    if mask_kind in {"shared_bool", "head_bool", "head_bias"}:
        head_count = 1 if mask_kind == "shared_bool" else 4
        allowed = torch.rand(2, head_count, 5, 7, device=device) > 0.3
        allowed[..., -1] = True
        mask = allowed
        if mask_kind == "head_bias":
            mask = torch.randn(2, head_count, 5, 7, device=device, dtype=dtype)
            mask = mask.masked_fill(~allowed, float("-inf"))
    kwargs = dict(attn_mask=mask, dropout_p=0.0, is_causal=mask_kind == "causal")
    actual = scaled_dot_product_attention(query, key, value, **kwargs)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key.repeat_interleave(4 // num_kv_heads, dim=1),
        value.repeat_interleave(4 // num_kv_heads, dim=1),
        **kwargs,
    )
    upstream = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (query, key, value), upstream)
    expected_gradients = torch.autograd.grad(expected, (query, key, value), upstream)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=rtol, atol=atol)


def test_mqa_trace_expands_only_the_explicit_attention_weight_result():
    encoder = _make_encoder()
    encoded = _make_encoded(torch.randn(1, 5, 8))

    *_, attentions = run_split_encoder(encoder, encoded, collect_attention=True)

    assert len(attentions) == 1
    assert attentions[0].shape == (1, 4, 5, 5)

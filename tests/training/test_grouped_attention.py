"""Grouped-query attention 的投影、SDPA 与 trace 回归测试。"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from common.policy.model.attention.attention_variants import manual_attention
from common.policy.model.attention.grouped_attention import (
    GroupedQueryAttention,
    scaled_dot_product_attention,
)
from common.policy.model.attention.position_encoding import RotaryPositionEncoding
from common.policy.model.execution.causal_encoder import (
    project_qkv,
    run_causal_encoder,
)
from common.policy.model.layers.transformer import TraceableTransformerEncoderLayer


def _make_encoder(
    *,
    d_model: int = 8,
    num_heads: int = 4,
    num_kv_heads: int = 1,
    head_dim: int | None = None,
    rope_head_dim: int | None = None,
    batch_first: bool = True,
):
    encoder = nn.TransformerEncoder(
        TraceableTransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dim_feedforward=16,
            dropout=0.0,
            batch_first=batch_first,
            norm_first=True,
        ),
        1,
        norm=nn.RMSNorm(d_model, eps=1e-5, elementwise_affine=False),
    ).eval()
    encoder.rotary_position_encoding = RotaryPositionEncoding(
        encoder.layers[0].self_attn.head_dim, rope_head_dim=rope_head_dim,
    )
    return encoder


def _make_encoded(tokens: torch.Tensor) -> dict[str, torch.Tensor | int]:
    return {
        "tokens": tokens,
        "valid": torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device),
    }


@pytest.mark.parametrize("d_model,head_dim", ((8, 2), (8, 6), (9, 6)))
def test_mqa_uses_small_kv_projections_and_runs_layer_forward(d_model, head_dim):
    torch.manual_seed(7)
    encoder = _make_encoder(d_model=d_model, head_dim=head_dim)
    layer = encoder.layers[0]
    values = torch.randn(2, 5, d_model, requires_grad=True)

    query, key, value = project_qkv(layer.self_attn, values)

    assert layer.self_attn.num_heads == 4
    assert layer.self_attn.num_kv_heads == 1
    assert query.shape == (2, 5, 4 * head_dim)
    assert key.shape == (2, 5, head_dim)
    assert value.shape == (2, 5, head_dim)
    assert key is value
    assert layer.self_attn.q_a.weight.shape == (2, d_model)
    assert layer.self_attn.q_b.weight.shape == (4 * head_dim, 2)
    assert layer.self_attn.out_proj.weight.shape == (d_model, 4 * head_dim)
    assert layer.self_attn.q_norm.eps == layer.self_attn.kv_norm.eps == 1e-20

    output = layer(values)
    output.square().mean().backward()

    assert output.shape == values.shape
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize("num_kv_heads", (1, 2, 4))
@pytest.mark.parametrize("head_dim,rope_head_dim", ((2, 2), (6, 4)))
def test_low_rank_query_shared_kv_and_inverse_rotation_match_formula(
    num_kv_heads, head_dim, rope_head_dim,
):
    """独立公式覆盖归一化位置、共享旋转 KV、输出逆旋转及全部投影梯度。"""
    torch.manual_seed(27)
    encoder = _make_encoder(
        num_kv_heads=num_kv_heads, head_dim=head_dim, rope_head_dim=rope_head_dim,
    )
    layer = encoder.layers[0]
    attention = layer.self_attn
    with torch.no_grad():
        attention.q_norm.weight.copy_(torch.linspace(0.6, 1.4, attention.q_lora_rank))
        attention.kv_norm.weight.copy_(torch.linspace(0.7, 1.3, attention.head_dim))
    tokens = torch.randn(2, 5, 8, requires_grad=True)
    positions = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 2, 3]])
    encoded = {**_make_encoded(tokens), "position_ids": positions}
    actual, _, _ = run_causal_encoder(encoder, encoded)

    values = layer.norm1(tokens)
    latent = F.linear(values, attention.q_a.weight, attention.q_a.bias)
    latent = latent.float() * torch.rsqrt(latent.float().square().mean(-1, keepdim=True) + 1e-20)
    query = F.linear(latent * attention.q_norm.weight, attention.q_b.weight, attention.q_b.bias)
    kv = F.linear(values, attention.kv_proj.weight, attention.kv_proj.bias)
    kv = kv.reshape(2, 5, num_kv_heads, attention.head_dim)
    kv = kv.float() * torch.rsqrt(kv.float().square().mean(-1, keepdim=True) + 1e-20)
    kv = (kv * attention.kv_norm.weight).transpose(1, 2)
    query = query.reshape(2, 5, attention.num_heads, attention.head_dim).transpose(1, 2)
    frequencies = positions.float()[..., None] * encoder.rotary_position_encoding.inv_freq
    cosine, sine = frequencies.cos()[:, None], frequencies.sin()[:, None]

    def rotate(values, inverse=False):
        pairs = values[..., -rope_head_dim:].float().reshape(*values.shape[:-1], -1, 2)
        first, second = pairs[..., 0], pairs[..., 1]
        direction = -1 if inverse else 1
        rotated = torch.stack((first * cosine - direction * second * sine,
                               direction * first * sine + second * cosine), dim=-1).flatten(-2)
        return torch.cat((values[..., :-rope_head_dim], rotated.to(values.dtype)), dim=-1)

    query, kv = rotate(query), rotate(kv)
    kv = kv.repeat_interleave(attention.num_heads // num_kv_heads, dim=1)
    scores = query @ kv.transpose(-1, -2) / attention.head_dim**0.5
    allowed = torch.ones(5, 5, dtype=torch.bool).tril()
    weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
    attended = rotate(weights @ kv, inverse=True).transpose(1, 2).reshape(2, 5, 4 * head_dim)
    hidden = tokens + F.linear(attended, attention.out_proj.weight, attention.out_proj.bias)
    expected = encoder.norm(hidden + layer._ff_block(layer.norm2(hidden)))
    upstream = torch.randn_like(actual)
    parameters = (tokens, *attention.parameters())
    actual_gradients = torch.autograd.grad(actual, parameters, upstream)
    expected_gradients = torch.autograd.grad(expected, parameters, upstream)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-4, atol=1e-5)


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


@pytest.mark.parametrize("training", (False, True))
@pytest.mark.parametrize("encoder_path", (False, True))
@pytest.mark.parametrize("num_kv_heads", (1, 2, 4))
def test_sdpa_batches_query_heads_with_compressed_kv_storage(
    monkeypatch, training, encoder_path, num_kv_heads,
):
    encoder = _make_encoder(num_kv_heads=num_kv_heads).train(training)
    attention = GroupedQueryAttention(
        8, 4, num_kv_heads, batch_first=True,
    ).train(training)
    values = torch.randn(2, 4, 8, requires_grad=training)
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
        if encoder_path:
            outputs = (run_causal_encoder(encoder, _make_encoded(values))[0],)
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
    assert call_count == (num_kv_heads if num_kv_heads < 4 else 1)
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
    # 非连续 Q/K/V 与投影后的布局一致；Q/K 长度不同，覆盖因果后缀及 cache 使用场景。
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
    kwargs = {"attn_mask": mask, "dropout_p": 0.0, "is_causal": mask_kind == "causal"}
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


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.bfloat16))
def test_trace_attention_zeroes_blocked_rows_and_gradients_without_scalar_reads(dtype, monkeypatch):
    """全屏蔽行应输出零且不传播梯度，单列可见行只聚合该列。"""
    query = torch.ones((1, 4, 2, 4), dtype=dtype, requires_grad=True)
    key = torch.ones((1, 2, 2, 4), dtype=dtype, requires_grad=True)
    value = torch.arange(16, dtype=dtype).reshape(1, 2, 2, 4).requires_grad_()
    allowed = torch.tensor([[[[False, False], [True, False]]]])

    def reject_scalar_read(_tensor):
        pytest.fail("trace attention 不应把设备 tensor 转为 Python 布尔值")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", reject_scalar_read)
        output, weights = manual_attention(query, key, value, allowed)
    output.sum().backward()

    expected = value.detach().repeat_interleave(2, dim=1).clone()
    expected[:, :, 0] = 0
    expected[:, :, 1] = value.detach()[:, :, 0].repeat_interleave(2, dim=1)
    expected_weights = torch.zeros((1, 4, 2, 2), dtype=dtype)
    expected_weights[:, :, 1, 0] = 1
    expected_value_gradient = torch.zeros_like(value)
    expected_value_gradient[:, :, 0] = 2
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(query.grad, torch.zeros_like(query), rtol=0, atol=0)
    torch.testing.assert_close(key.grad, torch.zeros_like(key), rtol=0, atol=0)
    torch.testing.assert_close(value.grad, expected_value_gradient, rtol=0, atol=0)

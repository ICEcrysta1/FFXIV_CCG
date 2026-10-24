"""RoPE 数学与逻辑位置边界测试。"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pytest
import torch

from common.policy.model.attention.position_encoding import RotaryPositionEncoding
from common.policy.model.execution.causal_encoder import (
    run_causal_encoder,
    run_causal_layer,
)
from common.policy.model.layers.transformer import TraceableTransformerEncoderLayer


def test_rope_zero_position_is_identity():
    rope = RotaryPositionEncoding(head_dim=4)
    values = torch.randn(2, 3, 5, 4)
    positions = torch.zeros((2, 5), dtype=torch.long)

    rotated = rope.apply(values, positions)

    torch.testing.assert_close(rotated, values)


def test_rope_uses_independent_query_and_key_positions():
    rope = RotaryPositionEncoding(head_dim=4, theta=10_000.0)
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 5, 4)
    query_positions = torch.tensor([[7, 8, 9]])
    key_positions = torch.tensor([[0, 1, 2, 3, 4]])

    rotated_query, rotated_key = rope(query, key, query_positions, key_positions)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    assert not torch.equal(rotated_query, query)
    assert not torch.equal(rotated_key, key)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
@pytest.mark.parametrize(("head_dim", "rope_head_dim"), ((4, 4), (512, 64)))
def test_rope_matches_adjacent_complex_rotation_and_gradients(dtype, head_dim, rope_head_dim):
    """以独立的复数乘法核对相邻配对、局部旋转及其梯度。"""
    rope = RotaryPositionEncoding(head_dim, rope_head_dim=rope_head_dim, theta=2.0)
    values = torch.linspace(0.1, 0.9, head_dim, dtype=dtype).reshape(1, 1, 1, head_dim)
    values.requires_grad_()
    reference = values.detach().clone().requires_grad_()
    positions = torch.tensor([[1]])

    rotated = rope.apply(values, positions)

    inv_freq = 1.0 / (2.0 ** (torch.arange(0, rope_head_dim, 2, dtype=torch.float32) / rope_head_dim))
    phase = positions.float()[..., None] * inv_freq
    rotation = torch.polar(torch.ones_like(phase), phase).unsqueeze(1)
    pairs = torch.view_as_complex(reference[..., -rope_head_dim:].float().unflatten(-1, (-1, 2)))
    expected_tail = torch.view_as_real(pairs * rotation).flatten(-2).to(dtype)
    expected = torch.cat(
        (reference[..., :-rope_head_dim], expected_tail), dim=-1,
    )

    torch.testing.assert_close(rotated, expected)
    probe = torch.linspace(-0.7, 0.9, head_dim, dtype=dtype).reshape(1, 1, 1, head_dim)
    (rotated * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(values.grad, reference.grad)


@pytest.mark.parametrize(("head_dim", "rope_head_dim"), ((512, 64), (5, 4)))
@pytest.mark.parametrize("merge_heads", (False, True))
def test_partial_rope_preserves_prefix_and_inverse_restores_input(head_dim, rope_head_dim, merge_heads):
    rope = RotaryPositionEncoding(head_dim=head_dim, rope_head_dim=rope_head_dim)
    values = torch.randn(2, 3, 5, head_dim, requires_grad=True)
    positions = torch.tensor([[0, 1, 3, 9, 50], [2, 4, 7, 12, 17]])
    cosine_sine = rope._cos_sin(positions)

    rotated = rope.apply(values, positions, cosine_sine)
    restored = rope.apply(rotated, positions, cosine_sine, inverse=True, merge_heads=merge_heads)

    torch.testing.assert_close(rotated[..., :-rope_head_dim], values[..., :-rope_head_dim], rtol=0, atol=0)
    expected = values.transpose(1, 2).flatten(2) if merge_heads else values
    torch.testing.assert_close(restored, expected)
    assert not torch.equal(rotated[..., -rope_head_dim:], values[..., -rope_head_dim:])
    restored.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("merge_heads", (False, True))
@pytest.mark.parametrize(("batch", "tokens", "heads"), ((2, 7, 6), (1, 1, 1)))
@pytest.mark.parametrize("head_dim", (256, 512))
def test_cuda_rope_preserves_strided_input_outputs_and_gradients(dtype, inverse, merge_heads, batch, tokens, heads, head_dim):
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    rope = RotaryPositionEncoding(head_dim, rope_head_dim=64).cuda().train()
    reference_rope = deepcopy(rope).eval()
    values = torch.randn(batch, tokens, heads, head_dim, device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()
    reference = values.detach().clone().requires_grad_()
    original = values.detach().clone()
    positions = torch.arange(batch * tokens, device="cuda").reshape(batch, tokens) * 17 + 1
    cosine_sine = rope._cos_sin(positions)
    if merge_heads:
        # 外部共享频表允许 batch 广播，并保留不同于连续表的通道/位置 stride。
        cosine_sine = tuple(value[:1].transpose(-1, -2).contiguous().transpose(-1, -2) for value in cosine_sine)

    actual = rope.apply(values, positions, cosine_sine, inverse=inverse, merge_heads=merge_heads)
    expected = reference_rope.apply(reference, positions, cosine_sine, inverse=inverse, merge_heads=merge_heads)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(values.grad, reference.grad)
    torch.testing.assert_close(values, original, rtol=0, atol=0)
    assert actual.is_contiguous()


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_rope_keeps_exact_float32_frequencies_and_rotation_after_dtype_conversion(dtype):
    rope = RotaryPositionEncoding(head_dim=512, rope_head_dim=64, theta=160_000.0)
    reference = deepcopy(rope)
    rope.to(dtype=dtype)
    values = torch.randn(1, 2, 3, 512, dtype=dtype)
    positions = torch.tensor([[1, 12345, 50000]])

    query, key = rope(values, values, positions, positions)
    expected = reference.apply(values.float(), positions).to(dtype)

    assert rope.inv_freq.dtype == torch.float32
    torch.testing.assert_close(rope.inv_freq, reference.inv_freq, rtol=0, atol=0)
    torch.testing.assert_close(query, expected, rtol=0, atol=0)
    torch.testing.assert_close(key, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rope_head_dim", (0, -2, 3, 10))
def test_rope_rejects_invalid_rotary_dimensions(rope_head_dim):
    with pytest.raises(ValueError, match="rope_head_dim"):
        RotaryPositionEncoding(head_dim=8, rope_head_dim=rope_head_dim)


@pytest.mark.parametrize(
    ("collect_attention", "checkpoint_attention"),
    ((False, False), (False, True), (True, False)),
)
def test_encoder_shares_rope_trigonometry_across_heads_layers_and_recomputation(
    monkeypatch, collect_attention, checkpoint_attention,
):
    torch.manual_seed(54)
    encoder = torch.nn.TransformerEncoder(
        TraceableTransformerEncoderLayer(
            d_model=8, nhead=2, num_kv_heads=1, dim_feedforward=12,
            activation="swiglu", batch_first=True, dropout=0.0,
        ),
        3, enable_nested_tensor=False,
    )
    encoder.rotary_position_encoding = RotaryPositionEncoding(4)
    reference = deepcopy(encoder)
    tokens = torch.randn(2, 5, 8, requires_grad=True)
    reference_tokens = tokens.detach().clone().requires_grad_()
    positions = torch.tensor([[0, 1, 4, 5, 8], [11, 10, 9, 6, 3]])
    valid = torch.ones((2, 5), dtype=torch.bool)
    expected = reference_tokens
    expected_layers = []
    for layer in reference.layers:
        expected = run_causal_layer(
            layer, expected, key_valid=valid, position_ids=positions,
            rotary_position_encoding=reference.rotary_position_encoding,
        )[0]
        expected_layers.append(expected)
    for layer in encoder.layers:
        layer.activation_checkpoint_attention = checkpoint_attention
        layer.activation_checkpoint_attention_block = checkpoint_attention

    counts = Counter()
    for name in ("cos", "sin"):
        original = getattr(torch.Tensor, name)

        def counted(values, original=original, name=name):
            assert values.dtype == torch.float32
            counts[name] += 1
            return original(values)

        monkeypatch.setattr(torch.Tensor, name, counted)

    actual, layer_hidden, _ = run_causal_encoder(
        encoder, {"tokens": tokens, "valid": valid, "position_ids": positions},
        collect_attention=collect_attention,
    )
    torch.testing.assert_close(actual, expected)
    if collect_attention:
        for actual_hidden, expected_hidden in zip(layer_hidden, expected_layers, strict=True):
            torch.testing.assert_close(actual_hidden, expected_hidden)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    assert counts == {"cos": 1, "sin": 1}
    torch.testing.assert_close(tokens.grad, reference_tokens.grad)
    for actual_parameter, reference_parameter in zip(encoder.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad)


@pytest.mark.parametrize(
    ("values_shape", "position_shape"),
    [((1, 2, 3), (1, 3)), ((1, 2, 3, 4), (2, 3))],
)
def test_rope_rejects_invalid_shapes(values_shape, position_shape):
    rope = RotaryPositionEncoding(head_dim=4)
    values = torch.randn(values_shape)
    positions = torch.zeros(position_shape, dtype=torch.long)

    with pytest.raises(ValueError, match="RoPE"):
        rope.apply(values, positions)

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


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_rope_matches_mainstream_half_split_formula(dtype):
    """与 LLaMA/HF/GPT-NeoX 的 half-split 正旋转公式逐元素一致。"""
    rope = RotaryPositionEncoding(head_dim=4, theta=2.0)
    values = torch.tensor([[[[0.3, 0.7, 0.2, 0.8]]]], dtype=dtype, requires_grad=True)
    reference = values.detach().clone().requires_grad_()
    positions = torch.tensor([[1]])

    rotated = rope.apply(values, positions)

    half = 2
    inv_freq = 1.0 / (2.0 ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4.0))
    phase = torch.tensor(1.0) * inv_freq
    cosine = phase.cos().to(dtype).view(1, 1, 1, half)
    sine = phase.sin().to(dtype).view(1, 1, 1, half)
    first, second = reference[..., :half], reference[..., half:]
    expected = torch.cat(
        (
            first * cosine - second * sine,
            first * sine + second * cosine,
        ),
        dim=-1,
    )

    torch.testing.assert_close(rotated, expected)
    probe = torch.tensor([[[[0.2, -0.7, 0.5, 0.9]]]], dtype=dtype)
    (rotated * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(values.grad, reference.grad)


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

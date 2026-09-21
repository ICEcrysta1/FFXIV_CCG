"""RoPE 数学与逻辑位置边界测试。"""

from __future__ import annotations

import pytest
import torch

from common.policy.model.position_encoding import RotaryPositionEncoding


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


def test_rope_rotates_in_standard_positive_direction():
    """锁定正旋转约定：基向量 (1, 0) 在位置 m 旋转后为 (cos, sin)。

    通道 0 的频率恒为 1（theta ** 0），位置 1 的相位即 1 rad；若实现
    误回负旋转 R(-θ)，第二半会得到 -sin(1)，形成镜像差异。
    """
    rope = RotaryPositionEncoding(head_dim=4)
    values = torch.zeros(1, 1, 1, 4)
    values[..., 0] = 1.0
    positions = torch.tensor([[1]])

    rotated = rope.apply(values, positions)

    torch.testing.assert_close(
        rotated[0, 0, 0], torch.tensor([torch.cos(torch.tensor(1.0)), 0.0, torch.sin(torch.tensor(1.0)), 0.0])
    )


def test_rope_matches_mainstream_half_split_formula():
    """与 LLaMA/HF/GPT-NeoX 的 half-split 正旋转公式逐元素一致。"""
    rope = RotaryPositionEncoding(head_dim=4, theta=2.0)
    values = torch.tensor([[[[0.3, 0.7, 0.2, 0.8]]]])
    positions = torch.tensor([[1]])

    rotated = rope.apply(values, positions)

    half = 2
    inv_freq = 1.0 / (2.0 ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4.0))
    phase = torch.tensor(1.0) * inv_freq
    cosine = phase.cos().view(1, 1, 1, half)
    sine = phase.sin().view(1, 1, 1, half)
    first, second = values[..., :half], values[..., half:]
    expected = torch.cat(
        (
            first * cosine - second * sine,
            first * sine + second * cosine,
        ),
        dim=-1,
    )

    torch.testing.assert_close(rotated, expected)


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

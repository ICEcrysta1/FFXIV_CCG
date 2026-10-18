"""FP32 输入投影及统一 RMSNorm 的数值、布局和梯度边界。"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from common.policy.config import ModelConfig
from common.policy.model.input_encoder import CausalInputEncoder
from common.pytorch.runtime import autocast_context, move_batch
from tests.training._causal_fixtures import make_batch, make_data_spec


def _encoder():
    torch.manual_seed(61)
    return CausalInputEncoder(
        make_data_spec(num_scene_types=2),
        ModelConfig(d_model=8, n_layers=1, n_heads=2, ff_dim=16, dropout=0.0),
        vocab_size=3,
    )


def _dense_batch(*, scene_length=2, history_length=2, dtype=torch.float32):
    batch = make_batch(
        make_data_spec(num_scene_types=2), batch_size=2,
        history_length=history_length, scene_length=scene_length, dtype=dtype)
    batch['scene_vectors'] = torch.arange(2 * scene_length * 4, dtype=dtype).reshape(
        2, scene_length, 4) / 7 - 0.3
    batch['history_state_vectors'] = torch.arange(2 * history_length * 4, dtype=dtype).reshape(
        2, history_length, 4) / 5 - 0.2
    batch['history_skill_features'] = torch.arange(2 * history_length * 2, dtype=dtype).reshape(
        2, history_length, 2) / 3 - 0.1
    batch['current_state_vectors'] = torch.tensor(
        [[0.2, -0.5, 0.8, 1.2], [-0.3, 0.4, 0.1, 0.9]], dtype=dtype)
    batch['history_state_vectors'] = torch.cat((batch['history_state_vectors'], torch.zeros((2, history_length, 4), dtype=dtype)), dim=-1)
    batch['current_state_vectors'] = torch.cat((batch['current_state_vectors'], torch.zeros((2, 4), dtype=dtype)), dim=-1)
    batch['current_state_null_mask'] = torch.tensor(
        [[True, False, False, False], [False, True, False, False]])
    if scene_length == 2:
        batch['scene_types'] = torch.tensor([[0, 1], [0, 1]])
        batch['scene_mask'] = torch.tensor([[True, False], [False, True]])
    if history_length == 2:
        batch['history_skill_ids'] = torch.tensor([[1, 2], [2, 0]])
        batch['history_mask'] = torch.tensor([[True, True], [True, False]])
        batch['history_state_null_mask'][0, 0, 2] = True
        # compact 路径正式补齐的无效历史为 0 + 全缺失标记。
        batch['history_skill_features'][1, 1] = 0
        batch['history_state_vectors'][1, 1] = 0
        batch['history_state_null_mask'][1, 1] = True
    return batch


def test_input_rms_is_applied_once_after_content_and_role():
    encoder = _encoder()
    batch = _dense_batch()
    encoded = encoder(batch)
    # 独立按类型取投影，避免将 scene 先归一化或误选类型也算作正确。
    scene = torch.stack([
        torch.stack([encoder.scene_proj[int(scene_type)](vector)
                     for scene_type, vector in zip(types, vectors)])
        for types, vectors in zip(batch['scene_types'], batch['scene_vectors'])])
    history = encoder.embed_history(batch)
    interleaved = torch.stack((history['state'], history['skill']), dim=2).reshape(2, 4, 8)
    state = encoder.state_proj(batch['current_state_vectors']) + encoder.state_null_proj(
        batch['current_state_null_mask'].float())
    # 独立按有效输入顺序构造预期内容，不让补齐行参与归一化。
    content = torch.zeros_like(encoded['tokens'])
    for row in range(2):
        prefix = torch.cat((scene[row, batch['scene_mask'][row]],
                            interleaved[row, batch['history_mask'][row].repeat_interleave(2)]))
        content[row, :prefix.shape[0]] = prefix
        content[row, -1] = state[row]
    raw = content + encoder.role_embed(encoded['role_ids'])
    expected = raw / (raw.square().mean(dim=-1, keepdim=True) + 1e-5).sqrt()
    assert encoded['tokens'].dtype == torch.float32
    torch.testing.assert_close(encoded['tokens'], expected, atol=1e-6, rtol=1e-6)
    # RMSNorm 不去均值；这个断言会拒绝把统一步骤换成 LayerNorm。
    assert torch.any(encoded['tokens'][encoded['valid']].mean(dim=-1).abs() > 0.1)
    assert not any(isinstance(module, nn.LayerNorm) for module in encoder.modules())
    assert not any('norm' in name for name, _ in encoder.named_parameters())


def test_zero_initialized_reset_projection_receives_gradients():
    encoder = _encoder()
    assert torch.count_nonzero(encoder.state_reset_proj.weight) == 0
    encoded = encoder(_dense_batch())
    (encoded['tokens'] * torch.arange(8, dtype=torch.float32)).sum().backward()
    assert torch.count_nonzero(encoder.state_reset_proj.weight.grad) > 0


@pytest.mark.parametrize('scene_length,history_length', ((0, 0), (0, 2), (2, 0)))
def test_input_rms_handles_empty_scene_and_history(scene_length, history_length):
    encoder = _encoder()
    batch = _dense_batch(scene_length=scene_length, history_length=history_length)
    encoded = encoder(batch)
    assert encoded['tokens'].shape == (2, scene_length + 2 * history_length + 1, 8)
    assert torch.isfinite(encoded['tokens']).all()
    assert encoded['current_state_positions'].tolist() == [scene_length + 2 * history_length] * 2
    assert encoded['valid'][:, -1].all()
    rms = encoded['tokens'][encoded['valid']].square().mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=2e-5, rtol=0)


def test_padding_content_cannot_change_valid_token_normalization():
    encoder = _encoder()
    batch = _dense_batch()
    original = encoder(batch)
    changed = copy.deepcopy(batch)
    changed['scene_vectors'][~changed['scene_mask']] = 1e4
    changed['history_state_vectors'][~changed['history_mask']] = -1e4
    changed['history_skill_features'][~changed['history_mask']] = 1e4
    changed['history_skill_ids'][~changed['history_mask']] = 2
    changed['history_state_null_mask'][~changed['history_mask']] = False
    altered = encoder(changed)
    valid = original['valid']
    torch.testing.assert_close(original['tokens'][valid], altered['tokens'][valid], atol=0, rtol=0)
    assert torch.equal(original['valid'], altered['valid'])
    assert torch.equal(original['position_ids'], altered['position_ids'])


def test_zero_content_and_role_have_finite_values_and_gradients():
    encoder = _encoder()
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.zero_()
    encoded = encoder(_dense_batch(scene_length=0, history_length=0))
    assert torch.equal(encoded['tokens'], torch.zeros_like(encoded['tokens']))
    weights = torch.linspace(-0.7, 1.3, 8, dtype=torch.float32)
    (encoded['tokens'] * weights).sum().backward()
    for parameter in (encoder.state_proj.weight, encoder.state_proj.bias, encoder.role_embed.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


@pytest.mark.parametrize('precision', ('float32', 'bf16'))
def test_input_rms_backpropagates_to_all_sources_and_preserves_padding_embedding(precision):
    if precision == 'bf16' and (
            not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        pytest.skip('CUDA BF16 is unavailable')
    device = torch.device('cuda' if precision == 'bf16' else 'cpu')
    encoder = _encoder().to(device=device)
    batch = _dense_batch(dtype=torch.float32)
    batch = move_batch(batch, device)
    with autocast_context(device, 'bf16' if precision == 'bf16' else 'float32'):
        encoded = encoder(batch)
        weights = torch.linspace(-0.7, 1.3, 8, device=device, dtype=encoded['tokens'].dtype)
        loss = (encoded['tokens'][encoded['valid']] * weights).sum()
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    assert torch.count_nonzero(encoder.skill_embed.weight.grad[0]) == 0


def test_all_input_projections_and_tokens_stay_fp32_under_bf16_autocast():
    encoder = _encoder()
    batch = _dense_batch()
    outputs = {}
    projections = {
        'state': encoder.state_proj,
        'null': encoder.state_null_proj,
        'reset': encoder.state_reset_proj,
        'skill': encoder.skill_feat_proj,
        **{f'scene_{index}': projection for index, projection in enumerate(encoder.scene_proj)},
    }
    handles = [projection.register_forward_hook(
        lambda _module, _inputs, result, name=name: outputs.setdefault(name, []).append(result)
    ) for name, projection in projections.items()]
    try:
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            encoded = encoder(batch)
            weights = torch.linspace(-0.7, 1.3, 8)
            loss = (encoded['tokens'][encoded['valid']] * weights).sum()
    finally:
        for handle in handles:
            handle.remove()
    assert outputs.keys() == projections.keys()
    assert all(value.dtype == torch.float32 for values in outputs.values() for value in values)
    assert encoded['tokens'].dtype == torch.float32
    loss.backward()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    assert torch.count_nonzero(encoder.skill_embed.weight.grad[0]) == 0


@pytest.mark.parametrize('route', ('state', 'skill', 'scene'))
@pytest.mark.parametrize('raw_values,scale', (((8000.0, 8001.0), 10000.0), ((30.0, 30.1), 270.0)))
def test_fp32_input_keeps_differences_that_bf16_would_merge(route, raw_values, scale):
    encoder = _encoder()
    batch = make_batch(make_data_spec(num_scene_types=2), batch_size=2, history_length=1)
    normalized = torch.tensor(raw_values, dtype=torch.float32) / scale
    assert normalized[0].to(torch.bfloat16) == normalized[1].to(torch.bfloat16)
    # 留一个独立角色分量，确保 RMSNorm 以后数值差异也能保留。
    with torch.no_grad():
        for parameter in encoder.parameters():
            parameter.zero_()
        encoder.role_embed.weight[:, 1] = 1
        if route == 'state':
            encoder.state_proj.weight[0, 0] = 1
        elif route == 'skill':
            encoder.skill_feat_proj.weight[0, 0] = 1
        else:
            for projection in encoder.scene_proj:
                projection.weight[0, 0] = 1
    if route == 'state':
        batch['current_state_vectors'][:, 0] = normalized
        position = 3
    elif route == 'skill':
        batch['history_skill_features'][:, 0, 0] = normalized
        position = 2
    else:
        batch['scene_vectors'][:, 0, 0] = normalized
        position = 0
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        encoded = encoder(batch)
    tokens = encoded['tokens'][:, position]
    assert tokens.dtype == torch.float32
    assert not torch.equal(tokens[0], tokens[1])
    # 两个非零分量之比消去 RMSNorm 的共同因子，独立恢复原始数值。
    torch.testing.assert_close(tokens[:, 0] / tokens[:, 1], normalized, atol=1e-7, rtol=1e-6)

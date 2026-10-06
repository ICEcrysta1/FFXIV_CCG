"""可学习普通残差的初始化、共享路径、梯度与重算契约。"""

from copy import deepcopy

import pytest
import torch
from torch import nn

from common.policy.config import ModelConfig
from common.policy.data.input_contract import RESIDUAL_MIX_CONFIG_FIELDS
from common.policy.model import CausalPolicyModel
from common.policy.model.causal_encoder import run_causal_encoder, run_causal_layer
from common.policy.model.residual_mix import LearnedResidualMix
from tests.training._causal_fixtures import make_batch, make_checkpoint, make_data_spec


def _config(**overrides):
    values = {"d_model": 16, "n_layers": 3, "n_heads": 4, "num_kv_heads": 1,
              "ff_dim": 32, "dropout": 0.0, "transformer_activation": "swiglu"}
    values.update(overrides)
    return ModelConfig(**values)


def _model(**overrides):
    torch.manual_seed(47)
    return CausalPolicyModel(make_data_spec(), _config(**overrides), 3)


@pytest.mark.parametrize("n_layers", (1, 6))
def test_residual_mix_initialization_is_linear_without_consuming_rng(n_layers):
    state = torch.get_rng_state().clone()
    mix = LearnedResidualMix(n_layers, r_start=1.15, r_end=1.05, a_start=0.20, a_end=0.05)
    assert torch.equal(torch.get_rng_state(), state)
    denominator = max(n_layers - 1, 1)
    torch.testing.assert_close(mix.r, torch.tensor([1.15 - 0.10 * i / denominator for i in range(n_layers)]))
    torch.testing.assert_close(mix.a, torch.tensor([0.20 - 0.15 * i / denominator for i in range(n_layers)]))
    assert dict(mix.named_parameters()).keys() == {"r", "a"}


def test_endpoint_config_roundtrip_and_custom_initialization():
    values = {"residual_mix_r_start": 0.9, "residual_mix_r_end": 1.1,
              "residual_mix_a_start": 0.3, "residual_mix_a_end": -0.1}
    config = ModelConfig.from_mapping(values)
    assert all(getattr(config, key) == value for key, value in values.items())
    model = _model(**values)
    torch.testing.assert_close(model.encoder.residual_mix.r, torch.tensor([0.9, 1.0, 1.1]))
    torch.testing.assert_close(model.encoder.residual_mix.a, torch.tensor([0.3, 0.1, -0.1]))


@pytest.mark.parametrize("key", ("residual_mix_r_start", "residual_mix_r_end",
                                 "residual_mix_a_start", "residual_mix_a_end"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_endpoint_config_rejects_nonfinite_values(key, value):
    with pytest.raises(ValueError, match=f"model.{key} must be finite"):
        ModelConfig.from_mapping({key: value})


@pytest.mark.parametrize("key", RESIDUAL_MIX_CONFIG_FIELDS)
@pytest.mark.parametrize("value", (True, False))
def test_endpoint_direct_config_rejects_boolean_values(key, value):
    """直接构造也不能把布尔端点当作 1/0，不能只保护 YAML 解析入口。"""
    with pytest.raises(ValueError, match=rf"model\.{key}"):
        ModelConfig(**{key: value})


@pytest.mark.parametrize("key", RESIDUAL_MIX_CONFIG_FIELDS)
@pytest.mark.parametrize("value", (True, False))
@pytest.mark.parametrize("full_attention_residuals", (False, True))
def test_checkpoint_endpoint_config_rejects_boolean_values(key, value, full_attention_residuals):
    """恢复 checkpoint 原始架构元数据时拒绝布尔端点，Full 路径也不补默认值。"""
    checkpoint = make_checkpoint(config=_config(full_attention_residuals=full_attention_residuals))
    checkpoint["model_config"][key] = value
    with pytest.raises(ValueError, match=rf"model\.{key}"):
        CausalPolicyModel.checkpoint_model_config(checkpoint)


@pytest.mark.parametrize("norm_first", (True, False))
def test_neutral_mixing_matches_ordinary_residual_forward_and_trace(norm_first):
    model = _model(transformer_norm_first=norm_first, residual_mix_r_start=1.0,
                   residual_mix_r_end=1.0, residual_mix_a_start=0.0, residual_mix_a_end=0.0).double().eval()
    reference = deepcopy(model)
    del reference.encoder.residual_mix
    batch = make_batch(batch_size=2, dtype=torch.float64)
    batch["history_mask"][0, -1] = False
    actual, expected = model(batch), reference(batch)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    actual_trace, expected_trace = model.trace(batch), reference.trace(batch)
    torch.testing.assert_close(actual_trace.hidden, expected_trace.hidden, rtol=0, atol=0)
    for values, expected_values in ((actual_trace.layer_hidden, expected_trace.layer_hidden),
                                    (actual_trace.attentions, expected_trace.attentions)):
        for value, expected_value in zip(values, expected_values, strict=True):
            torch.testing.assert_close(value, expected_value, rtol=0, atol=0)


def test_real_mix_changes_skip_once_per_layer_and_keeps_original_x0():
    model = _model().double().eval()
    for layer in model.encoder.layers:
        for projection in (layer.self_attn.out_proj, layer.linear2):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
    captured, calls = {}, []
    input_hook = model.input_encoder.register_forward_hook(
        lambda module, args, output: captured.update(encoded=output))
    mix_hook = model.encoder.residual_mix.register_forward_hook(
        lambda module, args, output: calls.append((args, output)))
    trace = model.trace(make_batch(dtype=torch.float64))
    input_hook.remove()
    mix_hook.remove()
    x0 = captured["encoded"]["tokens"]
    expected = x0
    assert len(calls) == model.config.n_layers
    for index, (args, mixed) in enumerate(calls):
        assert args[1] is x0 and args[2] == index
        expected = model.encoder.residual_mix.r[index] * expected + model.encoder.residual_mix.a[index] * x0
        torch.testing.assert_close(mixed, expected, rtol=0, atol=0)
        torch.testing.assert_close(trace.layer_hidden[index], expected, rtol=0, atol=0)


def test_x0_injection_keeps_gradients_when_previous_hidden_scale_is_zero():
    model = _model().double()
    with torch.no_grad():
        model.encoder.residual_mix.r.zero_()
        model.encoder.residual_mix.a.fill_(1.0)
        for layer in model.encoder.layers:
            for projection in (layer.self_attn.out_proj, layer.linear2):
                projection.weight.zero_()
                projection.bias.zero_()
    tokens = torch.randn(1, 3, 16, dtype=torch.float64, requires_grad=True)
    hidden, _, _ = run_causal_encoder(model.encoder, {
        "tokens": tokens, "valid": torch.ones((1, 3), dtype=torch.bool),
        "position_ids": torch.arange(3).unsqueeze(0),
    })
    (hidden * torch.arange(16, dtype=torch.float64)).sum().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert torch.count_nonzero(tokens.grad) == tokens.numel()


def test_full_attention_residual_has_no_unused_mix_parameters():
    model = _model(full_attention_residuals=True)
    assert not hasattr(model.encoder, "residual_mix")
    assert not any("residual_mix" in name for name, _ in model.named_parameters())
    assert hasattr(model.encoder, "attention_residual")


@pytest.mark.parametrize("attention_block", (False, True))
def test_activation_checkpoint_keeps_mix_output_and_all_parameter_gradients(attention_block):
    model = _model().double().train()
    checkpointed = deepcopy(model)
    checkpointed.enable_activation_checkpoint_attention(True, block=attention_block)
    checkpointed.enable_activation_checkpoint_ffn(True)
    batch = make_batch(dtype=torch.float64)
    output, expected = checkpointed(batch)["logits"], model(batch)["logits"]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.square().mean().backward()
    expected.square().mean().backward()
    for (name, parameter), (expected_name, reference) in zip(
        checkpointed.named_parameters(), model.named_parameters(), strict=True,
    ):
        assert name == expected_name and parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-12, atol=1e-12)
        if name.startswith("encoder.residual_mix."):
            assert torch.count_nonzero(parameter.grad) == parameter.numel()


def test_mixed_layer_requires_aligned_initial_tokens_and_excludes_full_attnres():
    model = _model()
    layer = model.encoder.layers[0]
    hidden = torch.randn(1, 2, 16)
    common = {"key_valid": torch.ones((1, 2), dtype=torch.bool),
              "position_ids": torch.arange(2).unsqueeze(0),
              "rotary_position_encoding": model.encoder.rotary_position_encoding,
              "residual_mix": model.encoder.residual_mix}
    with pytest.raises(ValueError, match="initial tokens aligned"):
        run_causal_layer(layer, hidden, **common)
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_causal_layer(layer, hidden, initial_tokens=hidden, attention_residual=object(), **common)

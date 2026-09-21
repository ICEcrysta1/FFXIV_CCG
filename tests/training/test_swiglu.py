"""SwiGLU 公式、参数预算与 GELU 兼容性回归。"""

from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from common.policy.config import ModelConfig
from training.config import load_run_config
from common.policy.model.model import CandidateTransformerModel
from common.policy.model.trace import TraceableTransformerEncoderLayer


@pytest.mark.parametrize("bias", (True, False))
def test_swiglu_matches_explicit_gate_formula_and_gradients(bias):
    torch.manual_seed(37)
    layer = TraceableTransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=24, dropout=0.0,
        activation="swiglu", bias=bias, dtype=torch.float64,
    )
    reference = deepcopy(layer)
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)

    actual = layer._ff_block(x)
    # 展开 sigmoid 公式，独立验证门控、逐元素乘法与三个投影的梯度。
    gate = F.linear(reference_x, reference.gate_proj.weight, reference.gate_proj.bias)
    up = F.linear(reference_x, reference.linear1.weight, reference.linear1.bias)
    expected = F.linear(
        (gate * torch.sigmoid(gate)) * up,
        reference.linear2.weight,
        reference.linear2.bias,
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(x.grad, reference_x.grad, atol=1e-12, rtol=1e-12)
    for name in ("gate_proj", "linear1", "linear2"):
        for parameter_name, parameter in getattr(layer, name).named_parameters():
            reference_parameter = getattr(getattr(reference, name), parameter_name)
            assert parameter.grad is not None
            torch.testing.assert_close(
                parameter.grad, reference_parameter.grad, atol=1e-12, rtol=1e-12,
            )


def test_swiglu_matches_gelu_matrix_parameter_budget():
    kwargs = dict(d_model=8, nhead=2, dim_feedforward=3072)
    gelu = TraceableTransformerEncoderLayer(**kwargs, activation="gelu")
    swiglu = TraceableTransformerEncoderLayer(**kwargs, activation="swiglu")

    assert gelu.linear1.out_features == 3072
    assert swiglu.linear1.out_features == 2048
    assert swiglu.gate_proj.out_features == 2048
    assert swiglu.linear2.in_features == 2048
    assert sum(
        projection.weight.numel()
        for projection in (swiglu.linear1, swiglu.gate_proj, swiglu.linear2)
    ) == sum(projection.weight.numel() for projection in (gelu.linear1, gelu.linear2))
    assert gelu.gate_proj is None


@pytest.mark.parametrize("activation", ("gelu", "relu"))
@pytest.mark.parametrize("norm_first", (True, False))
def test_existing_ffn_keeps_torch_weights_forward_and_gradients(activation, norm_first):
    kwargs = dict(
        d_model=8, nhead=2, dim_feedforward=24, dropout=0.0,
        activation=activation, norm_first=norm_first,
        batch_first=True, dtype=torch.float64,
    )
    torch.manual_seed(41)
    original = nn.TransformerEncoderLayer(**kwargs)
    torch.manual_seed(41)
    current = TraceableTransformerEncoderLayer(**kwargs)

    assert current.state_dict().keys() == original.state_dict().keys()
    for name, tensor in current.state_dict().items():
        torch.testing.assert_close(tensor, original.state_dict()[name], atol=0, rtol=0)
    x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    original_x = x.detach().clone().requires_grad_(True)
    actual, expected = current(x), original(original_x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(x.grad, original_x.grad, atol=0, rtol=0)
    for (name, parameter), (original_name, original_parameter) in zip(
        current.named_parameters(), original.named_parameters(), strict=True,
    ):
        assert name == original_name
        torch.testing.assert_close(parameter.grad, original_parameter.grad, atol=0, rtol=0)


@pytest.mark.parametrize("norm_first", (True, False))
def test_swiglu_eval_does_not_bypass_gate_with_fused_gelu_path(norm_first, monkeypatch):
    layer = TraceableTransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=24, dropout=0.0,
        activation="swiglu", batch_first=True, norm_first=norm_first,
        dtype=torch.float64,
    ).eval()
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    expected = layer(x)

    def reject_gelu_fastpath(*args, **kwargs):
        pytest.fail("SwiGLU must not use the fused ReLU/GELU encoder path")

    monkeypatch.setattr(torch, "_transformer_encoder_layer_fwd", reject_gelu_fastpath)
    with torch.no_grad():
        actual = layer(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("activation", ("gelu", "relu", "swiglu"))
def test_activation_config_and_checkpoint_roundtrip(tmp_path, activation):
    path = tmp_path / "config.yaml"
    path.write_text(
        f'model:\n  transformer_activation: " {activation.upper()} "\n  ff_dim: 3072\n',
        encoding="utf-8", newline="\n",
    )
    config = load_run_config(path).model
    assert config.transformer_activation == activation
    assert config.ff_dim == 3072
    assert CandidateTransformerModel.checkpoint_model_config(
        {"model_config": asdict(config)}
    ) == config


def test_legacy_checkpoint_without_activation_retains_gelu_default():
    config = asdict(ModelConfig())
    config.pop("transformer_activation")
    assert CandidateTransformerModel.checkpoint_model_config(
        {"model_config": config}
    ).transformer_activation == "gelu"


def test_checkpoint_rejects_unknown_activation():
    config = asdict(ModelConfig())
    config["transformer_activation"] = "swish"
    with pytest.raises(ValueError, match="transformer_activation must be"):
        CandidateTransformerModel.checkpoint_model_config({"model_config": config})

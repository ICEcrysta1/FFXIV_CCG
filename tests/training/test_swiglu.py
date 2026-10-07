"""SwiGLU 公式、参数预算与 GELU 兼容性回归。"""

from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from common.policy.config import TRANSFORMER_ACTIVATIONS, ModelConfig
from common.policy.data.spec import DataSpec
from training.config import load_run_config
from common.policy.model.activation import (
    activation_hidden,
    resolve_pointwise_activation,
)
from common.policy.model.input_encoder import CausalInputEncoder
from common.policy.model.model import CausalPolicyModel
from common.policy.model.trace import TraceableTransformerEncoderLayer
from tests.training._causal_fixtures import make_checkpoint


def _activation_spec() -> DataSpec:
    return DataSpec(
        job_tag="black_mage",
        num_actions=2,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=1,
        num_scene_types=1,
        action_keys=("fire_iii", "fire_iv"),
        skill_feature_names=("potency",),
        action_to_vocab_id=(1, 2),
        action_is_gcd=(True, True),
    )


def _activation_config(activation: str) -> ModelConfig:
    return ModelConfig(
        d_model=8,
        n_layers=1,
        n_heads=2,
        num_kv_heads=1,
        ff_dim=16,
        dropout=0.0,
        transformer_activation=activation,
    )


def _activation_batch() -> dict[str, torch.Tensor]:
    return {
        "history_skill_ids": torch.ones((1, 1), dtype=torch.long),
        "history_skill_features": torch.zeros((1, 1, 1)),
        "history_state_vectors": torch.zeros((1, 1, 3)),
        "history_state_null_mask": torch.zeros((1, 1, 3), dtype=torch.bool),
        "history_state_reset_mask": torch.ones((1, 1, 3), dtype=torch.bool),
        "history_mask": torch.ones((1, 1), dtype=torch.bool),
        "current_state_vectors": torch.zeros((1, 3)),
        "current_state_null_mask": torch.zeros((1, 3), dtype=torch.bool),
        "current_state_reset_mask": torch.zeros((1, 3), dtype=torch.bool),
        "action_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.zeros((1, 1, 2)),
        "scene_types": torch.zeros((1, 1), dtype=torch.long),
        "scene_mask": torch.ones((1, 1), dtype=torch.bool),
    }


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
    # 参考层只替换归一化；投影初始化、FFN、attention 和残差布局继续由 PyTorch 实现。
    original.norm1 = nn.RMSNorm(8, eps=1e-5, elementwise_affine=False, dtype=torch.float64)
    original.norm2 = nn.RMSNorm(8, eps=1e-5, elementwise_affine=False, dtype=torch.float64)
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


@pytest.mark.parametrize("activation", TRANSFORMER_ACTIVATIONS)
@pytest.mark.parametrize("norm_first", (True, False))
def test_eval_does_not_bypass_rms_norm_with_fused_layer_norm_path(activation, norm_first, monkeypatch):
    layer = TraceableTransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=24, dropout=0.0,
        activation=activation, batch_first=True, norm_first=norm_first,
        dtype=torch.float64,
    ).eval()
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    expected = layer(x)

    def reject_gelu_fastpath(*args, **kwargs):
        pytest.fail("RMSNorm must not use the fused LayerNorm encoder path")

    monkeypatch.setattr(torch, "_transformer_encoder_layer_fwd", reject_gelu_fastpath)
    with torch.no_grad():
        actual = layer(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("activation", TRANSFORMER_ACTIVATIONS)
@pytest.mark.parametrize("norm_first", (True, False))
def test_outer_encoder_eval_keeps_unparameterized_rms_path(activation, norm_first, monkeypatch):
    """外层 Encoder 也不能把无参数 RMS 的层交给 LayerNorm 专用融合路径。"""
    layer = TraceableTransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=24, dropout=0.0,
        activation=activation, batch_first=True, norm_first=norm_first,
        dtype=torch.float64,
    )
    encoder = nn.TransformerEncoder(
        layer, 2,
        norm=nn.RMSNorm(8, eps=1e-5, elementwise_affine=False, dtype=torch.float64),
    ).eval()
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    expected = encoder(x)

    def reject_layer_norm_fastpath(*args, **kwargs):
        pytest.fail("RMSNorm encoder must not use the fused LayerNorm path")

    monkeypatch.setattr(torch, "_transformer_encoder_layer_fwd", reject_layer_norm_fastpath)
    with torch.no_grad():
        actual = encoder(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_layer_rms_norm_honors_custom_epsilon_without_affine_parameters():
    layer = TraceableTransformerEncoderLayer(
        d_model=8, nhead=2, layer_norm_eps=2e-4, dtype=torch.float64,
    )
    for norm in (layer.norm1, layer.norm2):
        assert isinstance(norm, nn.RMSNorm)
        assert norm.eps == 2e-4
        assert norm.elementwise_affine is False
        assert list(norm.parameters()) == []


@pytest.mark.parametrize("activation", TRANSFORMER_ACTIVATIONS)
def test_model_rms_norm_preserves_nonzero_mean_and_shared_forward_gradients(activation):
    """以非零均值输入区分 RMS 与 LN，并验证正式前向和 trace 共用归一化。"""
    torch.manual_seed(57)
    model = CausalPolicyModel(
        _activation_spec(), _activation_config(activation), vocab_size=4,
    ).double().eval()
    norms = [model.encoder.norm]
    for layer in model.encoder.layers:
        norms.extend((layer.norm1, layer.norm2))
    values = torch.arange(1, 9, dtype=torch.float64).reshape(1, 8).requires_grad_(True)
    expected = values / torch.sqrt(values.square().mean(dim=-1, keepdim=True) + 1e-5)
    for norm in norms:
        assert norm.elementwise_affine is False
        assert norm.eps == 1e-5
        assert list(norm.parameters()) == []
        actual = norm(values)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        assert actual.mean() > 0
        gradient = torch.autograd.grad(actual.square().sum(), values)[0]
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0

    batch = {
        key: value.double() if value.is_floating_point() else value
        for key, value in _activation_batch().items()
    }
    output = model(batch)
    traced = model.trace(batch)
    torch.testing.assert_close(
        output["logits"], model.score_hidden(traced.encoded, traced.hidden, batch),
        atol=1e-12, rtol=1e-12,
    )
    output["logits"].square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


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
    assert CausalPolicyModel.checkpoint_model_config(
        make_checkpoint(config=config)
    ) == config


def test_checkpoint_without_new_input_contract_is_rejected():
    with pytest.raises(ValueError, match="missing input_contract"):
        CausalPolicyModel.checkpoint_model_config({"model_config": asdict(ModelConfig())})


def test_checkpoint_rejects_unknown_activation():
    checkpoint = make_checkpoint()
    checkpoint["model_config"]["transformer_activation"] = "swish"
    with pytest.raises(ValueError, match="transformer_activation must be"):
        CausalPolicyModel.checkpoint_model_config(checkpoint)


def test_checkpoint_rejects_removed_cls_architecture():
    model = CausalPolicyModel(_activation_spec(), _activation_config("swiglu"), vocab_size=4)
    weights = {**model.state_dict(), "input_encoder.cls_token": torch.zeros(1, 1, 4)}
    with pytest.raises(RuntimeError, match="Unexpected key"):
        model.load_state_dict(weights, strict=True)


@pytest.mark.parametrize("activation", TRANSFORMER_ACTIVATIONS)
def test_independent_output_head_matches_fp32_softcap_and_gradients(activation):
    torch.manual_seed(23)
    model = CausalPolicyModel(_activation_spec(), _activation_config(activation), vocab_size=4).double().eval()
    reference = deepcopy(model)
    current_hidden = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    reference_hidden = current_hidden.detach().clone().requires_grad_(True)
    actual = model._score_current_hidden(current_hidden, {})
    scale = reference.config.logit_softcap
    expected = scale * torch.tanh(F.linear(reference_hidden, reference.output_head.weight).float() / scale)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(
        current_hidden.grad, reference_hidden.grad, atol=1e-12, rtol=1e-12,
    )
    actual_parameter = model.output_head.weight
    reference_parameter = reference.output_head.weight
    assert actual_parameter.grad is not None
    torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad, atol=1e-12, rtol=1e-12)
    assert model.input_encoder.skill_embed.weight.grad is None


@pytest.mark.parametrize("activation", TRANSFORMER_ACTIVATIONS)
def test_input_semantic_table_and_output_head_are_independent(activation):
    """输入技能表保留，动作头复制其动作行后独立学习。"""
    model = CausalPolicyModel(_activation_spec(), _activation_config(activation), vocab_size=4)
    names = dict(model.named_parameters())
    assert not hasattr(model, "scorer")
    assert not hasattr(model, "output_adapter")
    semantic = model.input_encoder.skill_embed.weight
    assert semantic.shape == (4, model.config.d_model)
    assert sum(parameter is semantic for parameter in names.values()) == 1
    assert "input_encoder.skill_embed.weight" in names
    assert "output_head.weight" in names
    assert model.output_head.bias is None
    assert model.output_head.weight.shape == (model.data_spec.num_actions, model.config.d_model)
    assert model.output_head.weight is not semantic
    torch.testing.assert_close(model.output_head.weight, semantic[model.action_to_vocab_id], atol=0, rtol=0)


@pytest.mark.parametrize("activation", TRANSFORMER_ACTIVATIONS)
def test_independent_skill_and_state_embeddings_match_explicit_formula(activation):
    torch.manual_seed(29)
    encoder = CausalInputEncoder(
        _activation_spec(),
        _activation_config(activation),
        vocab_size=4,
    ).double().eval()
    batch = _activation_batch()
    batch["history_skill_features"] = torch.randn(1, 1, 1, dtype=torch.float64)
    batch["history_state_vectors"] = torch.randn(1, 1, 3, dtype=torch.float64)
    batch["history_state_null_mask"] = torch.tensor([[[True, False, True]]])
    actual = encoder.embed_history(batch)
    skill_content = F.embedding(batch["history_skill_ids"], encoder.skill_embed.weight) + F.linear(
        batch["history_skill_features"], encoder.skill_feat_proj.weight, encoder.skill_feat_proj.bias,
    )
    state_content = F.linear(batch["history_state_vectors"], encoder.state_proj.weight, encoder.state_proj.bias) + F.linear(
        batch["history_state_null_mask"].double(), encoder.state_null_proj.weight,
    )
    # 独立内容投影不先归一化；role 相加后的统一 RMS 由完整编码入口负责。
    torch.testing.assert_close(actual["skill"], skill_content, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual["state"], state_content, atol=1e-12, rtol=1e-12)
    assert not any("pair_fusion" in name for name, _ in encoder.named_parameters())


@pytest.mark.parametrize("activation", TRANSFORMER_ACTIVATIONS)
def test_model_resolves_one_activation_for_every_activation_site(activation):
    """保留全部主干激活，支路与最终输出统一使用无参数 RMSNorm。"""
    model = CausalPolicyModel(
        _activation_spec(),
        _activation_config(activation),
        vocab_size=4,
    )
    expected = resolve_pointwise_activation(activation)

    assert model.encoder.layers[0].activation is expected
    assert not any(isinstance(module, nn.LayerNorm) for module in model.input_encoder.modules())
    assert all(isinstance(layer.norm1, nn.RMSNorm) and isinstance(layer.norm2, nn.RMSNorm)
               for layer in model.encoder.layers)
    assert isinstance(model.encoder.norm, nn.RMSNorm)
    assert not any(isinstance(module, nn.LayerNorm) for module in model.encoder.modules())
    model.eval()
    assert model(_activation_batch())["logits"].shape == (1, 2)


def test_every_configured_activation_resolves():
    """配置白名单与激活映射表不能各写一份取值。"""
    for activation in TRANSFORMER_ACTIVATIONS:
        assert callable(resolve_pointwise_activation(activation))
    with pytest.raises(ValueError, match="activation must be"):
        resolve_pointwise_activation("swish")


def test_activation_hidden_merges_gate_and_pointwise_paths():
    """门控合成只在 activation.py 实现一次，两条分支都要整段重合。"""
    up = torch.randn(2, 3, dtype=torch.float64)
    gate = torch.randn(2, 3, dtype=torch.float64)

    torch.testing.assert_close(
        activation_hidden(F.gelu, up), F.gelu(up), atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        activation_hidden(F.silu, up, gate=gate),
        F.silu(gate) * up,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("key", ["scorer.network.0.weight", "scorer.up_proj.weight"])
def test_checkpoint_rejects_removed_candidate_scorer_layout(key):
    model = CausalPolicyModel(_activation_spec(), _activation_config("swiglu"), vocab_size=4)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        model.load_state_dict({**model.state_dict(), key: torch.zeros(1)}, strict=True)

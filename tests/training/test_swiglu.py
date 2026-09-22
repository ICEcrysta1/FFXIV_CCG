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
    uses_gate,
)
from common.policy.model.candidate_scorer import CandidateScorer
from common.policy.model.input_encoder import CandidateInputEncoder
from common.policy.model.model import CandidateTransformerModel
from common.policy.model.trace import TraceableTransformerEncoderLayer


def _activation_spec() -> DataSpec:
    return DataSpec(
        job_tag="black_mage",
        num_candidates=2,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=1,
        num_scene_types=1,
        candidate_action_keys=("fire_iii", "fire_iv"),
        skill_feature_names=("potency",),
    )


def _activation_config(activation: str) -> ModelConfig:
    return ModelConfig(
        d_model=8,
        pair_embedding_dim=4,
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
        "history_mask": torch.ones((1, 1), dtype=torch.bool),
        "candidate_skill_ids": torch.ones((1, 2), dtype=torch.long),
        "candidate_skill_features": torch.zeros((1, 2, 1)),
        "candidate_state_vectors": torch.zeros((1, 2, 3)),
        "candidate_legal_mask": torch.ones((1, 2), dtype=torch.bool),
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


@pytest.mark.parametrize("activation", TRANSFORMER_ACTIVATIONS)
def test_candidate_scorer_matches_explicit_formula_and_gradients(activation):
    torch.manual_seed(23)
    scorer = CandidateScorer(
        d_model=8, dropout=0.0, activation=activation
    ).double().eval()
    reference = deepcopy(scorer)
    cls_hidden = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    candidate_hidden = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    reference_cls = cls_hidden.detach().clone().requires_grad_(True)
    reference_candidates = candidate_hidden.detach().clone().requires_grad_(True)

    actual = scorer(cls_hidden=cls_hidden, candidate_hidden=candidate_hidden)
    paired = torch.cat(
        (
            reference_cls.unsqueeze(1).expand(-1, 3, -1),
            reference_candidates,
        ),
        dim=-1,
    )
    up = F.linear(paired, reference.up_proj.weight, reference.up_proj.bias)
    if uses_gate(activation):
        # 展开 sigmoid 门控公式，独立验证门控、逐元素乘法与三个投影的梯度。
        gate = F.linear(paired, reference.gate_proj.weight, reference.gate_proj.bias)
        hidden = (gate * torch.sigmoid(gate)) * up
        projection_names = ("gate_proj", "up_proj", "down_proj")
    else:
        hidden = resolve_pointwise_activation(activation)(up)
        projection_names = ("up_proj", "down_proj")
    expected = F.linear(
        hidden,
        reference.down_proj.weight,
        reference.down_proj.bias,
    ).squeeze(-1)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(cls_hidden.grad, reference_cls.grad, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        candidate_hidden.grad, reference_candidates.grad, atol=1e-12, rtol=1e-12,
    )
    for name in projection_names:
        for parameter_name, parameter in getattr(scorer, name).named_parameters():
            reference_parameter = getattr(getattr(reference, name), parameter_name)
            assert parameter.grad is not None
            torch.testing.assert_close(
                parameter.grad, reference_parameter.grad, atol=1e-12, rtol=1e-12,
            )


@pytest.mark.parametrize("activation", TRANSFORMER_ACTIVATIONS)
def test_candidate_scorer_keeps_two_layer_mlp_matrix_parameter_budget(activation):
    """折算后打分头矩阵参数量与原来的两层 MLP 近似相等。"""
    scorer = CandidateScorer(d_model=768, dropout=0.1, activation=activation)
    legacy_parameters = 2 * 768 * 768 + 768 + 768 + 1
    scorer_parameters = sum(parameter.numel() for parameter in scorer.parameters())

    assert abs(scorer_parameters - legacy_parameters) / legacy_parameters < 0.01
    if uses_gate(activation):
        assert scorer.gate_proj.out_features == 384
        assert scorer.up_proj.out_features == 384
        assert scorer.down_proj.in_features == 384
    else:
        assert scorer_parameters == legacy_parameters
        assert not hasattr(scorer, "gate_proj")


@pytest.mark.parametrize("activation", TRANSFORMER_ACTIVATIONS)
def test_pair_fusion_follows_configured_activation(activation):
    torch.manual_seed(29)
    encoder = CandidateInputEncoder(
        _activation_spec(),
        _activation_config(activation),
        vocab_size=4,
    ).double().eval()
    paired = torch.randn(2, 3, 8, dtype=torch.float64)

    hidden = F.linear(
        paired, encoder.pair_fusion_up.weight, encoder.pair_fusion_up.bias
    )
    if uses_gate(activation):
        gate = F.linear(
            paired, encoder.pair_fusion_gate.weight, encoder.pair_fusion_gate.bias
        )
        hidden = (gate * torch.sigmoid(gate)) * hidden
    else:
        hidden = resolve_pointwise_activation(activation)(hidden)
    expected = encoder.pair_fusion_down(encoder.pair_fusion_norm(hidden))

    torch.testing.assert_close(
        encoder._fuse_pair(paired), expected, atol=1e-12, rtol=1e-12
    )
    assert encoder.pair_fusion_gated is uses_gate(activation)


@pytest.mark.parametrize("activation", TRANSFORMER_ACTIVATIONS)
def test_model_resolves_one_activation_for_every_activation_site(activation):
    """主干 FFN、候选打分头与 pair 融合必须解析出同一个配置激活。"""
    model = CandidateTransformerModel(
        _activation_spec(),
        _activation_config(activation),
        vocab_size=4,
    )
    expected = resolve_pointwise_activation(activation)

    assert model.encoder.layers[0].activation is expected
    assert model.scorer.activation is expected
    assert model.input_encoder.pair_fusion_activation is expected
    assert model.scorer.gated is uses_gate(activation)
    assert model.input_encoder.pair_fusion_gated is uses_gate(activation)
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


def test_checkpoint_rejects_removed_candidate_scorer_layout():
    config = asdict(ModelConfig())
    with pytest.raises(ValueError, match="removed candidate scorer layout"):
        CandidateTransformerModel.checkpoint_model_config(
            {
                "model_config": config,
                "model_state_dict": {"scorer.network.0.weight": torch.zeros(1)},
            }
        )
    # 当前打分头的 checkpoint 仍然可以被正常解析。
    assert CandidateTransformerModel.checkpoint_model_config(
        {
            "model_config": config,
            "model_state_dict": {"scorer.up_proj.weight": torch.zeros(1)},
        }
    ).transformer_activation == ModelConfig.transformer_activation

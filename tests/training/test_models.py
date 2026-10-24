"""公共训练模型测试。"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

from common.policy.config import ModelConfig, resolve_policy_cache_dir
from common.policy.data import DataSpec, SkillVocab
from common.policy.model import (
    CausalPolicyModel,
    build_causal_attention_mask,
)
from common.policy.model.layers.input_encoder import build_position_ids
from common.pytorch.runtime import autocast_context, model_dtype, move_batch
from scripts.convert_fflogs.cache import select_training_raw_paths
from tests.training._causal_fixtures import make_batch, make_data_spec
from tests.training._common_fixtures import make_dataset as _make_dataset
from tests.training._common_fixtures import make_demo_pt as _make_demo_pt
from training import TrainingCollator
from training.config import (
    ValuePreferenceConfig,
    load_run_config,
)
from training.loop.dataloaders import _load_skill_values
from training.loop.losses.primary import primary_loss
from training.loop.losses.value_preference import (
    compute_value_preference_loss,
)


@pytest.mark.parametrize("num_kv_heads", (4, 2, 1))
@pytest.mark.parametrize("head_dim", (8, 12))
def test_latent_qkv_projection_matches_formula_and_learnable_gamma_gradients(num_kv_heads, head_dim):
    """独立公式锁定低秩 Q 与每头共享 KV 的归一化位置、FP32 统计及梯度。"""
    torch = pytest.importorskip("torch")
    from copy import deepcopy

    from common.policy.model.attention.grouped_attention import GroupedQueryAttention

    torch.manual_seed(819)
    attention = GroupedQueryAttention(32, 4, num_kv_heads, q_lora_rank=8, head_dim=head_dim, bias=False)
    assert attention.q_norm.eps == attention.kv_norm.eps == 1e-20
    torch.testing.assert_close(attention.q_norm.weight, torch.ones(8))
    torch.testing.assert_close(attention.kv_norm.weight, torch.ones(head_dim))
    with torch.no_grad():
        attention.q_norm.weight.copy_(torch.linspace(0.4, 1.7, 8))
        attention.kv_norm.weight.copy_(torch.linspace(1.6, 0.3, head_dim))
    reference = deepcopy(attention)
    inputs = torch.randn(2, 4, 32)
    inputs[:, 0] = 0
    inputs[:, 1] *= 1e-12
    inputs[:, 2] *= 1e4
    actual_inputs = inputs.clone().requires_grad_()
    reference_inputs = inputs.clone().requires_grad_()
    actual = attention.project_qkv(actual_inputs)
    assert actual[1] is actual[2]

    def normalize(values, gamma):
        normalized = values.float() * (values.float().square().mean(-1, keepdim=True) + 1e-20).rsqrt()
        return (normalized * gamma).to(values.dtype)

    q_latent = torch.nn.functional.linear(reference_inputs, reference.q_a.weight)
    query = torch.nn.functional.linear(normalize(q_latent, reference.q_norm.weight), reference.q_b.weight)
    kv_latent = torch.nn.functional.linear(reference_inputs, reference.kv_proj.weight)
    kv = normalize(kv_latent.unflatten(-1, (num_kv_heads, head_dim)), reference.kv_norm.weight).flatten(-2)
    expected = (query, kv, kv)
    for output, target in zip(actual, expected):
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, target, rtol=2e-6, atol=2e-6)
    probes = tuple(torch.randn_like(output) for output in actual)
    sum((output * probe).sum() for output, probe in zip(actual, probes)).backward()
    sum((output * probe).sum() for output, probe in zip(expected, probes)).backward()
    torch.testing.assert_close(actual_inputs.grad, reference_inputs.grad, rtol=2e-6, atol=2e-6)
    for (name, parameter), (_, expected_parameter) in zip(attention.named_parameters(), reference.named_parameters()):
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=2e-6, atol=2e-6)
        if name in {"q_norm.weight", "kv_norm.weight"}:
            assert torch.count_nonzero(parameter.grad) > 0


@pytest.mark.parametrize("num_kv_heads,full_attention_residuals", [
    (4, False), (2, False), (1, False), (1, True),
])
@pytest.mark.parametrize("head_dim,rope_head_dim", ((8, 8), (12, 4)))
def test_shared_rotated_kv_and_inverse_rope_match_trace_attention(
    monkeypatch, num_kv_heads, full_attention_residuals, head_dim, rope_head_dim,
):
    """非均匀 gamma 下验证共享旋转 KV、输出逆旋转及 trace 权重。"""
    torch = pytest.importorskip("torch")
    from common.policy.model.execution import causal_encoder

    torch.manual_seed(820)
    spec = make_data_spec()
    model = CausalPolicyModel(
        spec, ModelConfig(
            d_model=32, n_layers=2, n_heads=4, num_kv_heads=num_kv_heads,
            q_lora_rank=8, head_dim=head_dim, rope_head_dim=rope_head_dim,
            ff_dim=48, dropout=0.0, transformer_activation="swiglu",
            full_attention_residuals=full_attention_residuals,
        ), vocab_size=3,
    ).eval()
    with torch.no_grad():
        for layer in model.encoder.layers:
            layer.self_attn.q_norm.weight.copy_(torch.linspace(0.4, 1.7, 8))
            layer.self_attn.kv_norm.weight.copy_(torch.linspace(1.6, 0.3, head_dim))
    batch = make_batch(spec, batch_size=2, history_length=2, scene_length=2,
                       dtype=torch.float32)
    batch["scene_mask"][1, -1] = False
    batch["history_mask"][1, -1] = False
    attention_calls = []
    original_attention = causal_encoder.run_head_attention

    def observe_attention(layer, query, key, value, **kwargs):
        assert key is value
        assert query.shape[1] == 4
        assert key.shape[1] == num_kv_heads
        expanded_key = key.repeat_interleave(4 // num_kv_heads, dim=1)
        scores = query @ expanded_key.transpose(-2, -1) / query.shape[-1] ** 0.5
        count = query.shape[2]
        causal = torch.arange(count).view(-1, 1) >= torch.arange(count).view(1, -1)
        allowed = kwargs["key_valid"][:, None, None, :] & causal
        expected_weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        attended = expected_weights @ expanded_key
        positions = kwargs["position_ids"].float()
        frequencies = positions[..., None] / (
            10000 ** (torch.arange(0, rope_head_dim, 2).float() / rope_head_dim)
        )
        cosine, sine = frequencies.cos()[:, None], frequencies.sin()[:, None]
        pairs = attended[..., -rope_head_dim:].float().unflatten(-1, (-1, 2))
        first, second = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack((first * cosine + second * sine,
                               second * cosine - first * sine), dim=-1).flatten(-2)
        unrotated = torch.cat((attended[..., :-rope_head_dim], rotated.to(attended.dtype)), dim=-1)
        expected_output = layer.self_attn.out_proj(unrotated.transpose(1, 2).flatten(-2))
        output, weights = original_attention(layer, query, key, value, **kwargs)
        torch.testing.assert_close(output, expected_output, rtol=3e-6, atol=3e-6)
        if kwargs["collect_attention"]:
            torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=2e-6)
        attention_calls.append(kwargs["collect_attention"])
        return output, weights

    monkeypatch.setattr(causal_encoder, "run_head_attention", observe_attention)
    with torch.no_grad():
        logits = model(batch)["logits"]
        trace = model.trace(batch)
        traced_logits = model.score_hidden(trace.encoded, trace.hidden, batch)
    assert attention_calls == [False, False, True, True]
    torch.testing.assert_close(logits, traced_logits, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("num_kv_heads,full_attention_residuals,attention_block", [
    (4, False, False), (2, False, True), (1, True, False), (1, True, True),
])
def test_qk_normalization_preserves_recomputed_forward_and_gradients(
    num_kv_heads, full_attention_residuals, attention_block,
):
    """非均匀潜在表示尺度下，SDPA/整块/Full AttnRes 重算保留全部梯度。"""
    torch = pytest.importorskip("torch")
    from common.policy.model.execution.causal_encoder import run_causal_encoder

    torch.manual_seed(821)
    spec = make_data_spec()
    config = ModelConfig(
        d_model=16, n_layers=2, n_heads=4, num_kv_heads=num_kv_heads,
        ff_dim=32, dropout=0.1, transformer_activation="swiglu",
        full_attention_residuals=full_attention_residuals, q_lora_rank=4,
        head_dim=12, rope_head_dim=4,
    )
    eager = CausalPolicyModel(spec, config, vocab_size=3).float().train()
    with torch.no_grad():
        for layer in eager.encoder.layers:
            layer.self_attn.q_norm.weight.copy_(torch.linspace(0.5, 1.5, 4))
            layer.self_attn.kv_norm.weight.copy_(torch.linspace(1.5, 0.5, 12))
    recomputed = CausalPolicyModel(spec, config, vocab_size=3).float().train()
    recomputed.load_state_dict(eager.state_dict())
    recomputed.enable_activation_checkpoint_attention(True, block=attention_block)
    recomputed.enable_activation_checkpoint_ffn(True)
    template = torch.randn(2, 5, 16, dtype=torch.float32)
    target = torch.randn_like(template)
    valid = torch.tensor([[True, True, True, True, True],
                          [True, True, False, True, True]])
    positions = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 0, 2, 3]])

    def run(model):
        tokens = template.detach().clone().requires_grad_(True)
        # 相同 dropout 起点，同时覆盖会消费 RNG 的层内和外层重算。
        torch.manual_seed(822)
        hidden, _, _ = run_causal_encoder(model.encoder, {
            "tokens": tokens, "valid": valid, "position_ids": positions,
        })
        (hidden * target).sum().backward()
        gradients = {name: parameter.grad.detach().clone()
                     for name, parameter in model.encoder.named_parameters()
                     if parameter.grad is not None}
        return hidden.detach(), tokens.grad.detach(), gradients

    expected, expected_input_grad, expected_grads = run(eager)
    actual, actual_input_grad, actual_grads = run(recomputed)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual_input_grad, expected_input_grad, rtol=2e-6, atol=2e-6)
    assert actual_grads.keys() == expected_grads.keys()
    for name, gradient in actual_grads.items():
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, expected_grads[name], rtol=2e-6, atol=2e-6)
    for name in ("q_a.weight", "q_b.weight", "kv_proj.weight", "q_norm.weight", "kv_norm.weight"):
        assert torch.count_nonzero(actual_grads[f"layers.0.self_attn.{name}"]) > 0


def test_future_current_state_cannot_change_earlier_context_hidden():
    torch = pytest.importorskip("torch")
    torch.manual_seed(73)
    spec = make_data_spec()
    model = CausalPolicyModel(
        spec,
        ModelConfig(d_model=16, n_layers=2,
                    n_heads=4, ff_dim=32, dropout=0.0),
        vocab_size=3,
    ).eval()
    batch = make_batch(spec)
    first = model.trace(batch)
    altered = dict(batch, current_state_vectors=batch["current_state_vectors"] + 7.0)
    second = model.trace(altered)
    current = first.encoded["current_state_position"]
    torch.testing.assert_close(first.hidden[:, :current], second.hidden[:, :current])
    assert not torch.allclose(first.hidden[:, current], second.hidden[:, current])


def test_action_supervision_metadata_does_not_enter_transformer():
    torch = pytest.importorskip("torch")
    torch.manual_seed(74)
    spec = make_data_spec()
    model = CausalPolicyModel(
        spec,
        ModelConfig(d_model=16, n_layers=1,
                    n_heads=4, ff_dim=32, dropout=0.0),
        vocab_size=3,
    ).eval()
    batch = make_batch(spec)
    other = dict(batch, action_legal_mask=~batch["action_legal_mask"],
                 action_values=torch.tensor([[100.0, -100.0]]))
    with torch.no_grad():
        torch.testing.assert_close(model(batch)["logits"], model(other)["logits"])
    assert model.input_encoder(batch)["tokens"].shape[1] == 6


def test_select_training_raw_paths_uses_directory_proportions(tmp_path):
    directory_counts = {"FRU": 4, "M5s": 6, "M11s": 10, "M12s": 8}
    for directory_name, count in directory_counts.items():
        directory = tmp_path / directory_name
        directory.mkdir()
        for index in range(count):
            (directory / f"sample_{index:03d}.json.br").touch()

    selected = select_training_raw_paths(tmp_path, max_files=14)

    assert len(selected) == 14
    assert Counter(path.parent.name for path in selected) == {
        "FRU": 2,
        "M5s": 3,
        "M11s": 5,
        "M12s": 4,
    }


def test_common_model_uses_pt_dimensions_and_job_route(tmp_path):
    pytest.importorskip("torch")
    pt_path = _make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="model_demo")
    dataset = _make_dataset([pt_path])
    data_spec = DataSpec.from_dataset(dataset)

    model = CausalPolicyModel(
        data_spec,
        ModelConfig(d_model=32, n_layers=1, n_heads=4, ff_dim=64, dropout=0.0),
        vocab_size=SkillVocab.build_from_job_tag(data_spec.job_tag).size(),
    ).eval()
    batch = TrainingCollator()([dataset[0], dataset[1]])
    from common.policy.data.encoding.context_encoding import ContextEncoder

    batch = ContextEncoder(dataset.normalizer, dataset.schema, layout=dataset.state_layout).encode(batch)

    output = model(batch)
    assert data_spec.job_tag == "black_mage"
    assert output["logits"].shape == (2, data_spec.num_actions)
    assert output["top3_accuracy"].ndim == 0

    inference_batch = {key: value for key, value in batch.items() if key != "label_index"}
    predictions, logits = model.predict(inference_batch)
    assert predictions.shape == (2,)
    assert logits.shape == (2, data_spec.num_actions)


@pytest.mark.parametrize("precision", ("float32", "bf16"))
def test_swiglu_training_updates_all_ffn_projections_with_checkpointing(tmp_path, precision):
    torch = pytest.importorskip("torch")
    if precision == "bf16" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("CUDA BF16 is unavailable")
    device = torch.device("cuda" if precision == "bf16" else "cpu")
    torch.manual_seed(53)
    pt_path = _make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="swiglu_train")
    dataset = _make_dataset([pt_path])
    data_spec = DataSpec.from_dataset(dataset)
    model = CausalPolicyModel(
        data_spec,
        ModelConfig(
            d_model=32, n_layers=2, n_heads=4, num_kv_heads=1,
            ff_dim=96, dropout=0.1, transformer_activation="swiglu",
        ),
        vocab_size=SkillVocab.build_from_job_tag(data_spec.job_tag).size(),
    ).to(device=device).train()
    model.encoder.to(dtype=model_dtype(precision))
    model.output_head.to(dtype=model_dtype(precision))
    model.enable_activation_checkpoint_attention()
    model.enable_activation_checkpoint_ffn()
    batch = move_batch(TrainingCollator()([dataset[0], dataset[1]]), device)
    from common.policy.data.encoding.context_encoding import ContextEncoder

    batch = ContextEncoder(dataset.normalizer, dataset.schema, layout=dataset.state_layout).to(device=device).encode(batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    weights_before = {
        (index, name): getattr(layer, name).weight.detach().clone()
        for index, layer in enumerate(model.encoder.layers)
        for name in ("linear1", "gate_proj", "linear2")
    }
    with autocast_context(device, precision):
        loss = primary_loss(model(batch), batch)
    assert torch.isfinite(loss)
    loss.backward()
    for layer in model.encoder.layers:
        for name in ("linear1", "gate_proj", "linear2"):
            gradient = getattr(layer, name).weight.grad
            assert gradient is not None
            assert torch.isfinite(gradient).all()
            assert torch.count_nonzero(gradient) > 0
    optimizer.step()
    for (index, name), before in weights_before.items():
        after = getattr(model.encoder.layers[index], name).weight
        assert torch.isfinite(after).all()
        assert not torch.equal(before, after)


def test_position_ids_are_logical_sequential_when_all_tokens_are_valid():
    torch = pytest.importorskip("torch")
    position_ids = build_position_ids(prefix_valid=torch.ones((1, 7), dtype=torch.bool))[0].tolist()
    assert position_ids == [0, 1, 2, 3, 4, 5, 6, 7]


def test_position_ids_ignore_right_padding_per_sample():
    torch = pytest.importorskip("torch")
    position_ids = build_position_ids(prefix_valid=torch.tensor([
        [True] * 4 + [False] * 7,
        [True] * 9 + [False] * 2,
    ]))
    assert position_ids.tolist() == [
        [0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 9],
    ]


def test_position_ids_count_valid_tokens_without_right_padding():
    torch = pytest.importorskip("torch")
    position_ids = build_position_ids(prefix_valid=torch.tensor([
        [False, True, True, False, False, True, True, False, False, True, True],
    ]))
    assert position_ids.tolist() == [[0, 0, 1, 0, 0, 2, 3, 0, 0, 4, 5, 6]]


def test_rope_logits_are_invariant_to_other_samples_right_padding():
    torch = pytest.importorskip("torch")
    data_spec = make_data_spec(
        base_state_dim=3, scene_dim=2, skill_feature_dim=1,
        action_keys=("fire_iii", "fire_iv"),
        skill_feature_names=("potency",),
    )
    model = CausalPolicyModel(
        data_spec,
        ModelConfig(
            d_model=8,
            n_layers=1,
            n_heads=2,
            ff_dim=16,
            dropout=0.0,
        ),
        vocab_size=4,
    ).eval()

    short_batch = {
        "history_skill_ids": torch.tensor([[1]]),
        "history_skill_features": torch.tensor([[[0.25]]]),
        "history_state_vectors": torch.nn.functional.pad(torch.tensor([[[0.1, 0.2, 0.3]]]), (0, 12)),
        "history_state_null_mask": torch.zeros((1, 1, 3), dtype=torch.bool),
        "history_dead_positions": torch.empty((1, 0), dtype=torch.long),
        "history_dead_mask": torch.zeros((1, 0), dtype=torch.bool),
        "history_mask": torch.ones((1, 1), dtype=torch.bool),
        'current_state_vectors': torch.nn.functional.pad((torch.tensor(
            [[[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]]
        ))[:, 0, :], (0, 12)),
        'current_state_null_mask': (torch.zeros((1, 2, 3), dtype=torch.bool))[:, 0, :],
        "action_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]),
        "scene_types": torch.zeros((1, 2), dtype=torch.long),
        "scene_mask": torch.ones((1, 2), dtype=torch.bool),
        "history_state_reset_mask": torch.zeros_like(torch.tensor([[[0.1, 0.2, 0.3]]]), dtype=torch.bool),
        "current_state_reset_mask": torch.zeros_like((torch.tensor(
            [[[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]]
        ))[:, 0, :], dtype=torch.bool),
    }
    padded_batch = {
        key: torch.cat((value, value), dim=0)
        for key, value in short_batch.items()
    }
    padded_batch["scene_vectors"] = torch.cat(
        (
            torch.cat((short_batch["scene_vectors"], torch.zeros((1, 1, 2))), dim=1),
            torch.tensor([[[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]]),
        ),
        dim=0,
    )
    padded_batch["scene_types"] = torch.zeros((2, 3), dtype=torch.long)
    padded_batch["scene_mask"] = torch.tensor(
        [[True, True, False], [True, True, True]]
    )
    padded_batch["history_skill_ids"] = torch.tensor([[1, 0], [2, 3]])
    padded_batch["history_skill_features"] = torch.tensor(
        [[[0.25], [0.0]], [[0.6], [0.7]]]
    )
    padded_batch["history_state_vectors"] = torch.nn.functional.pad(torch.tensor(
        [
            [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]],
            [[0.8, 0.9, 1.0], [1.1, 1.2, 1.3]],
        ]
    ), (0, 12))
    padded_batch["history_state_null_mask"] = torch.tensor(
        [
            [[False, False, False], [True, True, True]],
            [[False, False, False], [False, False, False]],
        ]
    )
    padded_batch["history_mask"] = torch.tensor(
        [[True, False], [True, True]]
    )
    padded_batch["history_state_reset_mask"] = torch.zeros_like(
        padded_batch["history_state_vectors"][..., :data_spec.base_state_dim], dtype=torch.bool,
    )

    with torch.no_grad():
        standalone_logits = model(short_batch)["logits"]
        mixed_logits = model(padded_batch)["logits"]

    torch.testing.assert_close(
        standalone_logits[0], mixed_logits[0], rtol=1e-5, atol=1e-6
    )


def test_causal_attention_mask_allows_only_current_and_earlier_tokens():
    torch = pytest.importorskip("torch")
    mask = build_causal_attention_mask(token_count=6, device=torch.device("cpu"))
    assert mask.shape == (6, 6)
    torch.testing.assert_close(mask, torch.ones((6, 6), dtype=torch.bool).triu(1))
    assert not mask[-1].any()


def test_model_encode_with_attention_returns_per_head_weights():
    torch = pytest.importorskip("torch")
    data_spec = make_data_spec(
        base_state_dim=3, scene_dim=2, skill_feature_dim=1,
        action_keys=("fire_iii", "fire_iv"),
        skill_feature_names=("cast_time.seconds",),
    )
    model = CausalPolicyModel(
        data_spec,
        ModelConfig(
            d_model=8,
            n_layers=2,
            n_heads=2,
            ff_dim=16,
            dropout=0.0,
        ),
        vocab_size=4,
    ).eval()
    batch = make_batch(data_spec, history_length=1, scene_length=1)
    batch["history_state_reset_mask"] = torch.zeros_like(batch["history_state_reset_mask"])
    batch.pop("label_index")

    encoded, hidden, attentions = model.encode_with_attention(batch)

    assert hidden.shape == encoded["tokens"].shape
    assert len(attentions) == 2
    assert attentions[0].shape == (1, 2, encoded["tokens"].shape[1], encoded["tokens"].shape[1])
    assert encoded["current_state_position"] == 3
    assert encoded["tokens"].shape[1] == 4


def test_job_model_config_loads_training_precision(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "output_dir: artifacts/checkpoints/test\n"
        "model:\n"
        "  d_model: 192\n"
        "  full_attention_residuals: true\n"
        "training:\n"
        "  precision: bf16\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.precision == "bf16"
    assert config.model.d_model == 192
    assert config.model.full_attention_residuals is True
    assert config.model.transformer_norm_first is True
    assert config.model.transformer_activation == "gelu"


def test_job_model_config_loads_ffn_activation_checkpoint_switch(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  activation_checkpoint_ffn: true\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.activation_checkpoint_ffn is True


def test_job_model_config_loads_attention_activation_checkpoint_switch(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  activation_checkpoint_attention: true\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.activation_checkpoint_attention is True
    assert config.activation_checkpoint_attention_block is False


def test_job_model_config_loads_attention_checkpoint_block_switch(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  activation_checkpoint_attention: true\n"
        "  activation_checkpoint_attention_block: true\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.activation_checkpoint_attention_block is True


def test_job_model_config_loads_runtime_debug_switch(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  runtime_debug:\n"
        "    enabled: true\n"
        "    max_steps: 3\n"
        "    synchronize: false\n"
        "    output_filename: memory-debug.jsonl\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.runtime_debug.enabled is True
    assert config.runtime_debug.max_steps == 3
    assert config.runtime_debug.synchronize is False
    assert config.runtime_debug.output_filename == "memory-debug.jsonl"


def test_black_mage_artzip_uses_current_mainline_architecture():
    config = load_run_config(Path("config/models/black_mage/artzip/config.yaml"))

    assert config.model.d_model == 384
    assert config.model.n_layers == 6
    assert config.model.n_heads == 6
    assert config.model.num_kv_heads == 1
    assert config.model.ff_dim == 1536
    assert config.model.transformer_norm_first is True
    assert config.model.transformer_activation == "swiglu"
    assert config.model.context_capacity == 768
    assert config.model.full_attention_residuals is False
    assert "candidate_shuffle_enabled" not in config.__dict__
    assert "candidate_order_file" not in config.__dict__


def test_job_model_config_loads_history_truncation_switch(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  history_truncation:\n"
        "    enabled: true\n"
        "    probability: 0.25\n"
        "    min_recent: 2\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.history_truncation_enabled is True
    assert config.history_truncation_probability == 0.25
    assert config.history_min_recent == 2


def test_job_model_config_rejects_removed_candidate_shuffle(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n  candidate_shuffle:\n    enabled: false\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="candidate_shuffle"):
        load_run_config(config_path)


def test_job_model_config_rejects_removed_candidate_order(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n  candidate_order_file: candidate_order.yaml\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="candidate_order_file"):
        load_run_config(config_path)


def test_training_collator_requires_current_state_action_values(tmp_path):
    pt_path = _make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="value_batch_demo")
    dataset = _make_dataset([pt_path])
    sample = dict(dataset[0])
    assert "value" not in dataset.skill_feature_names
    sample.pop("action_values")
    with pytest.raises(KeyError, match="action_values"):
        TrainingCollator()([sample])


def test_training_collator_preserves_dynamic_action_values_from_state(tmp_path):
    torch = pytest.importorskip("torch")
    pt_path = _make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="dynamic_value_batch_demo")
    sample = dict(_make_dataset([pt_path])[0])
    runtime_values = torch.arange(len(sample["action_keys"]), dtype=sample["action_values"].dtype)
    sample["action_values"] = runtime_values
    batch = TrainingCollator()([sample])
    torch.testing.assert_close(batch["action_values"][0], runtime_values)


def test_job_model_config_loads_value_preference_switch(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  value_preference:\n"
        "    enabled: true\n"
        "    loss_weight: 0.1\n"
        "    margin_scale: 0.5\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.value_preference.enabled is True
    assert config.value_preference.loss_weight == 0.1
    assert config.value_preference.margin_scale == 0.5


def test_value_preference_only_pushes_high_value_label_against_lower_value_legal_actions():
    torch = pytest.importorskip("torch")
    config = ValuePreferenceConfig(enabled=True, loss_weight=0.05, margin_scale=0.25)
    batch = {
        "action_values": torch.tensor([[2.0, 1.0, 1.0]]),
        "action_legal_mask": torch.tensor([[True, True, True]]),
        "label_index": torch.tensor([0]),
    }

    unranked = compute_value_preference_loss(
        torch.zeros((1, 3)), batch, config
    )
    ranked = compute_value_preference_loss(
        torch.tensor([[1.0, 0.0, 0.0]]), batch, config
    )

    assert ranked < unranked

    batch["label_index"] = torch.tensor([1])
    assert compute_value_preference_loss(
        torch.zeros((1, 3)), batch, config
    ).item() == pytest.approx(0.0)


def test_value_preference_requires_runtime_job_values():
    torch = pytest.importorskip("torch")
    config = ValuePreferenceConfig(enabled=True, loss_weight=0.05, margin_scale=0.25)

    with pytest.raises(ValueError, match="runtime action values"):
        compute_value_preference_loss(
            torch.zeros((1, 1)),
            {"label_index": torch.tensor([0]), "action_legal_mask": torch.ones((1, 1), dtype=torch.bool)},
            config,
        )


def test_value_preference_loads_values_from_job_yaml():
    values = _load_skill_values("machinist")

    assert values["full_metal_field"] == pytest.approx(2.0)
    assert values["heated_split_shot"] == pytest.approx(1.0)
    assert values["ogcd_wait"] == pytest.approx(1.0)


def test_training_collator_truncates_only_early_history_and_preserves_scene(tmp_path):
    torch = pytest.importorskip("torch")
    pt_path = _make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="history_demo")
    dataset = _make_dataset([pt_path])
    sample = dict(dataset[1])
    state_width = sample["history_bank_state_abs_values"].shape[-1]
    feature_width = sample["history_bank_skill_features"].shape[-1]
    sample["history_action_keys"] = ["old", "middle", "latest"]
    sample["history_end"] = 4
    sample["history_length"] = 3
    sample["history_bank_action_keys"] = ("", "old", "middle", "latest")
    sample["history_bank_skill_ids"] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    sample["history_bank_skill_features"] = torch.zeros((4, feature_width))
    sample["history_bank_state_abs_values"] = torch.zeros((4, state_width))
    sample["history_bank_state_abs_values"][:, 0] = torch.tensor([0.0, 10.0, 20.0, 30.0])
    sample["history_bank_state_null_mask"] = torch.zeros((4, state_width), dtype=torch.bool)
    sample["history_bank_state_delta_values"] = torch.zeros((4, state_width))
    sample["history_bank_state_delta_reset_mask"] = torch.zeros((4, state_width), dtype=torch.bool)
    request_time_index = dataset.state_layout.request_time_index
    sample["_history_budget"] = {
        **sample["_history_budget"],
        "total_history": 3,
        "request_times": tuple(sample["history_bank_state_abs_values"][1:, request_time_index].tolist()),
    }
    scene_before = sample["scene_abs_values"].clone()
    decision_before = {
        key: sample[key].clone()
        for key in (
            "current_state_abs_values",
            "current_state_null_mask",
            "action_values",
            "action_legal_mask",
        )
    }

    batch = TrainingCollator(
        history_truncation_enabled=True,
        history_truncation_probability=1.0,
        history_min_recent=1,
        rng=random.Random(0),
    )([sample])

    history_length = int(batch["history_lengths"][0].item())
    assert 1 <= history_length < 3
    assert batch["history_action_keys"][0][-1] == "latest"
    assert batch["history_bank_state_abs_values"][batch["history_ends"][0] - 1, 0].item() == 30.0
    assert torch.equal(batch["scene_abs_values"][0, : scene_before.shape[0]], scene_before)
    for key, expected in decision_before.items():
        assert torch.equal(batch[key][0], expected)


def test_training_collator_preserves_fixed_output_order_and_label_mapping(tmp_path):
    torch = pytest.importorskip("torch")
    pt_path = _make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="fixed_action_demo")
    dataset = _make_dataset([pt_path])
    samples = [dataset[1], dataset[0]]
    batch = TrainingCollator(rng=random.Random(0))(samples)
    for index, sample in enumerate(samples):
        assert tuple(batch["action_keys"][index]) == dataset.action_keys
        label_index = batch["label_index"][index].item()
        assert batch["action_keys"][index][label_index] == sample["label_action_key"]
        for name in ("action_values", "action_legal_mask", "current_state_abs_values", "current_state_null_mask"):
            torch.testing.assert_close(batch[name][index], sample[name])


def test_job_model_config_loads_repetition_policy(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  repetition:\n"
        "    mode: whitelist\n"
        "    skills: [fire_iv, flare]\n"
        "    penalty: 1.25\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.repetition.mode == "whitelist"
    assert config.repetition.skills == ("fire_iv", "flare")
    assert config.repetition.penalty == 1.25


def test_low_precision_autocast_requires_cuda():
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="requires a CUDA device"):
        autocast_context(torch.device("cpu"), "bf16")


def test_job_model_config_loads_bounded_compiled_cache(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  compiled_cache_shard_size: 128\n"
        "  compiled_cache_max_shards: 3\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.compiled_cache_shard_size == 128
    assert config.compiled_cache_max_shards == 3


def test_training_cache_dir_is_scoped_by_job(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAINING_CACHE_ROOT", str(tmp_path))

    assert resolve_policy_cache_dir("black_mage") == tmp_path / "black_mage" / ".cache"

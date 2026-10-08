"""独立状态/技能 token 的因果顺序、位置和共享参数验收。"""

from __future__ import annotations

import pytest
import torch

from common.policy.config import ModelConfig
from common.policy.model import CausalPolicyModel
from common.policy.model.input_encoder import ROLE_SCENE, ROLE_SKILL, ROLE_STATE
from tests.training._causal_fixtures import make_batch, make_data_spec


def _model(*, full_attention_residuals=False, history_capacity=4, scene_capacity=3):
    spec = make_data_spec(num_scene_types=2)
    config = ModelConfig(d_model=8, n_heads=2, num_kv_heads=1, n_layers=2,
                         ff_dim=16, dropout=0.0, scene_capacity=scene_capacity,
                         history_capacity=history_capacity,
                         history_reset_keep=min(8, history_capacity),
                         full_attention_residuals=full_attention_residuals)
    return CausalPolicyModel(spec, config, vocab_size=3).eval()


@pytest.mark.parametrize("changed_field", ["history_skill_ids", "history_skill_features",
                                         "history_state_vectors", "history_state_null_mask"])
def test_skill_and_state_inputs_do_not_cross_before_attention(changed_field):
    torch.manual_seed(911)
    model = _model()
    batch = make_batch(model.data_spec)
    original = model.input_encoder.embed_history(batch)
    changed = dict(batch)
    if changed_field == "history_skill_ids":
        changed[changed_field] = torch.full_like(batch[changed_field], 2)
    elif changed_field.endswith("null_mask"):
        changed[changed_field] = ~batch[changed_field]
    else:
        changed[changed_field] = torch.randn_like(batch[changed_field])
    updated = model.input_encoder.embed_history(changed)
    modified_kind = "skill" if changed_field.startswith("history_skill") else "state"
    fixed_kind = "state" if modified_kind == "skill" else "skill"
    torch.testing.assert_close(original[fixed_kind], updated[fixed_kind], atol=0, rtol=0)
    assert not torch.allclose(original[modified_kind], updated[modified_kind])


def test_current_and_history_state_share_encoder_and_role():
    torch.manual_seed(912)
    model = _model()
    batch = make_batch(model.data_spec, batch_size=2)
    batch["history_state_vectors"] = torch.randn_like(batch["history_state_vectors"])
    batch["history_state_null_mask"] = torch.tensor([[[True, False, True, False], [False, True, False, True]],
                                                     [[False, False, True, True], [True, True, False, False]]])
    batch["current_state_vectors"] = batch["history_state_vectors"][:, -1].clone()
    batch["current_state_null_mask"] = batch["history_state_null_mask"][:, -1].clone()
    encoded = model.input_encoder(batch)
    last_history_state = encoded["history_state_positions"][0, -1].item()
    current = encoded["current_state_position"]
    torch.testing.assert_close(encoded["tokens"][:, last_history_state], encoded["tokens"][:, current], atol=0, rtol=0)
    assert encoded["role_ids"][:, last_history_state].tolist() == [ROLE_STATE] * 2
    assert encoded["role_ids"][:, current].tolist() == [ROLE_STATE] * 2
    assert model.input_encoder.state_null_proj.bias is None
    assert not hasattr(model.input_encoder, "segment_embed")
    assert "segment_ids" not in encoded


@pytest.mark.parametrize("full_attention_residuals", [False, True])
@pytest.mark.parametrize("changed_field", ["history_skill_ids", "history_skill_features"])
def test_skill_cannot_change_its_preceding_request_state(full_attention_residuals, changed_field):
    torch.manual_seed(913)
    model = _model(full_attention_residuals=full_attention_residuals)
    batch = make_batch(model.data_spec)
    original = model.trace(batch)
    changed = dict(batch)
    changed[changed_field] = batch[changed_field].clone()
    if changed_field == "history_skill_ids":
        changed[changed_field][:, 0] = 2
    else:
        changed[changed_field][:, 0] = torch.tensor([3.0, -4.0])
    updated = model.trace(changed)
    skill_position = original.encoded["history_skill_positions"][0, 0].item()
    state_position = original.encoded["history_state_positions"][0, 0].item()
    assert skill_position == state_position + 1
    torch.testing.assert_close(original.hidden[:, :skill_position], updated.hidden[:, :skill_position], atol=0, rtol=0)
    assert not torch.allclose(original.hidden[:, skill_position], updated.hidden[:, skill_position])


@pytest.mark.parametrize("full_attention_residuals", [False, True])
def test_request_state_is_visible_to_its_skill_and_later_tokens(full_attention_residuals):
    torch.manual_seed(914)
    model = _model(full_attention_residuals=full_attention_residuals)
    batch = make_batch(model.data_spec)
    original = model.trace(batch)
    changed = dict(batch)
    changed["history_state_vectors"] = batch["history_state_vectors"].clone()
    changed["history_state_vectors"][:, 0] = torch.tensor([3.0, -4.0, 5.0, 2.0])
    updated = model.trace(changed)
    state_position = original.encoded["history_state_positions"][0, 0].item()
    skill_position = original.encoded["history_skill_positions"][0, 0].item()
    torch.testing.assert_close(original.hidden[:, :state_position], updated.hidden[:, :state_position], atol=0, rtol=0)
    assert not torch.allclose(original.hidden[:, state_position], updated.hidden[:, state_position])
    assert not torch.allclose(original.hidden[:, skill_position], updated.hidden[:, skill_position])
    assert not torch.allclose(original.hidden[:, -1], updated.hidden[:, -1])


def test_interleaved_layout_keeps_action_positions_and_excludes_padding_from_rope():
    model = _model()
    batch = make_batch(model.data_spec, batch_size=2, scene_length=3, history_length=3)
    batch["scene_mask"] = torch.tensor([[True, False, True], [True, True, True]])
    batch["history_mask"] = torch.tensor([[False, True, False], [True, True, False]])
    encoded = model.input_encoder(batch)
    assert encoded["tokens"].shape == (2, 10, 8)
    assert encoded["history_length"] == 3
    assert encoded["history_token_length"] == 6
    assert encoded["prefix_length"] == encoded["current_state_position"] == 9
    assert encoded["current_state_positions"].tolist() == [9, 9]
    assert encoded["history_skill_positions"].tolist() == [[4, 6, 8]] * 2
    assert encoded["history_state_positions"].tolist() == [[3, 5, 7]] * 2
    assert encoded["role_ids"].tolist() == [[ROLE_SCENE] * 3 + [ROLE_STATE, ROLE_SKILL] * 3 + [ROLE_STATE]] * 2
    assert encoded["position_ids"].tolist() == [[0, 0, 1, 0, 0, 2, 3, 0, 0, 4],
                                                [0, 1, 2, 3, 4, 5, 6, 0, 0, 7]]
    assert encoded["valid"].tolist() == [[True, False, True, False, False, True, True, False, False, True],
                                        [True, True, True, True, True, True, True, False, False, True]]


@pytest.mark.parametrize("history_capacity", [300, 384])
def test_history_capacity_still_counts_actions(history_capacity):
    model = _model(history_capacity=history_capacity, scene_capacity=4)
    batch = make_batch(model.data_spec, history_length=history_capacity, scene_length=4)
    encoded = model.input_encoder(batch)
    assert model.config.history_capacity == history_capacity
    assert model.input_encoder.max_token_count == 4 + 2 * history_capacity + 1
    assert encoded["history_length"] == history_capacity
    assert encoded["history_token_length"] == 2 * history_capacity
    assert encoded["tokens"].shape[1] == 4 + 2 * history_capacity + 1
    excessive = make_batch(model.data_spec, history_length=history_capacity + 1, scene_length=4)
    with pytest.raises(ValueError, match="model.history_capacity"):
        model.input_encoder(excessive)


@pytest.mark.parametrize("history_length", [0, 2])
def test_empty_or_fully_masked_history_keeps_only_current_state_visible(history_length):
    model = _model()
    batch = make_batch(model.data_spec, scene_length=2, history_length=history_length)
    batch["scene_mask"].zero_()
    batch["history_mask"].zero_()
    encoded = model.input_encoder(batch)
    assert encoded["position_ids"].count_nonzero().item() == 0
    assert encoded["valid"].sum().item() == 1
    assert encoded["history_skill_positions"].shape == (1, history_length)
    with torch.no_grad():
        assert torch.isfinite(model(batch)["logits"]).all()


@pytest.mark.parametrize("legacy_key", ["output_adapter.weight", "input_encoder.pair_fusion_up.weight",
                                       "input_encoder.token_embedding.0.weight", "input_encoder.segment_embed.weight",
                                       "_learned_residual_mix_experiment_guard"])
def test_unexpected_architecture_weights_are_rejected_by_strict_loading(legacy_key):
    model = _model()
    weights = {**model.state_dict(), legacy_key: torch.zeros(1)}
    with pytest.raises(RuntimeError, match="Unexpected key"):
        model.load_state_dict(weights, strict=True)

"""公共训练模型测试。"""

from __future__ import annotations

from collections import Counter
import random
from pathlib import Path

import pytest

from common.torch_runtime import autocast_context, model_dtype, move_batch
from scripts.convert_fflogs.cache import select_training_raw_paths
from tests.training._common_fixtures import make_dataset as _make_dataset
from tests.training._common_fixtures import make_demo_pt as _make_demo_pt
from common.policy.config import ModelConfig
from common.policy.data import DataSpec, SkillVocab
from common.policy.model import (
    CausalPolicyModel,
    RepetitionConfig,
    build_causal_attention_mask,
)
from training import TrainingCollator
from training.config import (
    ValuePreferenceConfig,
)
from common.policy.config import resolve_policy_cache_dir
from training.config import load_run_config
from training.loop.losses.value_preference import (
    compute_value_preference_loss,
)
from training.data.skill_values import load_skill_values
from training.loop.losses.primary import primary_loss
from common.policy.model.input_encoder import build_position_ids
from common.policy.model.repetition import apply_repetition_penalty
from tests.training._causal_fixtures import make_batch, make_data_spec


@pytest.mark.parametrize("scale", (1.2, 0.9))
def test_qk_rmsnorm_matches_independent_formula_and_gradients_at_small_norms(scale):
    """零向量、极小向量与不同 head 幅度都遵循 epsilon 公式，并保持有限梯度。"""
    torch = pytest.importorskip("torch")
    from common.policy.model.attention_utils import normalize_qk

    torch.manual_seed(819)
    raw_query = torch.randn(1, 4, 3, 8, dtype=torch.float64)
    raw_key = torch.randn(1, 2, 3, 8, dtype=torch.float64)
    raw_query[:, 0] = 0
    raw_query[:, 1] *= 1e-10
    raw_query[:, 2] *= 1e4
    raw_key[:, 0] *= 1e-10
    actual_inputs = tuple(tensor.clone().requires_grad_(True) for tensor in (raw_query, raw_key))
    reference_inputs = tuple(tensor.clone().requires_grad_(True) for tensor in (raw_query, raw_key))
    actual = normalize_qk(*actual_inputs, scale=scale)
    epsilon = torch.finfo(torch.float64).eps
    reference = tuple(
        tensor * scale / (tensor.square().mean(-1, keepdim=True) + epsilon).sqrt()
        for tensor in reference_inputs
    )
    for left, right in zip(actual, reference):
        assert torch.isfinite(left).all()
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(actual[0][:, 0]) == 0
    # epsilon 会衰减极小向量，不把所有 head 无条件顶到固定 RMS。
    assert (actual[0][:, 1].square().mean(-1).sqrt() < scale / 10).all()
    targets = tuple(torch.randn_like(tensor) for tensor in actual)
    actual_grad = torch.autograd.grad(sum((value * target).sum() for value, target in zip(actual, targets)), actual_inputs)
    reference_grad = torch.autograd.grad(sum((value * target).sum() for value, target in zip(reference, targets)), reference_inputs)
    for left, right in zip(actual_grad, reference_grad):
        assert torch.isfinite(left).all()
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-9)


@pytest.mark.parametrize("num_kv_heads", (4, 2, 1))
@pytest.mark.parametrize("full_attention_residuals", (False, True))
@pytest.mark.parametrize("scale", (1.2, 0.7))
def test_qk_head_rmsnorm_follows_rope_and_matches_trace_attention(
    monkeypatch, num_kv_heads, full_attention_residuals, scale,
):
    """用实际 attention 边界与独立公式验证 head 范数、位置和 trace 权重。"""
    torch = pytest.importorskip("torch")
    from common.policy.model import causal_encoder

    torch.manual_seed(820)
    spec = make_data_spec()
    model = CausalPolicyModel(
        spec, ModelConfig(
            d_model=32, n_layers=2, n_heads=4, num_kv_heads=num_kv_heads,
            ff_dim=48, dropout=0.0, transformer_activation="swiglu",
            full_attention_residuals=full_attention_residuals, qk_norm_scale=scale,
        ), vocab_size=3,
    ).double().eval()
    batch = make_batch(spec, batch_size=2, history_length=2, scene_length=2,
                       dtype=torch.float64)
    batch["scene_mask"][1, -1] = False
    batch["history_mask"][1, -1] = False
    rotated, attention_calls = [], []
    original_attention = causal_encoder.run_head_attention

    def perturb_rope_output(_module, _args, output):
        # 正常 RoPE 与 RMS 在数学上可交换；非均匀 hook 才能锁定真实先后顺序。
        query, key = output
        channels = torch.arange(1, query.shape[-1] + 1, dtype=query.dtype)
        changed = (query * channels, key * channels.flip(0))
        rotated.append(tuple(value.detach().clone() for value in changed))
        return changed

    def observe_attention(layer, query, key, value, **kwargs):
        raw_query, raw_key = rotated[-1]
        epsilon = torch.finfo(torch.float64).eps
        expected = tuple(
            tensor * scale / (tensor.square().mean(-1, keepdim=True) + epsilon).sqrt()
            for tensor in (raw_query, raw_key)
        )
        torch.testing.assert_close(query, expected[0], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(key, expected[1], rtol=1e-12, atol=1e-12)
        assert query.shape[1] == 4
        assert key.shape[1] == value.shape[1] == num_kv_heads
        for tensor in (query, key):
            head_rms = tensor.square().mean(-1).sqrt()
            torch.testing.assert_close(head_rms, torch.full_like(head_rms, scale),
                                       rtol=1e-12, atol=1e-12)
        output, weights = original_attention(layer, query, key, value, **kwargs)
        if kwargs["collect_attention"]:
            # 独立展开 MQA/GQA，按合法 key 与严格因果 mask 计算完整逐 head 权重。
            expanded_key = key.repeat_interleave(4 // num_kv_heads, dim=1)
            scores = query @ expanded_key.transpose(-2, -1) / query.shape[-1] ** 0.5
            count = query.shape[2]
            causal = torch.arange(count).view(-1, 1) >= torch.arange(count).view(1, -1)
            allowed = kwargs["key_valid"][:, None, None, :] & causal
            expected_weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
            torch.testing.assert_close(weights, expected_weights, rtol=1e-12, atol=1e-12)
        attention_calls.append(kwargs["collect_attention"])
        return output, weights

    monkeypatch.setattr(causal_encoder, "run_head_attention", observe_attention)
    handle = model.encoder.rotary_position_encoding.register_forward_hook(perturb_rope_output)
    try:
        with torch.no_grad():
            logits = model(batch)["logits"]
            trace = model.trace(batch)
            traced_logits = model.score_hidden(trace.encoded, trace.hidden, batch)
    finally:
        handle.remove()
    assert attention_calls == [False, False, True, True]
    torch.testing.assert_close(logits, traced_logits, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("num_kv_heads", (4, 2, 1))
@pytest.mark.parametrize("full_attention_residuals", (False, True))
@pytest.mark.parametrize("attention_block", (False, True))
def test_qk_normalization_preserves_recomputed_forward_and_gradients(
    num_kv_heads, full_attention_residuals, attention_block,
):
    """非默认 Q/K 尺度下，SDPA/整块/Full AttnRes 重算保留输入和所有参数梯度。"""
    torch = pytest.importorskip("torch")
    from common.policy.model.causal_encoder import run_causal_encoder

    torch.manual_seed(821)
    spec = make_data_spec()
    config = ModelConfig(
        d_model=16, n_layers=2, n_heads=4, num_kv_heads=num_kv_heads,
        ff_dim=32, dropout=0.1, transformer_activation="swiglu",
        full_attention_residuals=full_attention_residuals, qk_norm_scale=0.85,
    )
    eager = CausalPolicyModel(spec, config, vocab_size=3).double().train()
    recomputed = CausalPolicyModel(spec, config, vocab_size=3).double().train()
    recomputed.load_state_dict(eager.state_dict())
    recomputed.enable_activation_checkpoint_attention(True, block=attention_block)
    recomputed.enable_activation_checkpoint_ffn(True)
    template = torch.randn(2, 5, 16, dtype=torch.float64)
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
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(actual_input_grad, expected_input_grad, rtol=1e-10, atol=1e-10)
    assert actual_grads.keys() == expected_grads.keys()
    for name, gradient in actual_grads.items():
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, expected_grads[name], rtol=1e-10, atol=1e-10)
    # 标准 MHA 使用 packed 投影，GQA/MQA 使用独立投影；两种布局都要保留 Q/K 梯度。
    if num_kv_heads == 4:
        for gradient in actual_grads["layers.0.self_attn.in_proj_weight"].chunk(3, dim=0)[:2]:
            assert torch.count_nonzero(gradient) > 0
    else:
        for name in ("q_proj", "k_proj"):
            assert torch.count_nonzero(actual_grads[f"layers.0.self_attn.{name}.weight"]) > 0


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
    torch = pytest.importorskip("torch")
    pt_path = _make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="model_demo")
    dataset = _make_dataset([pt_path])
    data_spec = DataSpec.from_dataset(dataset)

    model = CausalPolicyModel(
        data_spec,
        ModelConfig(d_model=32, n_layers=1, n_heads=4, ff_dim=64, dropout=0.0),
        vocab_size=SkillVocab.build_from_job_tag(data_spec.job_tag).size(),
    ).eval()
    batch = TrainingCollator()([dataset[0], dataset[1]])

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
    ).to(device=device, dtype=model_dtype(precision)).train()
    model.enable_activation_checkpoint_attention()
    model.enable_activation_checkpoint_ffn()
    batch = move_batch(TrainingCollator()([dataset[0], dataset[1]]), device)
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
    position_ids = build_position_ids(
        batch_size=1,
        scene_length=3,
        history_length=2,
        device=torch.device("cpu"),
    )[0].tolist()

    assert position_ids == [0, 1, 2, 3, 4, 5, 6, 7]
    assert position_ids[-1] == 7


def test_position_ids_ignore_right_padding_per_sample():
    torch = pytest.importorskip("torch")
    position_ids = build_position_ids(
        batch_size=2,
        scene_length=3,
        history_length=4,
        device=torch.device("cpu"),
        scene_mask=torch.tensor([[True, True, False], [True, True, True]]),
        history_mask=torch.tensor(
            [[True, False, False, False], [True, True, True, False]]
        ),
    )

    assert position_ids.tolist() == [
        [0, 1, 0, 2, 3, 0, 0, 0, 0, 0, 0, 4],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 9],
    ]


def test_position_ids_count_valid_tokens_without_right_padding():
    torch = pytest.importorskip("torch")
    position_ids = build_position_ids(
        batch_size=1,
        scene_length=3,
        history_length=4,
        device=torch.device("cpu"),
        scene_mask=torch.tensor([[False, True, True]]),
        history_mask=torch.tensor([[False, True, False, True]]),
    )

    assert position_ids.tolist() == [[0, 0, 1, 0, 0, 2, 3, 0, 0, 4, 5, 6]]


def test_current_state_rope_position_follows_scene_and_history():
    torch = pytest.importorskip("torch")
    position_ids = build_position_ids(
        batch_size=1,
        scene_length=2,
        history_length=2,
        device=torch.device("cpu"),
    )

    assert position_ids.tolist() == [[0, 1, 2, 3, 4, 5, 6]]


def test_rope_logits_are_invariant_to_other_samples_right_padding():
    torch = pytest.importorskip("torch")
    data_spec = DataSpec(
        job_tag="black_mage",
        num_actions=2,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=1,
        num_scene_types=1,
        action_to_vocab_id=(1, 2), action_is_gcd=(True, True), action_keys=("fire_iii", "fire_iv"),
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
        "history_state_vectors": torch.tensor([[[0.1, 0.2, 0.3]]]),
        "history_state_null_mask": torch.zeros((1, 1, 3), dtype=torch.bool),
        "history_mask": torch.ones((1, 1), dtype=torch.bool),
        'current_state_vectors': (torch.tensor(
            [[[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]]
        ))[:, 0, :],
        'current_state_null_mask': (torch.zeros((1, 2, 3), dtype=torch.bool))[:, 0, :],
        "action_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]),
        "scene_types": torch.zeros((1, 2), dtype=torch.long),
        "scene_mask": torch.ones((1, 2), dtype=torch.bool),
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
    padded_batch["history_state_vectors"] = torch.tensor(
        [
            [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]],
            [[0.8, 0.9, 1.0], [1.1, 1.2, 1.3]],
        ]
    )
    padded_batch["history_state_null_mask"] = torch.tensor(
        [
            [[False, False, False], [True, True, True]],
            [[False, False, False], [False, False, False]],
        ]
    )
    padded_batch["history_mask"] = torch.tensor(
        [[True, False], [True, True]]
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


def test_causal_model_appends_one_current_state_token():
    torch = pytest.importorskip("torch")
    data_spec = DataSpec(
        job_tag="black_mage",
        num_actions=2,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=1,
        num_scene_types=1,
        action_to_vocab_id=(1, 2), action_is_gcd=(True, True), action_keys=("fire_iii", "fire_iv"),
        skill_feature_names=("cast_time.seconds",),
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
    batch = {
        "history_skill_ids": torch.ones((1, 2), dtype=torch.int64),
        "history_skill_features": torch.zeros((1, 2, 1)),
        "history_state_vectors": torch.zeros((1, 2, 3)),
        "history_state_null_mask": torch.zeros((1, 2, 3), dtype=torch.bool),
        "history_mask": torch.ones((1, 2), dtype=torch.bool),
        'current_state_vectors': (torch.zeros((1, 2, 3)))[:, 0, :],
        'current_state_null_mask': (torch.zeros((1, 2, 3), dtype=torch.bool))[:, 0, :],
        "action_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.zeros((1, 1, 2)),
        "scene_types": torch.zeros((1, 1), dtype=torch.int64),
        "scene_mask": torch.ones((1, 1), dtype=torch.bool),
    }

    output = model(batch)
    assert output["logits"].shape == (1, 2)
    encoded = model.input_encoder(batch)
    assert "attention_mask" not in encoded
    assert encoded["history_length"] == 2
    assert encoded["history_token_length"] == 4
    assert encoded["prefix_length"] == 5
    assert encoded["current_state_position"] == 5
    assert encoded["tokens"].shape == (1, 6, 8)
    assert encoded["role_ids"].tolist() == [[0, 1, 2, 1, 2, 1]]
    assert encoded["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5]]
    assert not any(key.startswith("cls_") for key in encoded)


def test_input_encoder_derives_context_capacity_from_context_blocks():
    torch = pytest.importorskip("torch")
    data_spec = DataSpec(
        job_tag="black_mage",
        num_actions=2,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=1,
        num_scene_types=1,
        action_to_vocab_id=(1, 2), action_is_gcd=(True, True), action_keys=("fire_iii", "fire_iv"),
        skill_feature_names=("cast_time.seconds",),
    )
    model = CausalPolicyModel(
        data_spec,
        ModelConfig(
            d_model=8,
            n_layers=1,
            n_heads=2,
            ff_dim=16,
            dropout=0.0,
            scene_capacity=4,
            history_capacity=2,
        ),
        vocab_size=4,
    ).eval()
    # 历史容量按动作计数：scene 4 + 技能/状态 2 * 2 + 当前状态 1 = 9。
    assert model.input_encoder.max_token_count == 9
    batch = {
        "history_skill_ids": torch.ones((1, 2), dtype=torch.int64),
        "history_skill_features": torch.zeros((1, 2, 1)),
        "history_state_vectors": torch.zeros((1, 2, 3)),
        "history_state_null_mask": torch.zeros((1, 2, 3), dtype=torch.bool),
        "history_mask": torch.ones((1, 2), dtype=torch.bool),
        'current_state_vectors': (torch.zeros((1, 2, 3)))[:, 0, :],
        'current_state_null_mask': (torch.zeros((1, 2, 3), dtype=torch.bool))[:, 0, :],
        "action_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.zeros((1, 4, 2)),
        "scene_types": torch.zeros((1, 4), dtype=torch.int64),
        "scene_mask": torch.ones((1, 4), dtype=torch.bool),
    }

    with torch.no_grad():
        output = model(batch)
    assert output["logits"].shape == (1, 2)

    batch["scene_vectors"] = torch.zeros((1, 5, 2))
    batch["scene_types"] = torch.zeros((1, 5), dtype=torch.int64)
    batch["scene_mask"] = torch.ones((1, 5), dtype=torch.bool)
    with pytest.raises(ValueError, match="model.scene_capacity"):
        model(batch)


def test_model_encode_with_attention_returns_per_head_weights():
    torch = pytest.importorskip("torch")
    data_spec = DataSpec(
        job_tag="black_mage",
        num_actions=2,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=1,
        num_scene_types=1,
        action_to_vocab_id=(1, 2), action_is_gcd=(True, True), action_keys=("fire_iii", "fire_iv"),
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
    batch = {
        "history_action_keys": [["fire_iii"]],
        "action_keys": [["fire_iii", "fire_iv"]],
        "history_skill_ids": torch.ones((1, 1), dtype=torch.int64),
        "history_skill_features": torch.zeros((1, 1, 1)),
        "history_state_vectors": torch.zeros((1, 1, 3)),
        "history_state_null_mask": torch.zeros((1, 1, 3), dtype=torch.bool),
        "history_mask": torch.ones((1, 1), dtype=torch.bool),
        'current_state_vectors': (torch.zeros((1, 2, 3)))[:, 0, :],
        'current_state_null_mask': (torch.zeros((1, 2, 3), dtype=torch.bool))[:, 0, :],
        "action_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.zeros((1, 1, 2)),
        "scene_types": torch.zeros((1, 1), dtype=torch.int64),
        "scene_mask": torch.ones((1, 1), dtype=torch.bool),
    }

    encoded, hidden, attentions = model.encode_with_attention(batch)

    assert hidden.shape == encoded["tokens"].shape
    assert len(attentions) == 2
    assert attentions[0].shape == (1, 2, encoded["tokens"].shape[1], encoded["tokens"].shape[1])
    assert encoded["current_state_position"] == 3
    assert encoded["tokens"].shape[1] == 4


def test_history_uses_independent_tokens_and_current_state_has_no_skill():
    torch = pytest.importorskip("torch")
    data_spec = DataSpec(
        job_tag="black_mage",
        num_actions=2,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=1,
        num_scene_types=1,
        action_to_vocab_id=(1, 2), action_is_gcd=(True, True), action_keys=("fire_iii", "fire_iv"),
        skill_feature_names=("cast_time.seconds",),
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
    batch = {
        "history_skill_ids": torch.ones((1, 2), dtype=torch.int64),
        "history_skill_features": torch.zeros((1, 2, 1)),
        "history_state_vectors": torch.zeros((1, 2, 3)),
        "history_state_null_mask": torch.zeros((1, 2, 3), dtype=torch.bool),
        "history_mask": torch.ones((1, 2), dtype=torch.bool),
        'current_state_vectors': (torch.zeros((1, 2, 3)))[:, 0, :],
        'current_state_null_mask': (torch.zeros((1, 2, 3), dtype=torch.bool))[:, 0, :],
        "action_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.zeros((1, 1, 2)),
        "scene_types": torch.zeros((1, 1), dtype=torch.int64),
        "scene_mask": torch.ones((1, 1), dtype=torch.bool),
    }

    encoded = model.input_encoder(batch)

    assert encoded["tokens"].shape[1] == 1 + 2 * 2 + 1
    assert set(model.input_encoder.embed_history(batch)) == {"skill", "state"}
    assert encoded["current_state_position"] == 5
    assert encoded["history_skill_positions"].tolist() == [[2, 4]]
    assert encoded["history_state_positions"].tolist() == [[1, 3]]


def test_input_encoder_routes_all_sources_through_one_post_role_rms(monkeypatch):
    torch = pytest.importorskip("torch")
    data_spec = DataSpec(
        job_tag="black_mage",
        num_actions=2,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=1,
        num_scene_types=1,
        action_to_vocab_id=(1, 2), action_is_gcd=(True, True), action_keys=("fire_iii", "fire_iv"),
        skill_feature_names=("cast_time.seconds",),
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
    batch = {
        "history_skill_ids": torch.ones((1, 2), dtype=torch.int64),
        "history_skill_features": torch.zeros((1, 2, 1)),
        "history_state_vectors": torch.zeros((1, 2, 3)),
        "history_state_null_mask": torch.zeros((1, 2, 3), dtype=torch.bool),
        "history_mask": torch.ones((1, 2), dtype=torch.bool),
        'current_state_vectors': (torch.zeros((1, 2, 3)))[:, 0, :],
        'current_state_null_mask': (torch.zeros((1, 2, 3), dtype=torch.bool))[:, 0, :],
        "action_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.zeros((1, 1, 2)),
        "scene_types": torch.zeros((1, 1), dtype=torch.int64),
        "scene_mask": torch.ones((1, 1), dtype=torch.bool),
    }
    captured = []
    original_rms = torch.nn.functional.rms_norm

    def capture_rms(values, normalized_shape, *, eps):
        captured.append((values.detach().clone(), normalized_shape, eps))
        return original_rms(values, normalized_shape, eps=eps)

    monkeypatch.setattr(torch.nn.functional, "rms_norm", capture_rms)
    encoded = model.input_encoder(batch)

    # 全部场景、交错历史和最新状态合成后，仅进入同一处 RMSNorm。
    assert len(captured) == 1
    values, normalized_shape, eps = captured[0]
    assert values.shape == (1, 6, 8)
    assert normalized_shape == (8,)
    assert eps == 1e-5
    scene_content = model.input_encoder.scene_proj[0](batch["scene_vectors"])
    scene_role = model.input_encoder.role_embed(encoded["role_ids"][:, :1])
    torch.testing.assert_close(values[:, :1], scene_content + scene_role)
    assert not any(isinstance(module, torch.nn.LayerNorm)
                   for module in model.input_encoder.modules())
    assert model.input_encoder.scene_proj[0].out_features == 8
    assert model.input_encoder.role_embed.num_embeddings == 3
    assert "input_encoder.cls_token" not in model.state_dict()
    assert not hasattr(model.input_encoder, "token_embedding")
    assert not hasattr(model.input_encoder, "segment_embed")
    assert not hasattr(model, "output_adapter")


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
    assert config.model.history_capacity == 300
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
    values = load_skill_values("machinist")

    assert values["full_metal_field"] == pytest.approx(2.0)
    assert values["heated_split_shot"] == pytest.approx(1.0)
    assert values["ogcd_wait"] == pytest.approx(1.0)


def test_training_collator_truncates_only_early_history_and_preserves_scene(tmp_path):
    torch = pytest.importorskip("torch")
    pt_path = _make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="history_demo")
    sample = dict(_make_dataset([pt_path])[1])
    state_width = sample["history_bank_state_vectors"].shape[-1]
    feature_width = sample["history_bank_skill_features"].shape[-1]
    sample["history_action_keys"] = ["old", "middle", "latest"]
    sample["history_end"] = 4
    sample["history_length"] = 3
    sample["history_bank_action_keys"] = ("", "old", "middle", "latest")
    sample["history_bank_skill_ids"] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    sample["history_bank_skill_features"] = torch.zeros((4, feature_width))
    sample["history_bank_state_vectors"] = torch.zeros((4, state_width))
    sample["history_bank_state_vectors"][:, 0] = torch.tensor([0.0, 10.0, 20.0, 30.0])
    sample["history_bank_state_null_mask"] = torch.zeros((4, state_width), dtype=torch.bool)
    scene_before = sample["scene_vectors"].clone()
    decision_before = {
        key: sample[key].clone()
        for key in (
            "current_state_vectors",
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
    assert batch["history_bank_state_vectors"][batch["history_ends"][0] - 1, 0].item() == 30.0
    assert torch.equal(batch["scene_vectors"][0, : scene_before.shape[0]], scene_before)
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
        for name in ("action_values", "action_legal_mask", "current_state_vectors", "current_state_null_mask"):
            torch.testing.assert_close(batch[name][index], sample[name])


def test_repetition_whitelist_only_penalizes_non_whitelisted_repeat():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 4))
    batch = {
        "action_legal_mask": torch.ones((1, 4), dtype=torch.bool),
        "history_action_keys": [["ogcd_wait", "fire_iii"]],
        "action_keys": [["fire_iii", "fire_iv", "xenoglossy", "flare"]],
    }

    adjusted = apply_repetition_penalty(
        logits,
        batch,
        RepetitionConfig(
            mode="whitelist",
            skills=("fire_iv", "xenoglossy", "flare"),
            penalty=1.0,
        ),
    )

    assert adjusted.tolist() == [[-1.0, 0.0, 0.0, 0.0]]


def test_repetition_blacklist_only_penalizes_listed_repeat():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 2))
    batch = {
        "action_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "history_action_keys": [["fire_iv"]],
        "action_keys": [["fire_iv", "fire_iii"]],
    }

    adjusted = apply_repetition_penalty(
        logits,
        batch,
        RepetitionConfig(mode="blacklist", skills=("fire_iv",), penalty=0.75),
    )

    assert adjusted.tolist() == [[-0.75, 0.0]]


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


def _obsolete_job_model_config_loads_bounded_sample_cache(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  sample_cache_size: 64\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.sample_cache_size == 64


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


def test_job_model_config_rejects_legacy_compiled_cache_workers(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  compiled_cache_workers: 4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CONVERT_FFLOGS_WORKERS"):
        load_run_config(config_path)


def test_training_cache_dir_is_scoped_by_job(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAINING_CACHE_ROOT", str(tmp_path))

    assert resolve_policy_cache_dir("black_mage") == tmp_path / "black_mage" / ".cache"

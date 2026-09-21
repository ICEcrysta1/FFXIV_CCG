"""纯 Tensor ONNX policy adapter 测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.onnx_export import CapacityContract, OnnxPolicy
from scripts.onnx_export.policy.policy import stable_masked_softmax
from scripts.onnx_export.runtime.validation import validate_pytorch_matrix
from common.policy.config import ModelConfig
from common.policy.data import DataSpec
from common.policy.model import (
    CandidateTransformerModel,
    RepetitionConfig,
)


TENSOR_INPUT_KEYS = (
    "scene_vectors",
    "scene_types",
    "scene_mask",
    "history_skill_ids",
    "history_skill_features",
    "history_state_vectors",
    "history_state_null_mask",
    "history_mask",
    "candidate_skill_ids",
    "candidate_skill_features",
    "candidate_state_vectors",
    "candidate_state_null_mask",
)


def _make_model(
    *,
    repetition: RepetitionConfig | None = None,
    full_attention_residuals: bool = False,
    activation: str = "gelu",
):
    data_spec = DataSpec(
        job_tag="black_mage",
        num_candidates=3,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=2,
        num_scene_types=4,
        candidate_action_keys=("fire_iii", "fire_iv", "blizzard_iii"),
        skill_feature_names=("potency", "cast_time.seconds"),
    )
    return CandidateTransformerModel(
        data_spec,
        ModelConfig(
            d_model=16,
            pair_embedding_dim=8,
            n_layers=2,
            n_heads=2,
            ff_dim=32,
            dropout=0.0,
            full_attention_residuals=full_attention_residuals,
            transformer_activation=activation,
        ),
        vocab_size=8,
        repetition=repetition,
    ).eval()


def _make_batch(
    scene_types: tuple[int, ...] = (3, 0, 2, 1),
    *,
    history_length: int = 2,
):
    scene_length = len(scene_types)
    return {
        "scene_vectors": torch.arange(
            scene_length * 2,
            dtype=torch.float32,
        ).reshape(1, scene_length, 2),
        "scene_types": torch.tensor([scene_types], dtype=torch.long),
        "scene_mask": torch.ones((1, scene_length), dtype=torch.bool),
        "history_skill_ids": torch.ones((1, history_length), dtype=torch.long),
        "history_skill_features": torch.arange(
            history_length * 2,
            dtype=torch.float32,
        ).reshape(1, history_length, 2),
        "history_state_vectors": torch.arange(
            history_length * 3,
            dtype=torch.float32,
        ).reshape(1, history_length, 3),
        "history_state_null_mask": torch.zeros(
            (1, history_length, 3),
            dtype=torch.bool,
        ),
        "history_mask": torch.ones((1, history_length), dtype=torch.bool),
        "candidate_skill_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "candidate_skill_features": torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
        ),
        "candidate_state_vectors": torch.tensor(
            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]],
        ),
        "candidate_state_null_mask": torch.zeros((1, 3, 3), dtype=torch.bool),
        "candidate_legal_mask": torch.ones((1, 3), dtype=torch.bool),
    }


def _tensor_args(batch):
    return tuple(batch[key] for key in TENSOR_INPUT_KEYS)


def _right_pad_context(
    batch: dict[str, torch.Tensor],
    *,
    scene_capacity: int,
    history_capacity: int,
    fill_value: float = 0.0,
) -> dict[str, torch.Tensor]:
    padded = {key: value.clone() for key, value in batch.items()}
    scene_length = batch["scene_vectors"].shape[1]
    history_length = batch["history_skill_ids"].shape[1]
    assert scene_capacity >= scene_length
    assert history_capacity >= history_length

    scene_padding = scene_capacity - scene_length
    history_padding = history_capacity - history_length
    padded["scene_vectors"] = torch.nn.functional.pad(
        batch["scene_vectors"],
        (0, 0, 0, scene_padding),
        value=fill_value,
    )
    padded["scene_types"] = torch.nn.functional.pad(
        batch["scene_types"],
        (0, scene_padding),
    )
    padded["scene_mask"] = torch.nn.functional.pad(
        batch["scene_mask"],
        (0, scene_padding),
        value=False,
    )
    padded["history_skill_ids"] = torch.nn.functional.pad(
        batch["history_skill_ids"],
        (0, history_padding),
    )
    padded["history_skill_features"] = torch.nn.functional.pad(
        batch["history_skill_features"],
        (0, 0, 0, history_padding),
        value=fill_value,
    )
    padded["history_state_vectors"] = torch.nn.functional.pad(
        batch["history_state_vectors"],
        (0, 0, 0, history_padding),
        value=fill_value,
    )
    padded["history_state_null_mask"] = torch.nn.functional.pad(
        batch["history_state_null_mask"],
        (0, 0, 0, history_padding),
        value=True,
    )
    padded["history_mask"] = torch.nn.functional.pad(
        batch["history_mask"],
        (0, history_padding),
        value=False,
    )
    return padded


@pytest.mark.parametrize("scene_types", ((3, 0, 2, 1), (2, 0, 2)))
def test_scene_projection_selects_each_type_without_data_dependent_branch(scene_types):
    model = _make_model()
    batch = _make_batch(scene_types)
    captured = {}

    def capture(_module, inputs, _output):
        captured["content"] = inputs[0].detach()

    handle = model.input_encoder.token_embedding.register_forward_hook(capture)
    try:
        model.input_encoder(batch)
    finally:
        handle.remove()

    expected = torch.stack(
        [
            model.input_encoder.scene_proj[scene_type](batch["scene_vectors"][:, index])
            for index, scene_type in enumerate(scene_types)
        ],
        dim=1,
    )
    torch.testing.assert_close(captured["content"][:, : len(scene_types)], expected)


@pytest.mark.parametrize("full_attention_residuals", (False, True))
@pytest.mark.parametrize("activation", ("gelu", "swiglu"))
def test_onnx_policy_matches_raw_model_logits_and_ignores_string_policy_metadata(
    full_attention_residuals: bool,
    activation: str,
):
    repetition = RepetitionConfig(
        mode="blacklist",
        skills=("fire_iii",),
        penalty=2.0,
    )
    policy_model = _make_model(
        activation=activation,
        repetition=repetition,
        full_attention_residuals=full_attention_residuals,
    )
    raw_model = _make_model(full_attention_residuals=full_attention_residuals, activation=activation)
    raw_model.load_state_dict(policy_model.state_dict())
    batch = _make_batch()
    batch["history_action_keys"] = [["fire_iii"]]
    batch["candidate_action_keys"] = [["fire_iii", "fire_iv", "blizzard_iii"]]

    with torch.no_grad():
        raw_logits = raw_model(batch)["logits"]
        policy_logits = OnnxPolicy(policy_model)(*_tensor_args(batch))
        strategy_logits = policy_model(batch)["logits"]

    torch.testing.assert_close(policy_logits, raw_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(strategy_logits[:, 0], raw_logits[:, 0] - 2.0)
    torch.testing.assert_close(strategy_logits[:, 1:], raw_logits[:, 1:])


@pytest.mark.parametrize("full_attention_residuals", (False, True))
@pytest.mark.parametrize("activation", ("gelu", "swiglu"))
def test_onnx_policy_exports_with_tensor_only_user_inputs(full_attention_residuals: bool, activation: str):
    batch = _make_batch()
    policy = OnnxPolicy(
        _make_model(full_attention_residuals=full_attention_residuals, activation=activation)
    )
    inputs = _tensor_args(batch)
    exported = torch.export.export(policy, inputs)

    with torch.no_grad():
        exported_logits = exported.module()(*inputs)
        pytorch_logits = policy(*inputs)
    torch.testing.assert_close(exported_logits, pytorch_logits, rtol=1e-5, atol=1e-6)

    user_inputs = [
        spec.arg.name
        for spec in exported.graph_signature.input_specs
        if spec.kind.name == "USER_INPUT"
    ]
    assert user_inputs == list(TENSOR_INPUT_KEYS)
    assert "history_action_keys" not in exported.graph_module.code
    assert "candidate_action_keys" not in exported.graph_module.code


def test_masked_softmax_returns_zero_for_fully_blocked_rows():
    scores = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    blocked = torch.tensor([[[[True, True], [False, True]]]])

    weights = stable_masked_softmax(scores, blocked)

    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights[0, 0, 0], torch.zeros(2))
    torch.testing.assert_close(weights[0, 0, 1], torch.tensor([1.0, 0.0]))


@pytest.mark.parametrize(
    ("scene_types", "history_length", "scene_capacity", "history_capacity"),
    (
        ((), 0, 1, 1),
        ((), 2, 2, 3),
        ((2,), 0, 2, 2),
        ((3, 1), 1, 4, 3),
    ),
)
def test_padded_context_is_finite_for_empty_or_short_context(
    scene_types,
    history_length,
    scene_capacity,
    history_capacity,
):
    # 完全因果布局下 position id 按容量生成，固定容量输入与动态长度
    # 输入不再逐 token 等价；这里只验证边界上下文不会产生 NaN/Inf。
    model = _make_model()
    policy = OnnxPolicy(model)
    dynamic_batch = _make_batch(scene_types, history_length=history_length)
    padded_batch = _right_pad_context(
        dynamic_batch,
        scene_capacity=scene_capacity,
        history_capacity=history_capacity,
        fill_value=37.0,
    )

    with torch.no_grad():
        dynamic_logits = policy(*_tensor_args(dynamic_batch))
        padded_logits = policy(*_tensor_args(padded_batch))

    assert torch.isfinite(dynamic_logits).all()
    assert torch.isfinite(padded_logits).all()


def test_padding_values_do_not_affect_logits():
    policy = OnnxPolicy(_make_model())
    batch = _make_batch((), history_length=0)
    zero_padding = _right_pad_context(
        batch,
        scene_capacity=2,
        history_capacity=2,
        fill_value=0.0,
    )
    nonzero_padding = _right_pad_context(
        batch,
        scene_capacity=2,
        history_capacity=2,
        fill_value=91.0,
    )

    with torch.no_grad():
        zero_logits = policy(*_tensor_args(zero_padding))
        nonzero_logits = policy(*_tensor_args(nonzero_padding))

    torch.testing.assert_close(nonzero_logits, zero_logits, rtol=1e-5, atol=1e-6)


def test_padding_matrix_validates_hidden_attention_logits_and_argmax():
    model = _make_model()
    results = validate_pytorch_matrix(
        OnnxPolicy(model),
        model.data_spec,
        CapacityContract(
            scene_capacity=3,
            history_capacity=4,
            candidate_count=model.data_spec.num_candidates,
        ),
        vocab_size=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert {(row["scene_valid"], row["history_valid"]) for row in results} == {
        (0, 0),
        (0, 1),
        (0, 4),
        (1, 0),
        (2, 3),
        (3, 4),
    }


class _PathSeparatedScorer(torch.nn.Module):
    """为门禁回归测试提供固定的 raw logits。"""

    def forward(self, **_kwargs):
        return torch.tensor([[2.0, 1.0, 0.0]])


class _PathSeparatedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = _PathSeparatedScorer()

    def trace(self, _batch):
        return _path_separated_trace()


def _path_separated_trace():
    return SimpleNamespace(
        encoded={
            "candidate_positions": torch.tensor([0, 1, 2]),
        },
        layer_hidden=(),
        hidden=torch.zeros((1, 3, 1)),
        attentions=(),
        logits=torch.tensor([[2.0, 1.0, 0.0]]),
    )


class _PathSeparatedPolicy(torch.nn.Module):
    """模拟 trace 与正式 forward 各自稳定但 top-1 不同的两条路径。"""

    def __init__(self):
        super().__init__()
        self.model = _PathSeparatedModel()

    def trace(self, *_inputs):
        return _path_separated_trace()

    def forward(self, *_inputs):
        # 这代表正式 SDPA/ONNX 路径；它和自己的 zero-padding 结果一致，
        # 但刻意与 trace 路径的 top-1 不同，复现 BF16 近似并列场景。
        return torch.tensor([[1.0, 2.0, 0.0]])


def test_padding_matrix_does_not_compare_trace_with_forward_argmax():
    policy = _PathSeparatedPolicy()
    results = validate_pytorch_matrix(
        policy,
        _make_model().data_spec,
        CapacityContract(
            scene_capacity=1,
            history_capacity=1,
            candidate_count=3,
        ),
        vocab_size=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert len(results) == 4
    assert all(row["argmax_match"] for row in results)


def test_onnx_policy_rejects_training_mode():
    policy = OnnxPolicy(_make_model())

    with pytest.raises(ValueError, match="only supports eval mode"):
        policy.train()

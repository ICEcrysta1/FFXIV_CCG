"""纯 Tensor ONNX policy adapter 测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.onnx_export import CapacityContract, OnnxPolicy
from scripts.onnx_export.runtime.validation import _assert_padding_keys_blocked, validate_pytorch_matrix
from common.policy.config import ModelConfig
from common.policy.data import DataSpec
from common.policy.model.attention_variants import run_head_attention
from common.policy.model import (
    CausalPolicyModel,
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
    "current_state_vectors",
    "current_state_null_mask",
    "history_state_reset_mask",
    "current_state_reset_mask",
)


def _make_model(
    *,
    repetition: RepetitionConfig | None = None,
    full_attention_residuals: bool = False,
    activation: str = "gelu",
    action_to_vocab_id: tuple[int, ...] = (1, 2, 3),
    history_capacity: int = 384,
    logit_softcap: float = 15.0,
):
    data_spec = DataSpec(
        job_tag="black_mage",
        num_actions=3,
        base_state_dim=3,
        state_dim=9,
        scene_dim=2,
        skill_feature_dim=2,
        num_scene_types=4,
        action_keys=("fire_iii", "fire_iv", "blizzard_iii"),
        action_to_vocab_id=action_to_vocab_id,
        action_is_gcd=(True, True, True),
        skill_feature_names=("potency", "cast_time.seconds"),
    )
    return CausalPolicyModel(
        data_spec,
        ModelConfig(
            d_model=16,
            n_layers=2,
            n_heads=2,
            ff_dim=32,
            dropout=0.0,
            history_capacity=history_capacity,
            history_reset_keep=min(8, history_capacity),
            scene_capacity=200,
            full_attention_residuals=full_attention_residuals,
            transformer_activation=activation,
            logit_softcap=logit_softcap,
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
        "history_state_vectors": torch.cat((torch.arange(history_length * 3, dtype=torch.float32).reshape(1, history_length, 3),
                                            torch.zeros((1, history_length, 6))), dim=-1),
        "history_state_null_mask": torch.zeros(
            (1, history_length, 3),
            dtype=torch.bool,
        ),
        "history_mask": torch.ones((1, history_length), dtype=torch.bool),
        "current_state_vectors": torch.tensor([[0.1, 0.2, 0.3, 1., 0., 1., 0., 1., 0.]]),
        "current_state_null_mask": torch.zeros((1, 3), dtype=torch.bool),
        "history_state_reset_mask": torch.zeros((1, history_length, 3), dtype=torch.bool),
        "current_state_reset_mask": torch.zeros((1, 3), dtype=torch.bool),
        "action_legal_mask": torch.ones((1, 3), dtype=torch.bool),
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
    padded["history_state_reset_mask"] = torch.nn.functional.pad(
        batch["history_state_reset_mask"], (0, 0, 0, history_padding), value=False,
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
    encoded = model.input_encoder(batch)

    expected = torch.stack(
        [
            model.input_encoder.scene_proj[scene_type](batch["scene_vectors"][:, index])
            for index, scene_type in enumerate(scene_types)
        ],
        dim=1,
    )
    expected = expected + model.input_encoder.role_embed(encoded["role_ids"][:, : len(scene_types)])
    # 场景投影先与角色相加，再和其他 token 共用一次无参数 RMS 归一化。
    expected = expected / torch.sqrt(expected.square().mean(dim=-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(encoded["tokens"][:, : len(scene_types)], expected)


@pytest.mark.parametrize("full_attention_residuals", (False, True))
@pytest.mark.parametrize("logit_softcap", (15.0, 2.5))
def test_raw_head_uses_independent_action_vectors_and_configured_softcap(
    full_attention_residuals: bool,
    logit_softcap: float,
):
    model = _make_model(
        action_to_vocab_id=(3, 1, 2),
        full_attention_residuals=full_attention_residuals,
        logit_softcap=logit_softcap,
    )
    torch.testing.assert_close(
        model.output_head.weight,
        model.input_encoder.skill_embed.weight[model.action_to_vocab_id],
        rtol=0,
        atol=0,
    )
    assert model.output_head.weight.data_ptr() != model.input_encoder.skill_embed.weight.data_ptr()
    assert model.output_head.bias is None
    policy = OnnxPolicy(model)
    batch = _make_batch()
    # 使用非动作词表行作为历史输入，让动作 embedding 的修改不影响 hidden。
    batch["history_skill_ids"].fill_(7)
    trace = policy.trace(*_tensor_args(batch))
    position = len(batch["scene_types"][0]) + 2 * batch["history_skill_ids"].shape[1]
    current_hidden = trace.hidden[:, position]
    raw_logits = logit_softcap * torch.tensor([[2.0, -3.0, 0.5]])
    # 构造会触发明显压缩的输出向量，避免小 logits 掩盖缺失的 softcap。
    with torch.no_grad():
        model.output_head.weight.copy_(
            raw_logits.T * current_hidden / current_hidden.square().sum()
        )
        embedding_rows = model.input_encoder.skill_embed.weight[model.action_to_vocab_id]
        model.input_encoder.skill_embed.weight[model.action_to_vocab_id] = embedding_rows + 100.0
    expected = logit_softcap * torch.tanh(raw_logits / logit_softcap)
    updated_trace = policy.trace(*_tensor_args(batch))
    torch.testing.assert_close(updated_trace.hidden, trace.hidden, rtol=0, atol=0)
    torch.testing.assert_close(updated_trace.logits, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(policy(*_tensor_args(batch)), expected, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(updated_trace.logits, raw_logits)
    assert model.input_encoder.skill_embed.embedding_dim == model.config.d_model
    assert not hasattr(model, "output_adapter")
    assert trace.encoded["history_skill_positions"].tolist() == [[5, 7]]
    assert trace.encoded["history_state_positions"].tolist() == [[4, 6]]
    assert trace.encoded["current_state_position"] == 8
    assert trace.encoded["role_ids"].tolist() == [[0, 0, 0, 0, 1, 2, 1, 2, 1]]


@pytest.mark.parametrize(
    ("dtype", "bf16_float_compute"),
    (
        (torch.float32, False),
        (torch.float16, False),
        (torch.bfloat16, False),
        (torch.bfloat16, True),
    ),
)
def test_forward_and_trace_delegate_action_readout_to_public_model_method(
    monkeypatch,
    dtype: torch.dtype,
    bf16_float_compute: bool,
):
    model = _make_model()
    model.to(dtype=dtype)
    # 非低精度格点的 FP32 分数同时验证最终输出的目标精度舍入。
    sentinel = torch.tensor([[7.0003, -3.0003, 2.0003]], dtype=torch.float32)
    calls = []

    def compute_action_logits(current_hidden):
        calls.append(current_hidden)
        return sentinel.clone()

    monkeypatch.setattr(model, "compute_action_logits", compute_action_logits)
    policy = OnnxPolicy(model, bf16_float_compute=bf16_float_compute)
    batch = {
        key: value.to(dtype=dtype) if value.is_floating_point() else value
        for key, value in _make_batch().items()
    }
    with torch.no_grad():
        actual = policy(*_tensor_args(batch))
        traced = policy.trace(*_tensor_args(batch))

    torch.testing.assert_close(actual, sentinel.to(dtype=dtype), rtol=0, atol=0)
    torch.testing.assert_close(traced.logits, sentinel.to(dtype=dtype), rtol=0, atol=0)
    assert len(calls) == 2
    assert calls[0].shape == calls[1].shape == (1, model.config.d_model)
    assert calls[0].dtype == (torch.float32 if bf16_float_compute else dtype)
    assert calls[1].dtype == dtype
    if bf16_float_compute:
        assert policy.model.output_head.weight.dtype == torch.bfloat16
        assert policy.compute_model.output_head.weight.dtype == torch.float32


@pytest.mark.parametrize("activation", ("gelu", "swiglu"))
@pytest.mark.parametrize("full_attention_residuals", (False, True))
def test_float16_policy_and_trace_cast_after_fp32_softcap(
    activation: str,
    full_attention_residuals: bool,
):
    model = _make_model(
        activation=activation,
        full_attention_residuals=full_attention_residuals,
        logit_softcap=2.5,
    ).to(dtype=torch.float16)
    policy = OnnxPolicy(model)
    batch = {
        key: value.half() if value.is_floating_point() else value
        for key, value in _make_batch().items()
    }
    with torch.no_grad():
        actual = policy(*_tensor_args(batch))
        traced = policy.trace(*_tensor_args(batch))
        current_hidden = traced.hidden[:, traced.encoded["current_state_position"]]
        raw_logits = torch.nn.functional.linear(current_hidden, model.output_head.weight)
        expected_fp32 = 2.5 * torch.tanh(raw_logits.float() / 2.5)
        expected_forward = model(batch)["logits"]

    assert actual.dtype == traced.logits.dtype == torch.float16
    assert expected_forward.dtype == model.compute_action_logits(current_hidden).dtype == torch.float32
    assert torch.isfinite(actual).all() and torch.isfinite(traced.logits).all()
    torch.testing.assert_close(traced.logits, expected_fp32.half(), rtol=0, atol=0)
    # 正式 SDPA 与显式 mask 入口沿用项目既有 FP16 门槛。
    from scripts.onnx_export.runtime.precision import precision_tolerances

    rtol, atol = precision_tolerances(torch.float16)
    torch.testing.assert_close(actual, expected_forward.half(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("logit_softcap", (
    torch.finfo(torch.float16).tiny,
    torch.finfo(torch.float16).max,
))
def test_float16_softcap_boundaries_keep_saturated_outputs_finite(
    monkeypatch,
    device: str,
    logit_softcap: float,
):
    """FP16 正常范围端点经正式 FP32 softcap 与部署 cast 后保持有限饱和值。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model = _make_model(logit_softcap=logit_softcap).to(device=device, dtype=torch.float16)
    raw_logits = torch.tensor([[-float("inf"), 0.0, float("inf")]],
                              device=device, dtype=torch.float16)

    def saturated_output_head(current_hidden):
        assert current_hidden.dtype == torch.float16
        assert current_hidden.device == raw_logits.device
        return raw_logits.expand(current_hidden.shape[0], -1)

    # 仅模拟输出投影溢出；softcap 保留正式 compute_action_logits 实现。
    monkeypatch.setattr(model.output_head, "forward", saturated_output_head)
    policy = OnnxPolicy(model)
    batch = {
        key: value.to(device=device, dtype=torch.float16 if value.is_floating_point() else value.dtype)
        for key, value in _make_batch().items()
    }
    with torch.no_grad():
        actual = policy(*_tensor_args(batch))
        traced = policy.trace(*_tensor_args(batch))
        current_hidden = traced.hidden[:, traced.encoded["current_state_position"]]
        fp32_logits = model.compute_action_logits(current_hidden)

    assert fp32_logits.dtype == torch.float32
    expected = torch.tensor([[-logit_softcap, 0.0, logit_softcap]],
                            device=device, dtype=torch.float16)
    for logits in (actual, traced.logits):
        assert logits.dtype == torch.float16
        assert torch.isfinite(logits).all()
        assert logits[0, 0] < 0 and logits[0, 2] > 0
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("history_capacity", (300, 384))
def test_maximum_action_window_uses_two_independent_tokens_per_action(history_capacity):
    model = _make_model(history_capacity=history_capacity)
    batch = _make_batch((0,) * 200, history_length=history_capacity)
    encoded = model.input_encoder(batch)
    total_token_count = 200 + 2 * history_capacity + 1
    assert model.input_encoder.max_token_count == total_token_count
    assert encoded["tokens"].shape == (1, total_token_count, model.config.d_model)
    assert encoded["current_state_position"] == total_token_count - 1
    assert encoded["history_skill_positions"][0, -1].item() == total_token_count - 2
    assert encoded["history_state_positions"][0, -1].item() == total_token_count - 3
    assert encoded["position_ids"].tolist() == [list(range(total_token_count))]
    with torch.no_grad():
        assert torch.isfinite(OnnxPolicy(model)(*_tensor_args(batch))).all()


@pytest.mark.parametrize("padding_key", (3, 4))
def test_padding_validation_checks_both_skill_and_state_keys(padding_key):
    # scene=1、历史容量=2、有效历史=1；无效状态和技能位于物理列 3/4。
    attention = torch.zeros((1, 1, 6, 6))
    attention[0, 0, 5, padding_key] = 1.0
    trace = SimpleNamespace(hidden=torch.zeros((1, 6, 2)), attentions=(attention,))
    with pytest.raises(AssertionError, match="padding attention keys"):
        _assert_padding_keys_blocked(trace, scene_valid=1, history_valid=1, contract=CapacityContract(1, 2))


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
    batch["action_keys"] = [["fire_iii", "fire_iv", "blizzard_iii"]]

    with torch.no_grad():
        raw_logits = raw_model(batch)["logits"]
        policy_logits = OnnxPolicy(policy_model)(*_tensor_args(batch))
        strategy_logits = policy_model(batch)["logits"]

    torch.testing.assert_close(policy_logits, raw_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(strategy_logits[:, 0], raw_logits[:, 0] - 2.0)
    torch.testing.assert_close(strategy_logits[:, 1:], raw_logits[:, 1:])


@pytest.mark.parametrize("full_attention_residuals", (False, True))
@pytest.mark.parametrize("activation", ("gelu", "swiglu"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_onnx_policy_exports_with_tensor_only_user_inputs(
    full_attention_residuals: bool,
    activation: str,
    dtype: torch.dtype,
):
    batch = {
        key: value.to(dtype=dtype) if value.is_floating_point() else value
        for key, value in _make_batch().items()
    }
    policy = OnnxPolicy(
        _make_model(full_attention_residuals=full_attention_residuals, activation=activation).to(dtype=dtype)
    )
    inputs = _tensor_args(batch)
    exported = torch.export.export(policy, inputs)

    with torch.no_grad():
        exported_logits = exported.module()(*inputs)
        pytorch_logits = policy(*inputs)
    assert exported_logits.dtype == pytorch_logits.dtype == dtype
    torch.testing.assert_close(exported_logits, pytorch_logits, rtol=1e-5, atol=1e-6)

    user_inputs = [
        spec.arg.name
        for spec in exported.graph_signature.input_specs
        if spec.kind.name == "USER_INPUT"
    ]
    assert user_inputs == list(TENSOR_INPUT_KEYS)
    assert "history_action_keys" not in exported.graph_module.code
    assert "action_keys" not in exported.graph_module.code
    assert any(node.target == torch.ops.aten.tanh.default for node in exported.graph.nodes)
    parameter_targets = {
        spec.target
        for spec in exported.graph_signature.input_specs
        if spec.kind.name == "PARAMETER"
    }
    assert "model.output_head.weight" in parameter_targets
    torch.testing.assert_close(
        exported.state_dict["model.output_head.weight"],
        policy.model.output_head.weight,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("collect_attention", [False, True])
def test_policy_attention_returns_zero_for_fully_blocked_rows(collect_attention):
    # 直接覆盖正式编码器的 SDPA/trace 路径，与 ONNX 共用显式 mask。
    layer = _make_model().eval().encoder.layers[0]
    attention = layer.self_attn
    query = torch.ones((2, attention.num_heads, 1, attention.head_dim))
    key = torch.ones((2, attention.num_kv_heads, 2, attention.head_dim))
    value = torch.ones_like(key)
    output, weights = run_head_attention(
        layer, query, key, value,
        key_valid=torch.tensor([[False, False], [True, False]]),
        causal=False,
        collect_attention=collect_attention,
        force_explicit_mask=True,
    )

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[0], torch.zeros_like(output[0]))
    if collect_attention:
        assert torch.isfinite(weights).all()
        torch.testing.assert_close(weights[0], torch.zeros_like(weights[0]))
        torch.testing.assert_close(weights[1, :, 0, 0], torch.ones(attention.num_heads))
        torch.testing.assert_close(weights[1, :, 0, 1], torch.zeros(attention.num_heads))
    else:
        assert weights is None


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
    # 每个有效 token 的逻辑位置相同，padding 不改变当前状态输出。
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
    torch.testing.assert_close(padded_logits, dynamic_logits, rtol=1e-5, atol=1e-6)


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


class _PathSeparatedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_head = torch.nn.Linear(1, 3, bias=False)
        with torch.no_grad():
            self.output_head.weight.copy_(torch.tensor([[2.0], [1.0], [0.0]]))

    def compute_action_logits(self, current_hidden):
        return self.output_head(current_hidden)

    def trace(self, _batch):
        return _path_separated_trace()


def _path_separated_trace():
    return SimpleNamespace(
        encoded={
            "current_state_position": 3,
        },
        layer_hidden=(),
        hidden=torch.ones((1, 4, 1)),
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

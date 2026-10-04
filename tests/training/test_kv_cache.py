"""模型推理 KV-Cache 测试。"""

from __future__ import annotations

import pytest
import torch

from common.policy.config import ModelConfig
from common.policy.data import DataSpec
from common.policy.model import CausalPolicyModel


def _make_model(
    *,
    norm_first: bool = True,
    full_attention_residuals: bool = False,
    activation: str = "gelu",
) -> CausalPolicyModel:
    data_spec = DataSpec(
        job_tag="black_mage",
        num_actions=3,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=2,
        num_scene_types=1,
        action_to_vocab_id=tuple(range(1, (3) + 1)), action_keys=("fire_iii", "fire_iv", "blizzard_iii"),
        action_is_gcd=(True, True, True),
        skill_feature_names=("potency", "cast_time.seconds"),
    )
    model = CausalPolicyModel(
        data_spec,
        ModelConfig(
            d_model=16,
            n_layers=2,
            n_heads=2,
            ff_dim=32,
            dropout=0.0,
            transformer_norm_first=norm_first,
            transformer_activation=activation,
            full_attention_residuals=full_attention_residuals,
        ),
        vocab_size=8,
    )
    return model.eval()


def _make_model_pair(
    *,
    norm_first: bool = True,
    full_attention_residuals: bool = False,
    activation: str = "gelu",
):
    cached_model = _make_model(
        activation=activation,
        norm_first=norm_first,
        full_attention_residuals=full_attention_residuals,
    )
    full_model = _make_model(
        activation=activation,
        norm_first=norm_first,
        full_attention_residuals=full_attention_residuals,
    )
    full_model.load_state_dict(cached_model.state_dict())
    return cached_model, full_model


def _make_batch(history_length: int, *, current_state_offset: float = 0.0, changed_history: bool = False):
    history_features = torch.arange(history_length * 2, dtype=torch.float32).reshape(
        1, history_length, 2
    )
    history_states = torch.arange(history_length * 3, dtype=torch.float32).reshape(
        1, history_length, 3
    )
    if changed_history and history_length:
        history_features[:, 0, 0] += 100.0
    current_state = torch.tensor([[0.1 + current_state_offset, 0.2, 0.3]])
    return {
        "history_skill_ids": torch.ones((1, history_length), dtype=torch.long),
        "history_skill_features": history_features,
        "history_state_vectors": history_states,
        "history_state_null_mask": torch.zeros(
            (1, history_length, 3), dtype=torch.bool
        ),
        "history_mask": torch.ones((1, history_length), dtype=torch.bool),
        "current_state_vectors": current_state,
        "current_state_null_mask": torch.zeros((1, 3), dtype=torch.bool),
        "action_legal_mask": torch.ones((1, 3), dtype=torch.bool),
        "scene_vectors": torch.tensor([[[0.25, 0.5], [0.75, 1.0]]]),
        "scene_types": torch.zeros((1, 2), dtype=torch.long),
        "scene_mask": torch.ones((1, 2), dtype=torch.bool),
    }


@pytest.mark.parametrize("norm_first", (True, False))
@pytest.mark.parametrize("activation", ("gelu", "swiglu"))
def test_kv_cache_matches_full_forward_when_history_appends(norm_first: bool, activation: str):
    model, full_model = _make_model_pair(norm_first=norm_first, activation=activation)
    model.enable_kv_cache(True)

    for history_length in (0, 1, 2):
        batch = _make_batch(history_length)
        full_logits = full_model(batch)["logits"]
        cached_output = model(batch)

        assert all(
            key.shape[1] == model.config.num_kv_heads
            and value.shape[1] == model.config.num_kv_heads
            for key, value in zip(
                model._kv_cache.key_cache,
                model._kv_cache.value_cache,
            )
        )

        torch.testing.assert_close(
            cached_output["logits"], full_logits, rtol=1e-5, atol=1e-6
        )

    assert model._kv_cache.prefix_tokens.shape[1] == 6
    assert model._kv_cache.key_cache[0].shape[2] == 6
    assert model._kv_cache.history_length == 2
    assert model._kv_cache.history_token_length == 4
    assert not hasattr(model._kv_cache, "layer_outputs")


@pytest.mark.parametrize("full_attention_residuals", (False, True))
def test_kv_cache_keeps_frozen_request_state_before_the_appended_skill(full_attention_residuals):
    """上次最新状态进入历史后仍是同一 token，后续技能不能反向改变该状态。"""
    torch.manual_seed(914)
    model, full_model = _make_model_pair(full_attention_residuals=full_attention_residuals)
    model.enable_kv_cache(True)
    previous = _make_batch(0)
    previous["current_state_null_mask"][0, 1] = True
    previous_trace = full_model.trace(previous)
    model(previous)

    following = _make_batch(1, current_state_offset=10.0)
    following["history_state_vectors"][:, 0] = previous["current_state_vectors"]
    following["history_state_null_mask"][:, 0] = previous["current_state_null_mask"]
    following_trace = full_model.trace(following)
    state_position = following_trace.encoded["history_state_positions"][0, 0].item()
    skill_position = following_trace.encoded["history_skill_positions"][0, 0].item()
    assert state_position == previous_trace.encoded["current_state_position"]
    assert skill_position == state_position + 1
    torch.testing.assert_close(
        following_trace.encoded["tokens"][:, state_position],
        previous_trace.encoded["tokens"][:, state_position], atol=0, rtol=0,
    )
    torch.testing.assert_close(
        following_trace.hidden[:, state_position],
        previous_trace.hidden[:, state_position], rtol=1e-5, atol=1e-6,
    )
    torch.testing.assert_close(
        model(following)["logits"], full_model(following)["logits"], rtol=1e-5, atol=1e-6,
    )
    cache = model._kv_cache

    # 新请求和缺失值标记只更新末尾最新状态，不回填已冻结的历史状态。
    following["current_state_vectors"][0, 1] += 20.0
    following["current_state_null_mask"][0, 2] = True
    torch.testing.assert_close(
        model(following)["logits"], full_model(following)["logits"], rtol=1e-5, atol=1e-6,
    )
    assert model._kv_cache is cache
    torch.testing.assert_close(
        cache.prefix_tokens[:, state_position], previous_trace.encoded["tokens"][:, state_position],
        atol=0, rtol=0,
    )


def test_kv_cache_recomputes_current_state_without_rebuilding_prefix():
    model, full_model = _make_model_pair()
    prefix_batch = _make_batch(2)
    model.enable_kv_cache(True)
    model(prefix_batch)
    original_cache = model._kv_cache
    changed_current = _make_batch(2, current_state_offset=10.0)
    changed_current["current_state_null_mask"][0, 1] = True

    full_logits = full_model(changed_current)["logits"]
    cached_output = model(changed_current)

    torch.testing.assert_close(
        cached_output["logits"], full_logits, rtol=1e-5, atol=1e-6
    )
    assert model._kv_cache is original_cache
    assert model._kv_cache.prefix_tokens.shape[1] == 6


@pytest.mark.parametrize("full_attention_residuals", (False, True))
@pytest.mark.parametrize(
    "changed_field",
    (
        "history_state_vectors",
        "history_state_null_mask",
        "history_mask",
        "scene_vectors",
        "scene_mask",
        "sliding_history",
    ),
)
def test_kv_cache_rebuilds_for_state_masks_and_sliding_window(
    changed_field: str, full_attention_residuals: bool,
):
    """历史状态、逻辑位置或固定窗口改变时，不沿用旧前缀。"""
    model, full_model = _make_model_pair(
        full_attention_residuals=full_attention_residuals,
    )
    model.enable_kv_cache(True)
    model(_make_batch(2))
    original_cache = model._kv_cache
    changed_batch = _make_batch(2)
    if changed_field == "sliding_history":
        for key in ("history_skill_features", "history_state_vectors"):
            changed_batch[key][:, 0] = changed_batch[key][:, 1].clone()
            changed_batch[key][:, 1] += 10.0
    elif changed_field in ("history_state_vectors", "scene_vectors"):
        changed_batch[changed_field][0, 0, 0] += 10.0
    elif changed_field == "history_state_null_mask":
        changed_batch[changed_field][0, 0, 0] = True
    else:
        changed_batch[changed_field][0, 0] = False

    full_logits = full_model(changed_batch)["logits"]
    cached_output = model(changed_batch)

    torch.testing.assert_close(
        cached_output["logits"], full_logits, rtol=1e-5, atol=1e-6,
    )
    assert model._kv_cache is not original_cache
    assert model._kv_cache.history_length == 2
    assert model._kv_cache.history_token_length == 4
    encoded = model.input_encoder(changed_batch)
    torch.testing.assert_close(model._kv_cache.prefix_valid, encoded["prefix_valid"])
    torch.testing.assert_close(
        model._kv_cache.prefix_position_ids, encoded["position_ids"][:, :6],
    )


@pytest.mark.parametrize("full_attention_residuals", (False, True))
def test_kv_cache_rebuilds_when_history_is_not_an_append(
    full_attention_residuals: bool,
):
    model, full_model = _make_model_pair(
        full_attention_residuals=full_attention_residuals,
    )
    model.enable_kv_cache(True)
    model(_make_batch(2))
    changed_history = _make_batch(2, changed_history=True)

    full_logits = full_model(changed_history)["logits"]
    cached_output = model(changed_history)

    torch.testing.assert_close(
        cached_output["logits"], full_logits, rtol=1e-5, atol=1e-6
    )
    assert model._kv_cache.prefix_tokens.shape[1] == 6


@pytest.mark.parametrize("activation", ("gelu", "swiglu"))
def test_full_attention_residual_kv_cache_matches_full_forward(activation):
    model, full_model = _make_model_pair(full_attention_residuals=True, activation=activation)
    model.enable_kv_cache(True)

    for history_length in (0, 1, 2):
        batch = _make_batch(history_length)
        full_logits = full_model(batch)["logits"]
        cached_logits = model(batch)["logits"]
        torch.testing.assert_close(cached_logits, full_logits, rtol=1e-5, atol=1e-6)

    trace = full_model.trace(_make_batch(2))
    assert len(trace.layer_hidden) == 2
    assert len(trace.attentions) == 2

    changed_current = _make_batch(2, current_state_offset=10.0)
    full_logits = full_model(changed_current)["logits"]
    cached_logits = model(changed_current)["logits"]
    torch.testing.assert_close(cached_logits, full_logits, rtol=1e-5, atol=1e-6)


def test_kv_cache_is_eval_only_and_does_not_add_checkpoint_parameters():
    model = _make_model()
    state_keys = set(model.state_dict())
    model.train()
    model.enable_kv_cache(True)
    model(_make_batch(0))
    assert model._kv_cache is None
    assert set(model.state_dict()) == state_keys


def test_kv_cache_attention_routes_through_sdpa(monkeypatch):
    """KV-cache 前缀追加与最新状态注意力经过标准 SDPA 入口。"""
    calls: list[tuple[tuple, dict]] = []
    original = torch.nn.functional.scaled_dot_product_attention

    def counting(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional,
        "scaled_dot_product_attention",
        counting,
    )
    model, full_model = _make_model_pair()
    model.enable_kv_cache(True)
    batch = _make_batch(2)

    with torch.no_grad():
        cached_output = model(batch)
        full_logits = full_model(batch)["logits"]

    assert calls
    torch.testing.assert_close(
        cached_output["logits"], full_logits, rtol=1e-5, atol=1e-6
    )


def test_kv_cache_matches_under_explicit_sdpa_backends():
    """CUDA 上限定 memory-efficient/flash backend 时 KV-cache 数值一致。"""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("SDPA backend selection requires CUDA")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    model, full_model = _make_model_pair()
    model.enable_kv_cache(True)
    model.to("cuda")
    full_model.to("cuda")
    batch = {
        key: value.to("cuda") if isinstance(value, torch.Tensor) else value
        for key, value in _make_batch(2).items()
    }

    with torch.no_grad():
        full_logits = full_model(batch)["logits"]
        default_logits = model(batch)["logits"]
        model.reset_kv_cache()
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            efficient_logits = model(batch)["logits"]
        model.reset_kv_cache()
        try:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                flash_logits = model(batch)["logits"]
        except RuntimeError:
            flash_logits = None
        cudnn_backend = getattr(SDPBackend, "CUDNN_ATTENTION", None)
        cudnn_logits = None
        if cudnn_backend is not None:
            model.reset_kv_cache()
            try:
                with sdpa_kernel(cudnn_backend):
                    cudnn_logits = model(batch)["logits"]
            except RuntimeError:
                cudnn_logits = None

    # 跨 SDPA backend（MHA 内部调度 vs KV-cache 手写入口）存在 kernel 级
    # 浮点差异，这里验证的是 backend 可用且数值合理，而非逐位一致。
    torch.testing.assert_close(
        default_logits, full_logits, rtol=1e-3, atol=1e-4
    )
    torch.testing.assert_close(
        efficient_logits, full_logits, rtol=1e-3, atol=1e-4
    )
    if flash_logits is not None:
        torch.testing.assert_close(flash_logits, full_logits, rtol=1e-3, atol=1e-4)
    if cudnn_logits is not None:
        torch.testing.assert_close(cudnn_logits, full_logits, rtol=1e-3, atol=1e-4)


def test_causal_attention_zeroes_fully_blocked_rows():
    """全屏蔽 query 行输出必须为 0，不能是 NaN。"""
    from common.policy.model.causal_encoder import run_head_attention, split_heads

    model = _make_model()
    layer = model.encoder.layers[0]
    attention = layer.self_attn
    d_model = model.config.d_model
    query = torch.randn(1, 3, d_model)
    current_key = torch.randn(1, 3, d_model)
    current_value = torch.randn(1, 3, d_model)
    prefix_key = split_heads(
        torch.randn(1, 2, d_model),
        attention.num_heads,
    )
    prefix_value = split_heads(
        torch.randn(1, 2, d_model),
        attention.num_heads,
    )
    # 前缀与当前块全部无效：每个 query 行都被完全屏蔽。
    fully_blocked_output, _ = run_head_attention(
        layer,
        split_heads(query, attention.num_heads),
        torch.cat((prefix_key, split_heads(current_key, attention.num_heads)), dim=2),
        torch.cat((prefix_value, split_heads(current_value, attention.num_heads)), dim=2),
        key_valid=torch.zeros((1, 5), dtype=torch.bool),
        causal=False,
        collect_attention=False,
    )
    assert torch.isfinite(fully_blocked_output).all()
    torch.testing.assert_close(
        fully_blocked_output,
        torch.zeros_like(fully_blocked_output),
    )
    # 全有效对照：输出有限且非零，路径本身正常。
    valid_output, _ = run_head_attention(
        layer,
        split_heads(query, attention.num_heads),
        torch.cat((prefix_key, split_heads(current_key, attention.num_heads)), dim=2),
        torch.cat((prefix_value, split_heads(current_value, attention.num_heads)), dim=2),
        key_valid=torch.ones((1, 5), dtype=torch.bool),
        causal=False,
        collect_attention=False,
    )
    assert torch.isfinite(valid_output).all()
    assert bool((valid_output != 0.0).any())

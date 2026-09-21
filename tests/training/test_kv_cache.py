"""模型推理 KV-Cache 测试。"""

from __future__ import annotations

import pytest
import torch

from common.policy.config import ModelConfig
from common.policy.data import DataSpec
from common.policy.model import CandidateTransformerModel


def _make_model(
    *,
    norm_first: bool = True,
    full_attention_residuals: bool = False,
    activation: str = "gelu",
) -> CandidateTransformerModel:
    data_spec = DataSpec(
        job_tag="black_mage",
        num_candidates=3,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=2,
        num_scene_types=1,
        candidate_action_keys=("fire_iii", "fire_iv", "blizzard_iii"),
        skill_feature_names=("potency", "cast_time.seconds"),
    )
    model = CandidateTransformerModel(
        data_spec,
        ModelConfig(
            d_model=16,
            pair_embedding_dim=8,
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


def _make_batch(history_length: int, *, candidate_offset: float = 0.0, changed_history: bool = False):
    history_features = torch.arange(history_length * 2, dtype=torch.float32).reshape(
        1, history_length, 2
    )
    history_states = torch.arange(history_length * 3, dtype=torch.float32).reshape(
        1, history_length, 3
    )
    if changed_history and history_length:
        history_features[:, 0, 0] += 100.0
    candidate_features = torch.tensor(
        [[[1.0 + candidate_offset, 2.0], [3.0, 4.0 + candidate_offset], [5.0, 6.0]]]
    )
    candidate_states = torch.tensor(
        [[[0.1 + candidate_offset, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]
    )
    return {
        "history_skill_ids": torch.ones((1, history_length), dtype=torch.long),
        "history_skill_features": history_features,
        "history_state_vectors": history_states,
        "history_state_null_mask": torch.zeros(
            (1, history_length, 3), dtype=torch.bool
        ),
        "history_mask": torch.ones((1, history_length), dtype=torch.bool),
        "candidate_skill_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "candidate_skill_features": candidate_features,
        "candidate_state_vectors": candidate_states,
        "candidate_state_null_mask": torch.zeros((1, 3, 3), dtype=torch.bool),
        "candidate_legal_mask": torch.ones((1, 3), dtype=torch.bool),
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

    assert model._kv_cache.prefix_tokens.shape[1] == 4
    assert model._kv_cache.key_cache[0].shape[2] == 4
    assert not hasattr(model._kv_cache, "layer_outputs")


def test_kv_cache_recomputes_dynamic_candidates_without_rebuilding_prefix():
    model, full_model = _make_model_pair()
    prefix_batch = _make_batch(2)
    model.enable_kv_cache(True)
    model(prefix_batch)
    changed_candidates = _make_batch(2, candidate_offset=10.0)

    full_logits = full_model(changed_candidates)["logits"]
    cached_output = model(changed_candidates)

    torch.testing.assert_close(
        cached_output["logits"], full_logits, rtol=1e-5, atol=1e-6
    )
    assert model._kv_cache.prefix_tokens.shape[1] == 4


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
    assert model._kv_cache.prefix_tokens.shape[1] == 4


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

    changed_candidates = _make_batch(2, candidate_offset=10.0)
    full_logits = full_model(changed_candidates)["logits"]
    cached_logits = model(changed_candidates)["logits"]
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
    """KV-cache 前缀追加与候选块注意力必须经过标准 SDPA 入口。"""
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


def test_split_attention_zeroes_fully_blocked_rows():
    """全屏蔽 query 行输出必须为 0，不能是 NaN。"""
    from common.policy.model.split_encoder import run_head_attention, split_heads

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

"""模型推理 KV-Cache 测试。"""

from __future__ import annotations

import pytest
import torch

from common.policy.config import ModelConfig
from common.policy.data import DataSpec
from common.policy.model import CausalPolicyModel
from tests.training._causal_fixtures import make_state_groups


def _make_model(
    *,
    norm_first: bool = True,
    full_attention_residuals: bool = False,
    activation: str = "gelu",
    n_heads: int = 2,
    num_kv_heads: int = 1,
    q_lora_rank: int = 4,
    head_dim: int | None = None,
    rope_head_dim: int | None = None,
) -> CausalPolicyModel:
    data_spec = DataSpec(
        job_tag="black_mage",
        num_actions=3,
        base_state_dim=3, state_dim=(3) + 2 * (3),
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
            n_heads=n_heads,
            num_kv_heads=num_kv_heads,
            q_lora_rank=q_lora_rank,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
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
    n_heads: int = 2,
    num_kv_heads: int = 1,
    q_lora_rank: int = 4,
    head_dim: int | None = None,
    rope_head_dim: int | None = None,
):
    cached_model = _make_model(
        activation=activation,
        norm_first=norm_first,
        full_attention_residuals=full_attention_residuals,
        n_heads=n_heads, num_kv_heads=num_kv_heads, q_lora_rank=q_lora_rank,
        head_dim=head_dim, rope_head_dim=rope_head_dim,
    )
    full_model = _make_model(
        activation=activation,
        norm_first=norm_first,
        full_attention_residuals=full_attention_residuals,
        n_heads=n_heads, num_kv_heads=num_kv_heads, q_lora_rank=q_lora_rank,
        head_dim=head_dim, rope_head_dim=rope_head_dim,
    )
    if full_attention_residuals:
        # 模拟已训练的不同深度权重，避免零 query 的均匀平均掩盖缓存聚合错误。
        residual = cached_model.encoder.attention_residual
        with torch.no_grad():
            residual.pseudo_queries.copy_(torch.linspace(
                -0.3, 0.4, residual.pseudo_queries.numel(),
            ).reshape_as(residual.pseudo_queries))
            for depth, key_norm in enumerate(residual.key_norms):
                key_norm.weight.copy_(torch.linspace(
                    0.7 + 0.02 * depth, 1.3 + 0.02 * depth,
                    key_norm.weight.numel(),
                ))
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
    history_resets = torch.zeros_like(history_states, dtype=torch.bool)
    history_resets[:, :1] = True
    return {
        "history_skill_ids": torch.ones((1, history_length), dtype=torch.long),
        "history_skill_features": history_features,
        "history_state_vectors": torch.cat((history_states, torch.zeros((1, history_length, 6))), dim=-1),
        "history_state_null_mask": torch.zeros(
            (1, history_length, 3), dtype=torch.bool
        ),
        "history_state_reset_mask": history_resets,
        "history_dead_positions": torch.empty((1, 0), dtype=torch.long),
        "history_dead_mask": torch.zeros((1, 0), dtype=torch.bool),
        "history_mask": torch.ones((1, history_length), dtype=torch.bool),
        "current_state_vectors": torch.cat((current_state, torch.zeros((1, 6))), dim=-1),
        "current_state_null_mask": torch.zeros((1, 3), dtype=torch.bool),
        "current_state_reset_mask": torch.full_like(current_state, history_length == 0, dtype=torch.bool),
        "action_legal_mask": torch.ones((1, 3), dtype=torch.bool),
        "scene_vectors": torch.tensor([[[0.25, 0.5], [0.75, 1.0]]]),
        "scene_types": torch.zeros((1, 2), dtype=torch.long),
        "scene_mask": torch.ones((1, 2), dtype=torch.bool),
    }


@pytest.mark.parametrize("full_attention_residuals", (False, True))
def test_shared_context_reanchor_and_scene_clip_rebuild_kv(full_attention_residuals):
    """分块重置重新选择 ABS 锚点和 scene view，完整前向与 KV 前向保持一致。"""
    from common.policy.data.encoding.context_encoding import (
        ContextEncoder,
        raw_state_delta,
    )
    from common.policy.data.encoding.history_window import history_window_length
    from common.policy.data.encoding.normalization import NormalizerConfig
    from common.policy.data.encoding.normalizer import Normalizer
    from common.policy.data.schema import SceneWindowSchema, TrainingSchema

    schema = TrainingSchema(
        serialization_format="test", sample_schema_version=11, context_schema_version=14,
        scene_context_mode="absolute", skill_history_fields=("kind",),
        state_groups=make_state_groups({"player_state": (
            "previous_action_after.time_seconds", "request_state.time_seconds", "request_state.mp",
        )}, ("fire_iii", "fire_iv", "blizzard_iii")), state_snapshots=("previous_action_after", "request_state"),
        scene_windows=(SceneWindowSchema.from_feature_keys(
            context_key="targetable", scene_type_id=0,
            feature_keys=("start_offset_seconds", "end_offset_seconds", "duration_seconds"),
        ),),
    )
    spec = DataSpec(
        job_tag="black_mage", num_actions=3, base_state_dim=3, state_dim=schema.state_layout(("fire_iii", "fire_iv", "blizzard_iii")).state_dim, scene_dim=3,
        skill_feature_dim=2, num_scene_types=1,
        action_keys=("fire_iii", "fire_iv", "blizzard_iii"), action_to_vocab_id=(1, 2, 3),
        action_is_gcd=(True, True, True), skill_feature_names=("potency", "cast_time.seconds"),
    )
    config = ModelConfig(
        d_model=16, n_layers=2, n_heads=2, num_kv_heads=1, ff_dim=32,
        dropout=0.0, context_capacity=12, history_reset_keep=2,
        full_attention_residuals=full_attention_residuals,
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(738)
        cached = CausalPolicyModel(spec, config, vocab_size=8).eval()
        full = CausalPolicyModel(spec, config, vocab_size=8).eval()
        # 模拟已经学会使用 ABS 标识，避免零初始化投影掩盖标识变化。
        with torch.no_grad():
            cached.input_encoder.state_reset_proj.weight.fill_(0.031)
        full.load_state_dict(cached.state_dict())
    cached.enable_kv_cache(True)
    normalizer = Normalizer(NormalizerConfig())
    normalizer = Normalizer.from_contract({
        **normalizer.normalization_contract, "skill_charge_limits": dict.fromkeys(spec.action_keys, 1),
    })
    context = ContextEncoder(normalizer, schema, layout=schema.state_layout(spec.action_keys))
    states = torch.tensor([[1199 + 3 * i, 1200 + 3 * i, 10000 - 400 * i]
                           for i in range(9)], dtype=torch.float32)
    nulls = torch.zeros_like(states, dtype=torch.bool)
    deltas, resets = raw_state_delta(
        states, nulls,
        torch.cat((torch.zeros_like(states[:1]), states[:-1])),
        torch.cat((torch.ones_like(nulls[:1]), nulls[:-1])),
    )
    previous_cache = None
    with torch.no_grad():
        for cursor in range(9):
            length = history_window_length(
                cursor, config.context_capacity, 2, request_times=states[:cursor, 1].tolist(),
                scene_end_times=[1204.0, 1220.0, 1230.0], current_request_time=float(states[cursor, 1]),
            )
            start = cursor - length
            raw = {
                "history_state_skill_state": torch.zeros((1, length, context.layout.skill_state_dim), dtype=torch.float32),
                "current_state_skill_state": torch.zeros((1, context.layout.skill_state_dim), dtype=torch.float32),
                "history_skill_ids": torch.ones((1, length), dtype=torch.long),
                "history_skill_features": torch.zeros((1, length, 2)),
                "history_state_abs_values": states[start:cursor][None],
                "history_state_delta_values": deltas[start:cursor][None],
                "history_state_null_mask": nulls[start:cursor][None],
                "history_state_delta_reset_mask": resets[start:cursor][None],
                "history_dead_positions": torch.empty((1, 0), dtype=torch.long),
                "history_dead_mask": torch.zeros((1, 0), dtype=torch.bool),
                "history_mask": torch.ones((1, length), dtype=torch.bool),
                "current_state_abs_values": states[cursor:cursor + 1],
                "current_state_delta_values": deltas[cursor:cursor + 1],
                "current_state_null_mask": nulls[cursor:cursor + 1],
                "current_state_delta_reset_mask": resets[cursor:cursor + 1],
                "scene_abs_values": torch.tensor([[[1195, 1204, 9], [1207, 1230, 23], [1213, 1220, 7]]], dtype=torch.float32),
                "scene_types": torch.zeros((1, 3), dtype=torch.int32),
                "scene_mask": torch.ones((1, 3), dtype=torch.bool),
            }
            batch = context.encode(raw)
            torch.testing.assert_close(cached(batch)["logits"], full(batch)["logits"], rtol=2e-5, atol=2e-5)
            if cursor in (5, 8):
                assert length == 2
                assert cached._kv_cache is not previous_cache
                assert cached._kv_cache.prefix_tokens.shape[1] == 6
                torch.testing.assert_close(batch["history_state_vectors"][0, 0, 1], torch.tensor(0.0))
                assert batch["history_state_reset_mask"][0, 0].tolist() == [True, True, False]
                assert batch["scene_vectors"].shape == (1, 2, 3)
                assert batch["scene_types"].shape == (1, 2)
                assert batch["scene_mask"].tolist() == [[True, True]]
                assert cached._kv_cache.scene_length == 2
            previous_cache = cached._kv_cache


@pytest.mark.parametrize("num_kv_heads", (4, 2, 1))
@pytest.mark.parametrize("full_attention_residuals", (False, True))
@pytest.mark.parametrize("head_dim,rope_head_dim", ((4, 4), (12, 4)))
def test_qk_normalized_kv_reuses_old_keys_once_and_matches_dense_trace(
    monkeypatch, num_kv_heads, full_attention_residuals, head_dim, rope_head_dim,
):
    """仅归一化新增共享 KV；非均匀 gamma 下缓存、dense 与 trace 一致。"""
    torch.manual_seed(823)
    model, dense = _make_model_pair(
        n_heads=4, num_kv_heads=num_kv_heads, q_lora_rank=4,
        head_dim=head_dim, rope_head_dim=rope_head_dim,
        full_attention_residuals=full_attention_residuals, activation="swiglu",
    )
    with torch.no_grad():
        for layer in model.encoder.layers:
            layer.self_attn.q_norm.weight.copy_(torch.linspace(0.4, 1.7, 4))
            layer.self_attn.kv_norm.weight.copy_(torch.linspace(1.6, 0.3, head_dim))
    dense.load_state_dict(model.state_dict())
    model.enable_kv_cache(True)
    calls = []
    handles = [layer.self_attn.kv_norm.register_forward_pre_hook(
        lambda _module, args: calls.append(args[0].shape[1]),
    ) for layer in model.encoder.layers]
    previous_keys = None
    requests = ((0, 0.0, (2, 1)), (1, 0.0, (2, 1)),
                (1, 7.0, (1,)), (2, 0.0, (2, 1)))
    try:
        with torch.no_grad():
            for history_length, offset, blocks in requests:
                batch = _make_batch(history_length, current_state_offset=offset)
                calls.clear()
                cached_logits = model(batch)["logits"]
                assert calls == [length for length in blocks for _ in range(model.config.n_layers)]
                keys = model._kv_cache.key_cache
                for key, value in zip(keys, model._kv_cache.value_cache):
                    assert key is value
                    assert key.shape[1] == num_kv_heads
                    assert key.shape[-1] == head_dim
                if previous_keys is not None:
                    for old, current in zip(previous_keys, keys):
                        torch.testing.assert_close(current[:, :, :old.shape[2]], old, rtol=0, atol=0)
                previous_keys = tuple(key.clone() for key in keys)
                trace = dense.trace(batch)
                torch.testing.assert_close(cached_logits, dense(batch)["logits"],
                                           rtol=2e-6, atol=2e-6)
                torch.testing.assert_close(cached_logits, dense.score_hidden(trace.encoded, trace.hidden, batch),
                                           rtol=2e-6, atol=2e-6)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("activation", ("gelu", "swiglu"))
def test_kv_cache_matches_full_forward_when_history_appends(activation: str):
    model, full_model = _make_model_pair(activation=activation)
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


@pytest.mark.parametrize("precision", ("float32", "bf16"))
def test_kv_hyper_connections_match_dense_for_prefix_append_current_and_rebuild(precision):
    """训练后的非均匀四路混合在构建、追加、当前重算及失效重建中保持一致。"""
    model, full_model = _make_model_pair(activation="swiglu")
    with torch.no_grad():
        for index, module in enumerate(model.encoder.residual_mix):
            module.scale.copy_(torch.tensor([0.4, -0.3, 0.6]))
            module.base.add_(torch.linspace(-0.7, 0.5, 24) * (index + 1))
    full_model.load_state_dict(model.state_dict())
    model.enable_kv_cache(True)
    requests = [(0, 0.0, False), (1, 0.0, False), (1, 10.0, False),
                (2, 0.0, False), (2, 0.0, True)]
    previous_cache = None
    previous_history_length = None
    tolerance = 0.01 if precision == "bf16" else 2e-6
    for history_length, current_offset, rebuild in requests:
        batch = _make_batch(history_length, current_state_offset=current_offset)
        if rebuild:
            batch["scene_vectors"][0, 0, 0] += 7.0
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"):
            actual = model(batch)["logits"]
            expected = full_model(batch)["logits"]
            trace = full_model.trace(batch)
            traced = full_model.score_hidden(trace.encoded, trace.hidden, batch)
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(actual, traced, rtol=tolerance, atol=tolerance)
        if previous_cache is not None:
            assert (model._kv_cache is previous_cache) is (
                history_length == previous_history_length and not rebuild
            )
        previous_cache = model._kv_cache
        previous_history_length = history_length


@pytest.mark.parametrize("full_attention_residuals", (False, True))
def test_kv_cache_handles_empty_prefix_then_append_reuse_and_rebuild(full_attention_residuals):
    """空前缀可追加历史；只改当前状态复用缓存，历史改动则重建。"""
    model, full_model = _make_model_pair(
        activation="swiglu", full_attention_residuals=full_attention_residuals,
    )
    model.enable_kv_cache(True)
    previous_cache = None
    requests = ((0, 0.0, False), (1, 0.0, False),
                (1, 10.0, False), (1, 10.0, True))
    with torch.no_grad():
        for step, (history_length, current_offset, changed_history) in enumerate(requests):
            batch = _make_batch(
                history_length, current_state_offset=current_offset,
                changed_history=changed_history,
            )
            for key in ("scene_vectors", "scene_types", "scene_mask"):
                batch[key] = batch[key][:, :0]
            torch.testing.assert_close(
                model(batch)["logits"], full_model(batch)["logits"], rtol=1e-5, atol=1e-6,
            )
            cache = model._kv_cache
            assert cache.prefix_tokens.shape[1] == 2 * history_length
            assert all(key.shape[2] == 2 * history_length for key in cache.key_cache)
            if step == 2:
                assert cache is previous_cache
            else:
                assert cache is not previous_cache
            previous_cache = cache


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

    original_cache = model._kv_cache
    changed_current = _make_batch(2, current_state_offset=10.0)
    full_logits = full_model(changed_current)["logits"]
    cached_logits = model(changed_current)["logits"]
    torch.testing.assert_close(cached_logits, full_logits, rtol=1e-5, atol=1e-6)
    assert model._kv_cache is original_cache


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
    from common.policy.model.execution.causal_encoder import (
        run_head_attention,
        split_heads,
    )

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
        rotary_position_encoding=model.encoder.rotary_position_encoding,
        position_ids=torch.zeros((1, 3), dtype=torch.long),
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
        rotary_position_encoding=model.encoder.rotary_position_encoding,
        position_ids=torch.zeros((1, 3), dtype=torch.long),
    )
    assert torch.isfinite(valid_output).all()
    assert bool((valid_output != 0.0).any())

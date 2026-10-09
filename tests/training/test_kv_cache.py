"""模型推理 KV-Cache 测试。"""

from __future__ import annotations

from tests.training._causal_fixtures import make_state_groups

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
    n_heads: int = 2,
    num_kv_heads: int = 1,
    qk_norm_scale: float = 1.2,
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
            qk_norm_scale=qk_norm_scale,
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
    qk_norm_scale: float = 1.2,
):
    cached_model = _make_model(
        activation=activation,
        norm_first=norm_first,
        full_attention_residuals=full_attention_residuals,
        n_heads=n_heads, num_kv_heads=num_kv_heads, qk_norm_scale=qk_norm_scale,
    )
    full_model = _make_model(
        activation=activation,
        norm_first=norm_first,
        full_attention_residuals=full_attention_residuals,
        n_heads=n_heads, num_kv_heads=num_kv_heads, qk_norm_scale=qk_norm_scale,
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
    from common.policy.data.context_encoding import ContextEncoder, raw_state_delta
    from common.policy.data.history_window import history_window_length
    from common.policy.data.normalization import NormalizerConfig
    from common.policy.data.normalizer import Normalizer
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
        job_tag="black_mage", num_actions=3, base_state_dim=3, state_dim=(3) + 2 * (3), scene_dim=3,
        skill_feature_dim=2, num_scene_types=1,
        action_keys=("fire_iii", "fire_iv", "blizzard_iii"), action_to_vocab_id=(1, 2, 3),
        action_is_gcd=(True, True, True), skill_feature_names=("potency", "cast_time.seconds"),
    )
    config = ModelConfig(
        d_model=16, n_layers=2, n_heads=2, num_kv_heads=1, ff_dim=32,
        dropout=0.0, history_capacity=4, history_reset_keep=2, scene_capacity=3,
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
    context = ContextEncoder(Normalizer(NormalizerConfig()), schema, config, layout=schema.state_layout(spec.action_keys))
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
            length = history_window_length(cursor, 4, 2)
            start = cursor - length
            raw = {
        "history_state_skill_availability": torch.zeros((*(states[start:cursor][None]).shape[:-1], 6), dtype=torch.bool),
        "current_state_skill_availability": torch.zeros((*(states[cursor:cursor + 1]).shape[:-1], 6), dtype=torch.bool),
                "history_skill_ids": torch.ones((1, length), dtype=torch.long),
                "history_skill_features": torch.zeros((1, length, 2)),
                "history_state_abs_values": states[start:cursor][None],
                "history_state_delta_values": deltas[start:cursor][None],
                "history_state_null_mask": nulls[start:cursor][None],
                "history_state_delta_reset_mask": resets[start:cursor][None],
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
                assert cached._kv_cache.history_length == 2
                torch.testing.assert_close(batch["history_state_vectors"][0, 0, 1], torch.tensor(0.0))
                assert batch["history_state_reset_mask"][0, 0].all()
                assert not batch["scene_mask"][0, 2]
            previous_cache = cached._kv_cache


@pytest.mark.parametrize("num_kv_heads", (4, 2, 1))
@pytest.mark.parametrize("full_attention_residuals", (False, True))
def test_qk_normalized_kv_reuses_old_keys_once_and_matches_dense_trace(
    monkeypatch, num_kv_heads, full_attention_residuals,
):
    """缓存保存已归一化 K，仅归一化新增块；非默认尺度下仍与 dense/trace 一致。"""
    from common.policy.model import causal_encoder

    torch.manual_seed(823)
    scale = 1.73
    model, dense = _make_model_pair(
        n_heads=4, num_kv_heads=num_kv_heads, qk_norm_scale=scale,
        full_attention_residuals=full_attention_residuals, activation="swiglu",
    )
    model.double()
    dense.double()
    model.enable_kv_cache(True)
    calls = []
    original = causal_encoder.normalize_qk

    def observe_normalization(query, key, *, scale):
        calls.append((query.shape[2], key.shape[2]))
        normalized = original(query, key, scale=scale)
        epsilon = torch.finfo(torch.float64).eps
        for raw, actual in zip((query, key), normalized):
            expected = raw * scale / (raw.square().mean(-1, keepdim=True) + epsilon).sqrt()
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        return normalized

    monkeypatch.setattr(causal_encoder, "normalize_qk", observe_normalization)
    previous_keys = None
    requests = ((0, 0.0, (2, 1)), (1, 0.0, (2, 1)),
                (1, 7.0, (1,)), (2, 0.0, (2, 1)))
    with torch.no_grad():
        for history_length, offset, blocks in requests:
            batch = _make_batch(history_length, current_state_offset=offset)
            batch = {key: value.double() if value.is_floating_point() else value
                     for key, value in batch.items()}
            calls.clear()
            cached_logits = model(batch)["logits"]
            assert calls == [(length, length) for length in blocks
                             for _ in range(model.config.n_layers)]
            keys = model._kv_cache.key_cache
            for key in keys:
                assert key.shape[1] == num_kv_heads
                rms = key.square().mean(-1).sqrt()
                torch.testing.assert_close(rms, torch.full_like(rms, scale),
                                           rtol=1e-12, atol=1e-12)
            if previous_keys is not None:
                for old, current in zip(previous_keys, keys):
                    # 旧 K 必须逐位复用；结合新增块调用统计，禁止再次归一化前缀。
                    torch.testing.assert_close(current[:, :, :old.shape[2]], old, rtol=0, atol=0)
            previous_keys = tuple(key.clone() for key in keys)
            trace = dense.trace(batch)
            torch.testing.assert_close(cached_logits, dense(batch)["logits"],
                                       rtol=1e-11, atol=1e-11)
            torch.testing.assert_close(cached_logits, dense.score_hidden(trace.encoded, trace.hidden, batch),
                                       rtol=1e-11, atol=1e-11)


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


@pytest.mark.parametrize("norm_first", (True, False))
def test_kv_mixing_aligns_x0_for_prefix_append_current_and_rebuild(norm_first):
    """不同于默认初值的真实 mix，必须对齐当前正在编码块的原始 token。"""
    model, full_model = _make_model_pair(norm_first=norm_first, activation="swiglu")
    with torch.no_grad():
        model.encoder.residual_mix.r.copy_(torch.tensor([0.7, 1.4]))
        model.encoder.residual_mix.a.copy_(torch.tensor([0.6, -0.2]))
    full_model.load_state_dict(model.state_dict())
    model.double()
    full_model.double()
    model.enable_kv_cache(True)
    captured, calls = {}, []
    input_hook = model.input_encoder.register_forward_hook(
        lambda module, args, output: captured.update(encoded=output))
    mix_hook = model.encoder.residual_mix.register_forward_hook(
        lambda module, args, output: calls.append(args))
    requests = [(0, 0.0, False), (1, 0.0, False), (1, 10.0, False),
                (2, 0.0, False), (2, 0.0, True)]
    previous_prefix_length = 0
    try:
        for step, (history_length, current_offset, rebuild) in enumerate(requests):
            calls.clear()
            batch = _make_batch(history_length, current_state_offset=current_offset)
            batch = {key: value.double() if value.is_floating_point() else value
                     for key, value in batch.items()}
            if rebuild:
                batch["scene_vectors"][0, 0, 0] += 7.0
            actual = model(batch)["logits"]
            torch.testing.assert_close(actual, full_model(batch)["logits"], rtol=1e-12, atol=1e-12)
            encoded = captured["encoded"]
            prefix_length = encoded["prefix_length"]
            if step == 0 or rebuild:
                expected_blocks = [encoded["tokens"][:, :prefix_length]]
            elif prefix_length > previous_prefix_length:
                expected_blocks = [encoded["tokens"][:, previous_prefix_length:prefix_length]]
            else:
                expected_blocks = []
            expected_blocks.append(encoded["tokens"][:, prefix_length:])
            assert len(calls) == len(expected_blocks) * model.config.n_layers
            for block_index, expected in enumerate(expected_blocks):
                for layer_index in range(model.config.n_layers):
                    args = calls[block_index * model.config.n_layers + layer_index]
                    assert args[2] == layer_index
                    torch.testing.assert_close(args[1], expected, rtol=0, atol=0)
            previous_prefix_length = prefix_length
    finally:
        input_hook.remove()
        mix_hook.remove()


def test_kv_mixing_handles_empty_scene_and_history_prefix():
    model, full_model = _make_model_pair(activation="swiglu")
    model.enable_kv_cache(True)
    batch = _make_batch(0)
    batch["scene_vectors"] = batch["scene_vectors"][:, :0]
    batch["scene_types"] = batch["scene_types"][:, :0]
    batch["scene_mask"] = batch["scene_mask"][:, :0]
    torch.testing.assert_close(model(batch)["logits"], full_model(batch)["logits"], rtol=1e-5, atol=1e-6)
    assert model._kv_cache.prefix_tokens.shape[1] == 0
    assert all(key.shape[2] == 0 for key in model._kv_cache.key_cache)


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

"""训练热路径优化的 CPU 等价性、调用次数与同步边界回归。"""

from collections import Counter
from copy import deepcopy

import pytest
import torch

from common.torch_runtime import move_batch
from training.data.collator import TrainingCollator
from common.policy.model import repetition as repetition_module
from common.policy.model import attention_variants as attention_variants_module
from common.policy.model.attention_residual import FullAttentionResidual
from common.policy.model.position_encoding import RotaryPositionEncoding
from common.policy.model import split_encoder as split
from common.policy.model.trace import TraceableTransformerEncoderLayer
from common.training.metrics import MetricAccumulator
from training.loop import training_loop
from tests.grpo.test_grpo import _decision


def _layer(*, norm_first=True, activation="swiglu", kv_heads=1, dropout=0.0):
    return TraceableTransformerEncoderLayer(
        d_model=8, nhead=4, num_kv_heads=kv_heads, dim_feedforward=12,
        norm_first=norm_first, activation=activation, batch_first=True, dropout=dropout,
    )


def _encoded(tokens, prefix_length):
    prefix_valid = torch.ones((2, prefix_length), dtype=torch.bool)
    prefix_valid[1] = False
    return {
        "tokens": tokens,
        "prefix_length": prefix_length,
        "candidate_count": 2,
        "prefix_valid": prefix_valid,
        "candidate_valid": torch.tensor([[True, False], [False, False]]),
        "cls_valid": torch.ones((2, 1), dtype=torch.bool),
        "position_ids": torch.arange(tokens.shape[1]).expand(2, -1),
    }


def _reference_split_layer(layer, encoded, rope):
    """保留优化前的逐段投影/FFN 作为独立参照，保护 attention 分区契约。"""
    lengths = (encoded["prefix_length"], 2, 1)
    parts = encoded["tokens"].split(lengths, dim=1)
    positions = encoded["position_ids"].split(lengths, dim=1)
    projected = []
    for hidden, ids in zip(parts, positions):
        values = layer.norm1(hidden) if layer.norm_first else hidden
        query, key, value = split.project_qkv(layer.self_attn, values)
        query, key = split.rotate_qk(
            rope, split.split_heads(query, 4),
            split.split_heads(key, split.kv_head_count(layer.self_attn)), ids, ids,
        )
        projected.append((query, key, split.split_heads(value, split.kv_head_count(layer.self_attn))))
    valid = (encoded["prefix_valid"], encoded["candidate_valid"], encoded["cls_valid"])
    outputs = []
    for index, hidden in enumerate(parts):
        attended, _ = split.run_head_attention(
            layer, projected[index][0],
            torch.cat([part[1] for part in projected[:index + 1]], dim=2),
            torch.cat([part[2] for part in projected[:index + 1]], dim=2),
            key_valid=torch.cat(valid[:index + 1], dim=1),
            causal=index == 0, collect_attention=False, force_explicit_mask=True,
        )
        outputs.append(split.finish_layer(layer, hidden, attended))
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize("norm_first", (True, False))
@pytest.mark.parametrize("activation", ("gelu", "swiglu"))
@pytest.mark.parametrize("kv_heads", (1, 2, 4))
@pytest.mark.parametrize("prefix_length", (0, 3))
def test_fused_segments_match_previous_outputs_and_gradients(norm_first, activation, kv_heads, prefix_length):
    torch.manual_seed(12)
    layer = _layer(norm_first=norm_first, activation=activation, kv_heads=kv_heads).double()
    reference = deepcopy(layer)
    tokens = torch.randn(2, prefix_length + 3, 8, dtype=torch.float64, requires_grad=True)
    reference_tokens = tokens.detach().clone().requires_grad_()
    rope = RotaryPositionEncoding(2)
    encoded = _encoded(tokens, prefix_length)
    prefix, candidate, cls = tokens.split((prefix_length, 2, 1), dim=1)
    actual = torch.cat(split.run_split_layer(
        layer, prefix, candidate, cls,
        prefix_valid=encoded["prefix_valid"], candidate_valid=encoded["candidate_valid"],
        cls_valid=encoded["cls_valid"], position_ids=encoded["position_ids"],
        rotary_position_encoding=rope, force_explicit_mask=True,
    )[:3], dim=1)
    expected = _reference_split_layer(reference, _encoded(reference_tokens, prefix_length), rope)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(tokens.grad, reference_tokens.grad, atol=1e-10, rtol=1e-8)
    for actual_param, expected_param in zip(layer.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_param.grad, expected_param.grad, atol=1e-10, rtol=1e-8)


def test_shared_projections_run_once_while_sdpa_keeps_three_regions(monkeypatch):
    encoder = torch.nn.TransformerEncoder(_layer(), 1, enable_nested_tensor=False)
    encoder.rotary_position_encoding = RotaryPositionEncoding(2)
    layer = encoder.layers[0]
    counts = Counter()
    handles = []
    for name, module in (
        ("norm1", layer.norm1), ("norm2", layer.norm2),
        ("q", layer.self_attn.q_proj), ("k", layer.self_attn.k_proj),
        ("v", layer.self_attn.v_proj), ("out", layer.self_attn.out_proj),
        ("ff_up", layer.linear1), ("ff_gate", layer.gate_proj), ("ff_down", layer.linear2),
    ):
        handles.append(module.register_forward_hook(lambda _m, _i, _o, name=name: counts.update([name])))
    original_sdpa = attention_variants_module.scaled_dot_product_attention

    def counted_sdpa(*args, **kwargs):
        counts["sdpa"] += 1
        return original_sdpa(*args, **kwargs)

    monkeypatch.setattr(attention_variants_module, "scaled_dot_product_attention", counted_sdpa)
    split.run_split_encoder(encoder, _encoded(torch.randn(2, 6, 8), 3))
    for handle in handles:
        handle.remove()
    assert counts.pop("sdpa") == 3
    assert len(counts) == 9
    assert set(counts.values()) == {1}


@pytest.mark.parametrize("full_residual", (False, True))
@pytest.mark.parametrize("checkpoint_ffn,checkpoint_attention", ((True, False), (False, True), (True, True)))
def test_fused_checkpoint_preserves_dropout_outputs_and_gradients(full_residual, checkpoint_ffn, checkpoint_attention):
    torch.manual_seed(31)
    encoder = torch.nn.TransformerEncoder(_layer(dropout=0.15), 2, enable_nested_tensor=False)
    encoder.rotary_position_encoding = RotaryPositionEncoding(2)
    if full_residual:
        encoder.attention_residual = FullAttentionResidual(d_model=8, num_queries=5)
    reference = deepcopy(encoder)
    for layer in encoder.layers:
        layer.set_activation_checkpoint_ffn(checkpoint_ffn)
        layer.set_activation_checkpoint_attention(checkpoint_attention)
    tokens = torch.randn(2, 6, 8, requires_grad=True)
    reference_tokens = tokens.detach().clone().requires_grad_()
    torch.manual_seed(32)
    actual = torch.cat(split.run_split_encoder(encoder, _encoded(tokens, 3))[:3], dim=1)
    torch.manual_seed(32)
    expected = torch.cat(split.run_split_encoder(reference, _encoded(reference_tokens, 3))[:3], dim=1)
    torch.testing.assert_close(actual, expected)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(tokens.grad, reference_tokens.grad)
    for actual_param, expected_param in zip(encoder.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_param.grad, expected_param.grad)


@pytest.mark.parametrize("mode", ("whitelist", "blacklist"))
def test_prepared_repetition_uses_cached_mask_with_reordered_candidates(mode, monkeypatch):
    config = repetition_module.RepetitionConfig(mode=mode, skills=("b",), penalty=1.25)
    batch = {
        "candidate_skill_ids": torch.zeros((5, 4), dtype=torch.int64),
        "candidate_action_keys": [["a", "b", "ogcd_wait", "a"], ["b", "a", "a", "ogcd_wait"]] * 2 + [["a", "b", "ogcd_wait", "a"]],
        "history_action_keys": [["a", "ogcd_wait"], ["a"], [], ["ogcd_wait"], ["b", "ogcd_wait", "ogcd_wait"]],
    }
    prepared = repetition_module.prepare_repetition_penalty(batch, config)
    assert "repetition_penalty_mask" not in batch
    expected = (
        [[True, False, False, True], [False, True, True, False], [False] * 4, [False] * 4, [False] * 4]
        if mode == "whitelist" else
        [[False] * 4, [False] * 4, [False] * 4, [False] * 4, [False, True, False, False]]
    )
    assert prepared["repetition_penalty_mask"].tolist() == expected
    moved = move_batch(prepared, torch.device("cpu"))

    def reject_rebuild(*_args, **_kwargs):
        pytest.fail("forward must reuse the prepared repetition mask")

    monkeypatch.setattr(repetition_module, "_build_repetition_mask_cpu", reject_rebuild)
    for _ in range(2):
        logits = torch.randn(5, 4, requires_grad=True)
        actual = repetition_module.apply_repetition_penalty(logits, moved, config)
        torch.testing.assert_close(actual, logits - torch.tensor(expected) * config.penalty)
        actual.sum().backward()
        assert torch.equal(logits.grad, torch.ones_like(logits))


def test_repetition_cache_is_bounded_and_policy_changes_rebuild_mask():
    first = repetition_module.RepetitionConfig("blacklist", ("a",), 1.0)
    second = repetition_module.RepetitionConfig("blacklist", ("b",), 2.0)
    batch = {"candidate_skill_ids": torch.zeros(1, 2), "candidate_action_keys": [["a", "b"]], "history_action_keys": [["a"]]}
    prepared = repetition_module.prepare_repetition_penalty(batch, first)
    assert not repetition_module.build_repetition_penalty_mask(prepared, second, device="cpu").any()
    assert repetition_module._candidate_penalty_indices.cache_info().maxsize == 128
    # 调用方修改输入后重新准备，不能复用旧的最近动作快照。
    prepared["history_action_keys"] = [["b"]]
    refreshed = repetition_module.prepare_repetition_penalty(prepared, first)
    assert not refreshed["repetition_penalty_mask"].any()


def _sample(history_length, scene_length):
    decision = _decision(history_length=history_length, scene_length=scene_length, action_index=0)
    sample = {key: value[0] for key, value in decision.batch.items()}
    sample.update(metadata={}, label_action_key="fire_iii", label_index=0)
    sample["history_state_vectors"] = torch.arange(history_length * 3, dtype=torch.float32).reshape(history_length, 3)
    sample["history_skill_ids"] = torch.arange(history_length, dtype=torch.int32)
    sample["scene_vectors"] = torch.arange(scene_length * 2, dtype=torch.float32).reshape(scene_length, 2)
    return sample


@pytest.mark.parametrize("lengths", (((0, 0), (0, 0)), ((1, 2), (3, 0)), ((2, 1), (2, 1))))
def test_native_padding_preserves_values_dtypes_empty_shapes_and_null_mask(lengths):
    samples = [_sample(*length) for length in lengths]
    originals = deepcopy(samples)
    batch = TrainingCollator()(samples)
    for kind, key in (("history", "history_skill_ids"), ("scene", "scene_vectors")):
        sizes = [sample[key].shape[0] for sample in samples]
        expected_mask = [[index < length for index in range(max(sizes))] for length in sizes]
        assert batch[f"{kind}_mask"].tolist() == expected_mask
    for key, pad_value in (("history_skill_ids", 0), ("history_skill_features", 0), ("history_state_vectors", 0), ("history_state_null_mask", True), ("scene_vectors", 0), ("scene_types", 0)):
        assert batch[key].dtype == samples[0][key].dtype
        for index, sample in enumerate(samples):
            length = sample[key].shape[0]
            torch.testing.assert_close(batch[key][index, :length], sample[key])
            assert (batch[key][index, length:] == pad_value).all()
            assert torch.equal(sample[key], originals[index][key])


def test_compact_padding_mask_and_history_truncation_stay_aligned():
    sample = _sample(3, 0)
    sample["history_action_keys"] = ["a", "b", "ogcd_wait"]
    compact = {key: value for key, value in sample.items() if not key.startswith("history_")}
    for key in ("skill_ids", "skill_features", "state_vectors", "state_null_mask"):
        compact[f"history_bank_{key}"] = sample[f"history_{key}"]
    compact.update(history_end=3, history_length=3, history_bank_id="test", history_bank_action_keys=sample["history_action_keys"])

    class KeepTwo:
        def random(self):
            return 0.0

        def randint(self, _start, _end):
            return 2

    collator = TrainingCollator(history_truncation_enabled=True, history_truncation_probability=1.0, rng=KeepTwo())
    batch = collator([compact, dict(compact, history_end=0, history_length=0)])
    assert batch["history_lengths"].tolist() == [2, 0]
    assert batch["history_mask"].tolist() == [[True, True], [False, False]]
    assert batch["history_action_keys"] == [["b", "ogcd_wait"], []]
    assert batch["history_bank_skill_ids"].data_ptr() == compact["history_bank_skill_ids"].data_ptr()


def test_repetition_preparation_follows_history_truncation_and_candidate_shuffle():
    class Augmentation:
        def random(self):
            return 0.0

        def randint(self, _start, _end):
            return 2

        def shuffle(self, values):
            values.reverse()

    sample = _sample(3, 0)
    sample["history_action_keys"] = ["a", "b", "ogcd_wait"]
    sample["candidate_action_keys"] = ["a", "b", "ogcd_wait"]
    collator = TrainingCollator(
        history_truncation_enabled=True, history_truncation_probability=1.0,
        candidate_shuffle_enabled=True, candidate_shuffle_probability=1.0,
        rng=Augmentation(),
    )
    config = repetition_module.RepetitionConfig("blacklist", ("b",), 1.0)
    prepared = repetition_module.prepare_repetition_penalty(collator([sample]), config)
    assert prepared["history_action_keys"] == [["b", "ogcd_wait"]]
    assert prepared["candidate_action_keys"] == [["ogcd_wait", "b", "a"]]
    assert prepared["repetition_penalty_mask"].tolist() == [[False, True, False]]


def test_metric_accumulator_reads_once_and_does_not_retain_graph(monkeypatch):
    totals = MetricAccumulator(("loss", "accuracy"), device="cpu")
    calls = []
    original_cpu = torch.Tensor.cpu

    def record_cpu(value, *args, **kwargs):
        calls.append(value.shape)
        return original_cpu(value, *args, **kwargs)

    def reject_item(*_args, **_kwargs):
        pytest.fail("metric accumulation must not call item")

    monkeypatch.setattr(torch.Tensor, "cpu", record_cpu)
    monkeypatch.setattr(torch.Tensor, "item", reject_item)
    totals.update({"loss": torch.tensor(2.0, requires_grad=True), "accuracy": torch.tensor(0.5)}, weight=3)
    totals.update({"loss": torch.tensor(4.0, requires_grad=True), "accuracy": torch.tensor(1.0)}, weight=1)
    assert not calls
    assert totals.totals.grad_fn is None
    assert not totals.totals.requires_grad
    assert totals.mean() == {"loss": 2.5, "accuracy": 0.625}
    assert calls == [torch.Size([2])]
    assert MetricAccumulator(("loss",), device="cpu").mean() == {"loss": 0.0}


@pytest.mark.parametrize("train", (True, False))
def test_epoch_metrics_keep_sample_weighting_and_prepare_repetition(train, monkeypatch):
    class Policy(torch.nn.Module):
        repetition = repetition_module.RepetitionConfig("blacklist", ("a",), 1.0)

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, batch):
            assert "repetition_penalty_mask" in batch
            count = batch["label_index"].shape[0]
            return {"loss": self.weight * count, "top1_accuracy": self.weight.detach() * (count / 4), "top3_accuracy": self.weight.detach()}

    def reject_item(*_args, **_kwargs):
        pytest.fail("epoch must not read individual device scalars")

    model = Policy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    batches = [{"label_index": torch.zeros(count, dtype=torch.long), "candidate_skill_ids": torch.zeros(count, 2)} for count in (3, 1)]
    monkeypatch.setattr(torch.Tensor, "item", reject_item)
    if train:
        metrics = training_loop.train_epoch(model, batches, optimizer, scheduler, torch.device("cpu"))
    else:
        metrics = training_loop.validate(model, batches, torch.device("cpu"))
    assert metrics == {"loss": 2.5, "cross_entropy_loss": 2.5, "value_preference_loss": 0.0, "top1_accuracy": 0.625, "top3_accuracy": 1.0}

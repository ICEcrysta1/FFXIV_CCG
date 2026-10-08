"""读侧重锚、状态差分和场景差分的共同数据边界。"""

from tests.training._causal_fixtures import make_state_groups

import math
from dataclasses import asdict

import pytest
import torch

from common.policy.config import ModelConfig
from common.policy.data.context_encoding import ContextEncoder, raw_state_delta
from common.policy.data.history_window import history_window_length
from common.policy.data.normalization import NormalizerConfig
from common.policy.data.normalizer import Normalizer
from common.policy.data.schema import SceneWindowSchema, TrainingSchema


def _encoder():
    schema = TrainingSchema(
        serialization_format="test", sample_schema_version=11, context_schema_version=14,
        scene_context_mode="absolute", skill_history_fields=("kind",),
        state_groups=make_state_groups({"player": (
            "request_state.mp", "request_state.is_moving", "request_state.time_seconds",
            "request_state.cumulative_potency", "previous_action_after.time_seconds",
            "previous_action_after.mp",
        )}, ("first", "second")), state_snapshots=("previous_action_after", "request_state"),
        scene_windows=(
            SceneWindowSchema.from_feature_keys(
                context_key="combat", scene_type_id=0,
                feature_keys=("start_offset_seconds", "end_offset_seconds", "duration_seconds", "targetable"),
            ),
            # 第二类的时间字段换序，防止实现依赖三列固定下标。
            SceneWindowSchema.from_feature_keys(
                context_key="count", scene_type_id=1,
                feature_keys=("target_count", "duration_seconds", "end_offset_seconds", "start_offset_seconds"),
            ),
        ),
    )
    return ContextEncoder(Normalizer(NormalizerConfig()), schema, ModelConfig(), layout=schema.state_layout(("first", "second")))


def _raw_batch():
    states = torch.tensor([[
        [8000, 1, 1200, 1000, 1198, 9000],
        [6000, 0, 1203, 1100, 1201, 8000],
    ]], dtype=torch.float32)
    nulls = torch.zeros_like(states, dtype=torch.bool)
    previous = torch.cat((torch.zeros_like(states[:, :1]), states[:, :-1]), dim=1)
    previous_nulls = torch.cat((torch.ones_like(nulls[:, :1]), nulls[:, :-1]), dim=1)
    delta, resets = raw_state_delta(states, nulls, previous, previous_nulls)
    current = torch.tensor([[7000, 1, 1207, 1050, 1205, 6000]], dtype=torch.float32)
    current_null = torch.zeros_like(current, dtype=torch.bool)
    current_delta, current_reset = raw_state_delta(current, current_null, states[:, -1], nulls[:, -1])
    return {
        "history_state_skill_availability": torch.zeros((*(states).shape[:-1], 4), dtype=torch.bool),
        "current_state_skill_availability": torch.zeros((*(current).shape[:-1], 4), dtype=torch.bool),
        "history_skill_ids": torch.tensor([[4, 5]]),
        "history_skill_features": torch.tensor([[[0.25], [0.5]]]),
        "history_state_abs_values": states,
        "history_state_delta_values": delta,
        "history_state_null_mask": nulls,
        "history_state_delta_reset_mask": resets,
        "history_mask": torch.tensor([[True, True]]),
        "current_state_abs_values": current,
        "current_state_delta_values": current_delta,
        "current_state_null_mask": current_null,
        "current_state_delta_reset_mask": current_reset,
        "scene_abs_values": torch.tensor([[[1100, 1300, 200, 1], [3, 10, 1250, 1240],
                                         [1150, 1199, 49, 0], [1220, 1600, 380, 1]]], dtype=torch.float32),
        "scene_types": torch.tensor([[0, 1, 0, 0]]),
        "scene_mask": torch.tensor([[True, True, True, True]]),
    }


def test_state_delta_signed_values_and_field_reset():
    previous = torch.tensor([[5, 1, 2, 0]], dtype=torch.float32)
    previous_null = torch.tensor([[False, False, False, True]])
    current = torch.tensor([[3, 0, 99, 7]], dtype=torch.float32)
    nulls = torch.tensor([[False, False, True, False]])
    delta, reset = raw_state_delta(current, nulls, previous, previous_null)
    torch.testing.assert_close(delta, torch.tensor([[-2, -1, 0, 7]], dtype=torch.float32))
    assert reset.tolist() == [[False, False, False, True]]
    absolute, first_reset = raw_state_delta(current, nulls)
    torch.testing.assert_close(absolute, torch.tensor([[3, 0, 0, 7]], dtype=torch.float32))
    assert first_reset.tolist() == [[True, True, False, True]]


def test_first_anchor_and_current_delta_are_distinct():
    encoder = _encoder()
    output = encoder.encode(_raw_batch())
    expected = torch.tensor([[[0.8, 1, 0, math.log1p(1000), -2 / 120, 0.9],
                              [-0.2, -1, 3 / 120, math.log1p(100), 3 / 120, -0.1]]])
    torch.testing.assert_close(output["history_state_vectors"][..., :6], expected)
    torch.testing.assert_close(output["current_state_vectors"][..., :6], torch.tensor([
        [0.1, 1, 4 / 120, -math.log1p(50), 4 / 120, -0.2],
    ]))
    assert output["history_state_reset_mask"].tolist() == [[[True] * 6, [False] * 6]]
    assert not output["current_state_reset_mask"].any()
    assert all(value.dtype == torch.float32 for key, value in output.items() if key.endswith("vectors"))
    assert not any("abs_values" in key or "delta_values" in key for key in output)
    assert not list(encoder.parameters())
    with pytest.raises(ValueError, match="encoded twice"):
        encoder.encode(output)


def test_scene_encoding_separates_raw_and_model_fields_without_mutating_raw():
    encoder = _encoder()
    batch = _raw_batch()
    absolute = batch["scene_abs_values"].clone()
    output = encoder.encode(batch)
    assert "scene_abs_values" not in output
    assert "scene_vectors" in output
    torch.testing.assert_close(batch["scene_abs_values"], absolute, rtol=0, atol=0)
    ambiguous = {**batch, "scene_vectors": absolute}
    with pytest.raises(ValueError, match="raw states and scenes"):
        encoder.encode(ambiguous)


def test_resources_timers_and_buff_refresh_reconstruct_raw_and_keep_signed_scales():
    """资源变化、timer 回绕与 Buff 过期/刷新先无损求差，再按各字段尺度编码。"""
    keys = (
        "request_state.time_seconds", "previous_action_after.time_seconds", "request_state.mp",
        "request_state.astral_fire", "request_state.polyglot", "request_state.polyglot_timer",
        "request_state.job.triplecast.active", "request_state.job.triplecast.remaining_seconds",
        "request_state.job.triplecast.stacks", "request_state.cumulative_potency",
    )
    normalizer = Normalizer.from_contract({
        "version": 2, "config": asdict(NormalizerConfig(remaining_seconds_max=60.0)),
        "resource_limits": {"astral_fire": 3.0, "polyglot": 3.0, "polyglot_timer": 30.0},
        "status_limits": {"job.triplecast": 3.0}, "job_tag": "black_mage",
    })
    schema = TrainingSchema(
        serialization_format="test", sample_schema_version=10, context_schema_version=14,
        scene_context_mode="absolute", skill_history_fields=("kind",),
        state_groups=make_state_groups({"state": keys}, ("first", "second")), state_snapshots=("previous_action_after", "request_state"),
        scene_windows=(SceneWindowSchema.from_feature_keys(
            context_key="combat", scene_type_id=0,
            feature_keys=("start_offset_seconds", "end_offset_seconds", "duration_seconds"),
        ),),
    )
    values = torch.tensor([
        [1200, 1198, 10000, 3, 2, 29, 1, 12, 3, 50000],
        [1203, 1201, 7600, 0, 3, 2, 1, 9, 2, 50700],
        [1207, 1205, 8000, 1, 0, 6, 0, 0, 0, 50650],
        [1210, 1208, 9200, 3, 1, 1, 1, 15, 3, 51300],
    ], dtype=torch.float32)
    nulls = torch.zeros_like(values, dtype=torch.bool)
    delta, reset = raw_state_delta(
        values, nulls,
        torch.cat((torch.zeros_like(values[:1]), values[:-1])),
        torch.cat((torch.ones_like(nulls[:1]), nulls[:-1])),
    )
    # 逐字段核对 raw 重建，防止某类资源/Buff 提前 clipping、取 log 或 BF16 丢精度。
    for index, key in enumerate(keys):
        reconstructed = torch.cat((delta[:1, index],
                                   delta[:1, index] + delta[1:, index].cumsum(dim=0)))
        torch.testing.assert_close(reconstructed, values[:, index], rtol=0, atol=0, msg=key)
    assert reset[0].all()
    assert not reset[1:].any()
    output = ContextEncoder(normalizer, schema, ModelConfig(), layout=schema.state_layout(("first", "second"))).encode({
        "history_state_skill_availability": torch.zeros((*(values[:3][None]).shape[:-1], 4), dtype=torch.bool),
        "current_state_skill_availability": torch.zeros((*(values[3:]).shape[:-1], 4), dtype=torch.bool),
        "history_skill_ids": torch.tensor([[1, 2, 3]]),
        "history_skill_features": torch.zeros((1, 3, 1)),
        "history_state_abs_values": values[:3][None],
        "history_state_delta_values": delta[:3][None],
        "history_state_null_mask": nulls[:3][None],
        "history_state_delta_reset_mask": reset[:3][None],
        "history_mask": torch.ones((1, 3), dtype=torch.bool),
        "current_state_abs_values": values[3:],
        "current_state_delta_values": delta[3:],
        "current_state_null_mask": nulls[3:],
        "current_state_delta_reset_mask": reset[3:],
        "scene_abs_values": torch.zeros((1, 0, 3)),
        "scene_types": torch.zeros((1, 0), dtype=torch.long),
        "scene_mask": torch.zeros((1, 0), dtype=torch.bool),
    })
    expected = torch.tensor([
        [0, -2/120, 1, 1, 2/3, 29/30, 1, 12/60, 1, math.log1p(50000)],
        [3/120, 3/120, -.24, -1, 1/3, -27/30, 0, -3/60, -1/3, math.log1p(700)],
        [4/120, 4/120, .04, 1/3, -1, 4/30, -1, -9/60, -2/3, -math.log1p(50)],
        [3/120, 3/120, .12, 2/3, 1/3, -5/30, 1, 15/60, 1, math.log1p(650)],
    ])
    actual = torch.cat((output["history_state_vectors"][0], output["current_state_vectors"]))
    for index, key in enumerate(keys):
        torch.testing.assert_close(actual[:, index], expected[:, index], msg=key)
    assert actual.dtype == torch.float32


def test_scene_clip_sort_and_separate_signed_start_end_deltas():
    output = _encoder().encode(_raw_batch())
    assert output["scene_mask"].tolist() == [[True, True, True, False]]
    assert output["scene_types"].tolist() == [[0, 0, 1, 0]]
    torch.testing.assert_close(output["scene_vectors"], torch.tensor([[
        [0, 100 / 120, 100 / 120, 1],
        [20 / 120, 300 / 120, 380 / 120, 1],
        [1, 10 / 120, -350 / 120, 20 / 120],
        [0, 0, 0, 0],
    ]]))


def test_future_first_scene_keeps_offset_and_equal_windows_use_stable_full_order():
    """未来首窗口不归零；相同起止按类型，再按原始行号稳定排序。"""
    batch = _raw_batch()
    # 第一类第4列是原始内容标签，第二类时间列换序；标签用于观察同类同起止稳定性。
    batch["scene_abs_values"] = torch.tensor([[
        [2, 40, 1260, 1220],
        [1220, 1260, 40, .25],
        [1220, 1250, 30, .5],
        [1220, 1260, 40, .75],
        [1240, 1280, 40, 1],
    ]], dtype=torch.float32)
    batch["scene_types"] = torch.tensor([[1, 0, 0, 0, 0]])
    batch["scene_mask"] = torch.ones((1, 5), dtype=torch.bool)
    output = _encoder().encode(batch)
    assert output["scene_types"].tolist() == [[0, 0, 0, 1, 0]]
    assert output["scene_mask"].all()
    torch.testing.assert_close(output["scene_vectors"], torch.tensor([[
        [20/120, 50/120, 30/120, .5],
        [0, 10/120, 40/120, .25],
        [0, 0, 40/120, .75],
        [2/3, 40/120, 0, 0],
        [20/120, 20/120, 40/120, 1],
    ]]))


def test_window_reanchors_from_selected_absolute_row_and_preserves_later_delta():
    batch = _raw_batch()
    for key in tuple(batch):
        if key.startswith("history_") and key != "history_cursor":
            batch[key] = batch[key][:, 1:]
    output = _encoder().encode(batch)
    torch.testing.assert_close(output["history_state_vectors"][0, 0, :6], torch.tensor(
        [0.6, 0, 0, math.log1p(1100), -2 / 120, 0.8],
    ))
    assert output["history_state_reset_mask"].all()
    assert output["current_state_vectors"][0, 2] == pytest.approx(4 / 120)
    assert output["scene_vectors"][0, 0, 1] == pytest.approx(97 / 120)


def test_empty_history_anchors_current_state_and_keeps_padded_scene_width():
    batch = _raw_batch()
    for key in tuple(batch):
        if key.startswith("history_") and key != "history_cursor":
            batch[key] = batch[key][:, :0]
    batch["scene_abs_values"][:, :, 1] = 0  # 所有 combat 窗口均已结束。
    batch["scene_mask"].fill_(False)
    output = _encoder().encode(batch)
    assert output["history_state_vectors"].shape == (1, 0, 10)
    assert output["current_state_reset_mask"].all()
    torch.testing.assert_close(output["current_state_vectors"][0, :6], torch.tensor(
        [0.7, 1, 0, math.log1p(1050), -2 / 120, 0.6],
    ))
    assert output["scene_vectors"].shape == (1, 4, 4)
    assert not output["scene_mask"].any()
    assert torch.count_nonzero(output["scene_vectors"]) == 0


def test_compact_gather_matches_dense_and_does_not_cross_bank_boundaries():
    dense = _raw_batch()
    compact = dict(dense)
    names = ("skill_ids", "skill_features", "state_abs_values", "state_delta_values",
             "state_null_mask", "state_delta_reset_mask", "state_skill_availability")
    for name in names:
        rows = compact.pop("history_" + name).squeeze(0)
        sentinel = torch.zeros_like(rows[:1])
        unrelated_rows = torch.full_like(rows, 1)
        compact["history_bank_" + name] = torch.cat((sentinel, unrelated_rows, sentinel, rows), dim=0)
    compact["history_lengths"] = torch.tensor([2])
    compact["history_ends"] = torch.tensor([6])
    encoder = _encoder()
    expected, actual = encoder.encode(dense), encoder.encode(compact)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
    assert not any(key.startswith("history_bank_") for key in actual)


def test_availability_is_absolute_and_does_not_change_skill_or_scene_encoding():
    """新增状态表不进入 DELTA，原基础段和非状态 token 使用原样编码。"""
    batch = _raw_batch()
    encoder = _encoder()
    original = encoder.encode(batch)
    batch["history_state_skill_availability"] = torch.tensor([[[1, 0, 0, 1], [0, 1, 1, 0]]], dtype=torch.bool)
    batch["current_state_skill_availability"] = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    changed = encoder.encode(batch)
    assert changed["history_state_vectors"][..., 6:].tolist() == [[[1, 0, 0, 1], [0, 1, 1, 0]]]
    assert changed["current_state_vectors"][..., 6:].tolist() == [[1, 1, 0, 0]]
    for key in ("history_state_vectors", "current_state_vectors"):
        torch.testing.assert_close(changed[key][..., :6], original[key][..., :6], rtol=0, atol=0)
    for key in ("history_skill_ids", "history_skill_features", "history_mask", "scene_vectors", "scene_types", "scene_mask"):
        torch.testing.assert_close(changed[key], original[key], rtol=0, atol=0)


def test_null_to_known_time_reset_is_reanchored_and_padding_is_inert():
    batch = _raw_batch()
    batch["history_state_delta_reset_mask"][0, 1, 4] = True
    batch["history_state_delta_values"][0, 1, 4] = 1201
    batch["history_state_null_mask"][0, 1, 0] = True
    output = _encoder().encode(batch)
    assert output["history_state_vectors"][0, 1, 4] == pytest.approx(1 / 120)
    assert output["history_state_reset_mask"][0, 1, 4]
    assert output["history_state_vectors"][0, 1, 0] == -1
    assert not output["history_state_reset_mask"][0, 1, 0]
    batch["history_mask"][0, 1] = False
    output = _encoder().encode(batch)
    assert torch.count_nonzero(output["history_state_vectors"][0, 1]) == 0
    assert output["history_state_null_mask"][0, 1].all()
    assert not output["history_state_reset_mask"][0, 1].any()


@pytest.mark.parametrize("total,expected", [(0, 0), (299, 299), (300, 300), (301, 8),
                                           (302, 9), (593, 300), (594, 8), (887, 8)])
def test_history_window_resets_instead_of_sliding(total, expected):
    assert history_window_length(total, 300, 8) == expected


@pytest.mark.parametrize("args", [(True, 300, 8), (-1, 300, 8), (1, 3, 4),
                                  (1, 3, 0), (1, 0, 1), (1, 300, False), (1.0, 300, 8)])
def test_history_window_rejects_invalid_parameters(args):
    with pytest.raises(ValueError):
        history_window_length(*args)


def test_no_history_window_is_explicit():
    assert history_window_length(1000, 0, 0) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_raw_encoding_rejects_precision_loss_before_subtraction(dtype):
    batch = _raw_batch()
    batch["current_state_abs_values"] = batch["current_state_abs_values"].to(dtype)
    with pytest.raises(ValueError, match="FP32"):
        _encoder().encode(batch)


def test_raw_delta_requires_both_predecessor_fields():
    values = torch.zeros(2, 3)
    nulls = torch.zeros_like(values, dtype=torch.bool)
    with pytest.raises(ValueError, match="provided together"):
        raw_state_delta(values, nulls, values)


def test_actual_cache_int32_scene_types_are_prepared_for_neural_gather():
    batch = _raw_batch()
    encoder = _encoder()
    expected = encoder.encode(batch)
    batch["scene_types"] = batch["scene_types"].to(torch.int32)
    output = encoder.encode(batch)
    assert output["scene_types"].dtype == torch.long
    torch.testing.assert_close(output["scene_types"], expected["scene_types"])
    torch.testing.assert_close(output["scene_vectors"], expected["scene_vectors"])
    torch.testing.assert_close(output["scene_mask"], expected["scene_mask"])


@pytest.mark.parametrize("bad", ["missing", "nan", "inf"])
@pytest.mark.parametrize("has_history", [False, True])
def test_cpu_raw_entry_rejects_unknown_or_nonfinite_request_anchor(bad, has_history):
    batch = _raw_batch()
    if not has_history:
        for key in tuple(batch):
            if key.startswith("history_") and key != "history_cursor":
                batch[key] = batch[key][:, :0]
    prefix, index = ("history", (0, 0, 2)) if has_history else ("current", (0, 2))
    if bad == "missing":
        batch[f"{prefix}_state_null_mask"][index] = True
    else:
        batch[f"{prefix}_state_abs_values"][index] = float(bad)
    with pytest.raises(ValueError, match="anchor request time must be known and finite"):
        _encoder().encode(batch)


@pytest.mark.parametrize("field,value", [("history_reset_keep", True), ("history_reset_keep", 8.5),
                                       ("history_reset_keep", "8"), ("history_reset_keep", 0),
                                       ("history_capacity", 7), ("time_delta_scale", False),
                                       ("time_delta_scale", 0), ("time_delta_scale", float("inf")),
                                       ("time_delta_scale", float("nan")), ("time_delta_scale", 1e-50)])
def test_model_context_config_is_strict(field, value):
    with pytest.raises(ValueError, match="model\\."):
        ModelConfig.from_mapping({field: value})


def test_model_context_defaults_and_no_history_boundary():
    assert ModelConfig().history_reset_keep == 8
    assert ModelConfig().time_delta_scale == 120
    assert ModelConfig(history_capacity=0, history_reset_keep=0).history_capacity == 0
    with pytest.raises(ValueError, match="must be zero"):
        ModelConfig(history_capacity=0)


@pytest.mark.parametrize("compact", [False, True])
def test_cuda_encoding_matches_cpu_without_device_scalar_reads(monkeypatch, compact):
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA 验证共享编码热路径")
    batch = _raw_batch()
    batch["scene_types"] = batch["scene_types"].to(torch.int32)
    if compact:
        for name in ("skill_ids", "skill_features", "state_abs_values", "state_delta_values",
                     "state_null_mask", "state_delta_reset_mask", "state_skill_availability"):
            values = batch.pop("history_" + name).squeeze(0)
            batch["history_bank_" + name] = torch.cat((values.new_zeros((2, *values.shape[1:])), values))
        batch["history_lengths"] = torch.tensor([2])
        batch["history_ends"] = torch.tensor([4])
    expected = _encoder().encode(batch)
    cuda_encoder = _encoder().to(device="cuda")
    cuda_batch = {key: value.cuda() for key, value in batch.items()}
    original_bool, original_item, original_cpu = torch.Tensor.__bool__, torch.Tensor.item, torch.Tensor.cpu

    def reject_device_read(method):
        def guarded(value, *args, **kwargs):
            if value.is_cuda:
                pytest.fail("上下文编码不能读取 CUDA 标量或回传 CPU")
            return method(value, *args, **kwargs)
        return guarded

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", reject_device_read(original_bool))
        patch.setattr(torch.Tensor, "item", reject_device_read(original_item))
        patch.setattr(torch.Tensor, "cpu", reject_device_read(original_cpu))
        actual = cuda_encoder.encode(cuda_batch)
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        torch.testing.assert_close(actual[key].cpu(), value, rtol=1e-6, atol=1e-6, msg=key)

"""读侧绝对状态、时间重锚和场景差分的共同数据边界。"""

from tests.training._causal_fixtures import make_state_groups

from dataclasses import asdict

import pytest
import torch

from common.policy.config import ModelConfig
from common.policy.data.context_encoding import ContextEncoder, raw_state_delta
from common.policy.data.history_window import fit_history_suffix_length, history_window_length
from common.policy.data.normalization import NormalizerConfig
from common.policy.data.normalizer import NORMALIZER_CONTRACT_VERSION, Normalizer
from common.policy.data.schema import SceneWindowSchema, TrainingSchema


def _encoder(normalizer_config=None):
    schema = TrainingSchema(
        serialization_format="test", sample_schema_version=11, context_schema_version=14,
        scene_context_mode="absolute", skill_history_fields=("kind",),
        state_groups=make_state_groups({"player": (
            "request_state.mp", "request_state.is_moving", "request_state.time_seconds",
            "request_state.current_potency", "previous_action_after.time_seconds",
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
    normalizer = Normalizer(normalizer_config or NormalizerConfig())
    normalizer = Normalizer.from_contract({**normalizer.normalization_contract, "skill_charge_limits": {"first": 2, "second": 2}})
    return ContextEncoder(normalizer, schema, layout=schema.state_layout(("first", "second")))


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
        "history_state_skill_state": torch.zeros((*(states).shape[:-1], 12), dtype=torch.float32),
        "current_state_skill_state": torch.zeros((*(current).shape[:-1], 12), dtype=torch.float32),
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


def test_only_time_coordinates_use_first_anchor_and_current_delta():
    encoder = _encoder()
    output = encoder.encode(_raw_batch())
    expected = torch.tensor([[[0.8, 1, 0, 1000 / 2000, -2 / 270, 0.9],
                              [0.6, 0, 3 / 270, 1100 / 2000, 3 / 270, 0.8]]])
    torch.testing.assert_close(output["history_state_vectors"][..., :6], expected)
    torch.testing.assert_close(output["current_state_vectors"][..., :6], torch.tensor([
        [0.7, 1, 4 / 270, 1050 / 2000, 4 / 270, 0.6],
    ]))
    assert output["history_state_reset_mask"].tolist() == [
        [[False, False, True, False, True, False], [False] * 6],
    ]
    assert not output["current_state_reset_mask"].any()
    assert all(value.dtype == torch.float32 for key, value in output.items() if key.endswith("vectors"))
    assert not any("abs_values" in key or "delta_values" in key for key in output)
    assert not list(encoder.parameters())
    with pytest.raises(ValueError, match="encoded twice"):
        encoder.encode(output)


def test_shared_normalization_stays_fp32_under_autocast():
    """技能、基础状态、技能状态和场景共用 FP32 运算，外层 autocast 不改变结果。"""
    encoder = _encoder()
    batch = _raw_batch()
    batch["history_state_skill_state"][..., 1::3] = 1
    batch["current_state_skill_state"][..., 2::3] = 30.0625
    raw_skills = torch.tensor([[1013.125], [1000.0625]], dtype=torch.float32)
    normalizer = Normalizer()
    expected_skills = normalizer.normalize_skill_features(raw_skills, ("potency",))
    expected_context = encoder.encode(batch)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        skills = normalizer.normalize_skill_features(raw_skills, ("potency",))
        context = encoder.encode(batch)
    assert skills.dtype == torch.float32
    torch.testing.assert_close(skills, expected_skills, rtol=0, atol=0)
    for key in ("history_state_vectors", "current_state_vectors", "scene_vectors"):
        assert context[key].dtype == torch.float32
        torch.testing.assert_close(context[key], expected_context[key], rtol=0, atol=0)


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


def test_resources_timers_and_buff_refresh_keep_absolute_states_and_separate_time_scales():
    """raw 差分保持无损，模型只对时间坐标求差，资源和有界时长保留绝对状态。"""
    keys = (
        "request_state.time_seconds", "previous_action_after.time_seconds", "request_state.mp",
        "request_state.astral_fire", "request_state.polyglot", "request_state.polyglot_timer",
        "request_state.job.triplecast.active", "request_state.job.triplecast.remaining_seconds",
        "request_state.job.triplecast.stacks", "request_state.current_potency",
    )
    normalizer = Normalizer.from_contract({
        "version": NORMALIZER_CONTRACT_VERSION, "config": asdict(NormalizerConfig(time_scale=60.0, duration_scale=30.0)),
        "resource_limits": {"astral_fire": 3.0, "polyglot": 3.0, "polyglot_timer": 30.0},
        "resource_time_fields": ("polyglot_timer",),
        "skill_charge_limits": {"first": 2, "second": 2},
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
        [1200, 1198, 10000, 3, 2, 29, 1, 12, 3, 1000],
        [1203, 1201, 7600, 0, 3, 2, 1, 9, 2, 1700],
        [1207, 1205, 8000, 1, 0, 6, 0, 0, 0, 1650],
        [1210, 1208, 9200, 3, 1, 1, 1, 15, 3, 2300],
    ], dtype=torch.float32)
    nulls = torch.zeros_like(values, dtype=torch.bool)
    delta, reset = raw_state_delta(
        values, nulls,
        torch.cat((torch.zeros_like(values[:1]), values[:-1])),
        torch.cat((torch.ones_like(nulls[:1]), nulls[:-1])),
    )
    # 逐字段核对 raw 重建，防止某类资源/Buff 提前截断或 BF16 丢精度。
    for index, key in enumerate(keys):
        reconstructed = torch.cat((delta[:1, index],
                                   delta[:1, index] + delta[1:, index].cumsum(dim=0)))
        torch.testing.assert_close(reconstructed, values[:, index], rtol=0, atol=0, msg=key)
    assert reset[0].all()
    assert not reset[1:].any()
    output = ContextEncoder(normalizer, schema, layout=schema.state_layout(("first", "second"))).encode({
        "history_state_skill_state": torch.zeros((*(values[:3][None]).shape[:-1], 12), dtype=torch.float32),
        "current_state_skill_state": torch.zeros((*(values[3:]).shape[:-1], 12), dtype=torch.float32),
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
        [0, -2/60, 1, 1, 2/3, 29/30, 1, 12/30, 1, 1000/2000],
        [3/60, 3/60, .76, 0, 1, 2/30, 1, 9/30, 2/3, 1700/2000],
        [4/60, 4/60, .8, 1/3, 0, 6/30, 0, 0, 0, 1650/2000],
        [3/60, 3/60, .92, 1, 1/3, 1/30, 1, 15/30, 1, 1],
    ])
    actual = torch.cat((output["history_state_vectors"][0], output["current_state_vectors"]))
    for index, key in enumerate(keys):
        torch.testing.assert_close(actual[:, index], expected[:, index], msg=key)
    assert actual.dtype == torch.float32


def test_scene_clip_sort_and_separate_signed_start_end_deltas():
    output = _encoder().encode(_raw_batch())
    assert output["scene_mask"].tolist() == [[True, True, True]]
    assert output["scene_types"].tolist() == [[0, 0, 1]]
    torch.testing.assert_close(output["scene_vectors"], torch.tensor([[
        [0, 100 / 270, 100 / 270, 1],
        [20 / 270, 300 / 270, 380 / 270, 1],
        [3 / 5, 10 / 270, -350 / 270, 20 / 270],
    ]]))


@pytest.mark.parametrize("count,expected", [(-3, 0), (0, 0), (1, 0.2), (3, 0.6), (5, 1), (8, 1)])
def test_scene_target_count_uses_default_five_target_limit(count, expected):
    """目标数按默认五目标上限截断，其他场景内容与类型索引照常保留。"""
    batch = _raw_batch()
    batch["scene_abs_values"] = torch.tensor([[[count, 10, 1250, 1240]]], dtype=torch.float32)
    batch["scene_types"] = torch.tensor([[1]])
    batch["scene_mask"] = torch.ones((1, 1), dtype=torch.bool)
    output = _encoder().encode(batch)
    assert output["scene_mask"].tolist() == [[True]]
    assert output["scene_types"].tolist() == [[1]]
    torch.testing.assert_close(output["scene_vectors"], torch.tensor([[[
        expected, 10 / 270, 50 / 270, 40 / 270,
    ]]]))
    assert batch["scene_abs_values"][0, 0, 0] == count


def test_relative_and_duration_scales_apply_independently_to_state_skill_and_scene():
    """时间坐标差分与绝对时长各用自己的尺度，场景长度仍允许超过一。"""
    batch = _raw_batch()
    batch["current_state_skill_state"][..., 2::3] = 30
    output = _encoder(NormalizerConfig(time_scale=30.0, duration_scale=60.0)).encode(batch)
    torch.testing.assert_close(output["history_state_vectors"][0, :, 2], torch.tensor([0, 3 / 30]))
    assert output["history_state_vectors"][0, 0, 4] == pytest.approx(-2 / 30)
    assert output["current_state_vectors"][0, 2] == pytest.approx(4 / 30)
    torch.testing.assert_close(output["current_state_vectors"][..., 8::3], torch.full((1, 4), 30 / 60))
    assert output["scene_vectors"][0, 1, 2] == pytest.approx(380 / 60)
    assert output["scene_vectors"][0, 2, 1] == pytest.approx(10 / 60)
    assert output["scene_vectors"][0, 2, 2] == pytest.approx(-350 / 30)


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
        [20/270, 50/270, 30/270, .5],
        [0, 10/270, 40/270, .25],
        [0, 0, 40/270, .75],
        [2/5, 40/270, 0, 0],
        [20/270, 20/270, 40/270, 1],
    ]]))


def test_window_reanchors_from_selected_absolute_row_and_preserves_later_delta():
    batch = _raw_batch()
    for key in tuple(batch):
        if key.startswith("history_") and key != "history_cursor":
            batch[key] = batch[key][:, 1:]
    output = _encoder().encode(batch)
    torch.testing.assert_close(output["history_state_vectors"][0, 0, :6], torch.tensor(
        [0.6, 0, 0, 1100 / 2000, -2 / 270, 0.8],
    ))
    assert output["history_state_reset_mask"].tolist() == [[[False, False, True, False, True, False]]]
    assert output["current_state_vectors"][0, 2] == pytest.approx(4 / 270)
    assert output["scene_vectors"][0, 0, 1] == pytest.approx(97 / 270)


@pytest.mark.parametrize("scene_width", (0, 4))
def test_empty_history_anchors_current_state_and_deletes_invalid_scenes(scene_width):
    batch = _raw_batch()
    for key in tuple(batch):
        if key.startswith("history_") and key != "history_cursor":
            batch[key] = batch[key][:, :0]
    for key in ("scene_abs_values", "scene_types", "scene_mask"):
        batch[key] = batch[key][:, :scene_width]
    batch["scene_abs_values"][:, :, 1] = 0  # 所有 combat 窗口均已结束。
    batch["scene_mask"].fill_(False)
    output = _encoder().encode(batch)
    assert output["history_state_vectors"].shape == (1, 0, 18)
    assert output["current_state_reset_mask"].tolist() == [[False, False, True, False, True, False]]
    torch.testing.assert_close(output["current_state_vectors"][0, :6], torch.tensor(
        [0.7, 1, 0, 1050 / 2000, -2 / 270, 0.6],
    ))
    assert output["scene_vectors"].shape == (1, 0, 4)
    assert output["scene_types"].dtype == torch.long
    assert output["scene_mask"].dtype == torch.bool
    assert not output["scene_mask"].any()
    assert torch.count_nonzero(output["scene_vectors"]) == 0


def test_expired_scenes_are_removed_and_only_mixed_batch_padding_remains():
    batch = {key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in _raw_batch().items()}
    batch["history_state_abs_values"][1, :, 2] += 200
    raw_scenes = batch["scene_abs_values"].clone()
    output = _encoder().encode(batch)
    assert output["scene_vectors"].shape == (2, 3, 4)
    assert output["scene_mask"].tolist() == [[True, True, True], [True, False, False]]
    for index, count in enumerate((3, 1)):
        separate = _encoder().encode({key: value[index:index + 1] for key, value in batch.items()})
        torch.testing.assert_close(output["scene_vectors"][index, :count], separate["scene_vectors"][0], rtol=0, atol=0)
        torch.testing.assert_close(output["scene_types"][index, :count], separate["scene_types"][0], rtol=0, atol=0)
    assert not output["scene_vectors"][1, 1:].any()
    torch.testing.assert_close(batch["scene_abs_values"], raw_scenes, rtol=0, atol=0)


def test_scene_ending_at_the_anchor_is_physically_deleted():
    batch = _raw_batch()
    batch["scene_abs_values"][0, 2, 1] = 1200.0
    assert _encoder().encode(batch)["scene_vectors"].shape[1] == 3


def test_compact_gather_matches_dense_and_does_not_cross_bank_boundaries():
    dense = _raw_batch()
    compact = dict(dense)
    names = ("skill_ids", "skill_features", "state_abs_values", "state_delta_values",
             "state_null_mask", "state_delta_reset_mask", "state_skill_state")
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


def test_skill_state_is_absolute_and_does_not_change_base_skill_or_scene_encoding():
    """充能和恢复时间直接编码各快照绝对值，不进入基础状态 DELTA。"""
    batch = _raw_batch()
    encoder = _encoder()
    original = encoder.encode(batch)
    batch["history_state_skill_state"] = torch.tensor([[
        [1, 1, 30, 0, 0, 60, 0, 1, 15, 1, 2, 0],
        [0, 0, 90, 1, 1, 60, 1, 2, 0, 0, 1, 30],
    ]], dtype=torch.float32)
    batch["current_state_skill_state"] = torch.tensor([[1, 2, 0, 1, 1, 30, 0, 0, 90, 0, 1, 15]], dtype=torch.float32)
    changed = encoder.encode(batch)
    torch.testing.assert_close(changed["history_state_vectors"][..., 6:], torch.tensor([[
        [1, .5, 30 / 270, 0, 0, 60 / 270, 0, .5, 15 / 270, 1, 1, 0],
        [0, 0, 90 / 270, 1, .5, 60 / 270, 1, 1, 0, 0, .5, 30 / 270],
    ]]))
    torch.testing.assert_close(changed["current_state_vectors"][..., 6:], torch.tensor([
        [1, 1, 0, 1, .5, 30 / 270, 0, 0, 90 / 270, 0, .5, 15 / 270],
    ]))
    for key in ("history_state_vectors", "current_state_vectors"):
        torch.testing.assert_close(changed[key][..., :6], original[key][..., :6], rtol=0, atol=0)
    for key in ("history_skill_ids", "history_skill_features", "history_mask", "scene_vectors", "scene_types", "scene_mask"):
        torch.testing.assert_close(changed[key], original[key], rtol=0, atol=0)


def test_null_to_known_only_resets_time_and_padding_is_inert():
    batch = _raw_batch()
    batch["history_state_delta_reset_mask"][0, 1, 4] = True
    batch["history_state_delta_values"][0, 1, 4] = 1201
    batch["history_state_null_mask"][0, 1, 0] = True
    output = _encoder().encode(batch)
    assert output["history_state_vectors"][0, 1, 4] == pytest.approx(1 / 270)
    assert output["history_state_reset_mask"][0, 1, 4]
    assert output["history_state_vectors"][0, 1, 0] == -1
    assert not output["history_state_reset_mask"][0, 1, 0]
    batch["history_state_null_mask"][0, 1, 0] = False
    batch["history_state_delta_reset_mask"][0, 1, 0] = True
    batch["history_state_delta_values"][0, 1, 0] = 6000
    output = _encoder().encode(batch)
    assert output["history_state_vectors"][0, 1, 0] == pytest.approx(.6)
    assert not output["history_state_reset_mask"][0, 1, 0]
    batch["history_mask"][0, 1] = False
    output = _encoder().encode(batch)
    assert torch.count_nonzero(output["history_state_vectors"][0, 1]) == 0
    assert output["history_state_null_mask"][0, 1].all()
    assert not output["history_state_reset_mask"][0, 1].any()


@pytest.mark.parametrize("total,expected", [(0, 0), (299, 299), (300, 300), (301, 8),
                                           (302, 9), (593, 300), (594, 8), (887, 8)])
def test_history_window_resets_instead_of_sliding(total, expected):
    assert history_window_length(
        total, 602, 8, request_times=range(total),
        scene_end_times=(1e9,), current_request_time=float(total),
    ) == expected


@pytest.mark.parametrize("args", [(True, 300, 8), (-1, 300, 8), (1, 3, 4),
                                  (1, 3, 0), (1, 0, 1), (1, 300, False), (1.0, 300, 8)])
def test_history_window_rejects_invalid_parameters(args):
    with pytest.raises(ValueError):
        history_window_length(*args, request_times=range(10), scene_end_times=(), current_request_time=0.0)


def test_shared_budget_recomputes_scene_at_each_reset():
    # 开始时八个场景占用预算，第一次溢出后均已过去，下一块可以保留更多历史。
    expected = {0: 0, 7: 7, 8: 2, 17: 11, 18: 2}
    for total, length in expected.items():
        assert history_window_length(
            total, 24, 2, request_times=range(30),
            scene_end_times=(4.0,) * 8, current_request_time=float(total),
        ) == length


@pytest.mark.parametrize("total,ends", [(0, (100.0,) * 18), (8, (100.0,) * 2)])
def test_shared_budget_rejects_scene_that_cannot_fit_with_required_history(total, ends):
    with pytest.raises(ValueError, match="cannot hold"):
        history_window_length(total, 17, 8, request_times=range(10),
                              scene_end_times=ends, current_request_time=float(total))


def test_shared_budget_1024_counts_scene_and_latest_state():
    for total, expected in ((411, 411), (412, 8), (413, 9)):
        assert history_window_length(
            total, 1024, 8, request_times=range(500),
            scene_end_times=(1e9,) * 200, current_request_time=float(total),
        ) == expected


def test_additional_history_crop_rechecks_budget_when_request_times_go_backwards():
    kwargs = dict(request_times=(0.1, 0.0), scene_end_times=(0.05,) * 7, current_request_time=0.2)
    assert history_window_length(2, 9, 2, **kwargs) == 2
    assert history_window_length(2, 9, 2, max_history=1, **kwargs) == 0


def test_death_markers_reset_at_their_first_overflowing_boundary():
    kwargs = dict(request_times=range(20), scene_end_times=(), current_request_time=5.0)
    marker = {"after_history_length": 5}
    assert history_window_length(5, 11, 2, **kwargs) == 5
    assert history_window_length(5, 11, 2, history_markers=(marker,), **kwargs) == 2
    assert history_window_length(8, 11, 2, history_markers=(marker,), **kwargs) == 2
    # 起点恰落在标记边界时仍需计入，不能在重置后把边界事件丢掉。
    assert history_window_length(12, 15, 2, history_markers=(marker,), **kwargs) == 2


def test_death_marker_budget_handles_zero_actions_and_repeated_same_cursor():
    kwargs = dict(request_times=(), scene_end_times=(), current_request_time=0.0)
    markers = ({"after_history_length": 0},) * 2
    assert history_window_length(0, 3, 1, history_markers=markers, **kwargs) == 0
    assert fit_history_suffix_length(0, 3, total_history=0, history_markers=markers, **kwargs) == 0
    with pytest.raises(ValueError, match="cannot hold"):
        history_window_length(0, 3, 1, history_markers=markers * 2, **kwargs)
    with pytest.raises(ValueError, match="cannot hold"):
        fit_history_suffix_length(0, 3, total_history=0, history_markers=markers * 2, **kwargs)


def test_death_markers_are_filtered_inclusively_when_fitting_a_suffix():
    markers = tuple({"after_history_length": position} for position in (0, 2, 2, 3))
    kwargs = dict(total_history=3, request_times=range(3), scene_end_times=(), current_request_time=3.0)
    assert fit_history_suffix_length(1, 6, history_markers=markers, **kwargs) == 1
    assert fit_history_suffix_length(1, 5, history_markers=markers, **kwargs) == 0


@pytest.mark.parametrize("positions", [(True,), (-1,), (2,), (1, 0)])
def test_history_window_rejects_invalid_death_boundaries(positions):
    with pytest.raises(ValueError, match="history marker"):
        history_window_length(
            1, 5, 1, request_times=(0.0,), scene_end_times=(), current_request_time=1.0,
            history_markers=tuple({"after_history_length": position} for position in positions),
        )


def test_death_window_planning_matches_sequential_action_and_marker_arrivals():
    import random

    rng = random.Random(731)
    for _ in range(300):
        total, capacity, keep = rng.randrange(13), rng.randrange(5, 20), rng.randrange(1, 3)
        times = tuple(rng.randrange(12) for _ in range(total))
        ends = tuple(sorted(rng.randrange(12) for _ in range(rng.randrange(5))))
        positions = sorted(rng.randrange(total + 1) for _ in range(rng.randrange(7)))
        markers = tuple({"after_history_length": position} for position in positions)

        def replay_arrivals():
            start = 0
            arrived = []

            def fit(cursor):
                nonlocal start
                while True:
                    anchor = times[start] if start < total else 12.0
                    scenes = sum(end > anchor for end in ends)
                    tokens = scenes + 2 * (cursor - start) + 1 + sum(position >= start for position in arrived)
                    if tokens <= capacity:
                        return
                    next_start = max(0, cursor - keep)
                    if next_start <= start:
                        raise ValueError("cannot hold required context")
                    start = next_start

            for cursor in range(total + 1):
                fit(cursor)
                for position in positions:
                    if position == cursor:
                        arrived.append(position)
                        fit(cursor)
            return total - start

        try:
            expected = replay_arrivals()
        except ValueError:
            with pytest.raises(ValueError, match="cannot hold"):
                history_window_length(
                    total, capacity, keep, request_times=times, scene_end_times=ends,
                    current_request_time=12.0, history_markers=markers,
                )
        else:
            assert history_window_length(
                total, capacity, keep, request_times=times, scene_end_times=ends,
                current_request_time=12.0, history_markers=markers,
            ) == expected


def test_random_history_crop_also_rechecks_budget_without_mutating_the_source():
    from training.data.collator import TrainingCollator

    sample = {
        "history_length": 2, "history_end": 3,
        "_history_budget": {
            "context_capacity": 9, "total_history": 2,
            "request_times": (0.1, 0.0), "scene_end_times": (0.05,) * 7,
            "current_request_time": 0.2,
        },
    }
    collator = TrainingCollator(history_truncation_enabled=True, history_truncation_probability=1.0)
    assert collator._truncate_early_history(sample)["history_length"] == 0
    assert sample["history_length"] == 2


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
                                       ("history_capacity", 7), ("time_delta_scale", 270)])
def test_model_context_config_is_strict(field, value):
    with pytest.raises(ValueError, match="model"):
        ModelConfig.from_mapping({field: value})


def test_model_context_defaults_and_no_history_boundary():
    assert ModelConfig().history_reset_keep == 8
    assert ModelConfig().context_capacity == 1024
    with pytest.raises(ValueError, match="context_capacity"):
        ModelConfig(context_capacity=16)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("scene_width, all_masked", [(0, False), (1, False), (4, False), (4, True)])
def test_cuda_encoding_matches_cpu_without_explicit_device_scalar_reads(monkeypatch, compact, scene_width, all_masked):
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA 验证共享编码热路径")
    batch = _raw_batch()
    batch["scene_types"] = batch["scene_types"].to(torch.int32)
    for key in ("scene_abs_values", "scene_types", "scene_mask"):
        batch[key] = batch[key][:, :scene_width]
    if all_masked:
        batch["scene_mask"].fill_(False)
    if compact:
        for name in ("skill_ids", "skill_features", "state_abs_values", "state_delta_values",
                     "state_null_mask", "state_delta_reset_mask", "state_skill_state"):
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

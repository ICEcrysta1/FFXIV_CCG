"""转换增量历史的连续性、冻结行复用及真实桥接等价回归。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.convert_fflogs import build_training_samples
from scripts.convert_fflogs.training.history_bank import HistoryPrefixInterner
from tests.helpers import (
    build_test_scene_context,
    forced_movement_window_token,
    raid_buff_window_token,
    targetable_window_token,
)


def _delta(start: int, end: int) -> dict[str, object]:
    return {
        "history_start": start,
        "history_cursor": end,
        "skill_history_context": [
            {"skill_key": f"skill_{index}", "kind": index % 2, "potency": float(index)}
            for index in range(start, end)
        ],
        "state_history_context": {
            "tokens": [{"player_state": [float(index), -0.0]} for index in range(start, end)],
            "execution_metrics": [
                {"cumulative_dot_potency": float(index * 60)} for index in range(start, end)
            ],
        },
        "current_state_context": {"tokens": [{"player_state": [float(end)]}]},
        "action_values": [1.0, 2.0],
    }


def _channels(context):
    return (
        context["skill_history_context"],
        context["state_history_context"]["tokens"],
        context["state_history_context"]["execution_metrics"],
    )


def _replace_channel(context, index, value):
    if index == 0:
        context["skill_history_context"] = value
    else:
        key = "tokens" if index == 1 else "execution_metrics"
        context["state_history_context"][key] = value


def _seeded_interner():
    interner = HistoryPrefixInterner()
    prefix = _delta(0, 2)
    interner.append_delta(prefix)
    return interner, prefix


def test_delta_append_supports_empty_equal_and_multiple_new_rows_without_mutating_old_lists():
    """新增数由游标决定；后续追加不能扩大之前样本的前缀列表。"""
    interner = HistoryPrefixInterner()
    observations = []
    snapshots = []
    unique_rows = [[], [], []]
    for start, end in ((0, 0), (0, 2), (2, 2), (2, 5), (5, 5)):
        context = _delta(start, end)
        delta_lists = _channels(context)
        new_rows = [list(rows) for rows in delta_lists]
        current = context["current_state_context"]
        action_values = context["action_values"]

        interner.append_delta(context)

        assert interner.cursor == end
        assert context["history_cursor"] == end
        assert "history_start" not in context
        assert context["current_state_context"] is current
        assert context["action_values"] is action_values
        for index, rows in enumerate(_channels(context)):
            unique_rows[index].extend(new_rows[index])
            assert rows is delta_lists[index]
            assert len(rows) == end
            assert all(row is expected for row, expected in zip(rows, unique_rows[index], strict=True))
            assert all(rows is not _channels(previous)[index] for previous in observations)
        assert observations == snapshots
        observations.append(context)
        snapshots.append(deepcopy(context))

    assert len({id(row) for context in observations for row in _channels(context)[1]}) == 5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("history_start", -1),
        ("history_start", 0),
        ("history_start", 1),
        ("history_start", 3),
        ("history_start", True),
        ("history_start", False),
        ("history_start", 2.0),
        ("history_start", "2"),
        ("history_start", None),
        ("history_cursor", -1),
        ("history_cursor", 1),
        ("history_cursor", True),
        ("history_cursor", False),
        ("history_cursor", 3.0),
        ("history_cursor", "3"),
        ("history_cursor", None),
    ],
)
def test_delta_append_rejects_wrong_or_non_integer_cursors_without_changing_collected_prefix(field, value):
    interner, prefix = _seeded_interner()
    context = _delta(2, 3)
    context[field] = value
    unchanged = deepcopy(context)
    prefix_snapshot = deepcopy(prefix)

    with pytest.raises(ValueError, match="incremental history cursor"):
        interner.append_delta(context)

    assert context == unchanged
    assert prefix == prefix_snapshot
    assert interner.cursor == 2
    # 错误包不能消费游标，下一份连续包仍可追加。
    interner.append_delta(_delta(2, 3))
    assert interner.cursor == 3


@pytest.mark.parametrize("field", ["history_start", "history_cursor"])
def test_delta_append_requires_both_transport_cursors(field):
    context = _delta(0, 0)
    del context[field]
    with pytest.raises(ValueError, match="incremental history cursor"):
        HistoryPrefixInterner().append_delta(context)


@pytest.mark.parametrize("channel", [0, 1, 2])
@pytest.mark.parametrize("invalid", ["missing_row", "extra_row", "tuple", "null"])
def test_delta_append_requires_three_list_channels_matching_cursor_span(channel, invalid):
    interner, prefix = _seeded_interner()
    context = _delta(2, 3)
    rows = _channels(context)[channel]
    if invalid == "missing_row":
        rows.pop()
    elif invalid == "extra_row":
        rows.append(deepcopy(rows[0]))
    else:
        _replace_channel(context, channel, tuple(rows) if invalid == "tuple" else None)
    unchanged = deepcopy(context)
    old_rows = [list(rows) for rows in _channels(prefix)]

    with pytest.raises(ValueError, match="incremental .* history length mismatch"):
        interner.append_delta(context)

    assert context == unchanged
    assert interner.cursor == 2
    for old, current in zip(old_rows, _channels(prefix), strict=True):
        assert all(a is b for a, b in zip(old, current, strict=True))


def test_delta_append_rejects_claimed_future_cursor_without_corresponding_rows():
    interner, _prefix = _seeded_interner()
    context = _delta(2, 3)
    context["history_cursor"] = 20
    with pytest.raises(ValueError, match="incremental .* history length mismatch"):
        interner.append_delta(context)
    assert interner.cursor == 2


@pytest.mark.parametrize("channel", [0, 1, 2])
@pytest.mark.parametrize("row", [None, [], "invalid"])
def test_delta_append_rejects_non_mapping_new_rows_atomically(channel, row):
    interner, prefix = _seeded_interner()
    context = _delta(2, 3)
    _channels(context)[channel][0] = row
    unchanged = deepcopy(context)
    prefix_snapshot = deepcopy(prefix)

    with pytest.raises(ValueError, match="incremental history rows must contain mappings"):
        interner.append_delta(context)

    assert context == unchanged
    assert prefix == prefix_snapshot
    assert interner.cursor == 2


@pytest.mark.parametrize("state_context", [None, [], "invalid"])
def test_delta_append_requires_state_context_mapping(state_context):
    context = _delta(0, 0)
    context["state_history_context"] = state_context
    with pytest.raises(TypeError, match="state_history_context must be a mapping"):
        HistoryPrefixInterner().append_delta(context)


def test_delta_append_never_deep_compares_already_frozen_rows():
    """旧行可以拒绝任意相等比较，增量追加仍必须只借用其引用。"""
    class UncomparableFrozenRow(dict):
        def __eq__(self, _other):
            raise AssertionError("已经冻结的历史行不可再次做深比较")

        def __ne__(self, _other):
            raise AssertionError("已经冻结的历史行不可再次做深比较")

    interner = HistoryPrefixInterner()
    first = _delta(0, 2)
    for rows in _channels(first):
        rows[:] = [UncomparableFrozenRow(row) for row in rows]
    interner.append_delta(first)
    old_rows = [list(rows) for rows in _channels(first)]

    for start, end in ((2, 2), (2, 5), (5, 5)):
        context = _delta(start, end)
        interner.append_delta(context)
        assert interner.cursor == end
        for old, current in zip(old_rows, _channels(context), strict=True):
            assert all(current[index] is row for index, row in enumerate(old))
    assert all(len(rows) == 2 for rows in _channels(first))


def _fight_payload(with_boundaries: bool):
    duration = 12.0
    targetable = [targetable_window_token(0.0, duration, targetable=True, segment_kind="combat")]
    movement = []
    raid_buffs = []
    if with_boundaries:
        # 所有真实请求均在 Boss 可选中的窗口；6 秒处统一恢复场景事实。
        targetable = [
            targetable_window_token(0.0, 4.0, targetable=True, segment_kind="combat"),
            targetable_window_token(4.0, 6.0, targetable=False, segment_kind="downtime"),
            targetable_window_token(6.0, duration, targetable=True, segment_kind="combat_final"),
        ]
        movement = [forced_movement_window_token(4.0, 6.0)]
        raid_buffs = [raid_buff_window_token(0.0, 6.0)]
    return {
        "fight_id": f"incremental_history_{with_boundaries}",
        "job_tag": "black_mage",
        "duration": duration,
        "scene_context": build_test_scene_context(
            targetable_tokens=targetable,
            forced_movement_tokens=movement,
            raid_buff_tokens=raid_buffs,
        ),
        "actions": [
            {"time_offset": timestamp, "request_time_offset": timestamp, "action_key": action,
             "time_gap": gap, "fight_remaining": duration - timestamp, "anchor": "combat"}
            for timestamp, action, gap in (
                (0.0, "fire_iii", 3.444),
                (3.444, "high_thunder", 0.0),
                (6.0, "blizzard_iii", 2.556),
                (9.5, "fire_iii", 3.5),
            )
        ],
    }


@pytest.mark.parametrize("with_boundaries", [False, True], ids=["dot_wait", "dot_wait_scene_boundaries"])
def test_real_incremental_replay_matches_full_context_and_transfers_each_history_row_once(
    cs_backend, cs_skill_book, monkeypatch, with_boundaries,
):
    """真实回放逐值相同；传输量只随新增 bank 行数增长。"""
    fight = _fight_payload(with_boundaries)
    cs_backend.init(actual_base_gcd=2.46, fight_remaining=12.0)
    baseline = build_training_samples(cs_backend, cs_skill_book, fight)
    cs_backend.init(actual_base_gcd=2.46, fight_remaining=12.0)
    replay_sequence = cs_backend.replay_policy_sequence
    transported = []
    replay_calls = []

    def tracked_sequence(requests, facts, *, fight_duration, policy_action_key):
        frames = replay_sequence(
            requests, facts, fight_duration=fight_duration, policy_action_key=policy_action_key,
        )
        replay_calls.append(len(requests))
        for frame in frames:
            context = frame["context"]
            start, end = context["history_start"], context["history_cursor"]
            lengths = tuple(len(rows) for rows in _channels(context))
            assert lengths == (end - start,) * 3
            # 保存数值计数，后续恢复完整前缀不得把观测记录也扩大。
            transported.append((start, end, lengths[0]))
        return frames

    monkeypatch.setattr(cs_backend, "replay_policy_sequence", tracked_sequence)
    incremental = build_training_samples(cs_backend, cs_skill_book, fight, deduplicate_history=True)

    assert incremental == baseline
    assert replay_calls == [len(fight["actions"])]
    assert "ogcd_wait" in incremental["resolved_sequence"]
    assert "high_thunder" in incremental["resolved_sequence"]
    contexts = [sample["context"] for sample in incremental["samples"]]
    assert len(transported) == len(contexts)
    assert transported[0][0] == 0
    assert all(left[1] == right[0] for left, right in zip(transported, transported[1:]))
    final_rows = len(contexts[-1]["skill_history_context"])
    assert sum(count for _start, _end, count in transported) == final_rows
    assert transported[-1][1] == final_rows
    assert final_rows < sum(len(context["skill_history_context"]) for context in contexts)
    assert all("history_start" not in context for context in contexts)
    assert any(
        metric["cumulative_dot_potency"] > 0
        for context in contexts for metric in context["state_history_context"]["execution_metrics"]
    )
    for old, new in zip(contexts, contexts[1:]):
        for old_rows, new_rows in zip(_channels(old), _channels(new), strict=True):
            assert old_rows is not new_rows
            assert all(old_row is new_row for old_row, new_row in zip(old_rows, new_rows))


def test_real_native_replay_preserves_queued_cast_overrides_and_supervision_metadata(
    cs_backend, cs_skill_book,
):
    """批量宿主必须保留排队接受、生效时刻和完整监督字段。"""
    fight = _fight_payload(False)
    fight["annotation_status"] = "partial"
    fight["ranking"] = {"percentile": 97.5, "percentile_bucket": "90-100"}
    fight["actions"] = [
        {
            "time_offset": timestamp,
            "request_time_offset": timestamp,
            "action_key": action,
            "actual_cast_seconds": cast_seconds,
            "raw_event_index": 100 + index,
            "quality_labels": [{"reason_id": "regression", "severity": "minor"}],
            "time_gap": gap,
            "anchor": "combat",
            "cast_timing_source": "begincast_duration",
            "request_order_adjustment_seconds": 0.0005,
        }
        for index, (timestamp, action, cast_seconds, gap) in enumerate((
            (0.0, "fire_iii", 3.0, 3.0),
            (2.6, "blizzard_iii", 1.0, 2.6),
            (6.0, "high_thunder", 0.0, 3.4),
            (9.5, "fire_iii", 1.25, 3.5),
        ))
    ]
    outputs = []
    for incremental in (False, True):
        cs_backend.init(actual_base_gcd=2.46, fight_remaining=12.0)
        outputs.append(build_training_samples(
            cs_backend, cs_skill_book, fight, deduplicate_history=incremental,
        ))

    assert outputs[0] == outputs[1]
    real_labels = [
        sample["label"] for sample in outputs[1]["samples"]
        if sample["label"]["action_key"] != "ogcd_wait"
    ]
    assert real_labels[1]["queued"] is True
    assert real_labels[1]["accepted_offset"] == pytest.approx(0.4)
    # 生效可能早于完整读条结束，不能用接受时刻加读条重新推算。
    assert real_labels[1]["accepted_offset"] < real_labels[1]["effect_offset"] < (
        real_labels[1]["accepted_offset"] + real_labels[1]["cast_time_seconds"]
    )
    assert [label["cast_time_seconds"] for label in real_labels] == [3.0, 1.0, 0.0, 1.25]
    assert [label["raw_event_index"] for label in real_labels] == list(range(100, 104))
    assert all(label["quality_labels"] == fight["actions"][0]["quality_labels"] for label in real_labels)


def test_real_native_replay_rejected_frame_keeps_public_error_step_and_reason(cs_backend, cs_skill_book):
    """拒绝的帧仍按 Python 转换层原有步号和文案失败。"""
    fight = _fight_payload(False)
    fight["actions"] = [
        {"time_offset": 0.0, "request_time_offset": 0.0, "action_key": "fire_iii"},
        {"time_offset": 0.1, "request_time_offset": 0.1, "action_key": "fire_iv"},
    ]
    errors = []
    for incremental in (False, True):
        cs_backend.init(actual_base_gcd=2.46, fight_remaining=12.0)
        with pytest.raises(ValueError, match="logged action was rejected") as error:
            build_training_samples(cs_backend, cs_skill_book, fight, deduplicate_history=incremental)
        errors.append(str(error.value))

    assert errors[0] == errors[1]
    assert "step=2, action=fire_iv, request_time=0.1000" in errors[1]


def _batch_source():
    from common.context.output import CANONICAL_CONTEXT_SCHEMA_VERSION
    from common.policy.data import ActionSpace
    from common.policy.data.schema import TRAINING_SAMPLE_SCHEMA_VERSION

    space = ActionSpace.from_job_tag("black_mage")
    metadata = {
        "player_state_feature_keys": [
            "previous_action_after.mp", "request_state.mp", "request_state.time_seconds",
        ],
        "buff_state_feature_keys": [],
        "target_buff_state_feature_keys": [
            "previous_action_after.target.current_gcd_dot_potency",
            "request_state.target.current_gcd_dot_potency",
        ],
        "resource_state_feature_keys": [],
        "skill_state_feature_keys": [
            f"{snapshot}.{key}.{field}"
            for snapshot in ("previous_action_after", "request_state") for key in space.action_keys
            for field in ("available", "current_charges", "next_cooldown_seconds")
        ],
    }

    def state(player, dot, index):
        return {
            "player_state": list(player), "buff_state": [], "target_buff_state": list(dot),
            "resource_state": [],
            "skill_state": [value for position in range(2 * len(space.action_keys))
                            for value in ((position + index) % 2, 1, 15.0 * index)],
        }

    history_tokens = [
        state(player, dot, index)
        for index, (player, dot) in enumerate((
            ((None, 10000.0, 0.0), (0.0, 0.0)),
            ((7500.0, 7600.0, 1.0), (30.0, 60.0)),
            ((7600.0, None, 2.0), (60.0, 120.0)),
            ((None, 7300.0, 3.0), (120.0, 180.0)),
            ((7000.0, 7100.0, 4.0), (180.0, 240.0)),
        ))
    ]
    current_tokens = [
        state(player, dot, index)
        for index, (player, dot) in enumerate((
            ((None, 9000.0, -0.0), (-0.0, 0.0)),
            ((8000.0, 8100.0, 1.0), (15.0, 30.0)),
            ((8300.0, None, 2.0), (30.0, 60.0)),
            ((8000.0, 8200.0, 3.0), (90.0, 150.0)),
            ((8100.0, 8300.0, 4.0), (120.0, 180.0)),
            ((8200.0, 8400.0, 5.0), (180.0, 240.0)),
            ((7100.0, 7200.0, 6.0), (240.0, 300.0)),
        ))
    ]
    scene = build_test_scene_context(
        targetable_tokens=[
            targetable_window_token(1800.0, 2400.0, targetable=True, segment_kind="combat"),
        ],
        forced_movement_tokens=[forced_movement_window_token(1805.0, 1808.0)],
        raid_buff_tokens=[raid_buff_window_token(1820.0, 1840.0)],
    )
    severities = ((), ("minor", "medium", "major"), (), ("major",), ("minor", "medium"), (), ("medium",))
    samples = []
    for index, length in enumerate((0, 1, 1, 3, 3, 4, 5)):
        history_skills = [
            {"skill_id": 0, "skill_key": "ogcd_wait", "kind": 0, "potency": row + 0.5}
            for row in range(length)
        ]
        context = {
            "job_tag": "black_mage", "schema_version": CANONICAL_CONTEXT_SCHEMA_VERSION,
            "history_cursor": length, "scene_context": deepcopy(scene),
            "skill_history_context": history_skills,
            "state_history_context": {
                **deepcopy(metadata), "tokens": deepcopy(history_tokens[:length]),
                "execution_metrics": [{"cumulative_dot_potency": float(row * 60)} for row in range(length)],
            },
            "current_state_context": {**deepcopy(metadata), "tokens": [current_tokens[index]]},
            "action_keys": list(space.action_keys), "action_legal_mask": [True] * len(space.action_keys),
            "action_values": [index + 1.0 + position / 7 for position in range(len(space.action_keys))],
        }
        samples.append({
            "step": index + 1, "source_step": index + 1, "time_offset": index + 0.25,
            "context": context,
            "label": {
                "action_key": "ogcd_wait", "action_index": space.action_keys.index("ogcd_wait"),
                "cast_time_seconds": None if index % 2 == 0 else index / 10,
                "raw_event_index": None if index in (0, 5) else 100 + index,
                "quality_labels": [{"reason_id": f"quality_{level}", "severity": level} for level in severities[index]],
            },
        })
    return {
        "sample_schema_version": TRAINING_SAMPLE_SCHEMA_VERSION, "job_tag": "black_mage",
        "fight_id": "incremental_batch_regression", "fight_scene_context": scene,
        "annotation_status": "partial", "ranking": {"percentile": 97.5, "percentile_bucket": "90-100"},
        "samples": samples,
    }


def _batch_components(int_dtype_name="int32", float_dtype_name="float32"):
    import torch

    from common.policy.data import SkillVocab
    from scripts.convert_fflogs.source.source_reader import TrainingSourceReader
    from scripts.convert_fflogs.training.history_bank import build_history_bank
    from scripts.convert_fflogs.training.sample_builder import TrainingSampleBuilder

    payload = _batch_source()
    reader = TrainingSourceReader(payload)
    int_dtype, float_dtype = getattr(torch, int_dtype_name), getattr(torch, float_dtype_name)
    bank = build_history_bank(
        reader, torch=torch, normalizer=None, skill_vocab=SkillVocab.build_from_job_tag(reader.job_tag),
        skill_feature_names=reader.skill_feature_names, int_dtype=int_dtype, float_dtype=float_dtype,
    )
    builder = TrainingSampleBuilder(
        torch=torch, int_dtype=int_dtype, float_dtype=float_dtype, num_actions=reader.num_actions,
        history_bank=bank, ranking=reader.ranking, annotation_status=reader.annotation_status,
    )
    return torch, payload, reader, bank, builder


def test_death_marker_prefix_is_independent_of_equal_skill_history_lengths():
    from scripts.convert_fflogs.source.source_reader import TrainingSourceReader

    payload = _batch_source()
    marker = {"token": "<dead/>", "timestamp": 2.0, "after_history_length": 1, "step": 3}
    payload["history_markers"] = [marker]
    reader = TrainingSourceReader(payload)
    assert reader.history_length(1) == reader.history_length(2) == 1
    assert reader.history_markers(1) == ()
    assert reader.history_markers(2) == (marker,)
    reader.history_markers(2)[0]["token"] = "changed"
    reader.history_marker_bank[0]["step"] = 1
    assert reader.history_markers(2) == (marker,)
    assert TrainingSourceReader(_batch_source()).history_marker_bank == ()


@pytest.mark.parametrize("change", [
    {"token": "fire_iii"}, {"timestamp": float("nan")}, {"timestamp": True},
    {"step": 0}, {"step": 8}, {"step": True},
    {"after_history_length": -1}, {"after_history_length": True}, {"after_history_length": 2},
])
def test_source_reader_rejects_invalid_death_marker_boundaries(change):
    from scripts.convert_fflogs.source.source_reader import TrainingSourceReader

    payload = _batch_source()
    payload["history_markers"] = [{
        "token": "<dead/>", "timestamp": 2.0, "after_history_length": 1, "step": 3, **change,
    }]
    with pytest.raises(ValueError, match="history marker"):
        TrainingSourceReader(payload)


@pytest.mark.parametrize("field,value", [("timestamp", 1.0), ("step", 2), ("after_history_length", 0)])
def test_source_reader_rejects_reordered_death_markers(field, value):
    from scripts.convert_fflogs.source.source_reader import TrainingSourceReader

    payload = _batch_source()
    payload["history_markers"] = [
        {"token": "<dead/>", "timestamp": 2.0, "after_history_length": 1, "step": 3},
        {"token": "<dead/>", "timestamp": 4.0, "after_history_length": 3, "step": 5, field: value},
    ]
    with pytest.raises(ValueError, match="history marker"):
        TrainingSourceReader(payload)


def test_death_markers_roundtrip_pt_without_entering_numeric_history(tmp_path):
    from common.policy.data.compiled_cache import CompiledCacheReader, CompiledShardCache
    from scripts.convert_fflogs.cache.cache_writer import write_compiled_cache_stream
    from scripts.convert_fflogs.source.source_reader import TrainingSourceReader

    torch, payload, _, bank, builder = _batch_components()
    payload["history_markers"] = [
        {"token": "<dead/>", "timestamp": 2.0, "after_history_length": 1, "step": 3},
        {"token": "<dead/>", "timestamp": 4.0, "after_history_length": 3, "step": 5},
    ]
    source_reader = TrainingSourceReader(payload)
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    cache = tmp_path / "markers.compiled.pt"
    written = write_compiled_cache_stream(
        cache, source, signature={}, config_dependencies={}, reader=source_reader,
        sample_batches=(builder.build_batch(source_reader, start, min(start + 2, source_reader.num_samples))
                        for start in range(0, source_reader.num_samples, 2)),
        num_samples=source_reader.num_samples, vocab_signature=(), shard_size=2, history_bank=bank,
    )
    stored = torch.load(cache, weights_only=False)
    assert stored["history_markers"] == tuple(payload["history_markers"])
    assert set(stored["history_bank"]) == set(bank)
    restored = CompiledCacheReader(stored, cache_path=cache, shard_cache=CompiledShardCache())
    for index in range(source_reader.num_samples):
        expected = source_reader.history_markers(index)
        assert written.history_markers(index) == restored.history_markers(index) == expected
        assert restored.sample(index).keys() == builder.build(source_reader, index).keys()
    assert restored.history_markers(1) == ()
    assert len(restored.history_markers(2)) == len(restored.history_markers(3)) == 1
    assert len(restored.history_markers(4)) == 2
    restored.history_markers(4)[0]["token"] = "changed"
    assert restored.history_markers(4) == tuple(payload["history_markers"])

    for invalid in (None, ({**payload["history_markers"][0], "after_history_length": 2},)):
        corrupted = dict(stored, history_markers=invalid)
        with pytest.raises(TypeError if invalid is None else ValueError, match="history marker"):
            CompiledCacheReader(corrupted, cache_path=cache, shard_cache=CompiledShardCache())
    corrupted = dict(stored)
    corrupted.pop("history_markers")
    with pytest.raises(KeyError, match="history_markers"):
        CompiledCacheReader(corrupted, cache_path=cache, shard_cache=CompiledShardCache())


def test_training_reads_visible_death_tokens_and_relocates_random_history_crops(tmp_path):
    import torch

    from common.policy.data import ActionSpace, Normalizer
    from tests.training._common_fixtures import write_test_compiled_cache
    from training import TrainingCollator, TrainingDataset

    payload = _batch_source()
    for sample in payload["samples"]:
        sample["label"]["quality_labels"] = []
        for skill in sample["context"]["skill_history_context"]:
            skill["max_charges"] = 1
    payload["history_markers"] = [
        {"token": "<dead/>", "timestamp": 0.0, "after_history_length": 0, "step": 1},
        {"token": "<dead/>", "timestamp": 2.0, "after_history_length": 1, "step": 3},
        {"token": "<dead/>", "timestamp": 2.1, "after_history_length": 1, "step": 3},
        {"token": "<dead/>", "timestamp": 3.0, "after_history_length": 2, "step": 4},
        {"token": "<dead/>", "timestamp": 4.0, "after_history_length": 3, "step": 5},
    ]
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    normalizer = Normalizer()
    options = dict(normalizer=normalizer, int_dtype=torch.int32, float_dtype=torch.float32)
    write_test_compiled_cache(source, payload, tmp_path / "cache", options)
    options.update(cache_dir=tmp_path / "cache", expected_action_space=ActionSpace.from_job_tag("black_mage"))
    dataset = TrainingDataset([source], context_capacity=32, history_reset_keep=1, **options)
    collator = TrainingCollator(int_dtype=torch.int32, index_dtype=torch.int64, float_dtype=torch.float32)
    samples = dataset.__getitems__([4, 2, 0, 1, 4])
    batch = collator(samples)
    assert batch["history_dead_positions"].dtype == torch.int64
    assert batch["history_dead_mask"].dtype == torch.bool
    assert [positions[mask].tolist() for positions, mask in zip(
        batch["history_dead_positions"], batch["history_dead_mask"], strict=True,
    )] == [[0, 1, 1, 2, 3], [0, 1, 1], [0], [0], [0, 1, 1, 2, 3]]
    assert samples[1]["history_length"] == samples[3]["history_length"] == 1
    assert samples[2]["history_length"] == 0

    class KeepOne:
        def random(self):
            return 0.0

        def randint(self, low, high):
            assert (low, high) == (1, 2)
            return 1

    cropped = TrainingCollator(
        history_truncation_enabled=True, history_truncation_probability=1.0, rng=KeepOne(),
        int_dtype=torch.int32, index_dtype=torch.int64, float_dtype=torch.float32,
    )([samples[0]])
    assert cropped["history_lengths"].tolist() == [1]
    assert cropped["history_dead_positions"].tolist() == [[0, 1]]
    assert cropped["history_dead_mask"].tolist() == [[True, True]]
    assert samples[0]["history_length"] == 3
    assert cropped["history_bank_skill_ids"] is batch["history_bank_skill_ids"]

    # 相同动作 cursor 的下一样本新收到死亡时可触发分块重置；乱序读取不改变结果。
    tight = TrainingDataset([source], context_capacity=14, history_reset_keep=1, **options)
    assert [sample["history_length"] for sample in tight.__getitems__([4, 3, 4, 3])] == [1, 3, 1, 3]
    tight_batch = collator([tight[4]])
    assert tight_batch["history_dead_positions"].tolist() == [[0, 1]]
    assert tight_batch["history_dead_mask"].tolist() == [[True, True]]

    bounded = TrainingDataset([source], context_capacity=32, history_reset_keep=1, max_history=1, **options)
    assert collator([bounded[4]])["history_dead_positions"].tolist() == [[0, 1]]
    current_only = TrainingDataset([source], context_capacity=32, history_reset_keep=1, max_history=0, **options)
    no_actions = collator([current_only[4], current_only[0]])
    assert no_actions["history_lengths"].tolist() == [0, 0]
    assert no_actions["history_dead_positions"].tolist() == [[0], [0]]
    assert no_actions["history_dead_mask"].tolist() == [[True], [True]]


def _assert_compiled_equal(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.shape == right.shape
        # FP32/低精度的符号零也须保留，不能只依赖浮点 ==。
        assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_compiled_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_compiled_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("float_dtype_name", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("normalize", [False, True])
def test_history_skill_features_keep_fp32_through_collation_and_context_encoding(
    float_dtype_name, normalize,
):
    """低精度缓存也必须保留技能数值差异，并能直接进入正式上下文编码。"""
    import torch

    from common.policy.data import Normalizer, SkillVocab
    from common.policy.data.context_encoding import ContextEncoder
    from common.policy.data.context_fields import ENCODING_BANK_FIELDS
    from common.policy.data.normalization import NormalizerConfig
    from scripts.convert_fflogs.source.source_reader import TrainingSourceReader
    from scripts.convert_fflogs.training.history_bank import build_history_bank
    from scripts.convert_fflogs.training.sample_builder import TrainingSampleBuilder
    from training.data.collator import TrainingCollator

    payload = _batch_source()
    # 这两个威力值在 FP16/BF16 原料阶段就会合并，事后转回 FP32 无法恢复。
    potencies = (1000.125, 1000.25)
    for sample in payload["samples"]:
        for index, row in enumerate(sample["context"]["skill_history_context"]):
            row["max_charges"] = 1
            if index < len(potencies):
                row["potency"] = potencies[index]
    reader = TrainingSourceReader(payload)
    float_dtype = getattr(torch, float_dtype_name)
    normalizer_config = NormalizerConfig()
    normalizer = Normalizer(normalizer_config)
    normalizer.ensure_job_resources(reader.job_tag)
    bank = build_history_bank(
        reader, torch=torch, normalizer=normalizer if normalize else None,
        skill_vocab=SkillVocab.build_from_job_tag(reader.job_tag),
        skill_feature_names=reader.skill_feature_names, int_dtype=torch.int32, float_dtype=float_dtype,
    )
    assert bank["skill_features"].dtype == torch.float32
    assert bank["skill_potencies"].dtype == bank["cumulative_dot_potencies"].dtype == float_dtype
    builder = TrainingSampleBuilder(
        torch=torch, int_dtype=torch.int32, float_dtype=float_dtype, num_actions=reader.num_actions,
        history_bank=bank, ranking=reader.ranking, annotation_status=reader.annotation_status,
    )
    samples = builder.build_batch(reader, 0, 4)
    for sample in samples:
        sample["scene_abs_values"] = sample.pop("scene_vectors")
        sample["history_bank_id"] = "fp32_skill_features"
        sample["history_bank_action_keys"] = bank["action_keys"]
        for field in ENCODING_BANK_FIELDS:
            sample[f"history_bank_{field.name}"] = bank[field.name]
    batch = TrainingCollator(
        int_dtype=torch.int32, index_dtype=torch.int64, float_dtype=float_dtype,
    )(samples)
    encoded = ContextEncoder(
        normalizer, reader.schema, layout=reader.schema.state_layout(reader.action_keys),
    ).encode(batch)
    assert encoded["history_skill_features"].dtype == torch.float32
    assert torch.count_nonzero(encoded["history_skill_features"][0]) == 0
    expected = torch.tensor(potencies, dtype=torch.float32)
    if normalize:
        expected /= normalizer_config.current_potency_max
    potency_index = reader.skill_feature_names.index("potency")
    actual = encoded["history_skill_features"][3, :2, potency_index]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[0] != actual[1]


@pytest.mark.parametrize("int_dtype_name", ["int32", "int64"])
@pytest.mark.parametrize("float_dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("shard_size", [1, 3, 7])
def test_batch_compile_matches_each_sample_bitwise_across_shards_and_dtypes(
    int_dtype_name, float_dtype_name, shard_size,
):
    """逐样本构建为基准，跨 shard 的批量向量化不能改变任何字段或 dtype。"""
    torch, _payload, reader, _bank, builder = _batch_components(int_dtype_name, float_dtype_name)
    expected = [builder.build(reader, index) for index in range(reader.num_samples)]
    actual = []
    for start in range(0, reader.num_samples, shard_size):
        actual.extend(builder.build_batch(reader, start, min(start + shard_size, reader.num_samples)))
    _assert_compiled_equal(expected, actual)

    assert actual[0]["history_length"] == 0
    assert torch.equal(actual[0]["current_state_delta_reset_mask"], ~actual[0]["current_state_null_mask"])
    assert actual[3]["current_state_delta_values"][:2].tolist() == [400.0, 8200.0]
    assert actual[3]["current_state_delta_reset_mask"][:2].tolist() == [False, True]
    assert actual[2]["current_state_null_mask"][1].item() is True
    assert actual[2]["current_state_delta_values"][1].item() == 0.0
    assert actual[1]["quality_label_levels"].tolist() == [1, 2, 3]
    assert [sample["metadata"]["quality_label_status"] for sample in actual] == [
        "synthetic", "attributed_label", "no_attributed_label", "attributed_label",
        "attributed_label", "synthetic", "attributed_label",
    ]
    assert actual[0]["scene_vectors"].dtype == torch.float32
    assert actual[0]["scene_vectors"][0, :3].tolist() == [1800.0, 2400.0, 600.0]
    assert actual[0]["scene_types"].unique().numel() == 3
    assert actual[0]["action_values"].dtype == getattr(torch, float_dtype_name)
    assert actual[0]["scene_types"].dtype == getattr(torch, int_dtype_name)


@pytest.mark.parametrize("int_dtype_name", ["int32", "int64"])
@pytest.mark.parametrize("float_dtype_name", ["float32", "bfloat16"])
def test_batch_compiled_tensor_mutation_keeps_other_samples_reader_source_and_bank_isolated(
    int_dtype_name, float_dtype_name,
):
    """行视图允许共享 storage，但不能重叠；场景与监督 tensor 也不得串改样本。"""
    torch, payload, reader, bank, builder = _batch_components(int_dtype_name, float_dtype_name)
    samples = builder.build_batch(reader, 0, 3) + builder.build_batch(reader, 3, reader.num_samples)
    unchanged_samples = deepcopy(samples)
    unchanged_payload, unchanged_bank = deepcopy(payload), deepcopy(bank)
    current_before = reader.current_state_matrix(1, dtype=torch.float32)
    scene_before = reader.scene_tokens(1, float_dtype=torch.float32, int_dtype=getattr(torch, int_dtype_name))

    for value in samples[1].values():
        if isinstance(value, torch.Tensor) and value.numel():
            flattened = value.view(-1)
            flattened[0] = not bool(flattened[0]) if value.dtype == torch.bool else flattened[0] + 1
    samples[1]["action_keys"].append("caller_only")
    samples[1]["quality_labels"].append({"severity": "caller_only"})

    for index in (0, 2, 3, 4, 5, 6):
        _assert_compiled_equal(samples[index], unchanged_samples[index])
    assert payload == unchanged_payload
    _assert_compiled_equal(bank, unchanged_bank)
    current_after = reader.current_state_matrix(1, dtype=torch.float32)
    for field in ("base_values", "base_null_mask", "skill_state"):
        _assert_compiled_equal(getattr(current_before, field), getattr(current_after, field))
    _assert_compiled_equal(scene_before, reader.scene_tokens(
        1, float_dtype=torch.float32, int_dtype=getattr(torch, int_dtype_name),
    ))


def test_cached_current_batch_returns_fresh_tensors_and_public_reader_still_reads_current_payload():
    """内部批量缓存不能通过返回值被改写，原公开单样本读取仍反映调用方修改。"""
    torch, payload, reader, _bank, _builder = _batch_components()
    first = reader.current_state_batch(0, 3, dtype=torch.float32)
    expected = [reader.current_state_matrix(index, dtype=torch.float32) for index in range(3)]
    first.base_values[0, 1] = -123.0
    first.base_null_mask[0, 1] = True
    first.skill_state[0, 0] = 1 - first.skill_state[0, 0]
    repeated = reader.current_state_batch(0, 3, dtype=torch.float32)

    for field in ("base_values", "base_null_mask", "skill_state"):
        _assert_compiled_equal(getattr(repeated, field), torch.cat([getattr(item, field) for item in expected]))
    payload["samples"][0]["context"]["current_state_context"]["tokens"][0]["player_state"][1] = 9123.0
    assert reader.current_state_matrix(0, dtype=torch.float32).base_values[0, 1].item() == 9123.0


def test_batch_compile_empty_ranges_do_not_create_samples():
    _torch, _payload, reader, _bank, builder = _batch_components()
    assert builder.build_batch(reader, 0, 0) == []
    assert builder.build_batch(reader, reader.num_samples, reader.num_samples) == []

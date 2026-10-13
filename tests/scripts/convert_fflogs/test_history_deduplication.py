"""转换内部冻结历史复用、读取隔离与存储规模回归测试。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.convert_fflogs import build_training_samples
from scripts.convert_fflogs.source.source_reader import TrainingSourceReader
from scripts.convert_fflogs.training.history_bank import HistoryPrefixInterner
from tests.helpers import build_test_scene_context, targetable_window_token


def _context(length: int) -> dict[str, object]:
    return {
        "skill_history_context": [
            {"skill_key": f"skill_{index}", "kind": index % 2, "potency": 100.0 + index}
            for index in range(length)
        ],
        "state_history_context": {
            "tokens": [
                {group: [float(index + offset) for offset in range(136)]
                 for group in ("player_state", "self_status", "enemy_status", "cooldowns")}
                for index in range(length)
            ],
            "execution_metrics": [
                {"cumulative_dot_potency": float(index * 60), "executed_gcd_count": index + 1}
                for index in range(length)
            ],
        },
        "current_state_context": {"tokens": [{"player_state": [float(length)]}]},
    }


def _history_lists(context):
    return (
        context["skill_history_context"],
        context["state_history_context"]["tokens"],
        context["state_history_context"]["execution_metrics"],
    )


def _intern(interner, context, index=0):
    interner.intern(context, sample_index=index, fight_id="history_regression")


def _reader(context):
    # 这里只验证读取的容器语义，无需创建无关的完整训练 schema。
    reader = TrainingSourceReader.__new__(TrainingSourceReader)
    reader._samples = [{"context": context}]
    reader._num_samples = 1
    return reader


def test_history_interner_preserves_content_and_lists_for_equal_and_growing_prefixes():
    """等长观测与一次新增多条均保留完整内容，只共享已冻结行。"""
    interner = HistoryPrefixInterner()
    previous = None
    for sample_index, length in enumerate((0, 2, 2, 5)):
        context = _context(length)
        original = deepcopy(context)
        original_lists = _history_lists(context)
        original_rows = [list(rows) for rows in original_lists]
        _intern(interner, context, sample_index)

        assert context == original
        for field_index, rows in enumerate(_history_lists(context)):
            assert type(rows) is list
            assert rows is original_lists[field_index]
            previous_rows = _history_lists(previous)[field_index] if previous else []
            if previous is not None:
                assert rows is not previous_rows
            for row_index, row in enumerate(rows):
                expected = (previous_rows[row_index] if row_index < len(previous_rows)
                            else original_rows[field_index][row_index])
                assert row is expected
        previous = context


@pytest.mark.parametrize("field_index", [0, 1, 2])
@pytest.mark.parametrize("change", ["value", "reorder"])
def test_history_interner_rejects_changed_or_reordered_prefixes(field_index, change):
    """三个历史通道的旧行均不可被回填或重排，失败不应部分改写新观测。"""
    interner = HistoryPrefixInterner()
    _intern(interner, _context(2))
    context = _context(3)
    rows = _history_lists(context)[field_index]
    if change == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    elif field_index == 0:
        rows[0]["potency"] += 1.0
    elif field_index == 1:
        rows[0]["player_state"][0] += 1.0
    else:
        rows[0]["cumulative_dot_potency"] += 1.0
    original_row_ids = [[id(row) for row in field] for field in _history_lists(context)]

    with pytest.raises(ValueError, match="history prefix changed"):
        _intern(interner, context, 1)

    assert [[id(row) for row in field] for field in _history_lists(context)] == original_row_ids


def test_history_interner_rejects_backwards_length():
    interner = HistoryPrefixInterner()
    _intern(interner, _context(2))
    with pytest.raises(ValueError, match="history length moved backwards"):
        _intern(interner, _context(1), 1)


@pytest.mark.parametrize("field_index", [0, 1, 2])
def test_history_interner_requires_matching_history_lengths(field_index):
    context = _context(2)
    _history_lists(context)[field_index].pop()
    with pytest.raises(ValueError, match="history length mismatch"):
        _intern(HistoryPrefixInterner(), context)


def test_history_interner_retains_one_large_state_vector_per_unique_history_row():
    """重复观测增加前缀引用，不能重复保留每行的大状态向量。"""
    interner = HistoryPrefixInterner()
    contexts = []
    lengths = [length for length in range(33) for _ in range(3)]
    for sample_index, length in enumerate(lengths):
        context = _context(length)
        _intern(interner, context, sample_index)
        contexts.append(context)

    state_rows = [row for context in contexts for row in _history_lists(context)[1]]
    assert len(state_rows) == sum(lengths)
    assert len({id(row) for row in state_rows}) == max(lengths)
    vectors = [vector for row in state_rows for vector in row.values()]
    assert len({id(vector) for vector in vectors}) == max(lengths) * 4
    for field_index in range(3):
        assert len({id(_history_lists(context)[field_index]) for context in contexts}) == len(contexts)
        assert len({id(row) for context in contexts for row in _history_lists(context)[field_index]}) == max(lengths)


def test_history_delta_copies_rows_by_default_and_borrows_only_on_request():
    context = _context(3)
    original = deepcopy(context)
    reader = _reader(context)
    copied_skills, copied_states = reader.history_delta(0, 1)
    borrowed_skills, borrowed_states = reader.history_delta(0, 1, copy_rows=False)

    for source, copied, borrowed in (
        (context["skill_history_context"], copied_skills, borrowed_skills),
        (context["state_history_context"]["tokens"], copied_states, borrowed_states),
    ):
        assert type(copied) is list and type(borrowed) is list
        assert copied is not source and borrowed is not source
        assert copied == borrowed == source[1:]
        assert all(row is not source[index + 1] for index, row in enumerate(copied))
        assert all(row is source[index + 1] for index, row in enumerate(borrowed))
        copied[0]["caller_only"] = True
        borrowed.pop()
    assert context == original
    # 默认契约维持浅复制，嵌套向量仍属于源状态行。
    assert copied_states[0]["player_state"] is context["state_history_context"]["tokens"][1]["player_state"]


def test_history_length_reads_length_without_iterating_or_copying_rows():
    class LengthOnlyHistory(list):
        def __iter__(self):
            raise AssertionError("history_length must not iterate or copy history rows")

        def __getitem__(self, _key):
            raise AssertionError("history_length must not access or slice history rows")

    context = _context(3)
    context["skill_history_context"] = LengthOnlyHistory(context["skill_history_context"])
    assert _reader(context).history_length(0) == 3


def _fight_payload():
    duration = 12.0
    return {
        "fight_id": "shared_history_regression", "job_tag": "black_mage", "duration": duration,
        "scene_context": build_test_scene_context(targetable_tokens=[
            targetable_window_token(0.0, duration, targetable=True, segment_kind="combat"),
        ]),
        "actions": [
            {"time_offset": effect, "request_time_offset": request, "action_key": action,
             "time_gap": gap, "fight_remaining": duration - effect, "anchor": "combat"}
            for effect, request, action, gap in (
                (3.444, 0.0, "fire_iii", 3.444),
                (3.444, 3.444, "high_thunder", 0.0),
                (6.0, 6.0, "blizzard_iii", 2.556),
                (9.5, 9.5, "fire_iii", 3.5),
            )
        ],
    }


def test_deduplicated_real_replay_matches_public_output_and_keeps_bridge_observations_independent(
    cs_backend, cs_skill_book,
):
    """真实含 policy wait 的回放逐值相同，桥接仍返回独立 Python 容器。"""
    fight = _fight_payload()
    cs_backend.init(actual_base_gcd=2.46, fight_remaining=12.0)
    baseline = build_training_samples(cs_backend, cs_skill_book, fight)
    cs_backend.init(actual_base_gcd=2.46, fight_remaining=12.0)
    shared = build_training_samples(cs_backend, cs_skill_book, fight, deduplicate_history=True)
    assert shared == baseline
    assert "ogcd_wait" in shared["resolved_sequence"]

    reused_rows = 0
    contexts = [sample["context"] for sample in shared["samples"]]
    for previous, current in zip(contexts, contexts[1:]):
        for previous_rows, current_rows in zip(_history_lists(previous), _history_lists(current), strict=True):
            assert previous_rows is not current_rows
            for old, new in zip(previous_rows, current_rows):
                assert old is new
                reused_rows += 1
    assert reused_rows > 0

    def observe():
        return cs_backend.observe_at(10.0, format="vector", next_observation_timestamp=10.0).context

    first, second = observe(), observe()
    original = deepcopy(second)
    assert first == second
    for first_rows, second_rows in zip(_history_lists(first), _history_lists(second), strict=True):
        assert first_rows and first_rows is not second_rows
        assert first_rows[0] is not second_rows[0]
        first_rows[0]["caller_only"] = True
    first["state_history_context"]["tokens"][0]["player_state"].append(-123.0)
    assert second == original
    assert observe() == original
    assert shared == baseline


def test_public_replay_default_preserves_mutation_isolation_between_samples(cs_backend, cs_skill_book):
    cs_backend.init(actual_base_gcd=2.46, fight_remaining=12.0)
    payload = build_training_samples(cs_backend, cs_skill_book, _fight_payload())
    contexts = [sample["context"] for sample in payload["samples"] if sample["context"]["skill_history_context"]]
    assert len(contexts) > 1
    unchanged = deepcopy(contexts[1:])
    for rows in _history_lists(contexts[0]):
        rows[0]["caller_only"] = True
    contexts[0]["state_history_context"]["tokens"][0]["player_state"].append(-123.0)
    assert contexts[1:] == unchanged

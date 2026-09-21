"""候选 canonical 顺序与训练样本重排测试。"""

from __future__ import annotations

import torch
import yaml

from common.policy.data.candidate_order import (
    candidate_permutation,
    load_candidate_order,
)
from training.data.dataset import TrainingDataset


def test_load_candidate_order_reads_one_based_mapping(tmp_path):
    path = tmp_path / "candidate_order.yaml"
    path.write_text(
        yaml.safe_dump(
            {"candidate_order": {1: "fire_iv", 2: "fire_iii"}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert load_candidate_order(
        path,
        expected_action_keys=("fire_iii", "fire_iv"),
    ) == ("fire_iv", "fire_iii")


def test_candidate_order_rejects_incomplete_or_unknown_mapping(tmp_path):
    path = tmp_path / "candidate_order.yaml"
    path.write_text(
        "candidate_order:\n  1: fire_iv\n  2: unknown\n",
        encoding="utf-8",
    )

    try:
        load_candidate_order(path, expected_action_keys=("fire_iii", "fire_iv"))
    except ValueError as exc:
        message = str(exc)
        assert "candidate_order must match" in message
        assert "update candidate_order.yaml" in message
        assert "re-stat the ranking" in message
        assert "missing=fire_iii" in message
        assert "unknown=unknown" in message
    else:
        raise AssertionError("invalid candidate order was accepted")


def test_training_dataset_reorders_aligned_candidate_fields_and_label():
    dataset = TrainingDataset.__new__(TrainingDataset)
    dataset._candidate_action_keys = ("fire_iv", "fire_iii")
    sample = {
        "candidate_action_keys": ["fire_iii", "fire_iv"],
        "candidate_skill_ids": torch.tensor([11, 22]),
        "candidate_skill_features": torch.tensor([[1.0], [2.0]]),
        "candidate_values": torch.tensor([1.0, 2.0]),
        "candidate_state_vectors": torch.tensor([[3.0], [4.0]]),
        "candidate_state_null_mask": torch.tensor([[False], [True]]),
        "candidate_legal_mask": torch.tensor([True, False]),
        "candidate_invalid_reasons": ["first", "second"],
        "label_index": 0,
    }

    reordered = dataset._reorder_sample(sample)

    assert reordered["candidate_action_keys"] == ["fire_iv", "fire_iii"]
    assert reordered["candidate_skill_ids"].tolist() == [22, 11]
    assert reordered["candidate_values"].tolist() == [2.0, 1.0]
    assert reordered["candidate_state_vectors"].tolist() == [[4.0], [3.0]]
    assert reordered["candidate_state_null_mask"].tolist() == [[True], [False]]
    assert reordered["candidate_legal_mask"].tolist() == [False, True]
    assert reordered["candidate_invalid_reasons"] == ["second", "first"]
    assert reordered["label_index"] == 1


def test_candidate_permutation_maps_target_order_to_source_indices():
    assert candidate_permutation(
        ("a", "b", "c"),
        ("c", "a", "b"),
    ) == (2, 0, 1)


def test_candidate_permutation_reports_missing_and_unknown_skills():
    try:
        candidate_permutation(("a", "c"), ("a", "b"))
    except ValueError as exc:
        assert str(exc) == (
            "source and target candidate orders differ: "
            "missing=b; unknown=c"
        )
    else:
        raise AssertionError("candidate order drift was accepted")

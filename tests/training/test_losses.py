"""训练主损失与可选辅助项的组合边界。"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from training.loop.loss import AuxiliaryLoss, compose_training_loss
from training.loop.losses.primary import primary_loss


def test_primary_loss_owns_cross_entropy_calculation():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]], requires_grad=True)
    batch = {"label_index": torch.tensor([0, 1])}
    output = {"logits": logits}

    loss = primary_loss(output, batch)
    expected = F.cross_entropy(logits, batch["label_index"])
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert logits.grad is not None


def test_primary_loss_promotes_bf16_logits_to_float32():
    logits = torch.tensor([[2.0, 0.0]], dtype=torch.bfloat16, requires_grad=True)
    batch = {"label_index": torch.tensor([0])}

    loss = primary_loss({"logits": logits}, batch)
    assert loss.dtype is torch.float32
    expected = F.cross_entropy(logits.float(), batch["label_index"])
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert logits.grad is not None


def test_auxiliary_loss_can_be_added_or_removed_without_changing_primary():
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
    auxiliary = torch.tensor(3.0, requires_grad=True)
    output = {"logits": logits}
    batch = {"label_index": torch.tensor([0])}

    baseline = compose_training_loss(output, batch)
    assert baseline.total is baseline.primary
    assert baseline.auxiliary == {}

    term = AuxiliaryLoss("example_loss", 0.25, lambda _output, _batch: auxiliary)
    composed = compose_training_loss(output, batch, (term,))
    assert composed.primary.item() == pytest.approx(baseline.primary.item())
    assert composed.total.item() == pytest.approx(baseline.primary.item() + 0.75)
    assert composed.auxiliary == {"example_loss": auxiliary}
    composed.total.backward()
    assert logits.grad is not None
    assert auxiliary.grad.item() == pytest.approx(0.25)


def test_disabled_auxiliary_is_not_evaluated_and_keeps_zero_metric():
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)

    def unexpected(_output, _batch):
        pytest.fail("disabled auxiliary loss must not read its inputs")

    composed = compose_training_loss(
        {"logits": logits},
        {"label_index": torch.tensor([0])},
        (AuxiliaryLoss("unused_loss", 0.0, unexpected),),
    )
    assert composed.total is composed.primary
    assert composed.auxiliary["unused_loss"].item() == 0.0


def test_auxiliary_metric_names_must_be_unique():
    output = {"logits": torch.zeros((1, 2))}
    batch = {"label_index": torch.tensor([0])}
    terms = (
        AuxiliaryLoss("duplicate", 0.0, lambda _output, _batch: torch.tensor(0.0)),
        AuxiliaryLoss("duplicate", 0.0, lambda _output, _batch: torch.tensor(0.0)),
    )
    with pytest.raises(ValueError, match="duplicate auxiliary loss metric"):
        compose_training_loss(output, batch, terms)

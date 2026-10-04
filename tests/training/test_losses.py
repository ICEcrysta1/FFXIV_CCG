"""训练主损失与可选辅助项的组合边界。"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from training.config import (
    ActionQualityLossConfig,
    ValuePreferenceConfig,
    load_run_config,
)
from training.config.action_quality import parse_action_quality_loss_config
from training.loop.loss import (
    AuxiliaryLoss,
    compose_training_loss,
    configured_auxiliary_losses,
)
from training.loop.losses.action_quality import action_quality_sample_weights
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

    term = AuxiliaryLoss("example_loss", 0.25, lambda _output, _batch, _weights: auxiliary)
    composed = compose_training_loss(output, batch, (term,))
    assert composed.primary.item() == pytest.approx(baseline.primary.item())
    assert composed.total.item() == pytest.approx(baseline.primary.item() + 0.75)
    assert composed.auxiliary == {"example_loss": auxiliary}
    composed.total.backward()
    assert logits.grad is not None
    assert auxiliary.grad.item() == pytest.approx(0.25)


def test_disabled_auxiliary_is_not_evaluated_and_keeps_zero_metric():
    logits = torch.tensor([[0.0, 0.0]], requires_grad=True)

    def unexpected(_output, _batch, _weights):
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
        AuxiliaryLoss("duplicate", 0.0, lambda _output, _batch, _weights: torch.tensor(0.0)),
        AuxiliaryLoss("duplicate", 0.0, lambda _output, _batch, _weights: torch.tensor(0.0)),
    )
    with pytest.raises(ValueError, match="duplicate auxiliary loss metric"):
        compose_training_loss(output, batch, terms)


def test_quality_config_uses_current_manifest_for_curve_and_severity():
    config = load_run_config("config/models/black_mage/artzip/config.yaml")
    assert config.action_quality_loss == ActionQualityLossConfig(
        enabled=True,
        scale=0.6,
        exponent=4.0,
        severity_weights=(0.25, 0.5, 1.0),
    )


def test_quality_weights_only_discount_tagged_actions_and_take_strongest_error():
    batch = {
        "label_index": torch.zeros(4, dtype=torch.long),
        "quality_label_levels": torch.tensor([[1, 2], [3, 0], [0, 0], [3, 0]]),
        "quality_label_mask": torch.tensor([[True, True], [True, False], [False, False], [True, False]]),
        "quality_annotation_available": torch.tensor([True, True, False, True]),
        "source_quality": torch.tensor([0.0, 0.0, -1.0, 0.9777]),
    }
    config = ActionQualityLossConfig(enabled=True)
    weights = action_quality_sample_weights(batch, config)
    expected_top = 1.0 - torch.exp(-torch.tensor((0.9777 / 0.6) ** 4))
    assert weights.tolist() == pytest.approx([0.5, 0.0, 1.0, expected_top.item()])


def test_quality_multiple_labels_take_maximum_configured_weight():
    batch = {
        "label_index": torch.tensor([0]),
        "quality_label_levels": torch.tensor([[1, 3]]),
        "quality_label_mask": torch.tensor([[True, True]]),
        "quality_annotation_available": torch.tensor([True]),
        "source_quality": torch.tensor([0.0]),
    }
    config = ActionQualityLossConfig(
        enabled=True, severity_weights=(0.8, 0.5, 0.3)
    )
    assert action_quality_sample_weights(batch, config).item() == pytest.approx(0.2)


def test_quality_empty_labels_preserve_regular_bc_weight():
    batch = {
        "label_index": torch.tensor([0, 1]),
        "quality_label_levels": torch.empty(2, 0, dtype=torch.int32),
        "quality_label_mask": torch.empty(2, 0, dtype=torch.bool),
        "quality_annotation_available": torch.tensor([False, True]),
        "source_quality": torch.tensor([-1.0, 0.5]),
    }
    assert action_quality_sample_weights(
        batch, ActionQualityLossConfig(enabled=True)
    ).tolist() == [1.0, 1.0]


def test_quality_config_rejects_invalid_curve_or_missing_severity():
    with pytest.raises(ValueError, match="function"):
        parse_action_quality_loss_config(
            {"action_quality_loss": {"percentile_decay": {"function": "unknown"}}},
            {"severity_weights": {"minor": 0.25, "medium": 0.5, "major": 1.0}},
        )
    with pytest.raises(TypeError, match="severity_weights.major"):
        parse_action_quality_loss_config(
            {"action_quality_loss": {"percentile_decay": {
                "function": "exp_negative_power", "scale": 0.6, "exponent": 4
            }}},
            {"severity_weights": {"minor": 0.25, "medium": 0.5}},
        )


def test_quality_loss_gates_ce_and_value_preference_together():
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)
    batch = {
        "label_index": torch.tensor([0, 0]),
        "quality_label_levels": torch.tensor([[3], [0]]),
        "quality_label_mask": torch.tensor([[True], [False]]),
        "quality_annotation_available": torch.tensor([True, False]),
        "source_quality": torch.tensor([0.0, -1.0]),
        "action_values": torch.tensor([[2.0, 1.0], [2.0, 1.0]]),
        "action_legal_mask": torch.ones(2, 2, dtype=torch.bool),
    }
    terms = configured_auxiliary_losses(
        value_preference=ValuePreferenceConfig(enabled=True, loss_weight=0.05)
    )
    loss = compose_training_loss(
        {"logits": logits}, batch, terms,
        action_quality=ActionQualityLossConfig(enabled=True),
    )
    assert loss.primary.item() == pytest.approx(torch.log(torch.tensor(2.0)).item() / 2)
    assert loss.auxiliary["value_preference_loss"].item() > 0
    loss.total.backward()
    assert torch.equal(logits.grad[0], torch.zeros(2))
    assert not torch.equal(logits.grad[1], torch.zeros(2))


@pytest.mark.parametrize("unknown_quality", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_quality_weights_ignore_unknown_percentile_for_untagged_target(unknown_quality):
    batch = {
        "label_index": torch.tensor([0, 0]),
        "quality_label_levels": torch.tensor([[3], [0]]),
        "quality_label_mask": torch.tensor([[True], [False]]),
        "quality_annotation_available": torch.tensor([True, False]),
        "source_quality": torch.tensor([0.5, unknown_quality]),
    }
    weights = action_quality_sample_weights(batch, ActionQualityLossConfig(enabled=True))
    expected = -torch.expm1(-torch.tensor((0.5 / 0.6) ** 4))
    assert weights.tolist() == pytest.approx([expected.item(), 1.0])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["empty", "untagged", "mixed"])
def test_quality_weights_do_not_read_device_scalars(device, dtype, layout):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    levels = torch.tensor([[1, 2], [3, 0], [0, 0]], device=device)
    mask = torch.tensor([[True, True], [True, False], [False, False]], device=device)
    if layout == "empty":
        levels, mask = levels[:, :0], mask[:, :0]
    elif layout == "untagged":
        mask.zero_()
    batch = {
        "label_index": torch.zeros(3, dtype=torch.long, device=device),
        "quality_label_levels": levels,
        "quality_label_mask": mask,
        "quality_annotation_available": torch.tensor([True, True, False], device=device),
        "source_quality": torch.tensor([0.0, 0.0, float("nan")], dtype=dtype, device=device),
    }

    class RejectScalarReads(TorchDispatchMode):
        """阻止 bool/item 等设备标量读取，覆盖 CPU 与 CUDA 的同一计算路径。"""

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten._local_scalar_dense.default:
                pytest.fail("quality weighting must not read tensor scalars on the host")
            return func(*args, **(kwargs or {}))

    with RejectScalarReads():
        weights = action_quality_sample_weights(batch, ActionQualityLossConfig(enabled=True))
    assert weights.dtype == torch.float32
    expected = [0.5, 0.0, 1.0] if layout == "mixed" else [1.0, 1.0, 1.0]
    assert weights.tolist() == pytest.approx(expected)

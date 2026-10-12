"""行为克隆监控的真实输入、输出复用与统计边界回归测试。"""

from __future__ import annotations

import pytest
import torch

from training.loop.training_loop import train_epoch, validate


class _ScalarRecorder:
    def __init__(self):
        self.values = []

    def add_scalar(self, tag, value, step):
        self.values.append((tag, value, step))


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.outputs = []

    def forward(self, batch):
        count = batch["label_index"].shape[0]
        margin = torch.log(torch.expm1(torch.tensor(float(count))))
        logits = torch.stack((torch.zeros(count), self.weight * margin.expand(count)), dim=1)
        self.outputs.append(logits)
        return {
            "logits": logits,
            "top1_accuracy": self.weight.detach() * (count / 4),
            "top3_accuracy": self.weight.detach(),
        }


def _batches(*counts):
    return [{"label_index": torch.zeros(count, dtype=torch.long)} for count in counts]


def test_step_scalars_average_all_samples_in_interval_and_flush_partial_interval():
    model = _Policy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    writer = _ScalarRecorder()
    train_epoch(
        model, _batches(3, 1, 2), optimizer, scheduler, torch.device("cpu"),
        tensorboard_writer=writer, tensorboard_log_every_steps=2, global_step_offset=8,
    )
    losses = [(value, step) for tag, value, step in writer.values if tag == "训练/记录区间/总损失"]
    assert [value for value, _ in losses] == pytest.approx([2.5, 2.0])
    assert [step for _, step in losses] == [10, 11]
    accuracies = [value for tag, value, _ in writer.values if tag == "训练/记录区间/Top-1准确率"]
    assert accuracies == pytest.approx([0.625, 0.5])


def test_step_learning_rate_is_the_rate_used_for_updates():
    model = _Policy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5 ** step)
    writer = _ScalarRecorder()
    train_epoch(
        model, _batches(3, 1), optimizer, scheduler, torch.device("cpu"),
        tensorboard_writer=writer, tensorboard_log_every_steps=2,
    )
    rates = [value for tag, value, _ in writer.values if tag == "训练/记录区间/本次更新学习率"]
    assert rates == pytest.approx([(0.2 * 3 + 0.1) / 4])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_monitor_receives_encoded_batch_gradients_before_update_and_actual_outputs():
    model = _Policy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    calls = []

    class Encoder:
        def encode(self, batch):
            encoded = {**batch, "encoded_marker": torch.tensor([17])}
            calls.append(("encode", encoded))
            return encoded

    class Monitor:
        def log_graph(self, batch):
            assert batch is calls[-1][1]
            assert batch["encoded_marker"].tolist() == [17]
            calls.append(("graph", batch))

        def log_gradients(self, global_step, grad_norm=None):
            assert model.weight.grad is not None
            assert model.weight.detach().item() == pytest.approx(1.0)
            assert grad_norm > 0
            calls.append(("gradient", global_step))

        def log_step(self, logits, *, global_step, labels=None):
            assert logits is model.outputs[-1]
            assert labels.tolist() == [0, 0]
            assert model.weight.detach().item() != pytest.approx(1.0)
            calls.append(("output", global_step))

    train_epoch(
        model, _batches(2), optimizer, scheduler, torch.device("cpu"),
        tensorboard_monitor=Monitor(), global_step_offset=6, context_encoder=Encoder(),
    )
    assert [name for name, _ in calls] == ["encode", "graph", "gradient", "output"]
    assert calls[-2:] == [("gradient", 7), ("output", 7)]
    assert len(model.outputs) == 1


def test_validation_monitor_collects_each_real_batch_and_ends_at_training_step():
    model = _Policy()
    observed = []

    class Monitor:
        def begin_validation(self):
            observed.append("begin")

        def update_validation(self, logits, labels):
            assert logits is model.outputs[-1]
            observed.append(labels.shape[0])

        def end_validation(self, *, global_step):
            observed.append(("end", global_step))

    validate(
        model, _batches(3, 1), torch.device("cpu"),
        tensorboard_monitor=Monitor(), global_step=29,
    )
    assert observed == ["begin", 3, 1, ("end", 29)]
    assert len(model.outputs) == 2


def test_invalid_direct_scalar_interval_is_rejected():
    model = _Policy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    with pytest.raises(ValueError, match="tensorboard_log_every_steps must be >= 1"):
        train_epoch(
            model, _batches(1), optimizer, scheduler, torch.device("cpu"),
            tensorboard_writer=_ScalarRecorder(), tensorboard_log_every_steps=0,
        )

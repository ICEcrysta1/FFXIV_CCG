"""GRPO 的混合优化器必须完整回滚矩阵动量、AdamW 状态与调度器。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import grpo.trainer as trainer_module
from common.policy.config import ModelConfig
from common.training.optimizer import build_optimizer
from common.training.optimizer_config import OptimizerConfig
from grpo.config import GrpoConfig, GrpoRunConfig
from grpo.trainer import _restore_grpo_rollback_state, _snapshot_optimizer_state


class _TinyPolicy(nn.Module):
    """保留生产参数路径的微型 Transformer，执行真实前向与反向更新。"""

    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=4, nhead=2, dim_feedforward=8, dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.scorer = nn.Linear(4, 2)

    def forward(self, inputs):
        return self.scorer(self.encoder(inputs))


@pytest.mark.parametrize("name", ["adamw", "muon"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grpo_training_builds_its_own_optimizer_and_closes_engine(tmp_path, monkeypatch, name, dtype):
    """正式入口使用 GRPO 参数建真实优化器，setup 中止时仍关闭共享引擎。"""
    config = GrpoRunConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path / "output",
        job_tag="black_mage",
        model=ModelConfig(d_model=4, n_heads=2, n_layers=1, ff_dim=8),
    )
    grpo = GrpoConfig(
        optimizer=OptimizerConfig(name=name, momentum=0.8, ns_steps=3),
        learning_rate=3e-5,
        weight_decay=0.04,
    )
    model = _TinyPolicy().to(dtype=dtype)
    model.config = config.model
    # 来源 checkpoint 故意声明相反算法及不同 LR/WD，热启动不能复用其优化器配置。
    checkpoint = {
        "run_config": {
            "optimizer": {"name": "adamw" if name == "muon" else "muon"},
            "learning_rate": 0.2,
            "weight_decay": 0.4,
        },
    }
    checkpoint_path = tmp_path / "source.pt"
    torch.save(checkpoint, checkpoint_path)
    scene_path = tmp_path / "scene.json"
    scene_path.write_text("{}", encoding="utf-8", newline="\n")
    backend = SimpleNamespace(
        model=model,
        data_spec=SimpleNamespace(job_tag="black_mage", num_candidates=2),
        input_contract=object(),
        checkpoint=checkpoint,
    )
    captured = {}

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            captured["close_count"] = 0

        def close(self):
            captured["close_count"] += 1

    class SetupComplete(Exception):
        """创建优化器后主动停止，不进入状态机或 rollout。"""

    def spy_builder(actual_model, optimizer_config, *, learning_rate, weight_decay):
        assert actual_model is model
        assert optimizer_config is grpo.optimizer
        assert learning_rate == grpo.learning_rate
        assert weight_decay == grpo.weight_decay
        optimizer = build_optimizer(
            actual_model, optimizer_config, learning_rate=learning_rate, weight_decay=weight_decay,
        )
        captured["optimizer"] = optimizer
        raise SetupComplete

    monkeypatch.setattr(trainer_module, "PyTorchPolicyBackend", lambda *_args, **_kwargs: backend)
    monkeypatch.setattr("scripts.autoregressive_replay.parallel.InProcessEngine", FakeSession)
    monkeypatch.setattr(trainer_module, "create_tensorboard_writer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer_module, "resolve_policy_cache_dir", lambda _job: tmp_path / "cache")
    monkeypatch.setattr(trainer_module, "build_optimizer", spy_builder)

    with pytest.raises(SetupComplete):
        trainer_module.run_grpo_training(
            config, grpo=grpo, checkpoint_path=checkpoint_path,
            raw_paths=(scene_path,), device_name="cpu",
        )

    optimizer = captured["optimizer"]
    assert captured["close_count"] == 1
    assert all(group["lr"] == grpo.learning_rate for group in optimizer.param_groups)
    assert all(group["weight_decay"] == grpo.weight_decay for group in optimizer.param_groups)
    if dtype == torch.bfloat16:
        assert optimizer.state_dict()["format"] == "fp32_master_v1"
        assert all(parameter.dtype == torch.float32 for group in optimizer.param_groups for parameter in group["params"])
    elif name == "adamw":
        assert type(optimizer) is torch.optim.AdamW
    if name == "muon":
        assert [group["optimizer_name"] for group in optimizer.param_groups] == ["muon", "adamw"]
        assert optimizer.param_groups[0]["momentum"] == grpo.optimizer.momentum
        assert optimizer.param_groups[0]["ns_steps"] == grpo.optimizer.ns_steps


def _step(model, optimizer, scheduler, seed):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn((2, 3, 4), generator=generator).to(next(model.parameters()).dtype)
    labels = torch.randint(2, (6,), generator=generator)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    nn.functional.cross_entropy(logits.reshape(-1, 2), labels).backward()
    optimizer.step()
    scheduler.step()


def _assert_state_equal(actual, expected):
    """递归比较嵌套原生状态，避免只看权重而漏掉第二套优化器动量。"""
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, Mapping):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_state_equal(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_state_equal(actual_value, expected_value)
    else:
        assert actual == expected


def _native_states(state):
    native = state.get("optimizer", state)
    if native.get("format") == "muon_adamw_v1":
        return [native[name]["state"] for name in ("muon", "adamw")]
    return [native["state"]]


@pytest.mark.parametrize("source", ["disk", "first_iteration"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("name", ["adamw", "muon"])
def test_grpo_rollback_restores_optimizer_and_next_update(tmp_path, source, dtype, name):
    torch.manual_seed(113)
    model = _TinyPolicy().to(dtype=dtype)
    baseline = deepcopy(model)
    grpo = GrpoConfig(
        optimizer=OptimizerConfig(name=name),
        learning_rate=0.01,
        weight_decay=0.02,
        warmup_steps=0,
    )
    optimizer = build_optimizer(
        model, grpo.optimizer, learning_rate=grpo.learning_rate, weight_decay=grpo.weight_decay,
    )
    baseline_optimizer = build_optimizer(
        baseline, grpo.optimizer, learning_rate=grpo.learning_rate, weight_decay=grpo.weight_decay,
    )
    schedule = lambda step: 1.0 / (step + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    baseline_scheduler = torch.optim.lr_scheduler.LambdaLR(baseline_optimizer, schedule)

    if source == "disk":
        # 后续轮次磁盘 checkpoint 已有两种优化器状态；首轮则从空状态回滚。
        for seed in (21, 22):
            _step(model, optimizer, scheduler, seed)
            _step(baseline, baseline_optimizer, baseline_scheduler, seed)
    expected_model = deepcopy(model.state_dict())
    expected_optimizer = _snapshot_optimizer_state(optimizer)
    expected_scheduler = deepcopy(scheduler.state_dict())
    for state in _native_states(expected_optimizer):
        assert bool(state) == (source == "disk")

    checkpoint_path = None
    if source == "disk":
        checkpoint_path = tmp_path / "latest.pt"
        torch.save({
            "model_state_dict": expected_model,
            "optimizer_state_dict": expected_optimizer,
            "scheduler_state_dict": expected_scheduler,
        }, checkpoint_path)

    for seed in (101, 102):
        _step(model, optimizer, scheduler, seed)
    assert scheduler.state_dict() != expected_scheduler
    assert any(
        not torch.equal(value, expected_model[name])
        for name, value in model.state_dict().items()
    )
    assert all(_native_states(optimizer.state_dict()))

    _restore_grpo_rollback_state(
        checkpoint_path=checkpoint_path,
        initial_model_state={} if source == "disk" else expected_model,
        initial_optimizer_state=None if source == "disk" else expected_optimizer,
        initial_scheduler_state=None if source == "disk" else expected_scheduler,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    _assert_state_equal(model.state_dict(), expected_model)
    _assert_state_equal(optimizer.state_dict(), expected_optimizer)
    _assert_state_equal(scheduler.state_dict(), expected_scheduler)

    # 实际重试更新必须与未发生失败的基线逐值相等，覆盖恢复后的分组引用与 LR。
    for seed in (31, 32):
        _step(model, optimizer, scheduler, seed)
        _step(baseline, baseline_optimizer, baseline_scheduler, seed)
        _assert_state_equal(model.state_dict(), baseline.state_dict())
        _assert_state_equal(optimizer.state_dict(), baseline_optimizer.state_dict())
        _assert_state_equal(scheduler.state_dict(), baseline_scheduler.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 检查回滚快照设备")
def test_first_grpo_rollback_snapshot_copies_master_weights_to_cpu():
    model = _TinyPolicy().to(device="cuda", dtype=torch.bfloat16)
    optimizer = build_optimizer(model, OptimizerConfig(), learning_rate=1e-6, weight_decay=0.0)
    snapshot = _snapshot_optimizer_state(optimizer)
    assert all(weight.device.type == "cpu" for weight in snapshot["master_weights"])
    before = snapshot["master_weights"][0].clone()
    optimizer.param_groups[0]["params"][0].add_(1)
    torch.testing.assert_close(snapshot["master_weights"][0], before, rtol=0, atol=0)

"""验证小更新的累计、低精度梯度生命周期及新旧 checkpoint 的完整恢复。"""

from copy import deepcopy

import pytest
import torch
from torch import nn

from common.training.master_weights import FP32MasterOptimizer
from common.training.optimizer import _MuonAdamW, _parameter_groups, build_optimizer
from common.training.optimizer_config import OptimizerConfig


class _TinyModel(nn.Module):
    def __init__(self, dtype=torch.bfloat16, device="cpu"):
        super().__init__()
        layer = nn.Module()
        layer.linear1 = nn.Linear(2, 2, bias=False, dtype=dtype, device=device)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([layer])
        self.scorer = nn.Linear(2, 1, bias=False, dtype=dtype, device=device)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(0.02)


def _build(model, name, *, lr=1e-5, weight_decay=0.0):
    return build_optimizer(
        model, OptimizerConfig(name=name), learning_rate=lr, weight_decay=weight_decay,
    )


def _native(model, name, *, lr=1e-5, weight_decay=0.0):
    """独立构造修复前的原生优化器，分别充当舍入复现和 FP32 参考。"""
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    muon, adamw = _parameter_groups(model)
    return _MuonAdamW(
        torch.optim.Muon([muon], lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw"),
        torch.optim.AdamW([adamw], lr=lr, weight_decay=weight_decay),
    )


def _step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.float().sum() for parameter in model.parameters()).backward()
    optimizer.step()


def _assert_model_equal(actual, expected):
    for parameter, reference in zip(actual.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(parameter, reference.to(parameter.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("name", ["adamw", "muon"])
def test_sub_bf16_updates_accumulate_like_fp32_instead_of_stalling(name):
    model = _TinyModel()
    stalled = deepcopy(model)
    reference = deepcopy(model).float()
    initial = deepcopy(model)
    optimizer = _build(model, name)
    old_optimizer = _native(stalled, name)
    reference_optimizer = _native(reference, name)
    for _ in range(128):
        _step(model, optimizer)
        _step(stalled, old_optimizer)
        _step(reference, reference_optimizer)
        _assert_model_equal(model, reference)
    _assert_model_equal(stalled, initial)
    assert all(not torch.equal(p, original) for p, original in zip(model.parameters(), initial.parameters()))
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert all(p.dtype == torch.float32 and p.grad is None for g in optimizer.param_groups for p in g["params"])
    assert all(
        value.dtype == torch.float32
        for state in optimizer.state.values() for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )


@pytest.mark.parametrize("name", ["adamw", "muon"])
def test_checkpoint_preserves_updates_smaller_than_bf16_and_scheduler(tmp_path, name):
    model = _TinyModel()
    optimizer = _build(model, name)
    schedule = lambda step: 1.0 / (1.0 + step / 100)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    for _ in range(3):
        _step(model, optimizer)
        scheduler.step()
    state = optimizer.state_dict()
    assert any(
        not torch.equal(master, parameter.float())
        for master, parameter in zip(state["master_weights"], model.parameters())
    )
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": state, "scheduler": scheduler.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored = _TinyModel()
    # 与 BC 一致：构造优化器后才恢复模型和优化器。
    restored_optimizer = _build(restored, name)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, schedule)
    restored.load_state_dict(saved["model"])
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_scheduler.load_state_dict(saved["scheduler"])
    for _ in range(20):
        _step(model, optimizer)
        _step(restored, restored_optimizer)
        scheduler.step()
        restored_scheduler.step()
        _assert_model_equal(model, restored)
        for actual, expected in zip(
            restored_optimizer.state_dict()["master_weights"], optimizer.state_dict()["master_weights"],
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert restored_scheduler.get_last_lr() == scheduler.get_last_lr()


@pytest.mark.parametrize("name", ["adamw", "muon"])
def test_legacy_state_migrates_from_restored_weights_and_upcasts_moments(name, caplog):
    legacy = _TinyModel()
    native = _native(legacy, name, lr=0.01)
    _step(legacy, native)
    saved = deepcopy(native.state_dict())
    model = _TinyModel()
    optimizer = _build(model, name)
    model.load_state_dict(legacy.state_dict())
    optimizer.load_state_dict(saved)
    assert "旧 optimizer checkpoint" in caplog.text
    reference = deepcopy(legacy).float()
    reference_optimizer = _native(reference, name)
    # 原生 load 对同 dtype 的 step 张量可能保留引用；参考优化器必须使用独立状态。
    reference_optimizer.load_state_dict(deepcopy(saved))
    for _ in range(3):
        _step(model, optimizer)
        _step(reference, reference_optimizer)
        _assert_model_equal(model, reference)
    assert all(
        value.dtype == torch.float32
        for state in optimizer.state.values() for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )


@pytest.mark.parametrize("set_to_none", [False, True])
def test_zero_grad_closure_and_missing_gradient_preserve_native_semantics(set_to_none):
    model = _TinyModel()
    model.scorer.float()  # 同时覆盖原本为 FP32 的参数，不能重复建立主权重。
    optimizer = _build(model, "adamw", weight_decay=0.1)
    before = deepcopy(model)
    calls = []

    def closure():
        calls.append(1)
        loss = model.scorer.weight.square().sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert calls == [1] and loss is not None
    assert torch.equal(model.encoder.layers[0].linear1.weight, before.encoder.layers[0].linear1.weight)
    assert not torch.equal(model.scorer.weight, before.scorer.weight)
    assert len(optimizer.state_dict()["master_weights"]) == 1
    _step(model, optimizer)
    optimizer.zero_grad(set_to_none=set_to_none)
    assert all(p.grad is None if set_to_none else torch.count_nonzero(p.grad) == 0 for p in model.parameters())


@pytest.mark.parametrize("corruption", ["format", "name", "count", "dtype", "shape"])
def test_corrupt_master_weights_are_rejected_before_mutation(corruption):
    model = _TinyModel()
    optimizer = _build(model, "adamw")
    state = deepcopy(optimizer.state_dict())
    if corruption == "format":
        state["format"] = "fp32_master_v999"
    elif corruption == "name":
        state["parameter_contract"][0]["parameters"][0]["name"] = "wrong.weight"
    elif corruption == "count":
        state["master_weights"].pop()
    elif corruption == "dtype":
        state["master_weights"][0] = state["master_weights"][0].bfloat16()
    else:
        state["master_weights"][0] = torch.zeros(1)
    before = deepcopy(model)
    with pytest.raises(ValueError, match="FP32 master optimizer"):
        optimizer.load_state_dict(state)
    _assert_model_equal(model, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 验证正式 BF16 训练路径")
@pytest.mark.parametrize("name", ["adamw", "muon"])
def test_cuda_bf16_master_updates_match_fp32_reference(name):
    model = _TinyModel(device="cuda")
    reference = deepcopy(model).float()
    optimizer = _build(model, name)
    reference_optimizer = _native(reference, name)
    for _ in range(16):
        _step(model, optimizer)
        _step(reference, reference_optimizer)
        _assert_model_equal(model, reference)
    assert isinstance(optimizer, FP32MasterOptimizer)

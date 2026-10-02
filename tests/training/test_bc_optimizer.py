"""共享混合优化器的参数边界、原生更新一致性与 BC 前后向回归。"""

from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import math

import pytest
import torch
from torch import nn

from common.policy.config import ModelConfig
from common.policy.data import DataSpec
from common.policy.model import CandidateTransformerModel
from common.training.optimizer import build_optimizer
from common.training.optimizer_config import OptimizerConfig


def _config(name="muon") -> OptimizerConfig:
    return OptimizerConfig(name=name)


def _build_optimizer(model, config=None, *, weight_decay=0.03):
    return build_optimizer(
        model, config or _config(), learning_rate=0.02, weight_decay=weight_decay,
    )


def _model(*, num_kv_heads=1, dtype=torch.float32) -> CandidateTransformerModel:
    spec = DataSpec(
        job_tag="black_mage", num_candidates=2, state_dim=3, scene_dim=2,
        skill_feature_dim=2, num_scene_types=1,
        candidate_action_keys=("first", "second"),
        skill_feature_names=("kind", "potency"),
    )
    config = ModelConfig(
        d_model=8, pair_embedding_dim=4, n_layers=1, n_heads=2,
        num_kv_heads=num_kv_heads, ff_dim=12, transformer_activation="swiglu",
        dropout=0.0, full_attention_residuals=True,
    )
    return CandidateTransformerModel(spec, config, vocab_size=3).to(dtype=dtype)


def _set_gradients(model: nn.Module, step: int) -> None:
    """固定非恒定梯度，覆盖矩阵方向变换及两类动量状态。"""
    for index, parameter in enumerate(model.parameters()):
        values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        parameter.grad = torch.sin(values + index + step * 0.37).to(parameter.dtype)


def _assert_models_equal(first: nn.Module, second: nn.Module) -> None:
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True,
    ):
        torch.testing.assert_close(first_parameter, second_parameter, rtol=0, atol=0)


def _native_pair(model: nn.Module, config: OptimizerConfig, parameter_names):
    named = dict(model.named_parameters())
    muon = torch.optim.Muon(
        [named[name] for name in parameter_names[0]], lr=0.02,
        weight_decay=0.03, momentum=config.momentum,
        nesterov=config.nesterov, ns_steps=config.ns_steps,
        adjust_lr_fn=config.adjust_lr_fn,
    )
    adamw = torch.optim.AdamW(
        [named[name] for name in parameter_names[1]],
        lr=0.02, weight_decay=0.03,
    )
    return muon, adamw


@pytest.mark.parametrize("num_kv_heads", (1, 2))
def test_muon_groups_only_transformer_projection_matrices(num_kv_heads):
    model = _model(num_kv_heads=num_kv_heads)
    optimizer = _build_optimizer(model)
    groups = {group["optimizer_name"]: group for group in optimizer.param_groups}
    muon_names = set(groups["muon"]["param_names"])
    attention = (
        {f"encoder.layers.0.self_attn.{name}.weight" for name in (
            "q_proj", "k_proj", "v_proj", "out_proj",
        )}
        if num_kv_heads == 1 else {
            "encoder.layers.0.self_attn.in_proj_weight",
            "encoder.layers.0.self_attn.out_proj.weight",
        }
    )
    assert muon_names == attention | {
        f"encoder.layers.0.{name}.weight" for name in ("linear1", "linear2", "gate_proj")
    }
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in model.parameters()}
    adamw_names = set(groups["adamw"]["param_names"])
    assert "encoder.attention_residual.pseudo_queries" in adamw_names
    assert "encoder.layers.0.norm1.weight" in adamw_names
    assert all(name in adamw_names for name, _ in model.named_parameters() if (
        name.startswith(("input_encoder.", "scorer.")) or name.endswith(".bias")
    ))


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_muon_step_and_scheduler_match_independent_native_optimizers(dtype):
    torch.manual_seed(71)
    actual = _model(dtype=dtype)
    expected = deepcopy(actual)
    config = _config()
    optimizer = _build_optimizer(actual, config)
    native_optimizers = _native_pair(
        expected, config, [group["param_names"] for group in optimizer.param_groups],
    )
    schedule = lambda step: 1.0 / (step + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    native_schedulers = [
        torch.optim.lr_scheduler.LambdaLR(native, schedule) for native in native_optimizers
    ]
    for step in range(3):
        _set_gradients(actual, step)
        _set_gradients(expected, step)
        optimizer.step()
        for native in native_optimizers:
            native.step()
        scheduler.step()
        for native_scheduler in native_schedulers:
            native_scheduler.step()
        _assert_models_equal(actual, expected)
        assert [group["lr"] for group in optimizer.param_groups] == [
            native.param_groups[0]["lr"] for native in native_optimizers
        ]
    optimizer.zero_grad(set_to_none=True)
    assert all(parameter.grad is None for parameter in actual.parameters())


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_muon_restore_continues_both_states_and_scheduler(dtype):
    torch.manual_seed(72)
    original = _model(dtype=dtype)
    optimizer = _build_optimizer(original)
    schedule = lambda step: 1.0 / (step + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    for step in range(2):
        _set_gradients(original, step)
        optimizer.step()
        scheduler.step()
    buffer = BytesIO()
    torch.save({
        "model": original.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = _model(dtype=dtype)
    restored.load_state_dict(saved["model"])
    restored_optimizer = _build_optimizer(restored)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, schedule)
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_scheduler.load_state_dict(saved["scheduler"])
    for step in range(2, 5):
        for model, current_optimizer, current_scheduler in (
            (original, optimizer, scheduler),
            (restored, restored_optimizer, restored_scheduler),
        ):
            _set_gradients(model, step)
            current_optimizer.step()
            current_scheduler.step()
        _assert_models_equal(original, restored)
        assert scheduler.get_last_lr() == restored_scheduler.get_last_lr()


def test_adamw_retains_native_optimizer_and_checkpoint_format():
    original = _model()
    expected = deepcopy(original)
    optimizer = _build_optimizer(original, _config("adamw"))
    native = torch.optim.AdamW(expected.parameters(), lr=0.02, weight_decay=0.03)
    assert type(optimizer) is torch.optim.AdamW
    assert set(optimizer.state_dict()) == {"state", "param_groups"}
    _set_gradients(original, 1)
    _set_gradients(expected, 1)
    optimizer.step()
    native.step()
    _assert_models_equal(original, expected)
    optimizer.load_state_dict(deepcopy(native.state_dict()))
    _set_gradients(original, 2)
    _set_gradients(expected, 2)
    optimizer.step()
    native.step()
    _assert_models_equal(original, expected)


@pytest.mark.parametrize("corruption", ("contract_name", "contract_shape", "native_names", "legacy"))
def test_muon_rejects_incompatible_optimizer_groups(corruption):
    optimizer = _build_optimizer(_model())
    state = deepcopy(optimizer.state_dict())
    if corruption == "contract_name":
        state["parameter_contract"][0]["parameters"][0]["name"] = "other.weight"
    elif corruption == "contract_shape":
        state["parameter_contract"][0]["parameters"][0]["shape"] = [1, 1]
    elif corruption == "native_names":
        state["adamw"]["param_groups"][0]["param_names"].reverse()
    else:
        state = torch.optim.AdamW(_model().parameters()).state_dict()
    with pytest.raises(ValueError, match="mismatch"):
        optimizer.load_state_dict(state)


def test_muon_keeps_weights_shared_with_input_encoder_in_adamw():
    model = _model()
    model.input_encoder.shared_projection = model.encoder.layers[0].linear1
    optimizer = _build_optimizer(model)
    target = model.encoder.layers[0].linear1.weight
    owners = [group["optimizer_name"] for group in optimizer.param_groups if any(
        parameter is target for parameter in group["params"]
    )]
    assert owners == ["adamw"]


def test_muon_fails_clearly_without_native_support(monkeypatch):
    monkeypatch.delattr(torch.optim, "Muon")
    with pytest.raises(RuntimeError, match="torch.optim.Muon"):
        _build_optimizer(_model())


def test_train_epoch_updates_muon_and_adamw_with_real_gradients():
    from training.loop.training_loop import train_epoch

    torch.manual_seed(73)
    model = _model()
    # 关闭衰减，确保检查到的权重变化确实来自前后向梯度。
    optimizer = _build_optimizer(model, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.8**step)
    batch = {
        "history_skill_ids": torch.ones((1, 1), dtype=torch.long),
        "history_skill_features": torch.zeros((1, 1, 2)),
        "history_state_vectors": torch.zeros((1, 1, 3)),
        "history_mask": torch.ones((1, 1), dtype=torch.bool),
        "candidate_skill_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "candidate_skill_features": torch.randn((1, 2, 2)),
        "candidate_state_vectors": torch.randn((1, 2, 3)),
        "candidate_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.zeros((1, 1, 2)),
        "scene_types": torch.zeros((1, 1), dtype=torch.long),
        "scene_mask": torch.ones((1, 1), dtype=torch.bool),
        "label_index": torch.zeros(1, dtype=torch.long),
    }
    matrices = (
        model.encoder.layers[0].linear1.weight,
        model.input_encoder.skill_embed.weight,
    )
    before = [parameter.detach().clone() for parameter in matrices]
    metrics = train_epoch(model, [batch], optimizer, scheduler, torch.device("cpu"))
    assert metrics and all(math.isfinite(value) for value in metrics.values())
    assert metrics["cross_entropy_loss"] > 0
    for original, parameter in zip(before, matrices, strict=True):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        assert not torch.equal(original, parameter)
    assert scheduler.get_last_lr() == [0.02 * 0.8, 0.02 * 0.8]

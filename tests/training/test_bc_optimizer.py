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
from common.policy.model import CausalPolicyModel
from common.training.optimizer import build_optimizer
from common.training.optimizer_config import OptimizerConfig


def _config(name="muon") -> OptimizerConfig:
    return OptimizerConfig(name=name)


def _build_optimizer(model, config=None, *, weight_decay=0.03):
    return build_optimizer(
        model, config or _config(), learning_rate=0.02, weight_decay=weight_decay,
    )


def _model(*, num_kv_heads=1, dtype=torch.float32) -> CausalPolicyModel:
    spec = DataSpec(
        job_tag="black_mage", num_actions=2, state_dim=3, scene_dim=2,
        skill_feature_dim=2, num_scene_types=1,
        action_keys=("first", "second"),
        skill_feature_names=("kind", "potency"),
        action_to_vocab_id=(1, 2),
        action_is_gcd=(True, True),
    )
    config = ModelConfig(
        d_model=8, pair_embedding_dim=4, n_layers=1, n_heads=2,
        num_kv_heads=num_kv_heads, ff_dim=12, transformer_activation="swiglu",
        dropout=0.0, full_attention_residuals=True,
    )
    return CausalPolicyModel(spec, config, vocab_size=3).to(dtype=dtype)


def _set_gradients(model: nn.Module, step: int) -> None:
    """固定非恒定梯度，覆盖矩阵方向变换及两类动量状态。"""
    for index, parameter in enumerate(model.parameters()):
        values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        parameter.grad = torch.sin(values + index + step * 0.37).to(parameter.dtype)


def _assert_models_equal(first: nn.Module, second: nn.Module) -> None:
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True,
    ):
        torch.testing.assert_close(
            first_parameter, second_parameter.to(first_parameter.dtype), rtol=0, atol=0,
        )


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
        name.startswith(("input_encoder.", "output_adapter.")) or name.endswith(".bias")
    ))


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_muon_step_and_scheduler_match_independent_native_optimizers(dtype):
    torch.manual_seed(71)
    actual = _model(dtype=dtype)
    # 低精度前向权重应等于独立 FP32 优化器权重的舍入结果。
    expected = deepcopy(actual).float()
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
        for parameter, reference in zip(actual.parameters(), expected.parameters(), strict=True):
            reference.grad = parameter.grad.float().clone()
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


def test_input_and_output_gradients_accumulate_on_one_embedding_parameter():
    model = _model()
    weight = model.input_encoder.skill_embed.weight
    input_loss = model.input_encoder.skill_embed(torch.tensor([1, 2])).sum()
    output_loss = model._score_current_hidden(torch.ones(1, model.config.d_model), {}).square().sum()
    input_grad = torch.autograd.grad(input_loss, weight, retain_graph=True)[0]
    output_grad = torch.autograd.grad(output_loss, weight, retain_graph=True)[0]
    (input_loss + output_loss).backward()
    torch.testing.assert_close(weight.grad, input_grad + output_grad)
    optimizer = _build_optimizer(model)
    assert sum(parameter is weight for group in optimizer.param_groups for parameter in group["params"]) == 1
    assert next(group["optimizer_name"] for group in optimizer.param_groups if any(parameter is weight for parameter in group["params"])) == "adamw"


@pytest.mark.parametrize("name", ["adamw", "muon"])
def test_shared_skill_embedding_has_exactly_one_fp32_master_across_restore(name):
    model = _model(dtype=torch.bfloat16)
    optimizer = _build_optimizer(model, _config(name))
    weight = model.input_encoder.skill_embed.weight
    assert sum(parameter is weight for parameter, _ in optimizer._master_pairs) == 1
    state = deepcopy(optimizer.state_dict())
    assert len(state["master_weights"]) == len(list(model.parameters()))
    embedding = [parameter for group in state["parameter_contract"] for parameter in group["parameters"]
                 if parameter["name"] == "input_encoder.skill_embed.weight"]
    assert len(embedding) == 1
    restored = _model(dtype=torch.bfloat16)
    restored_optimizer = _build_optimizer(restored, _config(name))
    restored_optimizer.load_state_dict(state)
    restored_weight = restored.input_encoder.skill_embed.weight
    assert sum(parameter is restored_weight for parameter, _ in restored_optimizer._master_pairs) == 1
    torch.testing.assert_close(weight, restored_weight, atol=0, rtol=0)


def test_muon_fails_clearly_without_native_support(monkeypatch):
    monkeypatch.delattr(torch.optim, "Muon")
    with pytest.raises(RuntimeError, match="torch.optim.Muon"):
        _build_optimizer(_model())


@pytest.mark.parametrize("name", ["adamw", "muon"])
@pytest.mark.parametrize("precision", ["float32", "bf16"])
def test_train_epoch_updates_both_parameter_groups_with_real_gradients(name, precision):
    from training.loop.training_loop import train_epoch

    if precision == "bf16" and not torch.cuda.is_available():
        pytest.skip("需要 CUDA 验证正式 BF16 训练路径")
    device = torch.device("cuda" if precision == "bf16" else "cpu")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    torch.manual_seed(73)
    model = _model(dtype=dtype).to(device)
    # 关闭衰减，确保检查到的权重变化确实来自前后向梯度。
    optimizer = _build_optimizer(model, _config(name), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.8**step)
    batch = {
        "history_skill_ids": torch.ones((1, 1), dtype=torch.long),
        "history_skill_features": torch.zeros((1, 1, 2)),
        "history_state_vectors": torch.zeros((1, 1, 3)),
        "history_state_null_mask": torch.zeros((1, 1, 3), dtype=torch.bool),
        "history_mask": torch.ones((1, 1), dtype=torch.bool),
        "current_state_vectors": torch.randn((1, 3)),
        "current_state_null_mask": torch.zeros((1, 3), dtype=torch.bool),
        "action_legal_mask": torch.ones((1, 2), dtype=torch.bool),
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
    metrics = train_epoch(model, [batch], optimizer, scheduler, device, precision)
    assert metrics and all(math.isfinite(value) for value in metrics.values())
    assert metrics["cross_entropy_loss"] > 0
    for original, parameter in zip(before, matrices, strict=True):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        assert not torch.equal(original, parameter)
    assert scheduler.get_last_lr() == [0.02 * 0.8] * (2 if name == "muon" else 1)
    if precision == "bf16":
        assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
        assert optimizer.state_dict()["format"] == "fp32_master_v1"

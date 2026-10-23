"""共享混合优化器的参数边界、原生更新一致性与 BC 前后向回归。"""

from __future__ import annotations

import math
from copy import deepcopy
from io import BytesIO

import pytest
import torch
from torch import nn

from common.policy.config import ModelConfig
from common.policy.data import DataSpec
from common.policy.model import CausalPolicyModel
from common.training.optimizer import build_optimizer
from common.training.optimizer_config import (
    AdamWConfig,
    MuonConfig,
    OptimizerConfig,
    SinkhornConfig,
    parse_optimizer_settings,
)
from common.training.sinkhorn import Sinkhorn


def _config(name="muon") -> OptimizerConfig:
    return OptimizerConfig(name=name)


def _build_optimizer(model, config=None, *, weight_decay=0.03):
    return build_optimizer(
        model, config or _config(), learning_rate=0.02, weight_decay=weight_decay,
    )


def _model(*, num_kv_heads=1, dtype=torch.float32, full_attention_residuals=True) -> CausalPolicyModel:
    spec = DataSpec(
        job_tag="black_mage", num_actions=2, base_state_dim=3, state_dim=(3) + 2 * (2), scene_dim=2,
        skill_feature_dim=2, num_scene_types=1,
        action_keys=("first", "second"),
        skill_feature_names=("kind", "potency"),
        action_to_vocab_id=(1, 2),
        action_is_gcd=(True, True),
    )
    config = ModelConfig(
        d_model=8, n_layers=1, n_heads=2,
        num_kv_heads=num_kv_heads, ff_dim=12, transformer_activation="swiglu",
        dropout=0.0, full_attention_residuals=full_attention_residuals,
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


def _native_pair(model: nn.Module, config: OptimizerConfig, parameter_groups):
    named = dict(model.named_parameters())
    groups = [{
        "params": [named[name] for name in group["param_names"]],
        "weight_decay": group["weight_decay"],
        "lr": group["lr"],
        "optimizer_name": group["optimizer_name"],
    } for group in parameter_groups]
    muon = torch.optim.Muon(
        [group for group in groups if group["optimizer_name"] == "muon"], lr=0.02,
        weight_decay=0.03, momentum=config.muon.momentum,
        nesterov=config.muon.nesterov, ns_steps=config.muon.ns_steps,
        adjust_lr_fn=config.muon.adjust_lr_fn, eps=config.muon.eps,
        ns_coefficients=config.muon.ns_coefficients,
    )
    adamw = torch.optim.AdamW(
        [group for group in groups if group["optimizer_name"] == "adamw"],
        lr=0.02, weight_decay=0.03, betas=config.adamw.betas,
        eps=config.adamw.eps, amsgrad=config.adamw.amsgrad,
    )
    sinkhorn = Sinkhorn(
        [group for group in groups if group["optimizer_name"] == "sinkhorn"], lr=0.02,
        momentum=config.sinkhorn.momentum, nesterov=config.sinkhorn.nesterov,
        steps=config.sinkhorn.iters, threshold=config.sinkhorn.threshold,
        eps=config.sinkhorn.eps, lr_scale=config.sinkhorn.lr_scale,
    )
    return muon, adamw, sinkhorn


@pytest.mark.parametrize("nesterov", (False, True))
def test_sinkhorn_momentum_masks_near_zero_rows_and_balances_l2(nesterov):
    parameter = nn.Parameter(torch.ones(3, 3))
    optimizer = Sinkhorn([parameter], lr=0.02, momentum=0.8, nesterov=nesterov, steps=3)
    expected = parameter.detach().clone()
    momentum = torch.zeros_like(parameter)
    for gradient in (
        torch.tensor([[3.0, -4.0, 2.0], [1.0, 2.0, -2.0], [1e-9, -1e-9, 0.0]]),
        torch.tensor([[-2.0, 1.0, 4.0], [4.0, -3.0, 1.0], [0.0, 0.0, 0.0]]),
    ):
        momentum = 0.8 * momentum + 0.2 * gradient
        update = 0.8 * momentum + 0.2 * gradient if nesterov else momentum.clone()
        row_norm = update.square().sum(dim=1, keepdim=True).sqrt()
        update[row_norm.squeeze(1) <= 1e-3 * row_norm.mean()] = 0.0
        # 三次迭代依次做行、列、行的 L2 归一化，末行被屏蔽后不能恢复。
        update = nn.functional.normalize(update, dim=1, eps=1e-20)
        update = nn.functional.normalize(update, dim=0, eps=1e-20)
        update = nn.functional.normalize(update, dim=1, eps=1e-20)
        expected -= 0.02 * 0.2 * math.sqrt(3) * update
        parameter.grad = gradient
        optimizer.step()
        torch.testing.assert_close(parameter, expected, rtol=0, atol=1e-7)
        torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], momentum)
        torch.testing.assert_close(parameter[2], torch.ones(3), rtol=0, atol=0)


def test_sinkhorn_zero_and_missing_gradients_do_not_change_weights():
    parameter = nn.Parameter(torch.ones(2, 3))
    missing = nn.Parameter(torch.ones(2, 3))
    optimizer = Sinkhorn([parameter, missing], lr=0.02)
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0)
    torch.testing.assert_close(missing, torch.ones_like(missing), rtol=0, atol=0)
    assert torch.isfinite(parameter).all()
    assert missing not in optimizer.state


@pytest.mark.parametrize("central", (False, True))
def test_auxiliary_learning_rates_keep_stage_ratio_after_primary_override(central):
    algorithm = {"name": "muon", "adamw": {"learning_rate": 2e-4}}
    policy = {"optimizers": {"bc": {"learning_rate": 1e-4, **algorithm}}} if central else {}
    legacy = {} if central else {"learning_rate": 1e-4, "optimizer": algorithm}
    settings = parse_optimizer_settings(
        policy, stage="bc", legacy=legacy,
        default_learning_rate=0.02, default_weight_decay=0.03, default_warmup_steps=0,
    )
    optimizer = build_optimizer(
        _model(), settings["optimizer"], learning_rate=2e-4, weight_decay=settings["weight_decay"],
    )
    rates = {group["optimizer_name"]: group["lr"] for group in optimizer.param_groups}
    assert rates == pytest.approx({"muon": 2e-4, "adamw": 4e-4, "sinkhorn": 2e-4})


@pytest.mark.parametrize("num_kv_heads", (1, 2))
@pytest.mark.parametrize("full_attention_residuals", (False, True))
@pytest.mark.parametrize("name", ("muon", "adamw"))
def test_optimizer_groups_projection_matrices_and_weight_decay(num_kv_heads, full_attention_residuals, name):
    model = _model(num_kv_heads=num_kv_heads, full_attention_residuals=full_attention_residuals)
    optimizer = _build_optimizer(model, _config(name))
    owners = {parameter_name: group for group in optimizer.param_groups
              for parameter_name in group["param_names"]}
    muon_names = {parameter_name for parameter_name, group in owners.items()
                  if group["optimizer_name"] == "muon"}
    attention = {f"encoder.layers.0.self_attn.{name}.weight" for name in (
        "q_a", "q_b", "kv_proj", "out_proj",
    )}
    matrix_names = attention | {
        f"encoder.layers.0.{name}.weight" for name in ("linear1", "linear2", "gate_proj")
    } | {
        f"input_encoder.{projection}.weight" for projection in (
            "skill_feat_proj", "state_proj", "state_null_proj", "state_reset_proj", "scene_proj.0",
        )
    }
    if not full_attention_residuals:
        matrix_names |= {f"encoder.residual_mix.{index}.fn" for index in range(2)}
    assert muon_names == (matrix_names if name == "muon" else set())
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in model.parameters()}
    embedding_names = {"input_encoder.skill_embed.weight", "input_encoder.dead_embed.weight",
                       "input_encoder.role_embed.weight", "output_head.weight"}
    exempt_names = set(embedding_names)
    exempt_names |= {parameter_name for parameter_name in owners if parameter_name.endswith("bias")}
    if full_attention_residuals:
        assert owners["encoder.attention_residual.pseudo_queries"]["optimizer_name"] == "adamw"
        assert owners["encoder.attention_residual.key_norms.0.weight"]["weight_decay"] == 0.03
    else:
        exempt_names |= {f"encoder.residual_mix.{index}.{field}"
                         for index in range(2) for field in ("base", "scale")}
    for parameter_name in exempt_names:
        assert owners[parameter_name]["optimizer_name"] == (
            "sinkhorn" if name == "muon" and parameter_name in embedding_names else "adamw"
        )
    for parameter_name in ("encoder.layers.0.norm1.weight", "encoder.layers.0.norm2.weight",
                           "encoder.norm.weight", "encoder.layers.0.self_attn.q_norm.weight",
                           "encoder.layers.0.self_attn.kv_norm.weight"):
        assert owners[parameter_name]["optimizer_name"] == "adamw"
        assert owners[parameter_name]["weight_decay"] == 0.03
    before = {parameter_name: parameter.detach().clone() for parameter_name, parameter in model.named_parameters()}
    for parameter_name, parameter in model.named_parameters():
        expected_decay = 0.0 if parameter_name in exempt_names else 0.03
        assert owners[parameter_name]["weight_decay"] == expected_decay
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    for parameter_name, parameter in model.named_parameters():
        expected_decay = 0.0 if parameter_name in exempt_names else 0.03
        torch.testing.assert_close(parameter, before[parameter_name] * (1 - 0.02 * expected_decay), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_muon_step_and_scheduler_match_independent_native_optimizers(dtype):
    torch.manual_seed(71)
    actual = _model(dtype=dtype)
    # 低精度前向权重应等于独立 FP32 优化器权重的舍入结果。
    expected = deepcopy(actual).float()
    config = OptimizerConfig(
        name="muon",
        muon=MuonConfig(learning_rate=0.01, momentum=0.8, nesterov=False,
                        ns_steps=3, adjust_lr_fn="original", eps=0.02,
                        ns_coefficients=(2.0, -1.5, 0.5)),
        adamw=AdamWConfig(learning_rate=0.003, weight_decay=0.07,
                          betas=(0.8, 0.9), eps=0.02, amsgrad=True),
        sinkhorn=SinkhornConfig(learning_rate=0.006, momentum=0.7, nesterov=False,
                               iters=3, threshold=1e-4, eps=1e-12, lr_scale=0.15),
    )
    optimizer = _build_optimizer(actual, config)
    expected_rates = {"muon": 0.02, "adamw": 0.006, "sinkhorn": 0.012}
    for group in optimizer.param_groups:
        assert group["lr"] == pytest.approx(expected_rates[group["optimizer_name"]])
        if group["optimizer_name"] == "adamw":
            assert group["weight_decay"] == (0.0 if any(name.endswith("bias") for name in group["param_names"]) else 0.07)
            assert group["betas"] == (0.8, 0.9)
            assert group["eps"] == 0.02
            assert group["amsgrad"] is True
    native_optimizers = _native_pair(
        expected, config, optimizer.param_groups,
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
            group["lr"] for native in native_optimizers for group in native.param_groups
        ]
    optimizer.zero_grad(set_to_none=True)
    assert all(parameter.grad is None for parameter in actual.parameters())


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_muon_restore_continues_both_states_and_scheduler(dtype):
    torch.manual_seed(72)
    original = _model(dtype=dtype)
    config = OptimizerConfig(
        name="muon", muon=MuonConfig(learning_rate=0.01),
        adamw=AdamWConfig(learning_rate=0.003, weight_decay=0.05,
                          betas=(0.8, 0.95), eps=1e-6, amsgrad=True),
        sinkhorn=SinkhornConfig(learning_rate=0.006, momentum=0.8, nesterov=False, iters=3),
    )
    optimizer = _build_optimizer(original, config)
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
    restored_optimizer = _build_optimizer(restored, config)
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
    config = OptimizerConfig(name="adamw", adamw=AdamWConfig(
        learning_rate=0.01, weight_decay=0.05, betas=(0.8, 0.95), eps=0.05, amsgrad=True,
    ))
    optimizer = _build_optimizer(original, config)
    expected_names = dict(expected.named_parameters())
    native = torch.optim.AdamW([{
        "params": [expected_names[name] for name in group["param_names"]],
        "weight_decay": group["weight_decay"],
    } for group in optimizer.param_groups], lr=0.02, weight_decay=0.03,
        betas=(0.8, 0.95), eps=0.05, amsgrad=True)
    assert type(optimizer) is torch.optim.AdamW
    assert all(group["lr"] == 0.02 for group in optimizer.param_groups)
    owners = {name: group for group in optimizer.param_groups for name in group["param_names"]}
    assert owners["encoder.layers.0.linear1.weight"]["weight_decay"] == 0.03
    assert owners["output_head.weight"]["weight_decay"] == 0.0
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


@pytest.mark.parametrize("corruption", ("contract_name", "contract_shape", "native_names", "native_groups", "legacy"))
def test_muon_rejects_incompatible_optimizer_groups(corruption):
    optimizer = _build_optimizer(_model())
    state = deepcopy(optimizer.state_dict())
    if corruption == "contract_name":
        state["parameter_contract"][0]["parameters"][0]["name"] = "other.weight"
    elif corruption == "contract_shape":
        state["parameter_contract"][0]["parameters"][0]["shape"] = [1, 1]
    elif corruption == "native_names":
        state["adamw"]["param_groups"][0]["param_names"].reverse()
    elif corruption == "native_groups":
        state["adamw"]["param_groups"].reverse()
    else:
        state = torch.optim.AdamW(_model().parameters()).state_dict()
    with pytest.raises(ValueError, match="mismatch"):
        optimizer.load_state_dict(state)


@pytest.mark.parametrize("alias", ("projection", "embedding", "head"))
def test_muon_deduplicates_shared_projections_and_exempts_embedding_or_head(alias):
    model = _model()
    target = model.encoder.layers[0].linear1.weight
    if alias == "projection":
        model.input_encoder.shared_projection = model.encoder.layers[0].linear1
    elif alias == "embedding":
        model.input_encoder.skill_embed.weight = target
    else:
        model.output_head.weight = target
    optimizer = _build_optimizer(model)
    owners = [group for group in optimizer.param_groups if any(
        parameter is target for parameter in group["params"]
    )]
    assert len(owners) == 1
    assert owners[0]["optimizer_name"] == ("muon" if alias == "projection" else "sinkhorn")
    assert owners[0]["weight_decay"] == (0.03 if alias == "projection" else 0.0)


def test_input_and_output_gradients_update_independent_sinkhorn_parameters():
    model = _model()
    weight = model.input_encoder.skill_embed.weight
    head = model.output_head.weight
    input_loss = model.input_encoder.skill_embed(torch.tensor([1, 2])).sum()
    output_loss = model._score_current_hidden(torch.ones(1, model.config.d_model), {}).square().sum()
    input_grad = torch.autograd.grad(input_loss, weight, retain_graph=True)[0]
    output_grad = torch.autograd.grad(output_loss, head, retain_graph=True)[0]
    assert torch.autograd.grad(output_loss, weight, allow_unused=True, retain_graph=True)[0] is None
    (input_loss + output_loss).backward()
    torch.testing.assert_close(weight.grad, input_grad)
    torch.testing.assert_close(head.grad, output_grad)
    optimizer = _build_optimizer(model)
    for parameter in (weight, head):
        assert sum(candidate is parameter for group in optimizer.param_groups for candidate in group["params"]) == 1
        assert next(group["optimizer_name"] for group in optimizer.param_groups
                    if any(candidate is parameter for candidate in group["params"])) == "sinkhorn"


@pytest.mark.parametrize("name", ["adamw", "muon"])
def test_input_embedding_and_independent_head_have_separate_fp32_masters_across_restore(name):
    model = _model(dtype=torch.bfloat16)
    optimizer = _build_optimizer(model, _config(name))
    weight = model.input_encoder.skill_embed.weight
    head = model.output_head.weight
    assert sum(parameter is weight for parameter, _ in optimizer._master_pairs) == 1
    assert sum(parameter is head for parameter, _ in optimizer._master_pairs) == 1
    assert next(master for parameter, master in optimizer._master_pairs if parameter is weight) is not next(
        master for parameter, master in optimizer._master_pairs if parameter is head)
    state = deepcopy(optimizer.state_dict())
    assert len(state["master_weights"]) == len(list(model.parameters()))
    embedding = [parameter for group in state["parameter_contract"] for parameter in group["parameters"]
                 if parameter["name"] == "input_encoder.skill_embed.weight"]
    assert len(embedding) == 1
    restored = _model(dtype=torch.bfloat16)
    restored_optimizer = _build_optimizer(restored, _config(name))
    restored_optimizer.load_state_dict(state)
    restored_weight = restored.input_encoder.skill_embed.weight
    restored_head = restored.output_head.weight
    assert sum(parameter is restored_weight for parameter, _ in restored_optimizer._master_pairs) == 1
    assert sum(parameter is restored_head for parameter, _ in restored_optimizer._master_pairs) == 1
    torch.testing.assert_close(weight, restored_weight, atol=0, rtol=0)
    torch.testing.assert_close(head, restored_head, atol=0, rtol=0)


def test_muon_fails_clearly_without_native_support(monkeypatch):
    monkeypatch.delattr(torch.optim, "Muon")
    with pytest.raises(RuntimeError, match="torch.optim.Muon"):
        _build_optimizer(_model())


@pytest.mark.parametrize("name", ["adamw", "muon"])
@pytest.mark.parametrize(("precision", "fp32_gradients"), [
    ("float32", False), ("bf16", False), ("bf16", True),
])
@pytest.mark.parametrize("full_attention_residuals", [False, True])
def test_train_epoch_updates_both_parameter_groups_with_real_gradients(
    name, precision, fp32_gradients, full_attention_residuals,
):
    from training.loop.epochs import train_epoch

    if precision == "bf16" and not torch.cuda.is_available():
        pytest.skip("需要 CUDA 验证正式 BF16 训练路径")
    device = torch.device("cuda" if precision == "bf16" else "cpu")
    dtype = torch.bfloat16 if precision == "bf16" and not fp32_gradients else torch.float32
    torch.manual_seed(73)
    model = _model(full_attention_residuals=full_attention_residuals).to(device)
    model.set_compute_dtype(dtype)
    # 关闭衰减，确保检查到的权重变化确实来自前后向梯度。
    optimizer = _build_optimizer(model, _config(name), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.8**step)
    batch = {
        "history_skill_ids": torch.ones((1, 1), dtype=torch.long),
        "history_skill_features": torch.zeros((1, 1, 2)),
        "history_state_vectors": torch.nn.functional.pad(torch.zeros((1, 1, 3)), (0, 4)),
        "history_state_null_mask": torch.zeros((1, 1, 3), dtype=torch.bool),
        "history_state_reset_mask": torch.ones((1, 1, 3), dtype=torch.bool),
        "history_dead_positions": torch.empty((1, 0), dtype=torch.long),
        "history_dead_mask": torch.zeros((1, 0), dtype=torch.bool),
        "history_mask": torch.ones((1, 1), dtype=torch.bool),
        "current_state_vectors": torch.nn.functional.pad(torch.randn((1, 3)), (0, 4)),
        "current_state_null_mask": torch.zeros((1, 3), dtype=torch.bool),
        "current_state_reset_mask": torch.zeros((1, 3), dtype=torch.bool),
        "action_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.zeros((1, 1, 2)),
        "scene_types": torch.zeros((1, 1), dtype=torch.long),
        "scene_mask": torch.ones((1, 1), dtype=torch.bool),
        "label_index": torch.zeros(1, dtype=torch.long),
    }
    matrices = (
        model.encoder.layers[0].linear1.weight,
        model.input_encoder.skill_embed.weight,
        model.output_head.weight,
    )
    if not full_attention_residuals:
        matrices += tuple(parameter for module in model.encoder.residual_mix for parameter in (module.base, module.scale))
    before = [parameter.detach().clone() for parameter in matrices]
    forward_dtypes = []
    hook = model.encoder.layers[0].linear1.register_forward_hook(
        lambda _module, _inputs, output: forward_dtypes.append(output.dtype)
    )

    class Monitor:
        def log_graph(self, batch):
            assert torch.is_autocast_enabled(device.type) == (precision == "bf16")

        def log_gradients(self, global_step, grad_norm=None):
            assert grad_norm.dtype == torch.float32
            assert all(parameter.grad.dtype == parameter.dtype for parameter in model.parameters()
                       if parameter.grad is not None)
            norms = [parameter.grad.float().norm() for parameter in model.parameters()
                     if parameter.grad is not None]
            # 旧 BF16 路径的范数及缩放系数有舍入误差；FP32 路径用更严格阈值。
            tolerance = 0.01 if dtype == torch.bfloat16 else 0.0001
            assert torch.stack(norms).norm() <= 1.0 + tolerance

        def log_step(self, *args, **kwargs):
            pass

    metrics = train_epoch(
        model, [batch], optimizer, scheduler, device, precision, tensorboard_monitor=Monitor(),
    )
    hook.remove()
    assert forward_dtypes == [torch.bfloat16 if precision == "bf16" else torch.float32]
    assert metrics and all(math.isfinite(value) for value in metrics.values())
    assert metrics["cross_entropy_loss"] > 0
    for original, parameter in zip(before, matrices, strict=True):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        assert torch.isfinite(parameter.grad).all()
        assert not torch.equal(original, parameter)
    assert scheduler.get_last_lr() == [0.02 * 0.8] * len(optimizer.param_groups)
    if precision == "bf16" and not fp32_gradients:
        assert all(parameter.dtype == (
            torch.float32 if parameter_name.startswith(('input_encoder.', 'encoder.residual_mix.')) else torch.bfloat16
        ) for parameter_name, parameter in model.named_parameters())
        assert optimizer.state_dict()["format"] == "fp32_master_v1"
    else:
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
        assert "master_weights" not in optimizer.state_dict()
        assert {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} == {
            id(parameter) for parameter in model.parameters()
        }


@pytest.mark.parametrize("name", ["adamw", "muon"])
def test_hyper_connection_coefficients_keep_fp32_updates_across_bf16_resume(name):
    model = _model(full_attention_residuals=False)
    model.set_compute_dtype(torch.bfloat16)
    mix = model.encoder.residual_mix
    optimizer = build_optimizer(
        model, _config(name), learning_rate=1e-4, weight_decay=0.0,
    )
    expected_names = {f"encoder.residual_mix.{index}.{field}"
                      for index in range(2) for field in ("fn", "base", "scale")}
    owners = {parameter_name: group for group in optimizer.param_groups
              for parameter_name in group["param_names"]}
    for parameter_name in expected_names:
        assert owners[parameter_name]["optimizer_name"] == (
            "muon" if name == "muon" and parameter_name.endswith(".fn") else "adamw"
        )
        assert owners[parameter_name]["lr"] == 1e-4
    assert all(parameter.dtype == torch.float32 for parameter in mix.parameters())
    original_scale = mix[0].scale.detach().clone()

    def step(current_model, current_optimizer):
        current_optimizer.zero_grad(set_to_none=True)
        for parameter in current_model.encoder.residual_mix.parameters():
            parameter.grad = torch.ones_like(parameter)
        current_optimizer.step()

    for _ in range(4):
        step(model, optimizer)
    # HC 参数直接保留 FP32，低精度主干的 master pair 不应接管这些系数。
    assert not torch.equal(mix[0].scale, original_scale)
    hc_parameters = {id(parameter) for parameter in mix.parameters()}
    assert all(id(parameter) not in hc_parameters for parameter, _ in optimizer._master_pairs)
    saved_model = deepcopy(model.state_dict())
    saved_optimizer = deepcopy(optimizer.state_dict())
    restored = _model(full_attention_residuals=False)
    restored.set_compute_dtype(torch.bfloat16)
    restored.load_state_dict(saved_model)
    restored_optimizer = build_optimizer(
        restored, _config(name), learning_rate=1e-4, weight_decay=0.0,
    )
    restored_optimizer.load_state_dict(saved_optimizer)
    for _ in range(50):
        step(model, optimizer)
        step(restored, restored_optimizer)
    for parameter, restored_parameter in zip(
        mix.parameters(), restored.encoder.residual_mix.parameters(), strict=True,
    ):
        torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)
    for first, second in zip(
        optimizer.state_dict()["master_weights"],
        restored_optimizer.state_dict()["master_weights"], strict=True,
    ):
        torch.testing.assert_close(first, second, rtol=0, atol=0)

"""独立动作头的复制初值、FP32 softcap、梯度与低精度连续恢复。"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from common.policy.config import ModelConfig
from common.policy.model import CausalPolicyModel, RepetitionConfig
from common.policy.model.repetition import apply_repetition_penalty
from common.training.master_weights import FP32MasterOptimizer
from common.training.optimizer import build_optimizer
from common.training.optimizer_config import OptimizerConfig
from tests.training._causal_fixtures import make_batch, make_checkpoint, make_data_spec


def _model(*, actions=(3, 1, 4), softcap=15.0, dtype=torch.float32, repetition=None):
    spec = make_data_spec(num_actions=len(actions), action_to_vocab_id=actions,
                          action_keys=tuple(f"action_{index}" for index in range(len(actions))),
                          action_is_gcd=(True,) * len(actions))
    config = ModelConfig(d_model=8, n_layers=2, n_heads=2, num_kv_heads=1, ff_dim=16,
                         dropout=0.0, transformer_activation="swiglu", logit_softcap=softcap)
    return CausalPolicyModel(spec, config, vocab_size=6, repetition=repetition).to(dtype=dtype)


def test_output_head_copies_action_row_order_without_extra_rng_or_shared_changes():
    models, states = [], []
    for actions in ((3, 1, 4), (5, 2)):
        torch.manual_seed(317)
        model = _model(actions=actions)
        models.append(model)
        states.append(torch.get_rng_state().clone())
        assert isinstance(model.output_head, torch.nn.Linear)
        assert model.output_head.bias is None
        assert model.output_head.weight.shape == (len(actions), model.config.d_model)
        selected = model.input_encoder.skill_embed.weight[torch.tensor(actions)]
        torch.testing.assert_close(model.output_head.weight, selected, atol=0, rtol=0)
    # 只改变动作表和独立头形状，不应消费额外初始化随机数或改变任何共享权重。
    assert torch.equal(states[0], states[1])
    shared = [dict(model.named_parameters()) for model in models]
    assert shared[0].keys() == shared[1].keys()
    for name in shared[0]:
        if name != "output_head.weight":
            torch.testing.assert_close(shared[0][name], shared[1][name], atol=0, rtol=0)


def test_output_head_storage_gradients_and_input_embedding_are_independent():
    model = _model()
    embedding, head = model.input_encoder.skill_embed.weight, model.output_head.weight
    assert head is not embedding and head.data_ptr() != embedding.data_ptr()
    hidden = torch.arange(1, 9, dtype=torch.float32).reshape(1, -1).requires_grad_(True)
    model.compute_action_logits(hidden).square().sum().backward()
    assert embedding.grad is None
    assert head.grad is not None and torch.count_nonzero(head.grad) > 0
    assert hidden.grad is not None and torch.count_nonzero(hidden.grad) > 0
    embedding_before = embedding.detach().clone()
    with torch.no_grad():
        head.add_(.1)
    assert torch.equal(embedding, embedding_before)
    head_before = head.detach().clone()
    model.zero_grad(set_to_none=True)
    model.input_encoder.skill_embed(torch.tensor([1, 3])).sum().backward()
    assert embedding.grad is not None and head.grad is None
    with torch.no_grad():
        embedding.add_(.2)
    assert torch.equal(head, head_before)


@pytest.mark.parametrize("scale", (15.0, 2.5))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_compute_action_logits_uses_fp32_cap_formula_and_independent_gradient(scale, dtype):
    model = _model(softcap=scale, dtype=dtype)
    reference = deepcopy(model)
    hidden = torch.linspace(-2, 2, 16, dtype=dtype).reshape(2, 8).requires_grad_(True)
    other = hidden.detach().clone().requires_grad_(True)
    actual = model.compute_action_logits(hidden)
    raw = F.linear(other, reference.output_head.weight).float()
    expected = scale * torch.tanh(raw / scale)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    target = torch.arange(1, actual.numel() + 1, dtype=torch.float32).reshape_as(actual)
    (actual * target).sum().backward()
    (expected * target).sum().backward()
    torch.testing.assert_close(hidden.grad, other.grad, atol=0, rtol=0)
    torch.testing.assert_close(model.output_head.weight.grad, reference.output_head.weight.grad, atol=0, rtol=0)
    assert model.input_encoder.skill_embed.weight.grad is None


def test_softcap_large_finite_logits_and_analytic_input_gradient():
    model = _model(actions=(1, 2), softcap=15.0)
    with torch.no_grad():
        model.output_head.weight.zero_()
        model.output_head.weight[0, 0] = 1
        model.output_head.weight[1, 0] = -1
    hidden = torch.zeros((5, 8), requires_grad=True)
    with torch.no_grad():
        hidden[:, 0] = torch.tensor([-1e20, -15., 0., 15., 1e20])
    actual = model.compute_action_logits(hidden)
    assert torch.isfinite(actual).all() and actual.abs().max() <= model.config.logit_softcap
    expected = 15 * torch.tanh(hidden[:, :1] / 15)
    torch.testing.assert_close(actual[:, :1], expected, atol=0, rtol=0)
    actual[:, 0].sum().backward()
    derivative = 1 - torch.tanh(hidden.detach()[:, 0] / 15).square()
    torch.testing.assert_close(hidden.grad[:, 0], derivative, atol=1e-7, rtol=1e-6)
    assert torch.count_nonzero(hidden.grad[:, 1:]) == 0
    assert torch.isfinite(model.output_head.weight.grad).all()


@pytest.mark.parametrize("scale", (1e-30, 1e38))
def test_extreme_valid_fp32_softcap_has_finite_outputs_and_hidden_gradients(scale):
    """合法 FP32 极端尺度既覆盖零点导数，也覆盖小幅非零输入。"""
    model = _model(actions=(1, 2), softcap=scale)
    with torch.no_grad():
        model.output_head.weight[0].fill_(.25)
        model.output_head.weight[1].fill_(-.125)
    hidden = torch.tensor([[0.] * 8, [.1] * 8], dtype=torch.float32, requires_grad=True)
    output = model.compute_action_logits(hidden)
    assert output.dtype == torch.float32 and torch.isfinite(output).all()
    assert output.abs().max() <= scale
    output.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert torch.count_nonzero(hidden.grad[0]) > 0
    assert model.output_head.weight.grad is not None
    assert torch.isfinite(model.output_head.weight.grad).all()


def test_repetition_is_applied_after_cap_and_action_legality_stays_external():
    repetition = RepetitionConfig(mode="blacklist", skills=("action_0",), penalty=7.0)
    model = _model(actions=(1, 2), repetition=repetition)
    with torch.no_grad():
        model.output_head.weight.fill_(1.)
    hidden = torch.ones((1, 8)) * 20
    batch = make_batch(model.data_spec)
    batch["history_action_keys"] = [["action_0", "action_0"]]
    capped = model.compute_action_logits(hidden)
    expected = apply_repetition_penalty(capped, batch, repetition)
    actual = model._score_current_hidden(hidden, batch)
    assert torch.equal(actual, expected)
    raw_penalized = apply_repetition_penalty(F.linear(hidden, model.output_head.weight), batch, repetition)
    assert not torch.equal(actual, 15 * torch.tanh(raw_penalized.float() / 15))
    assert actual[0, 0].item() == pytest.approx(capped[0, 0].item() - 7)
    illegal = {**batch, "action_legal_mask": torch.zeros_like(batch["action_legal_mask"])}
    assert torch.equal(model._score_current_hidden(hidden, illegal), actual)


@pytest.mark.parametrize("optimizer_name", ("adamw", "muon"))
def test_output_head_bf16_real_gradients_accumulate_master_and_continue_checkpoint(optimizer_name):
    model = _model(actions=(1, 2), dtype=torch.bfloat16)
    # 固定有限未饱和权重，让真正 softcap/CE 梯度产生小于 BF16 格点的更新。
    with torch.no_grad():
        model.output_head.weight.fill_(1.)
        model.output_head.weight[1].fill_(-.5)
    optimizer_config = OptimizerConfig(name=optimizer_name)
    optimizer = build_optimizer(model, optimizer_config, learning_rate=1e-5, weight_decay=0.0)
    assert isinstance(optimizer, FP32MasterOptimizer)
    groups = [group for group in optimizer.param_groups if "output_head.weight" in group["param_names"]]
    assert len(groups) == 1 and groups[0]["optimizer_name"] == "adamw" and groups[0]["lr"] == 1e-5
    weight = model.output_head.weight
    master = next(master for parameter, master in optimizer._master_pairs if parameter is weight)
    initial = weight.detach().clone()
    hidden = torch.linspace(.1, .8, 8, dtype=torch.bfloat16).reshape(1, 8)
    labels = torch.ones(1, dtype=torch.long)
    def step(current_model, current_optimizer):
        current_optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(current_model.compute_action_logits(hidden), labels).backward()
        gradient = current_model.output_head.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
        current_optimizer.step()
    previous = master.detach().clone()
    for _ in range(4):
        step(model, optimizer)
        assert not torch.equal(master, previous)
        previous = master.detach().clone()
    assert torch.equal(weight, initial)
    assert not torch.equal(master, initial.float())
    checkpoint = make_checkpoint(model.data_spec, model.config, model_state_dict=deepcopy(model.state_dict()))
    restored_config = CausalPolicyModel.checkpoint_model_config(checkpoint)
    restored = CausalPolicyModel(model.data_spec, restored_config, vocab_size=6).bfloat16()
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    restored_optimizer = build_optimizer(restored, optimizer_config, learning_rate=1e-5, weight_decay=0.0)
    restored_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    for _ in range(8):
        step(model, optimizer)
        step(restored, restored_optimizer)
    torch.testing.assert_close(model.output_head.weight, restored.output_head.weight, atol=0, rtol=0)
    for left, right in zip(optimizer.state_dict()["master_weights"], restored_optimizer.state_dict()["master_weights"], strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)

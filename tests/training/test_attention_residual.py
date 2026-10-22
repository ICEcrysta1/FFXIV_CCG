"""Full Attention Residual 聚合器的单元测试。"""

from __future__ import annotations

import pytest
import torch

import common.policy.model.layers.attention_residual as attention_residual_module
from common.policy.model.layers.attention_residual import FullAttentionResidual


def _stacked_reference(residual, sources, query_index):
    """独立按深度维堆叠计算论文公式，不使用待测的逐源聚合或重算入口。"""
    values = torch.stack(tuple(sources), dim=0)
    keys = residual.key_norms[query_index](values)
    logits = torch.einsum("d,sbtd->sbt", residual.pseudo_queries[query_index], keys)
    weights = torch.softmax(logits, dim=0)
    return (weights.unsqueeze(-1) * values).sum(dim=0)


def test_zero_initialized_query_starts_with_uniform_depth_average():
    residual = FullAttentionResidual(d_model=2, num_queries=1)
    sources = [
        torch.tensor([[[1.0, 3.0]]]),
        torch.tensor([[[5.0, 7.0]]]),
    ]

    output = residual(sources, 0)

    torch.testing.assert_close(output, torch.tensor([[[3.0, 5.0]]]))
    assert torch.count_nonzero(residual.pseudo_queries) == 0
    assert len(residual.key_norms) == 1
    torch.testing.assert_close(residual.key_norms[0].weight, torch.ones(2))


def test_full_attention_residual_query_receives_gradient():
    residual = FullAttentionResidual(d_model=4, num_queries=2)
    sources = [
        torch.randn(2, 3, 4, requires_grad=True),
        torch.randn(2, 3, 4, requires_grad=True),
    ]

    residual(sources, 1).square().mean().backward()

    assert residual.pseudo_queries.grad is not None
    assert torch.count_nonzero(residual.pseudo_queries.grad[1]) > 0


@pytest.mark.parametrize("training", [True, False])
def test_depth_attention_matches_stacked_reference_and_gradients(training):
    torch.manual_seed(13)
    residual = FullAttentionResidual(d_model=8, num_queries=2)
    reference = FullAttentionResidual(d_model=8, num_queries=2)
    with torch.no_grad():
        residual.pseudo_queries[1].normal_()
        residual.key_norms[1].weight.uniform_(0.7, 1.3)
    reference.load_state_dict(residual.state_dict())
    residual.train(training)
    sources = [torch.randn(2, 3, 8, requires_grad=True) for _ in range(4)]
    reference_sources = [source.detach().clone().requires_grad_() for source in sources]

    output = residual(sources, 1)
    expected = _stacked_reference(reference, reference_sources, 1)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(output)
    output.backward(upstream)
    expected.backward(upstream)

    for source, reference_source in zip(sources, reference_sources):
        torch.testing.assert_close(source.grad, reference_source.grad)
    for parameter, reference_parameter in zip(
        residual.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)


@pytest.mark.parametrize(
    ("training", "grad_enabled", "onnx_export", "expected_calls"),
    [
        (True, True, False, 1),
        (True, False, False, 0),
        (False, True, False, 0),
        (False, False, False, 0),
        (True, True, True, 0),
    ],
)
def test_local_recomputation_only_runs_in_training_with_gradients(
    monkeypatch, training, grad_enabled, onnx_export, expected_calls
):
    residual = FullAttentionResidual(d_model=4, num_queries=1).train(training)
    sources = [torch.randn(2, 3, 4, requires_grad=True) for _ in range(2)]
    original_checkpoint = attention_residual_module.checkpoint
    calls = []

    def record_checkpoint(function, *args, **kwargs):
        logits = original_checkpoint(function, *args, **kwargs)
        calls.append((args, kwargs, logits.shape))
        return logits

    monkeypatch.setattr(attention_residual_module, "checkpoint", record_checkpoint)
    monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: onnx_export)

    with torch.set_grad_enabled(grad_enabled):
        output = residual(sources, 0)

    assert len(calls) == expected_calls
    assert output.requires_grad == grad_enabled
    if calls:
        args, kwargs, checkpoint_output_shape = calls[0]
        assert all(actual is source for actual, source in zip(args, sources))
        assert len(args) == len(sources)
        assert kwargs == {"use_reentrant": False, "preserve_rng_state": False}
        # 重算边界只输出深度打分，不重复执行完整 value 聚合。
        assert checkpoint_output_shape == (len(sources), *sources[0].shape[:-1])


def test_local_recomputation_preserves_cpu_bfloat16_autocast_and_leaf_gradients():
    torch.manual_seed(23)
    residual = FullAttentionResidual(d_model=8, num_queries=2)
    reference = FullAttentionResidual(d_model=8, num_queries=2).eval()
    with torch.no_grad():
        residual.pseudo_queries.normal_()
        residual.key_norms[1].weight.uniform_(0.7, 1.3)
    reference.load_state_dict(residual.state_dict())
    # 与 FP32 叶子参数的实际前向一致：首个 embedding 源为 FP32，其余子层输出为 BF16。
    sources = [
        torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
        for dtype in (torch.float32, torch.bfloat16, torch.bfloat16)
    ]
    reference_sources = [source.detach().clone().requires_grad_() for source in sources]

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = residual(sources, 1)
        expected = reference(reference_sources, 1)

    assert output.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    upstream = torch.randn_like(output)
    # 反向在 autocast 之外，重算仍须恢复原前向的 autocast 精度。
    output.backward(upstream)
    expected.backward(upstream)
    for source, reference_source in zip(sources, reference_sources):
        assert source.grad.dtype == source.dtype
        torch.testing.assert_close(source.grad, reference_source.grad, rtol=0, atol=0)
    for parameter, reference_parameter in zip(
        residual.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)


def test_recomputation_keeps_single_source_query_and_gamma_gradients_present():
    residual = FullAttentionResidual(d_model=4, num_queries=1)
    with torch.no_grad():
        residual.pseudo_queries.normal_()
    source = torch.randn(2, 3, 4, requires_grad=True)

    output = residual([source], 0)
    output.sum().backward()

    torch.testing.assert_close(output, source)
    torch.testing.assert_close(source.grad, torch.ones_like(source))
    assert residual.pseudo_queries.grad is not None
    assert residual.key_norms[0].weight.grad is not None
    assert torch.count_nonzero(residual.pseudo_queries.grad) == 0
    assert torch.count_nonzero(residual.key_norms[0].weight.grad) == 0


def test_recomputation_preserves_state_dict_and_parameter_initialization():
    residual = FullAttentionResidual(d_model=4, num_queries=2)
    assert tuple(residual.state_dict()) == (
        "pseudo_queries",
        "key_norms.0.weight",
        "key_norms.1.weight",
    )
    assert residual.num_queries == 2
    with torch.no_grad():
        for parameter in residual.parameters():
            parameter.normal_()
    restored = FullAttentionResidual(d_model=4, num_queries=2)
    restored.load_state_dict(residual.state_dict(), strict=True)
    for name, parameter in residual.state_dict().items():
        torch.testing.assert_close(parameter, restored.state_dict()[name])

    restored.reset_parameters()
    assert torch.count_nonzero(restored.pseudo_queries) == 0
    for norm in restored.key_norms:
        torch.testing.assert_close(norm.weight, torch.ones_like(norm.weight))


def test_local_recomputation_does_not_save_independent_full_size_key_storage():
    torch.manual_seed(37)
    residual = FullAttentionResidual(d_model=16, num_queries=2)
    reference = FullAttentionResidual(d_model=16, num_queries=2)
    with torch.no_grad():
        residual.pseudo_queries.normal_()
    reference.load_state_dict(residual.state_dict())
    sources = [torch.randn(2, 7, 16, requires_grad=True) for _ in range(4)]
    reference_sources = [source.detach().clone().requires_grad_() for source in sources]

    def independent_saved_storage(module, values, forward):
        # 按底层 storage 去重，排除必须保留的原始源和参数，避免把 view 重复计为副本。
        original = {
            tensor.untyped_storage().data_ptr()
            for tensor in (*values, *module.parameters())
        }
        saved = {}

        def pack(tensor):
            storage = tensor.untyped_storage()
            pointer = storage.data_ptr()
            if pointer not in original and storage.nbytes():
                saved[pointer] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = forward(module, values, 1)
        return output, saved

    output, saved = independent_saved_storage(
        residual, sources, lambda module, values, index: module(values, index)
    )
    expected, reference_saved = independent_saved_storage(
        reference, reference_sources, _stacked_reference
    )
    one_source_bytes = sources[0].numel() * sources[0].element_size()

    torch.testing.assert_close(output, expected)
    assert not any(size >= one_source_bytes for size in saved.values())
    assert any(size >= len(sources) * one_source_bytes for size in reference_saved.values())
    assert sum(saved.values()) < sum(reference_saved.values())
    output.square().mean().backward()
    expected.square().mean().backward()

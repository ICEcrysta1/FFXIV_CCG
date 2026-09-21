"""Full Attention Residual 聚合器的单元测试。"""

from __future__ import annotations

import math

import torch

from common.policy.model.attention_residual import FullAttentionResidual


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


def test_depth_attention_scales_query_key_logits_by_model_width():
    residual = FullAttentionResidual(d_model=4, num_queries=1)
    with torch.no_grad():
        residual.pseudo_queries[0].copy_(torch.tensor([2.0, 0.0, 0.0, 0.0]))
    sources = [
        torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        torch.tensor([[[-1.0, 0.0, 0.0, 0.0]]]),
    ]

    output = residual(sources, 0)
    values = torch.stack(sources, dim=0)
    keys = residual.key_norms[0](values)
    logits = torch.einsum("d,sbtd->sbt", residual.pseudo_queries[0], keys)
    weights = torch.softmax(logits / math.sqrt(4), dim=0)
    expected = torch.einsum("sbt,sbtd->btd", weights, values)

    torch.testing.assert_close(output, expected)


def test_streaming_depth_attention_matches_stacked_reference():
    residual = FullAttentionResidual(d_model=8, num_queries=2)
    reference = FullAttentionResidual(d_model=8, num_queries=2)
    with torch.no_grad():
        residual.pseudo_queries[1].normal_()
    reference.load_state_dict(residual.state_dict())
    sources = [torch.randn(2, 3, 8, requires_grad=True) for _ in range(4)]
    reference_sources = [source.detach().clone().requires_grad_() for source in sources]

    output = residual(sources, 1)
    values = torch.stack(tuple(reference_sources), dim=0)
    keys = reference.key_norms[1](values)
    logits = torch.einsum("d,sbtd->sbt", reference.pseudo_queries[1], keys)
    logits = logits * reference._logit_scale
    weights = torch.softmax(logits, dim=0)
    expected = torch.einsum("sbt,sbtd->btd", weights, values)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    output.square().mean().backward()
    expected.square().mean().backward()

    for source, reference_source in zip(sources, reference_sources):
        torch.testing.assert_close(source.grad, reference_source.grad)
    for parameter, reference_parameter in zip(
        residual.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)

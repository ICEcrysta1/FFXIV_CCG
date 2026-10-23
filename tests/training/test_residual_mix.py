"""四路 Hyper-Connections 的公式、残差方向、梯度与重算契约。"""

from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from common.policy.config import ModelConfig
from common.policy.model import CausalPolicyModel
from common.policy.model.execution import causal_encoder
from common.policy.model.layers import residual_mix
from common.policy.model.layers.residual_mix import HyperConnections
from common.policy.model.layers.transformer import RMSNorm
from tests.training._causal_fixtures import make_batch, make_data_spec


def _model(**overrides):
    values = {"d_model": 16, "n_layers": 3, "n_heads": 4, "num_kv_heads": 1,
              "ff_dim": 32, "dropout": 0.0, "transformer_activation": "swiglu"}
    values.update(overrides)
    torch.manual_seed(47)
    return CausalPolicyModel(make_data_spec(), ModelConfig(**values), 3)


def _reference_mixes(module, streams):
    # 独立展开 flatten RMS、三组门控及源行/目标列 Sinkhorn。
    flat = streams.float().flatten(-2)
    values = F.linear(flat, module.fn.float()) / (
        flat.square().mean(-1, keepdim=True) + 1e-20
    ).sqrt()
    count, eps = module.hc_mult, module.hc_eps
    pre = (values[..., :count] * module.scale[0] + module.base[:count]).sigmoid() + eps
    post = 2 * (values[..., count:2 * count] * module.scale[1]
                + module.base[count:2 * count]).sigmoid()
    comb = (values[..., 2 * count:] * module.scale[2]
            + module.base[2 * count:]).unflatten(-1, (count, count)).softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(module.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


@pytest.mark.parametrize("dtype,amp", ((torch.float32, False), (torch.float32, True),
                                      (torch.bfloat16, True), (torch.bfloat16, False)))
def test_cuda_compiled_hyper_connections_preserve_public_outputs_gradients_and_state(dtype, amp):
    if not torch.cuda.is_available():
        pytest.skip("CUDA compilation requires a CUDA device")
    if amp and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 autocast requires BF16 support")
    # 每组精度独立验证，避免跨用例的形状/归一化组合占满编译缓存上限。
    torch._dynamo.reset()
    torch.manual_seed(52)
    module = HyperConnections(8, hc_sinkhorn_iters=3).cuda().train()
    with torch.no_grad():
        module.scale.copy_(torch.tensor([0.7, -0.4, 0.6], device="cuda"))
        module.base.copy_(torch.linspace(-0.4, 0.7, 24, device="cuda"))
    reference = deepcopy(module).eval()
    norm = RMSNorm(8, dtype=torch.float32 if amp else dtype).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.6, 1.4, 8, device="cuda"))
    reference_norm = deepcopy(norm)
    saved_state = {name: value.detach().clone() for name, value in module.state_dict().items()}
    # 首个布局保留 expand 的零 stride，随后改变 batch/长度并使用普通残差流。
    for batch_size, length, expanded in ((2, 5, True), (1, 9, False)):
        module.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        norm.zero_grad(set_to_none=True)
        reference_norm.zero_grad(set_to_none=True)
        shape = (batch_size, length, 8) if expanded else (batch_size, length, 4, 8)
        source = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        reference_source = source.detach().clone().requires_grad_()
        streams = source.unsqueeze(-2).expand(-1, -1, 4, -1) if expanded else source
        reference_streams = reference_source.unsqueeze(-2).expand(-1, -1, 4, -1) if expanded else reference_source
        if expanded:
            assert streams.stride(-2) == 0
        output = torch.randn(batch_size, length, 8, device="cuda", dtype=dtype, requires_grad=True)
        reference_output = output.detach().clone().requires_grad_()
        results = []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            for current, values, sublayer_output, current_norm in (
                (module, streams, output, norm), (reference, reference_streams, reference_output, reference_norm),
            ):
                pre, post, comb = current.mixes(values)
                collapsed = current.collapse(values, pre)
                normalized = current.collapse(values, pre, current_norm) if current is module else current_norm(collapsed)
                combined = current.combine(sublayer_output, values, post, comb)
                assert all(value.dtype == torch.float32 for value in (pre, post, comb))
                assert collapsed.dtype == normalized.dtype == combined.dtype == dtype
                results.append((pre, post, comb, collapsed, normalized, combined))
        for actual, expected in zip(*results, strict=True):
            tolerance = 0.016 if actual.dtype == torch.bfloat16 else 2e-5
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        upstream = tuple(torch.randn_like(value) for value in results[0])
        for values in results:
            sum((value.float() * gradient.float()).sum()
                for value, gradient in zip(values, upstream, strict=True)).backward()
        for actual, expected in ((source, reference_source), (output, reference_output),
                                 *zip(module.parameters(), reference.parameters(), strict=True),
                                 *zip(norm.parameters(), reference_norm.parameters(), strict=True)):
            assert actual.grad is not None and torch.isfinite(actual.grad).all()
            tolerance = 0.016 if dtype == torch.bfloat16 else 3e-5
            torch.testing.assert_close(actual.grad, expected.grad, rtol=tolerance, atol=tolerance)
        assert module.state_dict().keys() == saved_state.keys()
        for name, value in module.state_dict().items():
            assert value.dtype == saved_state[name].dtype == torch.float32
            torch.testing.assert_close(value, saved_state[name], rtol=0, atol=0)


def test_cuda_warmed_model_keeps_default_compile_limit_across_shapes_and_gradient_states(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA 验证完整模型的编译缓存与普通反向")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("需要 CUDA BF16 autocast 支持")
    from training.config import RunConfig
    from training.loop.training_loop import _warmup_compiled_training

    # 只在开始重置一次，保留预热、不同层和正式反向共享的编译缓存。
    torch._dynamo.reset()
    model = _model(d_model=8, n_layers=6, n_heads=2, ff_dim=16, context_capacity=32).cuda().train()
    reference = deepcopy(model).eval()
    config = RunConfig(
        raw_data_dir=tmp_path / "raw", output_dir=tmp_path / "output", job_tag="black_mage",
        model_variant="artzip", model=model.config, batch_size=2, precision="bf16", fp32_gradients=True,
    )
    assert all(parameter.dtype == torch.float32 and parameter.grad is None for parameter in model.parameters())
    _warmup_compiled_training(model, config)
    assert all(parameter.grad is None for parameter in model.parameters())

    norm_weights = [module.weight for module in model.modules() if isinstance(module, RMSNorm)]
    assert len(norm_weights) == 13
    checked_parameters = [*norm_weights, *model.encoder.residual_mix.parameters()]
    shapes = ((2, 0, 0), (1, 1, 1), (2, 3, 2), (1, 5, 1), (2, 2, 0), (2, 0, 1), (2, 0, 1))
    for index, (batch_size, history_length, scene_length) in enumerate(shapes):
        # 最后一批保留已有梯度，同时验证 None→Tensor 与 Tensor 上的梯度累加。
        if index < len(shapes) - 1:
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            assert all(parameter.grad is None for parameter in model.parameters())
        else:
            assert all(parameter.grad is not None for parameter in checked_parameters)
        batch = make_batch(batch_size=batch_size, history_length=history_length, scene_length=scene_length)
        batch = {name: value.cuda() if isinstance(value, torch.Tensor) else value for name, value in batch.items()}
        batch["current_state_vectors"] = torch.randn_like(batch["current_state_vectors"], requires_grad=True)
        reference_batch = deepcopy(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = model(batch)["logits"]
            expected = reference(reference_batch)["logits"]
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.016)
        actual.float().square().mean().backward()
        expected.float().square().mean().backward()
        torch.testing.assert_close(
            batch["current_state_vectors"].grad, reference_batch["current_state_vectors"].grad,
            rtol=0.025, atol=2e-4,
        )
        for (name, parameter), (expected_name, target) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True,
        ):
            assert name == expected_name
            if target.grad is None:
                assert parameter.grad is None
                continue
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, target.grad, rtol=0.025, atol=2e-4, msg=name)
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in checked_parameters)


def test_cuda_collapse_keeps_compile_cache_across_warmup_and_partial_batches():
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA 验证预热与尾批跨越归约尺寸阈值")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("需要 CUDA BF16 autocast 支持")
    torch._dynamo.reset()
    torch.manual_seed(109)
    module = HyperConnections(384).cuda().train()
    reference = deepcopy(module).eval()
    norm = RMSNorm(384).cuda()
    reference_norm = deepcopy(norm)

    def check_shape(batch_size, length, expanded):
        shape = (batch_size, length, 384) if expanded else (batch_size, length, 4, 384)
        source = torch.randn(shape, device="cuda", dtype=torch.float32 if expanded else torch.bfloat16,
                             requires_grad=True)
        if expanded:
            streams, pre = module.expand(source)
            inputs = (source,)
        else:
            streams = source
            pre = torch.rand(batch_size, length, 4, device="cuda", requires_grad=True)
            inputs = (source, pre)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = module.collapse(streams, pre, norm)
        upstream = torch.randn_like(actual)
        actual_gradients = torch.autograd.grad(actual, (*inputs, norm.weight), grad_outputs=upstream)
        # 先释放编译路径的反向图，再构造独立 eager 图，限制大形状的同时驻留量。
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = reference.collapse(streams, pre, reference_norm)
        expected_gradients = torch.autograd.grad(expected, (*inputs, reference_norm.weight), grad_outputs=upstream)
        tolerance = 3e-5 if expanded else 0.016
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        for index, (actual_gradient, expected_gradient) in enumerate(zip(actual_gradients, expected_gradients, strict=True)):
            assert torch.isfinite(actual_gradient).all()
            # γ 在约两万个 token 上做 FP32 归约，融合核的求和顺序允许少量绝对舍入误差。
            absolute_tolerance = 2e-4 if expanded and index == len(actual_gradients) - 1 else tolerance
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=tolerance, atol=absolute_tolerance)
        assert norm.weight.grad is None and reference_norm.weight.grad is None

    # 保持生产预热顺序；固定首层与可学习后续流各覆盖四种 batch/token 特化。
    for batch_size, length in ((20, 1023), (20, 1), (1, 1023), (1, 1), (10, 946), (20, 32)):
        for expanded in (True, False):
            check_shape(batch_size, length, expanded)


@pytest.mark.parametrize("mult,iterations,eps,uniform", (
    (2, 1, 1e-6, False), (3, 20, 1e-6, False), (4, 20, 1e-6, True), (4, 20, 1e-30, False),
))
def test_cuda_coefficients_kernel_preserves_tied_max_and_small_eps_gradients(mult, iterations, eps, uniform):
    if not torch.cuda.is_available():
        pytest.skip("CUDA compilation requires a CUDA device")
    from common.policy.model.layers.hyper_connections_cuda import hc_coefficients

    torch.manual_seed(107)
    size = mult * (mult + 2)
    mixes = torch.randn(2, 3, size, device="cuda", requires_grad=True)
    scale = torch.tensor([0.7, -0.4, 3.5], device="cuda", requires_grad=True)
    base = torch.linspace(-0.4, 0.7, size, device="cuda", requires_grad=True)
    with torch.no_grad():
        if uniform:
            mixes[..., 2 * mult:] = 1.25
            base[2 * mult:] = 0
        elif eps < 1e-20:
            # 制造接近零的列；合法小 eps 的反向不能引入倒数平方溢出。
            mixes[..., 2 * mult:] = torch.arange(mult, device="cuda").repeat(mult) * -100.0
    references = tuple(value.detach().clone().requires_grad_() for value in (mixes, scale, base))
    actual = torch.compile(hc_coefficients, fullgraph=True, dynamic=True)(mixes, scale, base, mult, iterations, eps)
    values, reference_scale, reference_base = references
    pre = (values[..., :mult] * reference_scale[0] + reference_base[:mult]).sigmoid() + eps
    post = 2 * (values[..., mult:2 * mult] * reference_scale[1] + reference_base[mult:2 * mult]).sigmoid()
    comb = (values[..., 2 * mult:] * reference_scale[2]
            + reference_base[2 * mult:]).unflatten(-1, (mult, mult)).softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    expected = (pre, post, comb)
    for value, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, target, rtol=2e-6, atol=2e-6)
    upstream = tuple(torch.randn_like(value) for value in actual)
    for result in (actual, expected):
        sum((value * gradient).sum() for value, gradient in zip(result, upstream, strict=True)).backward()
    for value, target in zip((mixes, scale, base), references, strict=True):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        torch.testing.assert_close(value.grad, target.grad, rtol=3e-5, atol=3e-6)


def test_hyper_connections_match_explicit_formula_and_all_gradients():
    torch.manual_seed(48)
    module = HyperConnections(8)
    reference = deepcopy(module)
    with torch.no_grad():
        module.scale.copy_(torch.tensor([0.7, -0.4, 0.6]))
        module.base.copy_(torch.linspace(-0.4, 0.7, 24))
        reference.load_state_dict(module.state_dict())
    streams = torch.randn(2, 3, 4, 8, requires_grad=True)
    reference_streams = streams.detach().clone().requires_grad_()
    actual = module.mixes(streams)
    expected = _reference_mixes(reference, reference_streams)
    for value, target in zip(actual, expected, strict=True):
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, target, rtol=2e-6, atol=2e-6)
    pre, post, comb = actual
    assert torch.all(pre > 0) and torch.all((post > 0) & (post < 2))
    torch.testing.assert_close(comb.sum(-1), torch.ones_like(pre), rtol=0, atol=4e-6)
    torch.testing.assert_close(comb.sum(-2), torch.ones_like(pre), rtol=0, atol=4e-6)
    upstream = [torch.randn_like(value) for value in actual]
    sum((value * gradient).sum() for value, gradient in zip(actual, upstream)).backward()
    sum((value * gradient).sum() for value, gradient in zip(expected, upstream)).backward()
    torch.testing.assert_close(streams.grad, reference_streams.grad, rtol=3e-6, atol=3e-6)
    for parameter, target in zip(module.parameters(), reference.parameters(), strict=True):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, target.grad, rtol=3e-6, atol=3e-6)


def test_expand_collapse_and_combine_use_source_to_target_axis():
    module = HyperConnections(3)
    tokens = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    streams, pre = module.expand(tokens)
    assert streams.shape == (1, 2, 4, 3)
    torch.testing.assert_close(pre, torch.tensor([1., 0., 0., 0.]).expand(1, 2, 4))
    torch.testing.assert_close(module.collapse(streams, pre), tokens, rtol=0, atol=0)
    residual = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3)
    # 非对称置换锁定源/目标轴，四流相同时无法发现转置错误。
    comb = F.one_hot(torch.tensor([1, 2, 3, 0]), 4).float().expand(1, 2, 4, 4)
    post = torch.tensor([0.2, 0.4, 0.6, 0.8]).expand(1, 2, 4)
    actual = module.combine(tokens, residual, post, comb)
    expected = residual[..., [3, 0, 1, 2], :] + tokens.unsqueeze(-2) * post.unsqueeze(-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hyper_connection_mixes_keep_fp32_under_cpu_bfloat16_autocast():
    torch.manual_seed(49)
    module = HyperConnections(8)
    streams = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    expected = module.mixes(streams)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = module.mixes(streams)
    for value, target in zip(actual, expected, strict=True):
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, target, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ("mixes", "collapse", "collapse_norm", "combine"))
def test_hyper_connection_recompute_saves_original_storage_and_preserves_gradients(operation):
    torch.manual_seed(50)
    module = HyperConnections(8).train()
    reference = deepcopy(module).eval()
    norm = RMSNorm(8)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.6, 1.4, 8))
    reference_norm = deepcopy(norm)
    streams = torch.randn(2, 8, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    pre = torch.randn(2, 8, 4, requires_grad=True)
    output = torch.randn(2, 8, 8, dtype=torch.bfloat16, requires_grad=True)
    comb = torch.randn(2, 8, 4, 4, requires_grad=True)
    inputs = {"mixes": (streams,), "collapse": (streams, pre), "collapse_norm": (streams, pre),
              "combine": (output, streams, pre, comb)}[operation]
    reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    results, storage_bytes = [], []
    for current, values, current_norm in ((module, inputs, norm), (reference, reference_inputs, reference_norm)):
        saved = {}

        def pack(value, saved=saved):
            storage = value.untyped_storage()
            saved[storage.data_ptr()] = storage.nbytes()
            return value

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            result = (current.collapse(*values, current_norm) if operation == "collapse_norm"
                      else getattr(current, operation)(*values))
        results.append(result if isinstance(result, tuple) else (result,))
        storage_bytes.append(sum(saved.values()))
    assert storage_bytes[0] < storage_bytes[1]
    for actual, expected in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = tuple(torch.randn_like(value) for value in results[0])
    for values in results:
        sum((value * gradient).float().sum() for value, gradient in zip(values, upstream, strict=True)).backward()
    for actual, expected in zip(inputs, reference_inputs, strict=True):
        assert actual.grad is not None and torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-6)
    if operation == "mixes":
        for actual, expected in zip(module.parameters(), reference.parameters(), strict=True):
            assert actual.grad is not None and torch.isfinite(actual.grad).all()
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-6)
    if operation == "collapse_norm":
        assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all()
        torch.testing.assert_close(norm.weight.grad, reference_norm.weight.grad, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mode", ("eval", "no_grad", "onnx"))
def test_hyper_connection_recompute_bypasses_inference_and_export(monkeypatch, mode):
    module = HyperConnections(8).train(mode != "eval")

    def unexpected_checkpoint(*args, **kwargs):
        pytest.fail("HC inference or export must bypass activation recomputation")

    monkeypatch.setattr(residual_mix, "checkpoint", unexpected_checkpoint)
    for operation in ("mixes", "collapse", "combine"):
        monkeypatch.setattr(module, f"_compiled_{operation}", unexpected_checkpoint)
    if mode == "onnx":
        monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
    streams = torch.randn(1, 2, 4, 8, requires_grad=True)
    with torch.set_grad_enabled(mode != "no_grad"):
        pre, post, comb = module.mixes(streams)
        module.combine(module.collapse(streams, pre), streams, post, comb)


def test_layer_uses_previous_pre_for_attention_then_attention_pre_for_ffn(monkeypatch):
    model = _model().eval()
    layer = model.encoder.layers[0]
    attn_mix, ffn_mix = model.encoder.residual_mix[:2]
    with torch.no_grad():
        for module in (attn_mix, ffn_mix):
            module.scale.fill_(0.7)
            module.base.add_(torch.linspace(-0.5, 0.6, 24))
    streams = torch.randn(1, 2, 4, 16)
    previous_pre = torch.tensor([0.1, 0.2, 0.3, 0.4]).expand(1, 2, 4)
    attended = torch.randn(1, 2, 16)
    attn_pre, attn_post, attn_comb = _reference_mixes(attn_mix, streams)
    after_attention = torch.einsum("bsij,bsid->bsjd", attn_comb, streams)
    after_attention = after_attention + attn_post.unsqueeze(-1) * attended.unsqueeze(-2)
    expected_pre, ffn_post, ffn_comb = _reference_mixes(ffn_mix, after_attention)
    attention_input = (streams * previous_pre.unsqueeze(-1)).sum(-2)
    ffn_input = (after_attention * attn_pre.unsqueeze(-1)).sum(-2)
    ffn_output = layer._ff_block(layer.norm2(ffn_input))
    expected = torch.einsum("bsij,bsid->bsjd", ffn_comb, after_attention)
    expected = expected + ffn_post.unsqueeze(-1) * ffn_output.unsqueeze(-2)
    captured = {}
    hooks = [
        layer.norm1.register_forward_pre_hook(lambda module, args: captured.update(attention=args[0])),
        layer.norm2.register_forward_pre_hook(lambda module, args: captured.update(ffn=args[0])),
    ]
    monkeypatch.setattr(causal_encoder, "run_head_attention", lambda *args, **kwargs: (attended, None))
    try:
        hidden, _, _, _, next_pre = causal_encoder.run_causal_layer(
            layer, streams, pre_mix=previous_pre, residual_mix=model.encoder.residual_mix,
            key_valid=torch.ones((1, 2), dtype=torch.bool),
            position_ids=torch.arange(2).unsqueeze(0),
            rotary_position_encoding=model.encoder.rotary_position_encoding,
        )
    finally:
        for hook in hooks:
            hook.remove()
    for value, target in ((captured["attention"], attention_input), (captured["ffn"], ffn_input),
                          (next_pre, expected_pre), (hidden, expected)):
        torch.testing.assert_close(value, target, rtol=2e-6, atol=2e-6)


def test_model_initialization_breaks_hyper_connection_symmetry_and_keeps_trace_3d():
    model = _model().eval()
    modules = model.encoder.residual_mix
    assert len(modules) == 2 * model.config.n_layers
    for module in modules:
        assert module.fn.shape == (24, 64)
        assert torch.count_nonzero(module.fn) > 0
        torch.testing.assert_close(module.scale, torch.full((3,), 0.01))
    batch = make_batch()
    output = model(batch)["logits"]
    trace = model.trace(batch)
    assert trace.hidden.ndim == 3 and all(hidden.ndim == 3 for hidden in trace.layer_hidden)
    torch.testing.assert_close(output, model.score_hidden(trace.encoded, trace.hidden, batch),
                               rtol=2e-6, atol=2e-6)
    output.square().mean().backward()
    for module in modules:
        assert module.fn.grad is not None and torch.count_nonzero(module.fn.grad) > 0


def test_full_attention_residual_has_no_unused_hyper_connection_parameters():
    model = _model(full_attention_residuals=True)
    assert not hasattr(model.encoder, "residual_mix")
    assert not any("residual_mix" in name for name, _ in model.named_parameters())
    assert hasattr(model.encoder, "attention_residual")


@pytest.mark.parametrize("attention_block", (False, True))
@pytest.mark.parametrize("precision", ("float32", "bf16"))
def test_activation_checkpoint_keeps_hyper_connection_output_and_all_gradients(attention_block, precision):
    model = _model().train()
    checkpointed = deepcopy(model)
    checkpointed.enable_activation_checkpoint_attention(True, block=attention_block)
    checkpointed.enable_activation_checkpoint_ffn(True)
    batch = make_batch()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=precision == "bf16"):
        output, expected = checkpointed(batch)["logits"], model(batch)["logits"]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.float().square().mean().backward()
    expected.float().square().mean().backward()
    for (name, parameter), (expected_name, reference) in zip(
        checkpointed.named_parameters(), model.named_parameters(), strict=True,
    ):
        assert name == expected_name and parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-6, atol=1e-6)

"""用真实因果模型与事件文件验证 TensorBoard 追踪的旁观语义。"""

from __future__ import annotations

import pytest
import torch

from common.policy.config import ModelConfig
from common.policy.model import CausalPolicyModel, RepetitionConfig
from common.policy.model.repetition import prepare_repetition_penalty
from common.training.tensorboard import TensorBoardConfig
from common.training.tensorboard_monitor import TensorBoardMonitor
from tests.training._causal_fixtures import make_batch, make_data_spec


@pytest.mark.parametrize(
    ("device", "dtype"),
    [("cpu", torch.float32), ("cuda", torch.bfloat16), ("cuda", torch.float32)],
)
def test_real_graph_preserves_repetition_rng_and_runtime_state(
    tmp_path, monkeypatch, device, dtype,
):
    """图包含重复惩罚，且 GQA、低精度追踪不改变正式训练的运行时状态。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("需要 CUDA 验证训练使用的 BF16 图追踪")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    spec = make_data_spec()
    repetition = RepetitionConfig(mode="blacklist", skills=("first",), penalty=2.5)
    model = CausalPolicyModel(
        spec,
        ModelConfig(
            d_model=16, n_layers=2, n_heads=4, num_kv_heads=2, ff_dim=24,
            dropout=0.2, transformer_activation="swiglu", full_attention_residuals=True,
        ),
        vocab_size=3,
        repetition=repetition,
    ).to(device=device)
    model.encoder.to(dtype=dtype)
    model.output_head.to(dtype=dtype)
    model.enable_activation_checkpoint_ffn()
    model.enable_activation_checkpoint_attention(block=True)
    batch = prepare_repetition_penalty(make_batch(spec), repetition)
    batch = {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device, dtype=torch.bfloat16, enabled=device == "cuda", cache_enabled=False,
    ):
        expected_logits = model(batch)["logits"].detach().clone()
        unpenalized = model({**batch, "repetition_penalty_mask": torch.zeros_like(batch["repetition_penalty_mask"])})["logits"]
    torch.testing.assert_close(
        unpenalized[:, 0] - expected_logits[:, 0],
        torch.full_like(expected_logits[:, 0], repetition.penalty),
    )

    model.train()
    # 混合子模块模式、非空 KV 和诊断哨兵可以发现 train()/eval() 的隐式副作用。
    model.encoder.layers[0].dropout1.eval()
    model.enable_kv_cache()
    cache_sentinel = object()
    model._kv_cache = cache_sentinel
    debug_sentinel = object()
    model.set_runtime_debug(debug_sentinel)
    modules = tuple(model.modules())
    modes = tuple(module.training for module in modules)
    debug_states = tuple(
        (module, module._runtime_debug)
        for module in modules if hasattr(module, "_runtime_debug")
    )
    parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None

    config = TensorBoardConfig(enabled=True)
    with SummaryWriter(str(tmp_path)) as writer:
        monitor = TensorBoardMonitor(writer, config, model=model)
        actual_add_graph = writer.add_graph
        captured_logits = []

        def capture_graph(graph_model, inputs, *args, **kwargs):
            with torch.no_grad():
                captured_logits.append(graph_model(*inputs).detach().clone())
            return actual_add_graph(graph_model, inputs, *args, **kwargs)

        monkeypatch.setattr(writer, "add_graph", capture_graph)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            monitor.log_graph(batch)
            monitor.log_graph(batch)

    assert len(captured_logits) == 1
    torch.testing.assert_close(captured_logits[0], expected_logits)
    accumulator = EventAccumulator(str(tmp_path)).Reload()
    assert accumulator.Tags()["graph"]
    assert len(accumulator.Graph().node) > 100
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert tuple(module.training for module in modules) == modes
    assert model._kv_cache_enabled is True
    assert model._kv_cache is cache_sentinel
    assert all(module._runtime_debug is original for module, original in debug_states)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in parameters.items())

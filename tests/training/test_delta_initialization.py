"""新增 ABS reset 投影不改变 4×256 模型公共参数的种子初始化。"""

import pytest
import torch
from torch import nn

from common.policy.config import ModelConfig
from common.policy.data.spec import DataSpec
from common.policy.model import input_encoder
from common.policy.model.model import CausalPolicyModel


@pytest.mark.parametrize("full_attention_residuals", [False, True])
def test_state_reset_projection_preserves_common_initialization(monkeypatch, full_attention_residuals):
    spec = DataSpec(
        job_tag="black_mage", num_actions=4, state_dim=86, scene_dim=7,
        skill_feature_dim=18, num_scene_types=4,
        action_keys=("a", "b", "c", "wait"),
        skill_feature_names=tuple(f"skill_{index}" for index in range(18)),
        action_to_vocab_id=(1, 2, 3, 4), action_is_gcd=(True, True, False, False),
    )
    config = ModelConfig(
        n_layers=4, d_model=256, n_heads=4, ff_dim=1024,
        transformer_activation="swiglu", history_capacity=300,
        full_attention_residuals=full_attention_residuals,
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        actual = CausalPolicyModel(spec, config, vocab_size=10)
        actual_rng = torch.random.get_rng_state()
        # 用无参数槽替代新增投影，重建此前拥有相同公共模块的模型。
        monkeypatch.setattr(input_encoder, "_StateResetProjection", lambda *args, **kwargs: nn.Identity())
        torch.random.default_generator.manual_seed(42)
        without_reset_projection = CausalPolicyModel(spec, config, vocab_size=10)
        expected_rng = torch.random.get_rng_state()

    reset_name = "input_encoder.state_reset_proj.weight"
    actual_parameters = dict(actual.named_parameters())
    reference_parameters = dict(without_reset_projection.named_parameters())
    assert actual_parameters.keys() - reference_parameters.keys() == {reset_name}
    assert reference_parameters.keys() - actual_parameters.keys() == set()
    assert torch.count_nonzero(actual_parameters[reset_name]) == 0
    for name, expected in reference_parameters.items():
        torch.testing.assert_close(actual_parameters[name], expected, rtol=0, atol=0, msg=name)
    torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)

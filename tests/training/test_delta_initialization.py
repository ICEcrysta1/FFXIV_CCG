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
        job_tag="black_mage", num_actions=4, base_state_dim=86, state_dim=(86) + 2 * (4), scene_dim=7,
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
    projection_rng = {}

    def capture_projection_rng(module, name, child):
        if isinstance(module, input_encoder.CausalInputEncoder) and name in {
            "role_embed", "state_reset_proj",
        }:
            projection_rng[name] = torch.random.get_rng_state()

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        hook = nn.modules.module.register_module_module_registration_hook(capture_projection_rng)
        try:
            actual = CausalPolicyModel(spec, config, vocab_size=10)
        finally:
            hook.remove()
        actual_rng = torch.random.get_rng_state()
        # 投影构造不能消耗随机数；参考模型在公共权重初始化前移除该投影。
        torch.testing.assert_close(
            projection_rng["role_embed"], projection_rng["state_reset_proj"], rtol=0, atol=0,
        )
        initialize_input_encoder = input_encoder.CausalInputEncoder.__init__

        def initialize_without_reset(self, *args, **kwargs):
            initialize_input_encoder(self, *args, **kwargs)
            del self.state_reset_proj

        monkeypatch.setattr(input_encoder.CausalInputEncoder, "__init__", initialize_without_reset)
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

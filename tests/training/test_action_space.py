"""固定动作输出空间与输入词表行的回归测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from common.config import load_project_config
from common.output_context_schema import CANONICAL_CONTEXT_SCHEMA_VERSION
from common.policy.data import ActionSpace, DataSpec, SkillVocab
from common.policy.data.schema import TRAINING_SAMPLE_SCHEMA_VERSION, TRAINING_SOURCE_FORMAT, TrainingSchema
from training.data.dataset import TrainingDataset
from common.policy.data.compiled_cache import CACHE_FORMAT, CompiledShardCache, load_compiled_cache


@pytest.mark.parametrize("job_tag", ["black_mage", "machinist"])
def test_output_space_is_sorted_enabled_and_excludes_padding(job_tag):
    config = load_project_config(job_tag=job_tag)
    vocab = SkillVocab.build_from_config(config)
    space = ActionSpace.from_config(config, skill_vocab=vocab)
    enabled = {skill.key: skill for skill in (*config.system.skills, *config.job.skills) if skill.enabled}
    assert space.action_keys == tuple(sorted(space.action_keys))
    assert set(space.action_keys) == set(enabled) | {"ogcd_wait"}
    assert len(set(space.action_to_vocab_id)) == len(space.action_keys)
    assert all(row > 0 for row in space.action_to_vocab_id)
    assert space.action_is_gcd[space.action_keys.index("ogcd_wait")] is False
    assert space.action_to_vocab_id[space.action_keys.index("ogcd_wait")] == vocab.require_lookup(0, context="wait")
    for key, row in zip(space.action_keys, space.action_to_vocab_id):
        if key in enabled:
            assert vocab.reverse_lookup(row) == enabled[key].game_id
            assert space.action_is_gcd[space.action_keys.index(key)] is (enabled[key].kind.value == "gcd")


def test_disabled_skill_remains_input_vocabulary_but_is_never_an_output_action():
    config = load_project_config(job_tag="black_mage")
    selected = next(skill for skill in config.job.skills if skill.enabled)
    job = replace(config.job, skills=tuple(replace(skill, enabled=False) if skill.key == selected.key else skill for skill in config.job.skills))
    modified = replace(config, job=job)
    vocab = SkillVocab.build_from_config(modified)
    assert vocab.lookup(selected.game_id) > 0
    assert selected.key not in ActionSpace.from_config(modified, skill_vocab=vocab).action_keys


def test_dataset_rejects_action_order_drift_instead_of_reordering():
    dataset = TrainingDataset.__new__(TrainingDataset)
    dataset._action_keys = ("a", "b")
    with pytest.raises(ValueError, match="action output order mismatch"):
        dataset._prepare_sample({"action_keys": ["b", "a"]})


@pytest.mark.parametrize("rows", [(0, 2), (1, 1), (1,)])
def test_data_spec_rejects_invalid_shared_vocabulary_mapping(rows):
    with pytest.raises(ValueError):
        DataSpec(job_tag="test", num_actions=2, state_dim=2, scene_dim=3,
                 skill_feature_dim=1, num_scene_types=1, action_keys=("a", "b"),
                 skill_feature_names=("kind",), action_to_vocab_id=rows, action_is_gcd=(True, False))


@pytest.mark.parametrize("drift", ["enabled", "kind", "order", "mapping"])
def test_cache_uses_caller_action_contract_when_current_configuration_changes(tmp_path, monkeypatch, drift):
    import common.policy.data.compiled_cache as cache_module

    space = ActionSpace.from_job_tag("black_mage")
    path = tmp_path / "manifest.pt"
    path.write_bytes(b"placeholder")
    # 先提供有效 manifest；随后只改变当前配置，保留模型保存的 DataSpec。
    spec = DataSpec("black_mage", len(space.action_keys), 1, 0, 0, 0,
                    space.action_keys, (), space.action_to_vocab_id, space.action_is_gcd)
    schema = TrainingSchema(
        serialization_format=TRAINING_SOURCE_FORMAT,
        sample_schema_version=TRAINING_SAMPLE_SCHEMA_VERSION,
        context_schema_version=CANONICAL_CONTEXT_SCHEMA_VERSION,
        scene_context_mode="absolute", scene_windows=(),
        state_group_feature_keys={"player_state": ("request_state.time_seconds",)}, skill_history_fields=(),
    )
    payload = {
        "cache_format": CACHE_FORMAT, "cache_signature": {}, "schema": schema,
        "job_tag": "black_mage", "num_samples": 0, "num_actions": len(space.action_keys),
        "skill_feature_names": (), "action_keys": space.action_keys,
        "action_to_vocab_id": space.action_to_vocab_id, "action_is_gcd": space.action_is_gcd, "vocab_signature": ((0, 1),),
        "shard_size": 1, "shard_files": [],
        "history_bank": {"skill_ids": torch.zeros(1, dtype=torch.long), "skill_features": torch.zeros(1, 0),
                         "state_abs_values": torch.zeros(1, 1), "state_delta_values": torch.zeros(1, 1),
                         "state_null_mask": torch.zeros(1, 1, dtype=torch.bool), "state_delta_reset_mask": torch.zeros(1, 1, dtype=torch.bool),
                         "action_keys": ("",), "skill_potencies": torch.zeros(1), "cumulative_dot_potencies": torch.zeros(1)},
    }
    monkeypatch.setattr(cache_module, "safe_torch_load", lambda *_args, **_kwargs: payload)
    def load(expected):
        return load_compiled_cache(
            path, tmp_path / "raw", signature={}, shard_cache=CompiledShardCache(),
            expected_action_space=expected,
        )

    assert load(space) is not None
    if drift == "enabled":
        changed = ActionSpace(space.action_keys[1:], space.action_to_vocab_id[1:], space.action_is_gcd[1:])
    elif drift == "kind":
        changed = replace(space, action_is_gcd=(not space.action_is_gcd[0], *space.action_is_gcd[1:]))
    elif drift == "order":
        changed = ActionSpace(space.action_keys[::-1], space.action_to_vocab_id[::-1], space.action_is_gcd[::-1])
    else:
        changed = replace(space, action_to_vocab_id=space.action_to_vocab_id[::-1])
    monkeypatch.setattr(ActionSpace, "from_job_tag", lambda _job: changed)
    # 新训练使用当前动作配置时仍拒绝漂移；模型恢复则继续使用保存的契约。
    assert load(ActionSpace.from_job_tag("black_mage")) is None
    assert load(ActionSpace.from_data_spec(spec)) is not None
    monkeypatch.setattr(ActionSpace, "from_job_tag", lambda _job: pytest.fail("通用 loader 不应读取 YAML"))
    assert load(ActionSpace.from_data_spec(spec)) is not None


@pytest.mark.parametrize("flags", [(), (True,), (True, 0)])
def test_data_spec_rejects_missing_or_nonboolean_action_kinds(flags):
    with pytest.raises(ValueError, match="action_is_gcd"):
        DataSpec(job_tag="test", num_actions=2, state_dim=2, scene_dim=3,
                 skill_feature_dim=1, num_scene_types=1, action_keys=("a", "b"),
                 skill_feature_names=("kind",), action_to_vocab_id=(1, 2), action_is_gcd=flags)

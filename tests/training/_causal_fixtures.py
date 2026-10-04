"""不依赖 C#、运行期职业 YAML 的微型因果模型测试契约。"""

from __future__ import annotations

from dataclasses import asdict

import torch

from common.policy.config import ModelConfig
from common.policy.data import DataSpec, ModelInputContract, Normalizer, SkillVocab
from common.policy.data.normalization import NormalizerConfig
from common.policy.data.normalizer import NORMALIZER_CONTRACT_VERSION
from common.policy.data.schema import SceneWindowSchema, TrainingSchema, TRAINING_SAMPLE_SCHEMA_VERSION
from common.output_context_schema import CANONICAL_CONTEXT_SCHEMA_VERSION


def make_data_spec(**overrides) -> DataSpec:
    values = dict(job_tag="black_mage", num_actions=2, state_dim=4, scene_dim=4,
                  skill_feature_dim=2, num_scene_types=1, action_keys=("first", "second"),
                  skill_feature_names=("kind", "potency"), action_to_vocab_id=(1, 2), action_is_gcd=(True, True))
    values.update(overrides)
    return DataSpec(**values)


def make_batch(data_spec=None, *, batch_size=1, history_length=2, scene_length=1,
               dtype=torch.float32) -> dict[str, object]:
    spec = data_spec or make_data_spec()
    return {
        "history_skill_ids": torch.ones((batch_size, history_length), dtype=torch.long),
        "history_skill_features": torch.zeros((batch_size, history_length, spec.skill_feature_dim), dtype=dtype),
        "history_state_vectors": torch.zeros((batch_size, history_length, spec.state_dim), dtype=dtype),
        "history_state_null_mask": torch.zeros((batch_size, history_length, spec.state_dim), dtype=torch.bool),
        "history_mask": torch.ones((batch_size, history_length), dtype=torch.bool),
        "current_state_vectors": torch.zeros((batch_size, spec.state_dim), dtype=dtype),
        "current_state_null_mask": torch.zeros((batch_size, spec.state_dim), dtype=torch.bool),
        "scene_vectors": torch.zeros((batch_size, scene_length, spec.scene_dim), dtype=dtype),
        "scene_types": torch.zeros((batch_size, scene_length), dtype=torch.long),
        "scene_mask": torch.ones((batch_size, scene_length), dtype=torch.bool),
        "action_legal_mask": torch.ones((batch_size, spec.num_actions), dtype=torch.bool),
        "action_values": torch.ones((batch_size, spec.num_actions), dtype=dtype),
        "action_keys": [list(spec.action_keys) for _ in range(batch_size)],
        "history_action_keys": [[spec.action_keys[0]] * history_length for _ in range(batch_size)],
        "label_index": torch.zeros((batch_size,), dtype=torch.long),
        "label_action_key": [spec.action_keys[0]] * batch_size,
    }


def make_input_contract(data_spec=None, *, vocab_size=None) -> ModelInputContract:
    spec = data_spec or make_data_spec()
    if spec.scene_dim and spec.scene_dim < 3:
        raise ValueError("scene test contract requires the three time fields")
    keys = ("start_offset_seconds", "end_offset_seconds", "duration_seconds",
            *(f"feature_{index}" for index in range(max(0, spec.scene_dim - 3))))
    windows = () if not spec.scene_dim else (SceneWindowSchema.from_feature_keys(
        context_key="targetable_window_context", feature_keys=keys, scene_type_id=0,
    ),)
    schema = TrainingSchema(serialization_format="test", sample_schema_version=TRAINING_SAMPLE_SCHEMA_VERSION,
                            context_schema_version=CANONICAL_CONTEXT_SCHEMA_VERSION, scene_context_mode="absolute",
                            scene_windows=windows,
                            state_group_feature_keys={"player_state": tuple(f"field_{index}" for index in range(spec.state_dim))},
                            skill_history_fields=("kind", "potency"))
    normalizer = Normalizer.from_contract({
        "version": NORMALIZER_CONTRACT_VERSION, "job_tag": spec.job_tag,
        "config": asdict(NormalizerConfig()), "resource_limits": {}, "status_limits": {},
    })
    normalizer.register_schema(schema)
    vocab_size = vocab_size or max(spec.action_to_vocab_id) + 1
    vocab = SkillVocab.from_entries([(1000 + row, row) for row in range(1, vocab_size)])
    return ModelInputContract.from_training(data_spec=spec, schema=schema, normalizer=normalizer, skill_vocab=vocab)


def make_checkpoint(data_spec=None, config=None, *, model_state_dict=None) -> dict[str, object]:
    spec = data_spec or make_data_spec()
    config = config or ModelConfig(d_model=8, n_layers=1, n_heads=2,
                                  num_kv_heads=1, ff_dim=16, dropout=0.0)
    embedding = (model_state_dict or {}).get("input_encoder.skill_embed.weight")
    vocab_size = None if embedding is None else int(embedding.shape[0])
    return {"model_config": asdict(config), "data_spec": asdict(spec),
            "input_contract": make_input_contract(spec, vocab_size=vocab_size).to_dict(),
            "model_state_dict": {} if model_state_dict is None else model_state_dict}

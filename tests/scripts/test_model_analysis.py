"""模型科研分析工具测试。"""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from matplotlib.colors import Normalize
from torch import nn

import common.cache_compilation as cache_compilation
from common.torch_serialization import safe_torch_load
from common.policy.model.input_encoder import (
    ROLE_SCENE, ROLE_STATE, ROLE_SKILL, build_position_ids, build_role_ids,
)
from scripts.model_analysis.common import pca_projection, skill_name_by_vocab_id
from scripts.model_analysis import common as analysis_common
from scripts.model_analysis.job_labels import decision_state_labels
from scripts.model_analysis.job_labels.black_mage import decision_state_labels as black_mage_labels
from scripts.model_analysis.outputs import attention as attention_output
from scripts.model_analysis.outputs import hidden_statistics as hidden_output
from scripts.model_analysis.outputs import loss_landscape as loss_output
from scripts.model_analysis.outputs import history_embedding as history_output
from scripts.model_analysis.outputs import pca_layers as pca_output
from scripts.model_analysis.outputs import skill_embedding as skill_output
from scripts.model_analysis.outputs.pca_layers import _strongest_pca_axis
from scripts.model_analysis.outputs.skill_embedding import _skill_labels
from scripts.model_analysis.token_metadata import build_token_metadata
from common.policy.config import (
    resolve_policy_checkpoint_path,
    resolve_policy_device,
    resolve_policy_model_config_path,
    resolve_policy_model_job_tag,
)
from training.config import RunConfig
from common.project_config import load_root_dotenv
from common.policy.data import ActionSpace, DataSpec, ModelInputContract, Normalizer, SkillVocab
from common.policy.data.schema import TrainingSchema, TRAINING_SAMPLE_SCHEMA_VERSION
from common.output_context_schema import CANONICAL_CONTEXT_SCHEMA_VERSION
from tests.training._causal_fixtures import make_state_groups
from training.loop import _save_checkpoint


def test_analysis_scene_uses_independent_env_and_cli_override(monkeypatch, tmp_path):
    monkeypatch.setattr(analysis_common, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("MODEL_ANALYSIS_SCENE_JSON", "analysis.json.br")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_JSON", "replay.json.br")
    kwargs = dict(raw_root=tmp_path, cache_dir=tmp_path / ".cache",
                  job_tag="black_mage", cache_shard_size=768,
                  normalizer=object(), skill_vocab=object(),
                  expected_action_space=ActionSpace(("a", "b"), (1, 2), (True, False)))
    assert analysis_common._resolve_analysis_source(None, **kwargs) == tmp_path / "analysis.json.br"
    assert analysis_common._resolve_analysis_source(Path("cli.json.br"), **kwargs) == tmp_path / "cli.json.br"


def test_analysis_scene_defaults_to_prepared_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_ANALYSIS_SCENE_JSON", " ")
    monkeypatch.setenv("AUTOREGRESSIVE_REPLAY_SCENE_JSON", "replay.json.br")
    expected = tmp_path / "90-100" / "scene.json.br"
    calls = []

    def select(root, **kwargs):
        calls.append((root, kwargs))
        return expected

    monkeypatch.setattr(analysis_common, "find_prepared_scene_source", select)
    action_space = ActionSpace(("a", "b"), (1, 2), (True, False))
    normalizer, vocab = object(), object()
    assert analysis_common._resolve_analysis_source(
        None, raw_root=tmp_path, cache_dir=tmp_path / ".cache",
        job_tag="black_mage", cache_shard_size=768,
        expected_action_space=action_space,
        normalizer=normalizer, skill_vocab=vocab,
    ) == expected
    assert calls == [(tmp_path, dict(cache_dir=tmp_path / ".cache",
                                   job_tag="black_mage", cache_shard_size=768,
                                   expected_action_space=action_space,
                                   normalizer=normalizer, expected_skill_vocab=vocab))]


def test_pca_projection_returns_coordinates_and_explained_variance():
    values = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
        ]
    )

    coordinates, explained = pca_projection(values, 3)

    assert coordinates.shape == (4, 3)
    assert explained.shape == (3,)
    assert np.isclose(explained.sum(), 1.0)
    assert explained[0] >= explained[1] >= explained[2]


def test_pca_axis_association_identifies_numeric_decision_axis():
    coordinates = np.array([
        [-2.0, 0.1],
        [-1.0, -0.1],
        [1.0, 0.0],
        [2.0, 0.2],
    ])
    values = np.array([0.0, 1.0, 3.0, 4.0])

    axis, score = _strongest_pca_axis(coordinates, values, numeric=True)

    assert axis == "PC1"
    assert score > 0.9


def test_model_analysis_decodes_mp_as_linear_0_to_10000_value():
    layout = _analysis_schema().state_layout(("a", "b"))

    batch = {
        "current_state_abs_values": torch.tensor([[2.0, 0.0, 5000.0, 0.0]]),
        "current_state_null_mask": torch.zeros((1, 4), dtype=torch.bool),
        "action_legal_mask": torch.tensor([[True]]),
    }

    elemental_state, mp = decision_state_labels("black_mage", 0, batch=batch, layout=layout)

    assert elemental_state == "AF2"
    assert mp == 5000.0


def test_training_model_defaults_use_root_env_and_model_output_dir(monkeypatch):
    monkeypatch.setenv("TRAINING_MODEL_CHECKPOINT", "best.pt")
    monkeypatch.setenv("FFXIV_JOB_TAG", "black_mage")
    config_path = resolve_policy_model_config_path()
    checkpoint_path = resolve_policy_checkpoint_path(config_path)

    assert config_path.name == "config.yaml"
    assert config_path.parent.name == "artzip"
    assert resolve_policy_model_job_tag(config_path) == "black_mage"
    assert checkpoint_path.parent.name == "artzip_bc"
    assert checkpoint_path.name == "best.pt"


def test_training_device_comes_from_root_env(monkeypatch):
    monkeypatch.setenv("TRAINING_DEVICE", "cpu")

    assert resolve_policy_device() == "cpu"


def test_load_root_dotenv_uses_python_dotenv_parser(monkeypatch, tmp_path: Path):
    quoted_key = "CODEX_TEST_DOTENV_QUOTED"
    exported_key = "CODEX_TEST_DOTENV_EXPORTED"
    monkeypatch.delenv(quoted_key, raising=False)
    monkeypatch.delenv(exported_key, raising=False)
    (tmp_path / ".env").write_text(
        f'{quoted_key}="hello world"\n'
        f'export {exported_key}=exported\n',
        encoding="utf-8",
    )

    assert load_root_dotenv(tmp_path) == tmp_path / ".env"
    assert os.environ[quoted_key] == "hello world"
    assert os.environ[exported_key] == "exported"


def test_checkpoint_exposes_job_tag_at_top_level(tmp_path: Path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    config = RunConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        job_tag=None,
        model_variant="artzip",
    )
    data_spec = DataSpec(
        job_tag="black_mage",
        num_actions=1,
        state_dim=3,
        base_state_dim=1,
        scene_dim=0,
        skill_feature_dim=1,
        num_scene_types=0,
        action_keys=("fire",),
        action_to_vocab_id=(1,),
        action_is_gcd=(True,),
        skill_feature_names=("id",),
    )
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    input_contract = ModelInputContract.from_training(
        skill_vocab=SkillVocab.from_entries([(1001, 1)]),
        data_spec=data_spec,
        schema=TrainingSchema(
            serialization_format="test",
            sample_schema_version=TRAINING_SAMPLE_SCHEMA_VERSION,
            context_schema_version=CANONICAL_CONTEXT_SCHEMA_VERSION,
            scene_context_mode="absolute",
            scene_windows=(),
            state_groups=make_state_groups({"player_state": ("request_state.time_seconds",)}, ("fire",)),
            state_snapshots=("previous_action_after", "request_state"),
            skill_history_fields=("id",),
        ),
        normalizer=normalizer,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"

    _save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        1,
        config,
        data_spec,
        {},
        input_contract=input_contract,
    )
    payload = safe_torch_load(checkpoint_path)

    assert payload["job_tag"] == "black_mage"
    assert payload["model_variant"] == "artzip"
    assert payload["data_spec"]["job_tag"] == "black_mage"


def test_skill_labels_reject_empty_dataset():
    context = type("EmptyContext", (), {"dataset": []})()

    with pytest.raises(ValueError, match="dataset is empty"):
        _skill_labels(context)


def test_skill_labels_use_chinese_names_from_project_config():
    class Vocab:
        @staticmethod
        def size():
            return 3

        @staticmethod
        def reverse_lookup(vocab_id):
            return {1: 0, 2: 152}[vocab_id]

    context = type(
        "AnalysisContext",
        (),
        {
            "dataset": [object()],
            "data_spec": type("DataSpec", (), {"job_tag": "black_mage"})(),
            "vocab": Vocab(),
        },
    )()

    assert skill_name_by_vocab_id(context) == {1: "空输出", 2: "爆炎"}


def test_sample_token_count_requires_mapping_and_sums_distinct_token_groups():
    sample = {"scene_abs_values": [[0.0]], "history_skill_ids": [1, 2]}
    assert attention_output._sample_token_count(sample) == 6
    with pytest.raises(TypeError, match="expected mapping sample"):
        attention_output._sample_token_count(object())
    with pytest.raises(TypeError, match="scene_abs_values"):
        attention_output._sample_token_count({"scene_abs_values": 1})


def test_sample_token_count_supports_compact_history_bank_samples():
    samples = [{"scene_abs_values": [[0.0]], "history_length": value} for value in (0, 2, 5)]
    assert [attention_output._sample_token_count(sample) for sample in samples] == [2, 6, 12]
    assert max(samples, key=attention_output._sample_token_count)["history_length"] == 5
    assert attention_output._sample_token_count({"scene_abs_values": [[0.0]] * 3, "history_length": 2}) == 8


def _analysis_schema(action_keys=("a", "b"), *, include_mp=True):
    player_keys = ("request_state.mp", "request_state.time_seconds") if include_mp else ("request_state.time_seconds",)
    return TrainingSchema(
        serialization_format="test", sample_schema_version=TRAINING_SAMPLE_SCHEMA_VERSION,
        context_schema_version=CANONICAL_CONTEXT_SCHEMA_VERSION, scene_context_mode="absolute",
        scene_windows=(),
        state_groups=make_state_groups({
            "resource_state": ("request_state.astral_fire", "request_state.umbral_ice"),
            "player_state": player_keys,
        }, action_keys),
        state_snapshots=("previous_action_after", "request_state"),
        skill_history_fields=("potency",),
    )


def _analysis_context(tmp_path):
    current_mask = np.array([False, False, True, True, True, True, False, False])
    roles = np.array([ROLE_SCENE, ROLE_STATE, ROLE_STATE, ROLE_STATE, ROLE_STATE, ROLE_STATE, ROLE_SKILL, ROLE_SCENE])
    vectors = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    features = {
        "fight_id": np.array(["f"] * 8),
        "step_index": np.arange(8, dtype=float),
        "label_index": np.array([-1.0, -1.0, 0.0, 1.0, 2.0, 3.0, -1.0, -1.0]),
        "prediction_index": np.array([-1.0, -1.0, 1.0, 1.0, 0.0, 3.0, -1.0, -1.0]),
        "skill_id": np.array([-1.0, -1.0, 1.0, 2.0, 3.0, 4.0, -1.0, -1.0]),
        "label_legal": np.array(["not_decision", "not_decision", "legal", "illegal", "legal", "legal", "not_decision", "not_decision"]),
        "elemental_state": np.array(["unknown", "unknown", "UI3", "UI3", "AF1", "AF1", "unknown", "unknown"]),
        "mp_bucket": np.array([np.nan, np.nan, 0.0, 2500.0, 5000.0, 10000.0, np.nan, np.nan]),
        "label_rank": np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, np.nan, np.nan]),
        "model_logit": np.array([np.nan, np.nan, 4.0, 3.0, 2.0, 1.0, np.nan, np.nan]),
    }
    layer_vectors = [vectors, vectors + 0.25]
    layer_roles = [roles, roles]
    layer_metadata = [features, features]

    class SkillVocab:
        @staticmethod
        def size():
            return 5

    model = SimpleNamespace(
        input_encoder=SimpleNamespace(skill_embed=nn.Embedding(5, 4)),
    )
    model.input_encoder.skill_embed.weight.data.copy_(torch.arange(20, dtype=torch.float32).reshape(5, 4))
    return SimpleNamespace(
        checkpoint_path=tmp_path / "checkpoint.pt",
        source_path=tmp_path / "data.json",
        output_dir=tmp_path / "plots",
        model=model,
        dataset=[object(), object()],
        data_spec=SimpleNamespace(job_tag="black_mage", action_keys=("fire_iii", "fire_iv")),
        vocab=SkillVocab(),
        device=torch.device("cpu"),
        precision="float32",
        autocast=nullcontext,
        encode_batch=lambda batch: batch,
        layer_vectors=layer_vectors,
        layer_roles=layer_roles,
        layer_current_state_masks=[current_mask, current_mask],
        layer_metadata=layer_metadata,
    )


def _history_input_encoding(skill, state, history_mask, *, scene_length=2):
    """完整编码结果替身：scene/current 占位与交错历史必须可明确区分。"""
    batch_size, history_length, dim = skill.shape
    current = scene_length + 2 * history_length
    tokens = torch.full((batch_size, current + 1, dim), -1000.0)
    state_positions = (scene_length + 2 * torch.arange(history_length)).unsqueeze(0).expand(batch_size, -1)
    skill_positions = state_positions + 1
    tokens.scatter_(1, state_positions.unsqueeze(-1).expand(-1, -1, dim), state)
    tokens.scatter_(1, skill_positions.unsqueeze(-1).expand(-1, -1, dim), skill)
    tokens[:, current] = 1000.0
    scene_mask = torch.ones((batch_size, scene_length), dtype=torch.bool)
    prefix_valid = torch.cat((scene_mask, history_mask.repeat_interleave(2, dim=1)), dim=1)
    valid = torch.cat((prefix_valid, torch.ones((batch_size, 1), dtype=torch.bool)), dim=1)
    return {
        'tokens': tokens, 'padding_mask': ~valid, 'valid': valid, 'prefix_valid': prefix_valid,
        'scene_length': scene_length, 'history_length': history_length,
        'history_token_length': 2 * history_length, 'prefix_length': current,
        'position_ids': build_position_ids(
            batch_size=batch_size, scene_length=scene_length, history_length=history_length,
            scene_mask=scene_mask, history_mask=history_mask),
        'role_ids': build_role_ids(
            batch_size=batch_size, scene_length=scene_length, history_length=history_length, device='cpu'),
        'history_skill_positions': skill_positions, 'history_state_positions': state_positions,
        'current_state_position': current,
        'current_state_positions': torch.full((batch_size,), current, dtype=torch.long),
    }


def test_model_analysis_metadata_and_black_mage_fallbacks():
    layout = _analysis_schema().state_layout(("a", "b"))
    samples = [{"metadata": {"fight_id": "fight-1", "step": 7}}]
    batch = {
        "history_skill_ids": torch.tensor([[9, 0]], dtype=torch.int32),
        "history_mask": torch.tensor([[True, False]]),
        "action_legal_mask": torch.tensor([[True, False]]),
        "current_state_abs_values": torch.tensor([[0.0, 1.0, 5000.0, 0.0]]),
        "current_state_null_mask": torch.zeros((1, 4), dtype=torch.bool),
        "label_index": torch.tensor([0]),
    }
    encoded = {
        "role_ids": torch.tensor([[ROLE_SCENE, ROLE_STATE, ROLE_SKILL, ROLE_STATE, ROLE_SKILL, ROLE_STATE]]),
        "current_state_positions": torch.tensor([5]),
        "history_skill_positions": torch.tensor([[2, 4]]),
    }
    metadata = build_token_metadata(
        samples, batch=batch, encoded=encoded, layout=layout,
        job_tag="black_mage", logits=np.array([[3.0, 1.0]]),
    )
    assert metadata["fight_id"][0, 0] == "fight-1"
    assert metadata["skill_id"][0, 2] == 9
    assert metadata["skill_id"][0, 4] == -1
    assert metadata["skill_id"][0, 5] == -1
    assert metadata["label_legal"][0, 5] == "legal"
    assert metadata["label_rank"][0, 5] == 1.0
    assert metadata["model_logit"][0, 5] == 3.0
    assert metadata["prediction_index"][0, 5] == 0
    assert np.isnan(metadata["label_rank"][0, 1])
    assert decision_state_labels("machinist", 0, batch=batch, layout=layout)[0] == "unknown"
    missing_layout = SimpleNamespace(group_slices={}, base_groups=())
    assert black_mage_labels(0, batch=batch, layout=missing_layout)[0] == "unknown"
    missing_values = dict(batch)
    missing_values["current_state_null_mask"] = torch.ones((1, 4), dtype=torch.bool)
    elemental, mp = black_mage_labels(0, batch=missing_values, layout=layout)
    assert elemental == "neutral"
    assert np.isnan(mp)


def test_model_analysis_grid_shape_matches_near_square_rule():
    expected = {
        1: (1, 1),
        2: (2, 1),
        3: (2, 2),
        4: (2, 2),
        7: (3, 3),
        8: (3, 3),
        12: (4, 3),
        16: (4, 4),
        19: (5, 4),
        20: (5, 4),
    }
    for tile_count, shape in expected.items():
        assert analysis_common.grid_shape(tile_count) == shape

    with pytest.raises(ValueError, match="must be positive"):
        analysis_common.grid_shape(0)

    fig, axes = analysis_common.create_grid_figure(19)
    try:
        assert axes.shape == (5, 4)
        analysis_common.hide_empty_tiles(axes, 19)
        assert axes.flat[18].get_visible() is True
        assert axes.flat[19].get_visible() is False
    finally:
        plt.close(fig)

    single_fig, single_ax = analysis_common.create_figure((6.0, 4.0))
    try:
        assert single_ax.get_figure() is single_fig
    finally:
        plt.close(single_fig)


def test_model_analysis_common_helpers_and_context_loading(monkeypatch, tmp_path):
    context = _analysis_context(tmp_path)
    empty_vectors = analysis_common._concat_arrays(
        [], empty_shape=(0, 0), dtype=np.float32
    )
    empty_roles = analysis_common._concat_arrays(
        [], empty_shape=(0,), dtype=np.int64
    )
    empty_metadata = analysis_common._concat_arrays(
        [], empty_shape=(0,), dtype=object
    )
    assert empty_vectors.shape == (0, 0)
    assert empty_vectors.dtype == np.float32
    assert empty_roles.shape == (0,)
    assert empty_roles.dtype == np.int64
    assert empty_metadata.shape == (0,)
    assert empty_metadata.dtype == object
    assert analysis_common._resolve_device("cpu") == torch.device("cpu")
    monkeypatch.setattr(analysis_common.torch.cuda, "is_available", lambda: False)
    assert analysis_common._resolve_device("cuda") == torch.device("cpu")
    with pytest.raises(ValueError, match="at least 3 rows"):
        pca_projection(np.zeros((2, 3)), 3)

    analysis_common.configure_matplotlib()
    assert context.output_dir != tmp_path

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "data_spec": {
                "job_tag": "black_mage",
                "num_actions": 2,
                "state_dim": 5,
                "base_state_dim": 1,
                "scene_dim": 1,
                "skill_feature_dim": 1,
                "num_scene_types": 1,
                "action_keys": ["a", "b"],
                "action_to_vocab_id": [1, 2],
                "action_is_gcd": [True, False],
                "skill_feature_names": ["potency"],
            },
            "model_state_dict": {},
        },
        checkpoint_path,
    )

    class FakeVocab:
        @staticmethod
        def build_from_job_tag(_job):
            return FakeVocab()

        @staticmethod
        def size():
            return 2

    class FakeDataset:
        job_tag = "black_mage"
        num_actions = 2
        state_dim = 5
        base_state_dim = 1
        scene_dim = 1
        num_scene_types = 1
        action_keys = ("a", "b")
        action_to_vocab_id = (1, 2)
        action_is_gcd = (True, False)
        skill_feature_names = ("potency",)
        schema = SimpleNamespace()

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"metadata": {"fight_id": "f", "step": 0}}

    class FakeModel:
        encoder = SimpleNamespace(layers=[0, 1])

        @staticmethod
        def checkpoint_model_config(_checkpoint):
            return SimpleNamespace()

        def __init__(self, *_args, **_kwargs):
            pass

        def load_state_dict(self, _state):
            return None

        def to(self, _device):
            return self

        def eval(self):
            return self

        def trace(self, _batch):
            encoded = {
                "padding_mask": torch.zeros((1, 4), dtype=torch.bool),
                "role_ids": torch.tensor([[ROLE_SCENE, ROLE_STATE, ROLE_SKILL, ROLE_STATE]]),
                "current_state_positions": torch.tensor([3]),
            }
            return SimpleNamespace(
                encoded=encoded,
                hidden=torch.zeros((1, 4, 2)),
                layer_hidden=[torch.ones((1, 4, 2)), torch.ones((1, 4, 2)) * 2],
                attentions=[],
            )

        @staticmethod
        def score_hidden(_encoded, _hidden, _batch):
            return torch.tensor([[1.0, 0.0]])

    class FakeNormalizer:
        def register_schema(self, _schema):
            pass

    monkeypatch.setattr(analysis_common, "resolve_project_job_tag", lambda **_kwargs: "black_mage")
    monkeypatch.setattr(analysis_common, "SkillVocab", FakeVocab)
    monkeypatch.setattr(analysis_common, "CausalPolicyModel", FakeModel)
    monkeypatch.setattr(analysis_common, "TrainingDataset", lambda *_args, **_kwargs: FakeDataset())
    monkeypatch.setattr(analysis_common, "Normalizer", FakeNormalizer)
    monkeypatch.setattr(analysis_common, "TrainingCollator", lambda: (lambda samples: {}))
    monkeypatch.setattr(analysis_common, "_resolve_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(analysis_common, "build_token_metadata", lambda *args, **kwargs: {
        feature: np.zeros((1, 4), dtype=object if feature in {"fight_id", "legal", "invalid_reason", "elemental_state"} else float)
        for feature in analysis_common.ANALYSIS_FEATURES
    })
    monkeypatch.setattr(analysis_common.DataSpec, "from_dataset", lambda _dataset: analysis_common.DataSpec(
        job_tag="black_mage", num_actions=2, state_dim=5, base_state_dim=1,
        scene_dim=1, skill_feature_dim=1, num_scene_types=1, action_keys=("a", "b"),
        skill_feature_names=("potency",), action_to_vocab_id=(1, 2), action_is_gcd=(True, False)
    ))
    monkeypatch.setattr(analysis_common.DataSpec, "assert_compatible_with", lambda self, other: None)
    monkeypatch.setattr(analysis_common, "move_batch", lambda batch, device: batch)

    # Avoid exercising reader internals here; the remaining helpers and output modules
    # are tested with focused fake contexts below.
    assert context.layer_vectors[0].shape == (8, 4)


@pytest.mark.parametrize("reordered_schema", [False, True])
def test_analysis_restores_saved_input_contract_without_current_yaml(monkeypatch, tmp_path, reordered_schema):
    from dataclasses import asdict, replace
    from tests.scripts.onnx_export.test_exporter import _write_small_checkpoint

    checkpoint = tmp_path / "model.pt"
    spec, saved = _write_small_checkpoint(checkpoint)
    seen = {}

    class Dataset:
        def __init__(self, sources, **kwargs):
            seen["dataset"] = kwargs
            for key, value in asdict(spec).items():
                setattr(self, key, value)
            self.schema = saved.schema
            if reordered_schema:
                self.schema = replace(saved.schema, state_groups=tuple(
                    replace(group, feature_keys=tuple(reversed(group.feature_keys)))
                    if group.encoding == "anchored_delta" else group
                    for group in saved.schema.state_groups
                ))

        def __len__(self):
            return 1

    def select(_root, **kwargs):
        seen["source"] = kwargs
        return tmp_path / "source.json.br"

    monkeypatch.setenv("MODEL_ANALYSIS_SCENE_JSON", "")
    monkeypatch.setattr(analysis_common, "TrainingDataset", Dataset)
    monkeypatch.setattr(analysis_common, "find_prepared_scene_source", select)
    monkeypatch.setattr(analysis_common, "Normalizer", lambda: pytest.fail("分析不得从当前 YAML 重建归一化器"))
    monkeypatch.setattr(SkillVocab, "build_from_job_tag", lambda *_a, **_kw: pytest.fail("分析不得重建当前技能词表"))
    kwargs = dict(checkpoint_path=checkpoint, source_path=None, raw_root=tmp_path,
                  cache_dir=tmp_path / ".cache", max_history=4, cache_shard_size=512,
                  cache_max_shards=2, output_dir=tmp_path / "analysis", device_name="cpu", precision="float32")
    if reordered_schema:
        with pytest.raises(ValueError, match="training state feature keys mismatch"):
            analysis_common.load_loss_landscape_context(**kwargs)
        return
    context = analysis_common.load_loss_landscape_context(**kwargs)
    assert context.vocab.to_dict() == saved.create_skill_vocab().to_dict()
    normalizer = seen["dataset"]["normalizer"]
    assert normalizer.normalization_contract == saved.normalizer_contract
    time_index = saved.schema.state_layout(spec.action_keys).base_feature_keys.index("previous_action_after.time_seconds")
    assert context.context_encoder.state_divisors[time_index] == pytest.approx(120.0)
    assert seen["source"]["normalizer"] is normalizer
    assert seen["source"]["expected_skill_vocab"] is seen["dataset"]["skill_vocab"]
    assert context.model.input_encoder.skill_embed.weight.shape[0] == 8


def test_model_analysis_retries_after_compiling_missing_cache(monkeypatch, tmp_path):
    dataset_calls = []
    compiled_calls = []
    sentinel_dataset = object()

    def fake_dataset(source_paths, **kwargs):
        dataset_calls.append((source_paths, kwargs))
        if len(dataset_calls) == 1:
            raise FileNotFoundError("missing compiled cache")
        return sentinel_dataset

    monkeypatch.setattr(analysis_common, "TrainingDataset", fake_dataset)
    monkeypatch.setattr(
        analysis_common,
        "compile_raw_training_caches",
        lambda **kwargs: compiled_calls.append(kwargs),
    )

    source_path = tmp_path / "source.json"
    cache_dir = tmp_path / "cache"
    action_space = ActionSpace(("a", "b"), (1, 2), (True, False))
    vocab = SkillVocab.from_entries([(1001, 1), (1002, 2)])
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    result = analysis_common._load_analysis_dataset(
        source_path=source_path,
        cache_dir=cache_dir,
        max_history=240,
        history_reset_keep=8,
        cache_shard_size=768,
        cache_max_shards=24,
        job_tag="black_mage",
        skill_vocab=vocab,
        normalizer=normalizer,
        expected_action_space=action_space,
    )

    assert result is sentinel_dataset
    assert len(dataset_calls) == 2
    assert all(call[1]["max_history"] == 240 for call in dataset_calls)
    assert all(call[1]["expected_action_space"] == action_space for call in dataset_calls)
    assert compiled_calls == [
        {
            "source_paths": [source_path],
            "cache_dir": cache_dir,
            "cache_shard_size": 768,
            "job_tag": "black_mage",
            "expected_action_space": action_space,
            "normalizer": normalizer,
            "expected_skill_vocab": vocab,
        }
    ]


def test_model_analysis_cache_compilation_uses_inprocess_batch_api(monkeypatch, tmp_path):
    calls = []
    def compile(paths, **kwargs):
        calls.append((paths, kwargs))
        return paths
    monkeypatch.setattr("scripts.convert_fflogs.cache.precompile_raw_training_caches", compile)
    source = tmp_path / "source.json"
    cache_compilation.compile_raw_training_caches(
        source_paths=[source], cache_dir=tmp_path / "cache", cache_shard_size=768, job_tag="black_mage",
    )
    shared = object()
    cache_compilation.compile_raw_training_caches(
        source_paths=[source, tmp_path / "second.json", source], cache_dir=tmp_path / "cache",
        cache_shard_size=768, job_tag="black_mage", engine=shared, workers=2,
    )
    assert calls[0][0] == [source.resolve()]
    assert calls[0][1]["shard_size"] == 768
    assert len(calls[1][0]) == 2
    assert calls[1][1]["engine"] is shared
    assert calls[1][1]["max_workers"] == 2


def test_model_analysis_outputs_generate_pngs(monkeypatch, tmp_path):
    context = _analysis_context(tmp_path)
    context.output_dir.mkdir(parents=True)
    monkeypatch.setattr(skill_output, "skill_name_by_vocab_id", lambda _context: {
        1: "爆炎", 2: "炽炎", 3: "冰封", 4: "悖论"
    })
    monkeypatch.setattr(history_output, "skill_name_by_vocab_id", lambda _context: {
        1: "爆炎", 2: "炽炎", 3: "冰封", 4: "悖论"
    })

    paths = hidden_output.plot_hidden_statistics(context)
    assert all(path.is_file() for path in paths)
    pca_paths = pca_output.plot_layer_pca(context)
    assert len(pca_paths) == 2 + len(analysis_common.ANALYSIS_FEATURES)
    assert pca_paths[0].is_file()
    assert pca_output._strongest_pca_axis(
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        np.array([1.0, 1.0, 1.0]),
        numeric=True,
    )[1] == 0.0
    assert pca_output._strongest_pca_axis(
        np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]),
        np.array(["a", "a", "b", "b"]),
        numeric=False,
    )[0] == "PC2"
    handles, labels = pca_output._legend_handles()
    assert len(handles) == len(labels) == 3

    skill_path = skill_output.plot_skill_embedding(context)
    assert skill_path.is_file()

    class FakeHistoryEncoder:
        def __call__(self, batch):
            size = batch["history_skill_ids"].shape[0]
            values = torch.arange(size * 2 * 3, dtype=torch.float32).reshape(size, 2, 3)
            return _history_input_encoding(values, values + 1.0, batch['history_mask'])

    context.model.input_encoder = FakeHistoryEncoder()
    monkeypatch.setattr(history_output, "TrainingCollator", lambda: (lambda samples: {
        "history_skill_ids": torch.tensor([[1, 2] for _ in samples], dtype=torch.int32),
        "history_mask": torch.ones((len(samples), 2), dtype=torch.bool),
    }))
    history_paths = history_output.plot_history_embeddings(context, batch_size=1, max_points=2)
    assert all(path.is_file() for path in history_paths)
    with pytest.raises(ValueError, match="batch size"):
        history_output.plot_history_embeddings(context, batch_size=0)
    with pytest.raises(ValueError, match="at least 2"):
        history_output.plot_history_embeddings(context, max_points=1)


def test_model_analysis_layer_pca_reuses_projection_cache(monkeypatch, tmp_path):
    context = _analysis_context(tmp_path)
    context.output_dir.mkdir(parents=True)
    calls = []
    original = pca_output.pca_projection

    def record_projection(values, components, **kwargs):
        calls.append((values.shape, components))
        return original(values, components, **kwargs)

    monkeypatch.setattr(pca_output, "pca_projection", record_projection)
    pca_output.plot_layer_pca(context)

    assert calls == [
        ((8, 4), 3),
        ((4, 4), 2),
        ((8, 4), 3),
        ((4, 4), 2),
    ]


def test_compact_causal_analysis_preserves_gathered_skill_ids_and_absolute_state_labels(monkeypatch, tmp_path):
    """正式 compact collator 分析必须同时读取已 gather 的历史身份和未归一化的当前状态。"""
    from common.policy.config import ModelConfig
    from common.policy.data.context_encoding import ContextEncoder
    from common.policy.model import CausalPolicyModel
    from tests.training._common_fixtures import make_dataset, make_demo_pt
    from training import TrainingCollator

    source = make_demo_pt(tmp_path, ["fire_iii", "fire_iv", "blizzard_iii"], fight_id="compact_analysis")
    dataset = make_dataset([source], max_history=4, history_reset_keep=2)
    samples = [dataset[index] for index in range(len(dataset))]
    raw_batch = TrainingCollator()(samples)
    assert "history_bank_skill_ids" in raw_batch
    assert "history_skill_ids" not in raw_batch
    data_spec = DataSpec.from_dataset(dataset)
    model = CausalPolicyModel(
        data_spec,
        ModelConfig(d_model=16, n_layers=1, n_heads=2, num_kv_heads=1, ff_dim=32,
                    dropout=0.0, history_capacity=4, history_reset_keep=2),
        vocab_size=SkillVocab.build_from_job_tag(data_spec.job_tag).size(),
    ).eval()
    layout = dataset.schema.state_layout(data_spec.action_keys)
    encoder = ContextEncoder(dataset.normalizer, dataset.schema, model.config, layout=layout)
    runtime = SimpleNamespace(
        model=model, dataset=dataset, data_spec=data_spec, device=torch.device("cpu"),
        checkpoint_path=tmp_path / "checkpoint.pt", source_path=source,
        output_dir=tmp_path, vocab=SimpleNamespace(), precision="float32", autocast=nullcontext,
        context_encoder=encoder,
    )
    captured = []

    def capture_metadata(samples, **kwargs):
        result = build_token_metadata(samples, **kwargs)
        captured.append((kwargs["batch"], kwargs["encoded"], result))
        return result

    monkeypatch.setattr(analysis_common, "_load_model_analysis_context", lambda **_kwargs: runtime)
    monkeypatch.setattr(analysis_common, "build_token_metadata", capture_metadata)
    context = analysis_common.load_analysis_context(
        checkpoint_path=runtime.checkpoint_path, source_path=source,
        raw_root=tmp_path, cache_dir=tmp_path / ".cache", max_history=4,
        cache_shard_size=1, cache_max_shards=1, output_dir=tmp_path,
        max_samples=len(dataset), max_tokens=512, batch_size=len(dataset),
        device_name="cpu", precision="float32",
    )
    assert len(captured) == 1
    metadata_batch, encoded, metadata = captured[0]
    prepared = encoder.encode(raw_batch)
    torch.testing.assert_close(metadata_batch["history_skill_ids"], prepared["history_skill_ids"])
    torch.testing.assert_close(metadata_batch["current_state_abs_values"], raw_batch["current_state_abs_values"])
    player_group = next(group for group in layout.base_groups if group.group_key == "player_state")
    request_mp_index = player_group.feature_keys.index("request_state.mp")
    player_slice = layout.group_slices["player_state"]
    for index, position in enumerate(encoded["current_state_positions"].tolist()):
        expected_mp = raw_batch["current_state_abs_values"][index, player_slice][request_mp_index].item()
        assert metadata["mp_bucket"][index, position] == pytest.approx(expected_mp)
        history_positions = encoded["history_skill_positions"][index]
        valid = prepared["history_mask"][index]
        np.testing.assert_array_equal(
            metadata["skill_id"][index, history_positions[valid].numpy()],
            prepared["history_skill_ids"][index, valid].numpy(),
        )
    assert np.nanmax(context.layer_metadata[0]["mp_bucket"]) > 1.0


def test_real_causal_trace_loads_metadata_and_keeps_current_query_under_token_cap(monkeypatch, tmp_path):
    """完整分析入口必须兼容真实 trace 和附加历史身份元数据。"""
    from tests.training.test_kv_cache import _make_model, _make_batch

    model = _make_model()
    sample = {"metadata": {"fight_id": "real-trace", "step": 0}}
    batch = _make_batch(1)
    batch["current_state_abs_values"] = batch["current_state_vectors"][:, :model.data_spec.base_state_dim]
    schema = _analysis_schema(model.data_spec.action_keys, include_mp=False)
    layout = schema.state_layout(model.data_spec.action_keys)
    batch["label_index"] = torch.tensor([0])
    runtime = SimpleNamespace(
        model=model, dataset=SimpleNamespace(schema=schema),
        data_spec=model.data_spec, device=torch.device("cpu"),
        checkpoint_path=tmp_path / "checkpoint.pt", source_path=tmp_path / "source.json",
        output_dir=tmp_path, vocab=SimpleNamespace(), precision="float32", autocast=nullcontext,
        context_encoder=SimpleNamespace(encode=lambda batch: batch, layout=layout),
    )
    class Dataset:
        def __init__(self):
            self.schema = schema
        def __len__(self):
            return 1
        def __getitem__(self, _index):
            return sample
    runtime.dataset = Dataset()
    monkeypatch.setattr(analysis_common, "_load_model_analysis_context", lambda **_kwargs: runtime)
    monkeypatch.setattr(analysis_common, "TrainingCollator", lambda: (lambda _samples: batch))
    context = analysis_common.load_analysis_context(
        checkpoint_path=runtime.checkpoint_path, source_path=runtime.source_path,
        raw_root=tmp_path, cache_dir=tmp_path, max_history=4, cache_shard_size=1,
        cache_max_shards=1, output_dir=tmp_path, max_samples=1, max_tokens=1,
        batch_size=1, device_name="cpu", precision="float32",
    )
    expected_logit = model(batch)["logits"][0, 0].item()
    for roles, metadata in zip(context.layer_roles, context.layer_metadata):
        assert roles.tolist() == [ROLE_STATE]
        assert set(metadata) == set(analysis_common.ANALYSIS_FEATURES)
        # trace 与 SDPA 存在 FP32 舍入差异；接近 0 时补绝对容差，保留原相对阈值。
        assert metadata["model_logit"][0] == pytest.approx(expected_logit, rel=1e-6, abs=1e-6)
    assert all(mask.tolist() == [True] for mask in context.layer_current_state_masks)
    layer_projections, query_projections = pca_output._build_pca_cache(context)
    assert layer_projections[0][0].shape == (1, 3)
    assert query_projections[0][0].shape == (1, 2)
    assert np.isfinite(query_projections[0][0]).all()


def test_history_embedding_views_exclude_padding_and_support_empty_history(monkeypatch, tmp_path):
    context = _analysis_context(tmp_path)
    context.output_dir.mkdir(parents=True)
    class Encoder:
        def __call__(self, batch):
            values = torch.tensor([[[1.0, 2.0], [float("nan"), float("nan")]]])
            return _history_input_encoding(values, values + 3.0, batch['history_mask'])
    context.model.input_encoder = Encoder()
    monkeypatch.setattr(history_output, "skill_name_by_vocab_id", lambda _context: {1: "fire"})
    values = {
        "history_skill_ids": torch.tensor([[1, 0]]),
        "history_mask": torch.tensor([[True, False]]),
    }
    monkeypatch.setattr(history_output, "TrainingCollator", lambda: (lambda _samples: values))
    assert all(path.is_file() for path in history_output.plot_history_embeddings(context, batch_size=1))
    values["history_mask"] = torch.zeros((1, 2), dtype=torch.bool)
    assert all(path.is_file() for path in history_output.plot_history_embeddings(context, batch_size=1))


def test_history_skill_and_state_pca_fit_distinct_valid_observations(monkeypatch, tmp_path):
    """显式位置只选归一化后的历史输入，scene/current 与 padding 都不能混入。"""
    context = _analysis_context(tmp_path)
    context.output_dir.mkdir(parents=True)
    observed = {
        "skill": torch.tensor([[[0.0, 1.0], [2.0, 3.0], [float("nan"), float("nan")]]]),
        "state": torch.tensor([[[10.0, 0.0], [0.0, 10.0], [float("nan"), float("nan")]]]),
    }
    class Encoder:
        def __call__(self, batch):
            encoded = _history_input_encoding(observed['skill'], observed['state'], batch['history_mask'])
            assert encoded['history_state_positions'].tolist() == [[2, 4, 6]]
            assert encoded['history_skill_positions'].tolist() == [[3, 5, 7]]
            assert encoded['current_state_positions'].tolist() == [8]
            return encoded

        @staticmethod
        def embed_history(_batch):
            pytest.fail('PCA 必须使用正式 forward 输出，不能直接读取原始 content')

    context.model.input_encoder = Encoder()
    monkeypatch.setattr(history_output, "skill_name_by_vocab_id", lambda _context: {1: "fire", 2: "ice"})
    monkeypatch.setattr(history_output, "TrainingCollator", lambda: (lambda _samples: {
        "history_skill_ids": torch.tensor([[1, 2, 0]]),
        "history_mask": torch.tensor([[True, True, False]]),
    }))
    fitted = []
    original = history_output.pca_projection

    def record(values, components):
        fitted.append(values.copy())
        return original(values, components)

    monkeypatch.setattr(history_output, "pca_projection", record)
    paths = history_output.plot_history_embeddings(context, batch_size=2)
    assert [path.name for path in paths] == [
        "05_history_skill_embedding_pca.png", "05_history_state_embedding_pca.png",
    ]
    assert all(path.is_file() for path in paths)
    assert len(fitted) == 2
    np.testing.assert_array_equal(fitted[0], observed["skill"][0, :2].numpy())
    np.testing.assert_array_equal(fitted[1], observed["state"][0, :2].numpy())


def test_history_pca_uses_actual_normalized_input_encoder_output(monkeypatch, tmp_path):
    """用真实编码器核对 post-role 输入，不在分析或测试中复制 RMS 公式。"""
    from common.policy.config import ModelConfig
    from common.policy.model.input_encoder import CausalInputEncoder
    from tests.training._causal_fixtures import make_batch, make_data_spec

    torch.manual_seed(81)
    encoder = CausalInputEncoder(
        make_data_spec(), ModelConfig(d_model=8, n_layers=1, n_heads=2, ff_dim=16), vocab_size=3)
    batch = make_batch(batch_size=2, scene_length=2, history_length=2)
    batch['history_skill_ids'] = torch.tensor([[1, 0], [2, 1]])
    batch['history_mask'] = torch.tensor([[True, False], [True, True]])
    batch['scene_mask'] = torch.tensor([[False, True], [True, True]])
    batch['history_state_vectors'] = torch.cat((
        torch.randn(2, 2, encoder.data_spec.base_state_dim),
        torch.randint(0, 2, (2, 2, encoder.data_spec.state_dim - encoder.data_spec.base_state_dim)).float(),
    ), dim=-1)
    batch['history_skill_features'] = torch.randn(2, 2, 2)
    batch['current_state_vectors'] = torch.cat((
        torch.randn(2, encoder.data_spec.base_state_dim),
        torch.randint(0, 2, (2, encoder.data_spec.state_dim - encoder.data_spec.base_state_dim)).float(),
    ), dim=-1)
    context = _analysis_context(tmp_path)
    context.output_dir.mkdir(parents=True)
    context.model.input_encoder = encoder
    monkeypatch.setattr(history_output, 'TrainingCollator', lambda: (lambda _samples: batch))
    monkeypatch.setattr(history_output, 'skill_name_by_vocab_id', lambda _context: {1: 'fire', 2: 'ice'})
    returned = []
    handle = encoder.register_forward_hook(lambda _module, _args, output: returned.append(output))
    fitted = []
    original = history_output.pca_projection

    def record(values, components):
        fitted.append(values.copy())
        return original(values, components)

    monkeypatch.setattr(history_output, 'pca_projection', record)
    try:
        paths = history_output.plot_history_embeddings(context, batch_size=2)
    finally:
        handle.remove()
    assert all(path.is_file() for path in paths)
    assert len(returned) == 1 and len(fitted) == 2
    encoded = returned[0]
    for actual, kind in zip(fitted, ('skill', 'state')):
        positions = encoded[f'history_{kind}_positions']
        expected = encoded['tokens'].gather(1, positions.unsqueeze(-1).expand(-1, -1, 8))
        np.testing.assert_array_equal(actual, expected[batch['history_mask']].numpy())
        np.testing.assert_allclose(np.sqrt(np.mean(actual ** 2, axis=1)), 1.0, atol=2e-5, rtol=0)


def test_loss_landscape_directions_are_filter_normalized_and_orthogonal():
    weight = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, -1.0, 0.5, 3.0],
        ],
        dtype=torch.float32,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)

    direction_x, direction_y = loss_output._filter_normalized_orthogonal_pair(
        weight,
        generator=generator,
    )

    expected_norms = torch.linalg.vector_norm(weight.double(), dim=1)
    np.testing.assert_allclose(
        torch.linalg.vector_norm(direction_x, dim=1).numpy(),
        expected_norms.numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        torch.linalg.vector_norm(direction_y, dim=1).numpy(),
        expected_norms.numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        (direction_x * direction_y).sum(dim=1).numpy(),
        np.zeros(2),
        atol=1e-12,
    )


def test_loss_landscape_exports_every_layer_and_restores_parameters(monkeypatch, tmp_path):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layers = nn.ModuleList(
                [
                    nn.Linear(2, 2),
                    nn.Linear(2, 2),
                ]
            )

        def forward(self, batch):
            hidden = batch["features"]
            hidden = torch.tanh(self.encoder.layers[0](hidden))
            logits = self.encoder.layers[1](hidden)
            return {"logits": logits}

    model = TinyModel()
    model.train()
    initial_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    dataset = [
        {"features": torch.tensor([1.0, 0.0]), "label_index": 0},
        {"features": torch.tensor([0.0, 1.0]), "label_index": 1},
        {"features": torch.tensor([1.0, 1.0]), "label_index": 0},
    ]
    context = SimpleNamespace(
        model=model,
        dataset=dataset,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        precision="float32",
        autocast=nullcontext,
        encode_batch=lambda batch: batch,
    )

    def collate(samples):
        return {
            "features": torch.stack([sample["features"] for sample in samples]),
            "label_index": torch.tensor(
                [sample["label_index"] for sample in samples],
                dtype=torch.int64,
            ),
        }

    monkeypatch.setattr(loss_output, "TrainingCollator", lambda: collate)
    paths = loss_output.plot_loss_landscape(
        context,
        resolution=3,
        radius=0.1,
        max_samples=3,
        batch_size=2,
        seed=7,
    )

    assert len(paths) == 4
    assert all(path.is_file() for path in paths)
    with np.load(tmp_path / "11_loss_landscape_values.npz") as values:
        assert values["losses"].shape == (2, 3, 3)
        assert values["losses"].dtype == np.float64
        assert np.isfinite(values["losses"]).all()
        np.testing.assert_allclose(
            values["losses"][:, 1, 1],
            np.repeat(values["baseline_loss"], 2),
            rtol=0.0,
            atol=0.0,
        )
    metadata = (tmp_path / "11_loss_landscape_metadata.json").read_text(encoding="utf-8")
    assert '"precision": "float32"' in metadata
    assert '"loss_accumulation_dtype": "float64"' in metadata
    assert '"layer": 2' in metadata
    assert model.training is True
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial_state[name], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"resolution": 4}, "odd integer"),
        ({"radius": 0.0}, "radius"),
        ({"batch_size": 0}, "batch size"),
    ],
)
def test_loss_landscape_rejects_invalid_options(kwargs, message):
    options = {
        "resolution": 3,
        "radius": 0.1,
        "max_samples": 1,
        "batch_size": 1,
    }
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        loss_output._validate_options(**options)


def test_opener_attention_uses_latest_state_query_and_excludes_padding(monkeypatch, tmp_path):
    """同类历史状态不能混作 query，逐样本位置和 padding 必须分别处理。"""
    weights = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.7, 0.0, 0.0, 0.0],
        [0.1, 0.1, 0.8, 0.0, 0.0],
        [0.6, 0.1, 0.1, 0.2, 0.0],
        [0.2, 0.1, 0.3, 0.9, 0.4],
    ]).reshape(1, 1, 5, 5).expand(2, 2, 5, 5)
    class Model:
        @staticmethod
        def trace(_batch):
            return SimpleNamespace(
                encoded={
                    "current_state_positions": torch.tensor([4, 3]),
                    "role_ids": torch.tensor([[ROLE_SCENE, ROLE_STATE, ROLE_SKILL, ROLE_STATE, ROLE_STATE]] * 2),
                    "padding_mask": torch.tensor([[False, False, False, True, False], [False, False, False, False, True]]),
                },
                hidden=torch.zeros(2, 5, 2), attentions=(weights,),
            )
        @staticmethod
        def score_hidden(_encoded, _hidden, _batch):
            return torch.tensor([[1.0, 0.0]] * 2)
    context = SimpleNamespace(
        model=Model(), data_spec=SimpleNamespace(action_keys=("a", "b")),
        dataset=[{"label_action_key": "a", "label_index": 0}] * 2,
        device=torch.device("cpu"), output_dir=tmp_path, autocast=nullcontext,
        encode_batch=lambda batch: batch,
    )
    captured = {}
    monkeypatch.setattr(attention_output, "TrainingCollator", lambda: (lambda _samples: {}))
    monkeypatch.setattr(attention_output, "_plot_query_attention",
                        lambda values, *_args: captured.update(query=values))
    monkeypatch.setattr(attention_output, "_plot_layer_attention",
                        lambda values, *_args: captured.update(layers=values))
    attention_output.plot_opener_attention(context, steps=2, batch_size=2)
    expected = np.array([[0.4, 1.0, 0.6], [1.0, 0.5, 1.0 / 6.0]])
    np.testing.assert_allclose(captured["query"], expected, rtol=1e-6)
    np.testing.assert_allclose(captured["layers"], expected.mean(axis=0, keepdims=True), rtol=1e-6)


@pytest.mark.parametrize(("scene_count", "history_count"), [(7, 300), (0, 300), (7, 0), (0, 0)])
def test_attention_structure_groups_interleaved_history_without_dense_grid(scene_count, history_count):
    """300 条交错历史也只显示区段边界，当前状态仍单独可辨。"""
    roles = np.array(
        [ROLE_SCENE] * scene_count + [ROLE_STATE, ROLE_SKILL] * history_count + [ROLE_STATE],
    )
    fig, ax = plt.subplots()
    try:
        ax.grid(True)
        attention_output._draw_attention_structure(ax, roles)
        section_count = int(scene_count > 0) + int(history_count > 0) + 1
        assert len(ax.get_xticklabels()) == section_count
        assert len(ax.get_yticklabels()) == section_count
        assert len(ax.lines) == 2 * (section_count - 1) + 1
        assert not any(line.get_visible() for line in (*ax.get_xgridlines(), *ax.get_ygridlines()))
        labels = [label.get_text() for label in ax.get_xticklabels()]
        assert labels[-1] == "current\nstate"
        if scene_count:
            assert labels[0] == f"scene\n{scene_count} tokens"
        if history_count:
            assert f"history: state / skill\n{2 * history_count} tokens" in labels
    finally:
        plt.close(fig)


def test_standard_attention_removes_padding_on_both_axes_without_changing_weights(monkeypatch, tmp_path):
    """失效场景和历史 padding 都裁掉，因果 mask、真实权重和 role 分母保持一致。"""
    roles = torch.tensor([[ROLE_SCENE, ROLE_SCENE, ROLE_STATE, ROLE_SKILL, ROLE_STATE, ROLE_SKILL, ROLE_STATE]])
    padding = torch.tensor([[False, True, False, False, True, True, False]])
    blocked = torch.triu(torch.ones((7, 7), dtype=torch.bool), diagonal=1)
    blocked[6, 0] = True
    weights = torch.arange(98, dtype=torch.float32).reshape(1, 2, 7, 7) / 100
    context = SimpleNamespace(
        dataset=[{"scene_abs_values": [[0.0]] * 2, "history_length": 2}],
        output_dir=tmp_path, device=torch.device("cpu"), encode_batch=lambda batch: batch,
        autocast=nullcontext,
        model=SimpleNamespace(trace=lambda _batch: SimpleNamespace(
            encoded={"role_ids": roles, "padding_mask": padding, "attention_mask": blocked},
            attentions=(weights,),
        )),
    )
    captured = {}
    monkeypatch.setattr(attention_output, "TrainingCollator", lambda: (lambda _samples: {}))
    monkeypatch.setattr(attention_output, "_plot_attention_matrix_grid",
                        lambda values, mask, ids, _path: captured.update(layers=values, mask=mask, roles=ids))
    monkeypatch.setattr(attention_output, "_plot_attention_head_grid",
                        lambda values, _mask, _ids, _path: captured.update(heads=values))
    monkeypatch.setattr(attention_output, "_plot_attention_role_blocks",
                        lambda values, mask, ids, _path: captured.update(
                            blocks=attention_output._role_block_means([values[0].mean(axis=0)], ids, mask)))
    attention_output.plot_standard_attention_outputs(context, steps=1)
    positions = np.array([0, 2, 3, 6])
    expected = weights[0].numpy()[:, positions, :][:, :, positions]
    np.testing.assert_array_equal(captured["layers"][0], expected)
    np.testing.assert_array_equal(captured["heads"], expected)
    np.testing.assert_array_equal(captured["mask"], blocked.numpy()[np.ix_(positions, positions)])
    np.testing.assert_array_equal(captured["roles"], [ROLE_SCENE, ROLE_STATE, ROLE_SKILL, ROLE_STATE])
    role_order = tuple(sorted(analysis_common.ROLE_NAMES))
    assert captured["blocks"][0, role_order.index(ROLE_STATE), role_order.index(ROLE_SCENE)] == pytest.approx(
        float(weights[0, :, 2, 0].mean()) / 2,
    )


def test_model_analysis_attention_output_and_main(monkeypatch, tmp_path):
    context = _analysis_context(tmp_path)
    context.output_dir.mkdir(parents=True)
    samples = [
        {
            "action_keys": ["fire_iii", "fire_iv"],
            "label_action_key": "fire_iii",
            "label_index": 0,
        }
        for _ in range(2)
    ]
    context.dataset = samples

    class FakeAttentionModel:
        @staticmethod
        def trace(_batch):
            encoded = {
                "current_state_positions": torch.tensor([2, 2]),
                "role_ids": torch.tensor([[ROLE_SCENE, ROLE_SKILL, ROLE_STATE]] * 2),
                "padding_mask": torch.zeros((2, 3), dtype=torch.bool),
            }
            attention = torch.ones((2, 2, 3, 3), dtype=torch.float32)
            return SimpleNamespace(
                encoded=encoded,
                hidden=torch.zeros((2, 3, 2)),
                attentions=[attention, attention * 2],
            )

        @staticmethod
        def score_hidden(_encoded, _hidden, _batch):
            return torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    context.model = FakeAttentionModel()
    monkeypatch.setattr(attention_output, "TrainingCollator", lambda: (lambda batch: {}))
    query_path, layer_path = attention_output.plot_opener_attention(
        context, steps=2, batch_size=1
    )
    assert query_path.is_file()
    assert layer_path.is_file()
    color_norm = attention_output._attention_color_norm()
    assert isinstance(color_norm, Normalize)
    assert float(color_norm(0.25)) == pytest.approx(0.25)
    query_relative = attention_output._normalize_query_attention(
        np.array([[0.1, 0.3, 0.6], [0.01, 0.02, 0.03]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        query_relative,
        [[1 / 6, 0.5, 1.0], [1 / 3, 2 / 3, 1.0]],
    )
    row_relative = attention_output._row_relative_attention(
        np.array([[0.1, 0.3, 0.6], [0.01, 0.02, 0.03]], dtype=np.float32),
        np.array([[False, False, False], [False, True, False]]),
    )
    np.testing.assert_allclose(row_relative, [[1 / 6, 0.5, 1.0], [1 / 3, 0.0, 1.0]])
    with pytest.raises(ValueError, match="steps"):
        attention_output.plot_opener_attention(context, steps=0)
    with pytest.raises(ValueError, match="batch size"):
        attention_output.plot_opener_attention(context, batch_size=0)

    standard_context = _analysis_context(tmp_path / "standard")
    standard_context.output_dir.mkdir(parents=True)
    standard_context.dataset = [
        {
            "scene_abs_values": [[0.0]],
            "history_skill_ids": [1, 2],
        }
    ]

    class StandardFakeAttentionModel:
        @staticmethod
        def trace(_batch):
            role_ids = torch.tensor([[0, 1, 1, 2, 2]], dtype=torch.int64)
            attention_mask = torch.zeros((5, 5), dtype=torch.bool)
            attention_mask[2, 0] = True
            padding_mask = torch.zeros((1, 5), dtype=torch.bool)
            attention = torch.ones((1, 2, 5, 5), dtype=torch.float32) / 5.0
            return SimpleNamespace(
                encoded={
                    "attention_mask": attention_mask,
                    "padding_mask": padding_mask,
                    "role_ids": role_ids,
                },
                attentions=(attention, attention * 2.0),
            )

    standard_context.model = StandardFakeAttentionModel()
    standard_paths = attention_output.plot_standard_attention_outputs(
        standard_context,
        steps=1,
    )
    assert len(standard_paths) == 3
    assert all(path.is_file() for path in standard_paths)

    import scripts.model_analysis.main as analysis_main

    fake_context = SimpleNamespace(
        checkpoint_path=tmp_path / "checkpoint.pt",
        source_path=tmp_path / "data.json",
        data_spec=SimpleNamespace(
            job_tag="black_mage",
            num_actions=2,
            state_dim=7,
            base_state_dim=3,
            scene_dim=1,
            skill_feature_dim=1,
        ),
        device=torch.device("cpu"),
        precision="float32",
        dataset=[1, 2, 3],
        layer_vectors=[np.zeros((2, 2))],
    )
    output_dir = tmp_path / "main-output"
    output_dir.mkdir(parents=True)
    torch.save({"model_variant": "artzip"}, tmp_path / "best.pt")
    monkeypatch.setattr(analysis_main, "resolve_policy_model_config_path", lambda _path: tmp_path / "config.yaml")
    monkeypatch.setattr(analysis_main, "resolve_policy_checkpoint_path", lambda _path: tmp_path / "best.pt")
    monkeypatch.setattr(analysis_main, "load_run_config", lambda _path: SimpleNamespace(
        raw_data_dir=tmp_path,
        model=SimpleNamespace(history_capacity=128),
        compiled_cache_shard_size=512,
        compiled_cache_max_shards=8,
        precision="float32",
    ))
    monkeypatch.setattr(analysis_main, "resolve_policy_model_job_tag", lambda _path: "black_mage")
    monkeypatch.setattr(analysis_main, "resolve_policy_model_variant", lambda _path: "artzip")
    monkeypatch.setattr(analysis_main, "resolve_policy_cache_dir", lambda _job: tmp_path / ".cache")
    analysis_context_calls = []
    monkeypatch.setattr(
        analysis_main,
        "load_analysis_context",
        lambda **kwargs: analysis_context_calls.append(kwargs) or fake_context,
    )
    loss_context_calls = []
    monkeypatch.setattr(
        analysis_main,
        "load_loss_landscape_context",
        lambda **kwargs: loss_context_calls.append(kwargs) or fake_context,
    )
    monkeypatch.setattr(analysis_main, "configure_matplotlib", lambda: None)
    monkeypatch.setattr(analysis_main, "plot_hidden_statistics", lambda _context: [tmp_path / "hidden.png"] * 2)
    monkeypatch.setattr(analysis_main, "plot_layer_pca", lambda _context: [tmp_path / "pca.png"])
    monkeypatch.setattr(analysis_main, "plot_opener_attention", lambda _context, **kwargs: (tmp_path / "a.png", tmp_path / "b.png"))
    monkeypatch.setattr(
        analysis_main,
        "plot_standard_attention_outputs",
        lambda _context, **kwargs: (tmp_path / "matrix.png", tmp_path / "heads.png", tmp_path / "blocks.png"),
    )
    monkeypatch.setattr(analysis_main, "plot_skill_embedding", lambda _context: tmp_path / "skill.png")
    monkeypatch.setattr(analysis_main, "plot_history_embeddings", lambda _context, **kwargs: (
        tmp_path / "history_skill.png", tmp_path / "history_state.png",
    ))
    monkeypatch.setattr(
        analysis_main,
        "plot_loss_landscape",
        lambda _context, **kwargs: [tmp_path / "loss.png", tmp_path / "loss.npz"],
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "model_analysis",
            "--output",
            str(output_dir),
            "--max-samples",
            "0",
            "--loss-landscape",
            "--device",
            "cpu",
        ],
    )
    analysis_main.main()
    metadata = (output_dir / "analysis_metadata.json").read_text(encoding="utf-8")
    assert '"job_tag": "black_mage"' in metadata
    assert '"enabled": true' in metadata
    assert '"resolution": 31' in metadata
    assert '"precision": "float32"' in metadata
    assert len(analysis_context_calls) == 1
    assert analysis_context_calls[0]["precision"] == "float32"
    assert len(loss_context_calls) == 1
    assert loss_context_calls[0]["precision"] == "float32"

    monkeypatch.setattr(
        "sys.argv",
        ["model_analysis", "--loss-landscape-only", "--output", str(output_dir), "--device", "cpu"],
    )
    analysis_main.main()
    assert len(analysis_context_calls) == 1
    assert len(loss_context_calls) == 2


def test_model_analysis_rejects_bf16_without_cuda(tmp_path):
    with pytest.raises(RuntimeError, match="bf16 requires CUDA"):
        analysis_common._load_model_analysis_context(
            checkpoint_path=tmp_path / "unused.pt",
            source_path=tmp_path / "unused.json",
            raw_root=tmp_path,
            cache_dir=tmp_path / "cache",
            max_history=1,
            cache_shard_size=1,
            cache_max_shards=1,
            output_dir=tmp_path / "output",
            device_name="cpu",
            precision="bf16",
        )


def test_model_analysis_rejects_old_checkpoint_contract_before_old_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(analysis_common, "safe_torch_load", lambda _path: {
        "input_contract": {"version": 9}, "data_spec": {"job_tag": "black_mage"},
    })
    with pytest.raises(ValueError, match="unsupported input contract version"):
        analysis_common._load_model_analysis_context(
            checkpoint_path=tmp_path / "old.pt", source_path=tmp_path / "source.json",
            raw_root=tmp_path, cache_dir=tmp_path, max_history=1,
            cache_shard_size=1, cache_max_shards=1, output_dir=tmp_path,
            device_name="cpu", precision="float32",
        )


@pytest.mark.parametrize("full_attention_residuals", [False, True])
@pytest.mark.parametrize("norm_first", [False, True])
def test_loss_landscape_cached_grid_matches_full_forward(full_attention_residuals, norm_first):
    from common.policy.model import RepetitionConfig
    from tests.training.test_kv_cache import _make_model, _make_batch

    if full_attention_residuals and not norm_first:
        pytest.skip("Full AttnRes requires Pre-LN")
    torch.manual_seed(19)
    model = _make_model(norm_first=norm_first, full_attention_residuals=full_attention_residuals)
    if not full_attention_residuals:
        # 使用差异明显的系数，确保缓存路径没有漏混合或错误复用第一层系数。
        with torch.no_grad():
            model.encoder.residual_mix.r.copy_(torch.tensor([0.8, 1.4]))
            model.encoder.residual_mix.a.copy_(torch.tensor([0.35, -0.15]))
    model.repetition = RepetitionConfig(mode="blacklist", skills=("fire_iv",), penalty=2.0)
    first = {key: torch.cat((value, value), dim=0) for key, value in _make_batch(3).items()}
    batches = (first, _make_batch(1))
    for index, batch in enumerate(batches):
        count = batch["current_state_vectors"].shape[0]
        batch["label_index"] = torch.full((count,), index)
        batch["action_keys"] = [("fire_iii", "fire_iv", "blizzard_iii")] * count
        batch["history_action_keys"] = [("fire_iv",)] * count
    context = SimpleNamespace(model=model, device=torch.device("cpu"), autocast=nullcontext,
                              encode_batch=lambda batch: batch)
    baseline = loss_output._mean_cross_entropy(context, batches)
    coordinates = np.linspace(-0.1, 0.1, 3)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    for layer in model.encoder.layers:
        directions = loss_output._build_layer_directions(layer, seed=7)
        expected = np.empty((3, 3))
        try:
            for iy, y in enumerate(coordinates):
                for ix, x in enumerate(coordinates):
                    loss_output._set_perturbed_parameters(directions, x_value=x, y_value=y)
                    expected[iy, ix] = loss_output._mean_cross_entropy(context, batches)
        finally:
            loss_output._restore_parameters(directions)
        actual = loss_output._evaluate_layer_grid(
            context, batches, directions, coordinates=coordinates,
            baseline_loss=baseline, center_index=1,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_loss_landscape_restores_parameters_on_forward_failure(monkeypatch):
    from tests.training.test_kv_cache import _make_model, _make_batch

    model = _make_model()
    batch = _make_batch(2)
    batch["label_index"] = torch.tensor([0])
    context = SimpleNamespace(model=model, device=torch.device("cpu"), autocast=nullcontext,
                              encode_batch=lambda batch: batch)
    directions = loss_output._build_layer_directions(model.encoder.layers[1], seed=7)

    def fail(_self):
        raise RuntimeError("deliberate failure")

    monkeypatch.setattr(loss_output._LayerForward, "logits", fail)
    with pytest.raises(RuntimeError, match="deliberate failure"):
        loss_output._evaluate_layer_grid(
            context, (batch,), directions, coordinates=np.linspace(-0.1, 0.1, 3),
            baseline_loss=0, center_index=1,
        )
    for parameter, base in zip(directions.parameters, directions.base_values):
        torch.testing.assert_close(parameter, base, rtol=0, atol=0)

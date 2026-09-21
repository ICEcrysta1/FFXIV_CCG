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
from common.policy.model.input_encoder import ROLE_CANDIDATE, ROLE_HISTORY
from scripts.model_analysis.common import pca_projection, skill_name_by_vocab_id
from scripts.model_analysis import common as analysis_common
from scripts.model_analysis.job_labels import decision_state_labels
from scripts.model_analysis.job_labels.black_mage import decision_state_labels as black_mage_labels
from scripts.model_analysis.outputs import attention as attention_output
from scripts.model_analysis.outputs import hidden_statistics as hidden_output
from scripts.model_analysis.outputs import loss_landscape as loss_output
from scripts.model_analysis.outputs import pair_embedding as pair_output
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
from common.policy.data import DataSpec, ModelInputContract, Normalizer
from common.policy.data.schema import TrainingSchema
from training.loop import _save_checkpoint


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
    class Schema:
        state_group_feature_keys = {
            "resource_state": ["before.astral_fire", "before.umbral_ice"],
            "player_state": ["before.mp"],
        }

        @staticmethod
        def state_group_slices():
            return {
                "resource_state": slice(0, 2),
                "player_state": slice(2, 3),
            }

    batch = {
        "candidate_state_vectors": torch.tensor([[[2.0, 0.0, 0.5]]]),
        "candidate_state_null_mask": torch.zeros((1, 1, 3), dtype=torch.bool),
        "candidate_legal_mask": torch.tensor([[True]]),
    }

    elemental_state, mp = decision_state_labels("black_mage", 0, batch=batch, schema=Schema())

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
    config = RunConfig(raw_data_dir=tmp_path, output_dir=tmp_path, job_tag=None)
    data_spec = DataSpec(
        job_tag="black_mage",
        num_candidates=1,
        state_dim=1,
        scene_dim=0,
        skill_feature_dim=1,
        num_scene_types=0,
        candidate_action_keys=("fire",),
        skill_feature_names=("id",),
    )
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    input_contract = ModelInputContract.from_training(
        data_spec=data_spec,
        schema=TrainingSchema(
            serialization_format="test",
            sample_schema_version=1,
            context_schema_version=1,
            scene_context_mode="absolute",
            scene_windows=(),
            state_group_feature_keys={"player_state": ("value",)},
            candidate_skill_fields=("id",),
            skill_history_fields=(),
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
    sample = {
        "scene_vectors": [[0.0]],
        "history_skill_ids": [1, 2],
        "candidate_skill_ids": [3, 4, 5],
    }

    assert attention_output._sample_token_count(sample) == 7

    with pytest.raises(TypeError, match="expected mapping sample"):
        attention_output._sample_token_count(object())

    with pytest.raises(TypeError, match="scene_vectors"):
        attention_output._sample_token_count({"scene_vectors": 1})


def test_sample_token_count_supports_compact_history_bank_samples():
    samples = [
        {
            "scene_vectors": [[0.0]],
            "history_length": history_length,
            "candidate_skill_ids": [3, 4, 5],
        }
        for history_length in (0, 2, 5)
    ]

    assert [attention_output._sample_token_count(sample) for sample in samples] == [
        5,
        7,
        10,
    ]
    assert max(samples, key=attention_output._sample_token_count)["history_length"] == 5


def _analysis_schema():
    class Schema:
        state_group_feature_keys = {
            "resource_state": ("before.astral_fire", "before.umbral_ice"),
            "player_state": ("before.mp",),
        }

        @staticmethod
        def state_group_slices():
            return {"resource_state": slice(0, 2), "player_state": slice(2, 3)}

        @staticmethod
        def state_vector_dim():
            return 3

        @staticmethod
        def scene_feature_dim():
            return 1

    return Schema()


def _analysis_context(tmp_path):
    candidate_mask = np.array([False, False, True, True, True, True, False, False])
    roles = np.array([0, ROLE_HISTORY, ROLE_CANDIDATE, ROLE_CANDIDATE, ROLE_CANDIDATE, ROLE_CANDIDATE, 3, 0])
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
        "candidate_index": np.array([-1.0, -1.0, 0.0, 1.0, 2.0, 3.0, -1.0, -1.0]),
        "skill_id": np.array([-1.0, -1.0, 1.0, 2.0, 3.0, 4.0, -1.0, -1.0]),
        "legal": np.array(["non_candidate", "non_candidate", "legal", "illegal", "legal", "legal", "non_candidate", "non_candidate"]),
        "invalid_reason": np.array(["", "", "", "requires_mp", "", "", "", ""]),
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
        data_spec=SimpleNamespace(job_tag="black_mage"),
        vocab=SkillVocab(),
        device=torch.device("cpu"),
        precision="float32",
        autocast=nullcontext,
        layer_vectors=layer_vectors,
        layer_roles=layer_roles,
        layer_metadata=layer_metadata,
    )


def test_model_analysis_metadata_and_black_mage_fallbacks():
    schema = _analysis_schema()
    samples = [{
        "metadata": {"fight_id": "fight-1", "step": 7},
        "candidate_invalid_reasons": ["", "requires_mp"],
    }]
    batch = {
        "history_skill_ids": torch.tensor([[9]], dtype=torch.int32),
        "candidate_skill_ids": torch.tensor([[10, 11]], dtype=torch.int32),
        "candidate_legal_mask": torch.tensor([[True, False]]),
        "candidate_state_vectors": torch.tensor([[[0.0, 1.0, 0.5], [0.0, 1.0, 0.5]]]),
        "candidate_state_null_mask": torch.zeros((1, 2, 3), dtype=torch.bool),
        "label_index": torch.tensor([0]),
    }
    encoded = {
        "role_ids": torch.tensor([[0, ROLE_HISTORY, ROLE_CANDIDATE, ROLE_CANDIDATE, 3]]),
        "candidate_positions": torch.tensor([2, 3]),
    }
    metadata = build_token_metadata(
        samples,
        batch=batch,
        encoded=encoded,
        schema=schema,
        job_tag="black_mage",
        logits=np.array([[3.0, 1.0]]),
    )
    assert metadata["fight_id"][0, 0] == "fight-1"
    assert metadata["skill_id"][0, 1] == 9
    assert metadata["legal"][0, 2] == "legal"
    assert metadata["invalid_reason"][0, 3] == "requires_mp"
    assert metadata["label_rank"][0, 2] == 1.0

    unknown = decision_state_labels("machinist", 0, batch=batch, schema=schema)
    assert unknown[0] == "unknown"
    missing_schema = SimpleNamespace(
        state_group_slices=lambda: {},
        state_group_feature_keys={},
    )
    assert black_mage_labels(0, batch=batch, schema=missing_schema)[0] == "unknown"

    no_legal = dict(batch)
    no_legal["candidate_legal_mask"] = torch.zeros((1, 2), dtype=torch.bool)
    no_legal["candidate_state_null_mask"] = torch.ones((1, 2, 3), dtype=torch.bool)
    elemental, mp = black_mage_labels(0, batch=no_legal, schema=schema)
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
    with pytest.raises(FileNotFoundError, match="no raw JSON"):
        analysis_common._find_default_raw(tmp_path / "missing")
    with pytest.raises(ValueError, match="at least 3 rows"):
        pca_projection(np.zeros((2, 3)), 3)

    analysis_common.configure_matplotlib()
    assert context.output_dir != tmp_path

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "data_spec": {
                "job_tag": "black_mage",
                "num_candidates": 2,
                "state_dim": 1,
                "scene_dim": 1,
                "skill_feature_dim": 1,
                "num_scene_types": 1,
                "candidate_action_keys": ["a", "b"],
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
        num_candidates = 2
        state_dim = 1
        scene_dim = 1
        num_scene_types = 1
        candidate_action_keys = ("a", "b")
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
                "role_ids": torch.tensor([[0, 1, ROLE_CANDIDATE, 3]]),
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
    monkeypatch.setattr(analysis_common, "CandidateTransformerModel", FakeModel)
    monkeypatch.setattr(analysis_common, "TrainingDataset", lambda *_args, **_kwargs: FakeDataset())
    monkeypatch.setattr(analysis_common, "Normalizer", FakeNormalizer)
    monkeypatch.setattr(analysis_common, "TrainingCollator", lambda: (lambda samples: {}))
    monkeypatch.setattr(analysis_common, "_resolve_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(analysis_common, "build_token_metadata", lambda *args, **kwargs: {
        feature: np.zeros((1, 4), dtype=object if feature in {"fight_id", "legal", "invalid_reason", "elemental_state"} else float)
        for feature in analysis_common.ANALYSIS_FEATURES
    })
    monkeypatch.setattr(analysis_common.DataSpec, "from_dataset", lambda _dataset: analysis_common.DataSpec(
        "black_mage", 2, 1, 1, 1, 1, ("a", "b"), ("potency",)
    ))
    monkeypatch.setattr(analysis_common.DataSpec, "assert_compatible_with", lambda self, other: None)
    monkeypatch.setattr(analysis_common, "move_batch", lambda batch, device: batch)

    # Avoid exercising reader internals here; the remaining helpers and output modules
    # are tested with focused fake contexts below.
    assert context.layer_vectors[0].shape == (8, 4)


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
        "Normalizer",
        lambda: SimpleNamespace(configure_job_resources=lambda _job_tag: None),
    )
    monkeypatch.setattr(
        analysis_common,
        "compile_raw_training_cache",
        lambda **kwargs: compiled_calls.append(kwargs),
    )

    source_path = tmp_path / "source.json"
    cache_dir = tmp_path / "cache"
    result = analysis_common._load_analysis_dataset(
        source_path=source_path,
        cache_dir=cache_dir,
        max_history=240,
        cache_shard_size=768,
        cache_max_shards=24,
        candidate_order_file=tmp_path / "candidate_order.yaml",
        job_tag="black_mage",
        skill_vocab=object(),
    )

    assert result is sentinel_dataset
    assert len(dataset_calls) == 2
    assert all(
        call[1]["candidate_order_file"] == tmp_path / "candidate_order.yaml"
        for call in dataset_calls
    )
    assert compiled_calls == [
        {
            "source_path": source_path,
            "cache_dir": cache_dir,
            "cache_shard_size": 768,
            "job_tag": "black_mage",
        }
    ]


def test_model_analysis_cache_compilation_calls_conversion_cli(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, *, cwd, check):
        calls.append((command, cwd, check))

    monkeypatch.setattr(cache_compilation.subprocess, "run", fake_run)
    source_path = tmp_path / "source.json"
    cache_dir = tmp_path / "cache"
    cache_compilation.compile_raw_training_cache(
        source_path=source_path,
        cache_dir=cache_dir,
        cache_shard_size=768,
        job_tag="black_mage",
    )

    assert len(calls) == 1
    command, cwd, check = calls[0]
    assert command[1:4] == ["-m", "scripts.convert_fflogs.cli", str(source_path.resolve())]
    assert "--job-tag" in command
    assert "--cache-root" in command
    assert cwd == cache_compilation.PROJECT_ROOT
    assert check is True


def test_model_analysis_outputs_generate_pngs(monkeypatch, tmp_path):
    context = _analysis_context(tmp_path)
    context.output_dir.mkdir(parents=True)
    monkeypatch.setattr(skill_output, "skill_name_by_vocab_id", lambda _context: {
        1: "爆炎", 2: "炽炎", 3: "冰封", 4: "悖论"
    })
    monkeypatch.setattr(pair_output, "skill_name_by_vocab_id", lambda _context: {
        1: "爆炎", 2: "炽炎", 3: "冰封", 4: "悖论"
    })

    paths = hidden_output.plot_hidden_statistics(context)
    assert all(path.is_file() for path in paths)
    pca_paths = pca_output.plot_layer_pca(context)
    assert len(pca_paths) == 12
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
    assert len(handles) == len(labels) == 4

    skill_path = skill_output.plot_skill_embedding(context)
    assert skill_path.is_file()

    class FakePairEncoder:
        @staticmethod
        def embed_pairs(batch):
            size = batch["candidate_skill_ids"].shape[0]
            return {"candidate": torch.arange(size * 2 * 3, dtype=torch.float32).reshape(size, 2, 3)}

    context.model.input_encoder = FakePairEncoder()
    monkeypatch.setattr(pair_output, "TrainingCollator", lambda: (lambda samples: {
        "candidate_skill_ids": torch.tensor([[1, 2] for _ in samples], dtype=torch.int32),
    }))
    pair_path = pair_output.plot_pair_embedding(context, batch_size=1, max_points=2)
    assert pair_path.is_file()
    with pytest.raises(ValueError, match="batch size"):
        pair_output.plot_pair_embedding(context, batch_size=0)
    with pytest.raises(ValueError, match="at least 2"):
        pair_output.plot_pair_embedding(context, max_points=1)


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


def test_model_analysis_attention_output_and_main(monkeypatch, tmp_path):
    context = _analysis_context(tmp_path)
    context.output_dir.mkdir(parents=True)
    samples = [
        {
            "candidate_action_keys": ["fire_iii", "fire_iv"],
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
                "candidate_positions": torch.tensor([1, 2]),
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
    candidate_path, layer_path = attention_output.plot_opener_attention(
        context, steps=2, batch_size=1
    )
    assert candidate_path.is_file()
    assert layer_path.is_file()
    color_norm = attention_output._attention_color_norm()
    assert isinstance(color_norm, Normalize)
    assert float(color_norm(0.25)) == pytest.approx(0.25)
    candidate_relative = attention_output._normalize_candidate_attention(
        np.array([[0.1, 0.3, 0.6], [0.01, 0.02, 0.03]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        candidate_relative,
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
            "scene_vectors": [[0.0]],
            "history_skill_ids": [1, 2],
            "candidate_skill_ids": [1, 2],
        }
    ]

    class StandardFakeAttentionModel:
        @staticmethod
        def trace(_batch):
            role_ids = torch.tensor([[0, 1, 1, 2, 2, 3]], dtype=torch.int64)
            attention_mask = torch.zeros((6, 6), dtype=torch.bool)
            attention_mask[2, 0] = True
            padding_mask = torch.zeros((1, 6), dtype=torch.bool)
            attention = torch.ones((1, 2, 6, 6), dtype=torch.float32) / 6.0
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
            num_candidates=2,
            state_dim=3,
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
        candidate_order_file=None,
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
    monkeypatch.setattr(analysis_main, "plot_pair_embedding", lambda _context, **kwargs: tmp_path / "pair.png")
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
            candidate_order_file=None,
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
    model.repetition = RepetitionConfig(mode="blacklist", skills=("fire_iv",), penalty=2.0)
    first = {key: torch.cat((value, value), dim=0) for key, value in _make_batch(3).items()}
    batches = (first, _make_batch(1))
    for index, batch in enumerate(batches):
        count = batch["candidate_skill_ids"].shape[0]
        batch["label_index"] = torch.full((count,), index)
        batch["candidate_action_keys"] = [("fire_iii", "fire_iv", "blizzard_iii")] * count
        batch["history_action_keys"] = [("fire_iv",)] * count
    context = SimpleNamespace(model=model, device=torch.device("cpu"), autocast=nullcontext)
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
    context = SimpleNamespace(model=model, device=torch.device("cpu"), autocast=nullcontext)
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

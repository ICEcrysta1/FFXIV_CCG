"""训练模型公共层的边界、配置和训练流程测试。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

import common.project_config as project_config_module
from common.torch_runtime import autocast_context, model_dtype, move_batch
from common.torch_serialization import safe_torch_load
from scripts.convert_fflogs import cache as cache_module
from scripts.convert_fflogs import cache_compile as cache_compile_module
from scripts.convert_fflogs import cache_paths as cache_paths_module
from common.policy.config import ModelConfig
from common.policy import config as policy_config_module
from common.policy.data import DataSpec, ModelInputContract, Normalizer
from common.policy.data.schema import SceneWindowSchema, TrainingSchema
from common.policy.model import RepetitionConfig, build_split_attention_mask
from training.config import RunConfig, ValuePreferenceConfig
from common.policy.model.input_encoder import CandidateInputEncoder
from common.policy.model.attention_residual import FullAttentionResidual
from common.policy.model.position_encoding import RotaryPositionEncoding
from common.policy.model.repetition import (
    apply_repetition_penalty,
    parse_repetition_config,
    repetition_config_from_checkpoint,
)
from common.policy.model.trace import TraceableTransformerEncoderLayer, trace_encoder
import training.config as config_module
import training.loop.dataloaders as dataloaders_module
import training.loop as training_module


def _attach_rope(encoder):
    """让直接构造的底层 encoder 使用生产模型的 RoPE 边界。"""
    encoder.rotary_position_encoding = RotaryPositionEncoding(
        encoder.layers[0].self_attn.head_dim
    )
    return encoder


def _write_config(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (["invalid"], "training config must be a mapping"),
        ({"model": [1]}, "model and training config sections must be mappings"),
        ({"training": [1]}, "model and training config sections must be mappings"),
        ({"model": {"state_dim": 1}}, "cache-derived dimensions"),
        ({"model": {"pair_embedding_dim": 0}}, "pair_embedding_dim must be positive"),
        (
            {"model": {"n_heads": 6, "num_kv_heads": 4}},
            "model.n_heads must be divisible by model.num_kv_heads",
        ),
        ({"model": {"transformer_activation": "swish"}}, "transformer_activation must be gelu, relu or swiglu"),
        ({"model": {"ff_dim": 0}}, "model.ff_dim must be positive"),
        (
            {
                "model": {
                    "full_attention_residuals": True,
                    "transformer_norm_first": False,
                }
            },
            "full_attention_residuals requires transformer_norm_first=true",
        ),
        (
            {"model": {"full_attention_residuals": "maybe"}},
            "model.full_attention_residuals must be a boolean",
        ),
        (
            {"model": {"full_attention_residuals": 2}},
            "model.full_attention_residuals must be a boolean",
        ),
        (
            {"model": {"position_encoding": "absolute"}},
            "model.position_encoding is removed",
        ),
        (
            {"model": {"max_sequence_length": 128}},
            "model.max_sequence_length is removed",
        ),
        (
            {"model": {"max_history": 128}},
            "model.max_history is removed",
        ),
        (
            {"model": {"history_capacity": -1}},
            "model.history_capacity must be >= 0",
        ),
        (
            {"model": {"scorer_use_raw_projection": True}},
            "model.scorer_use_raw_projection is removed",
        ),
        (
            {"training": {"max_history": 128}},
            "training.max_history is removed",
        ),
        ({"training": {"num_workers": -1}}, "num_workers must be >= 0"),
        ({"training": {"prefetch_factor": 0}}, "prefetch_factor must be >= 1"),
        ({"training": {"compiled_cache_shard_size": 0}}, "compiled_cache_shard_size must be >= 1"),
        ({"training": {"compiled_cache_shard_size": 0}}, "compiled_cache_shard_size must be >= 1"),
        ({"training": {"compiled_cache_max_shards": 0}}, "compiled_cache_max_shards must be >= 1"),
        ({"training": {"compiled_cache_workers": 0}}, "compiled_cache_workers must be >= 1"),
        ({"training": {"history_truncation": [1]}}, "history_truncation must be a mapping"),
        ({"training": {"history_truncation": {"probability": 1.1}}}, "probability must be between 0 and 1"),
        ({"training": {"history_truncation": {"min_recent": 0}}}, "history_min_recent must be >= 1"),
        ({"training": {"candidate_shuffle": [1]}}, "candidate_shuffle must be a mapping"),
        ({"training": {"candidate_shuffle": {"probability": -0.1}}}, "probability must be between 0 and 1"),
        ({"training": {"value_preference": [1]}}, "value_preference must be a mapping"),
        ({"training": {"value_preference": {"loss_weight": -1}}}, "loss_weight must be >= 0"),
        ({"training": {"value_preference": {"margin_scale": 0}}}, "margin_scale must be > 0"),
        ({"training": {"ppg": [1]}}, "training.ppg must be a mapping"),
        ({"training": {"ppg": {"gcd_count": 0}}}, "gcd_count must be >= 1"),
        ({"training": {"ppg": {"normalization": 0}}}, "normalization must be > 0"),
        ({"training": {"precision": "int8"}}, "precision must be one of"),
    ],
)
def test_load_run_config_rejects_invalid_values(tmp_path, payload, message):
    with pytest.raises(ValueError, match=message):
        config_module.load_run_config(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("true", True),
        ("YES", True),
        ("1", True),
        (1, True),
        ("false", False),
        (" No ", False),
        ("0", False),
        (0, False),
    ],
)
def test_load_run_config_parses_full_attention_residuals_string_aliases(
    tmp_path: Path,
    raw_value: object,
    expected: bool,
):
    config = config_module.load_run_config(
        _write_config(tmp_path, {"model": {"full_attention_residuals": raw_value}})
    )

    assert config.model.full_attention_residuals is expected


@pytest.mark.parametrize("raw_value", [False, 0, "false", "no"])
def test_load_run_config_ignores_removed_scorer_switches_during_migration(
    tmp_path: Path,
    raw_value: object,
):
    config = config_module.load_run_config(
        _write_config(
            tmp_path,
            {
                "model": {
                    "scorer_use_raw_projection": raw_value,
                    "scorer_use_candidate_hidden": False,
                }
            },
        )
    )

    assert "scorer_use_candidate_hidden" not in asdict(config.model)


@pytest.mark.parametrize("raw_value", [True, 1, "true", "yes"])
def test_load_run_config_rejects_enabled_removed_scorer_switch_aliases(
    tmp_path: Path,
    raw_value: object,
):
    with pytest.raises(
        ValueError,
        match="model.scorer_use_raw_projection is removed",
    ):
        config_module.load_run_config(
            _write_config(
                tmp_path,
                {"model": {"scorer_use_raw_projection": raw_value}},
            )
        )


def test_policy_config_resolvers_select_job_and_variant_from_env_and_cover_path_branches(monkeypatch, tmp_path):
    config_path = _write_config(
        tmp_path,
        {"raw_data_dir": "data", "output_dir": "artifacts/checkpoints"},
    )

    assert policy_config_module.resolve_policy_model_config_path(config_path) == config_path.resolve()

    monkeypatch.setenv(policy_config_module.PROJECT_JOB_TAG_ENV, "black_mage")
    monkeypatch.setenv(policy_config_module.PROJECT_MODEL_VARIANT_ENV, "artzip")
    selected_config = policy_config_module.resolve_policy_model_config_path()
    assert selected_config == (
        policy_config_module.PROJECT_ROOT
        / "config/models/black_mage/artzip/config.yaml"
    ).resolve()
    with pytest.raises(FileNotFoundError, match="not found"):
        policy_config_module.resolve_policy_model_config_path(tmp_path / "missing.yaml")

    black_mage_config = selected_config
    assert policy_config_module.resolve_policy_model_job_tag(black_mage_config) == "black_mage"
    assert policy_config_module.resolve_policy_model_variant(black_mage_config) == "artzip"

    with pytest.raises(ValueError, match="must live under"):
        policy_config_module.resolve_policy_model_job_tag(tmp_path / "config.yaml")
    with pytest.raises(ValueError, match="cannot derive"):
        policy_config_module.resolve_policy_model_job_tag(
            policy_config_module.PROJECT_ROOT / "config/models/config.yaml"
        )
    monkeypatch.setenv(policy_config_module.PROJECT_JOB_TAG_ENV, "machinist")
    with pytest.raises(ValueError, match="does not match"):
        policy_config_module.resolve_policy_model_job_tag(black_mage_config)

    assert policy_config_module.resolve_policy_device(" CPU ") == "cpu"
    with pytest.raises(ValueError, match="must be cuda or cpu"):
        policy_config_module.resolve_policy_device("metal")

    monkeypatch.setenv(policy_config_module.POLICY_CACHE_ROOT_ENV, str(tmp_path))
    assert policy_config_module.resolve_policy_cache_dir("black_mage") == tmp_path / "black_mage" / ".cache"

    absolute_checkpoint = tmp_path / "absolute.pt"
    assert policy_config_module.resolve_policy_checkpoint_path(
        black_mage_config,
        str(absolute_checkpoint),
    ) == absolute_checkpoint
    relative_checkpoint = policy_config_module.resolve_policy_checkpoint_path(
        black_mage_config,
        "model.pt",
    )
    assert relative_checkpoint.name == "model.pt"


@pytest.mark.parametrize("resolver_name", ["resolve_project_job_tag", "resolve_project_model_variant"])
def test_project_resolvers_reject_explicit_blank_instead_of_falling_back(
    monkeypatch,
    tmp_path,
    resolver_name,
):
    monkeypatch.setenv("FFXIV_JOB_TAG", "black_mage")
    monkeypatch.setenv("FFXIV_MODEL_VARIANT", "artzip")

    resolver = getattr(project_config_module, resolver_name)
    with pytest.raises(ValueError, match="missing"):
        resolver(project_root=tmp_path, explicit="  ")


def test_policy_config_resolver_requires_model_variant(monkeypatch):
    """未指定模型变体时不得回退到目录扫描。"""
    monkeypatch.setattr(policy_config_module, "load_root_dotenv", lambda _root: None)
    monkeypatch.setattr(project_config_module, "load_root_dotenv", lambda _root: None)
    monkeypatch.setenv(policy_config_module.PROJECT_JOB_TAG_ENV, "black_mage")
    monkeypatch.delenv(policy_config_module.PROJECT_MODEL_VARIANT_ENV, raising=False)

    with pytest.raises(
        ValueError,
        match="missing FFXIV_MODEL_VARIANT",
    ):
        policy_config_module.resolve_policy_model_config_path()


@pytest.mark.parametrize("invalid_variant", [".", "..", "nested/artzip", "C:/artzip"])
def test_policy_config_resolver_rejects_variant_path(monkeypatch, invalid_variant):
    """模型变体只能是职业目录下的单级目录名。"""
    monkeypatch.setenv(policy_config_module.PROJECT_JOB_TAG_ENV, "black_mage")
    monkeypatch.setenv(policy_config_module.PROJECT_MODEL_VARIANT_ENV, invalid_variant)

    with pytest.raises(ValueError, match="single model variant directory name"):
        policy_config_module.resolve_policy_model_config_path()


def test_load_run_config_parses_ppg_settings(tmp_path):
    config = config_module.load_run_config(
        _write_config(
            tmp_path,
            {"training": {"ppg": {"enabled": False, "gcd_count": 64, "normalization": 800}}},
        )
    )

    assert config.ppg.enabled is False
    assert config.ppg.gcd_count == 64
    assert config.ppg.normalization == pytest.approx(800.0)


def test_attention_rejects_negative_token_lengths():
    with pytest.raises(ValueError, match="non-negative"):
        build_split_attention_mask(
            prefix_length=-1,
            candidate_count=0,
            device=torch.device("cpu"),
        )


def test_data_spec_from_dict_and_mismatch_error():
    payload = {
        "job_tag": "black_mage",
        "num_candidates": 2,
        "state_dim": 3,
        "scene_dim": 1,
        "skill_feature_dim": 4,
        "num_scene_types": 1,
        "candidate_action_keys": ["fire_iii", "fire_iv"],
        "skill_feature_names": ["potency", "cast_time.seconds", "gcd_window.seconds", "value"],
    }
    spec = DataSpec.from_dict(payload)
    assert spec.candidate_action_keys == ("fire_iii", "fire_iv")
    assert spec.skill_feature_names[-1] == "value"

    other = DataSpec.from_dict({**payload, "state_dim": 4})
    with pytest.raises(ValueError, match=r"training data spec mismatch: state_dim: 3 != 4"):
        spec.assert_compatible_with(other)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("invalid", "must be a mapping"),
        ({"mode": "other"}, "mode must be none"),
        ({"skills": "fire_iv"}, "skills must be a list"),
        ({"penalty": -1}, "penalty must be >= 0"),
        ({"mode": "whitelist", "skills": []}, "skills must not be empty"),
    ],
)
def test_repetition_config_rejects_invalid_values(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_repetition_config(raw)


def test_repetition_config_checkpoint_and_disabled_mask_paths():
    assert repetition_config_from_checkpoint({"run_config": []}) == RepetitionConfig()
    assert repetition_config_from_checkpoint({}) == RepetitionConfig()

    batch = {
        "candidate_skill_ids": torch.zeros((1, 2), dtype=torch.long),
        "history_action_keys": "not-a-batch",
        "candidate_action_keys": [["fire_iv", "fire_iii"]],
    }
    logits = torch.zeros((1, 2))
    config = RepetitionConfig(mode="blacklist", skills=("fire_iv",), penalty=1.0)
    assert torch.equal(apply_repetition_penalty(logits, batch, RepetitionConfig()), logits)
    assert torch.equal(
        apply_repetition_penalty(logits, batch, config),
        logits,
    )

    with pytest.raises(ValueError, match="batch lengths differ"):
        apply_repetition_penalty(
            logits,
            {
                "candidate_skill_ids": torch.zeros((1, 1), dtype=torch.long),
                "history_action_keys": [["fire_iv"], ["fire_iv"]],
                "candidate_action_keys": [["fire_iv"]],
            },
            config,
        )


def _encoder_spec() -> DataSpec:
    return DataSpec(
        job_tag="black_mage",
        num_candidates=2,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=1,
        num_scene_types=1,
        candidate_action_keys=("fire_iii", "fire_iv"),
        skill_feature_names=("potency",),
    )


def _encoder_batch() -> dict[str, torch.Tensor]:
    return {
        "history_skill_ids": torch.ones((1, 1), dtype=torch.long),
        "history_skill_features": torch.zeros((1, 1, 1)),
        "history_state_vectors": torch.zeros((1, 1, 3)),
        "history_mask": torch.ones((1, 1), dtype=torch.bool),
        "candidate_skill_ids": torch.ones((1, 2), dtype=torch.long),
        "candidate_skill_features": torch.zeros((1, 2, 1)),
        "candidate_state_vectors": torch.zeros((1, 2, 3)),
        "candidate_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.zeros((1, 1, 2)),
        "scene_types": torch.zeros((1, 1), dtype=torch.long),
        "scene_mask": torch.ones((1, 1), dtype=torch.bool),
    }


def test_input_encoder_handles_null_state_and_rejects_bad_shapes():
    spec = _encoder_spec()
    encoder = CandidateInputEncoder(
        spec,
        ModelConfig(d_model=8, pair_embedding_dim=4, n_layers=1, n_heads=2, ff_dim=16),
        vocab_size=4,
    )
    assert encoder._embed_state(torch.zeros((1, 3)), None).shape == (1, 4)

    with pytest.raises(ValueError, match="non-empty scene"):
        CandidateInputEncoder(
            DataSpec(**{**spec.__dict__, "scene_dim": 0}),
            ModelConfig(d_model=8, pair_embedding_dim=4, n_layers=1, n_heads=2, ff_dim=16),
            vocab_size=4,
        )

    invalid_batches = [
        ("history skill/state lengths", "history_state_vectors", torch.zeros((1, 2, 3))),
        ("candidate count", "candidate_skill_ids", torch.ones((1, 1), dtype=torch.long)),
        ("candidate skill/state counts", "candidate_state_vectors", torch.zeros((1, 1, 3))),
        ("candidate state dimension", "candidate_state_vectors", torch.zeros((1, 2, 4))),
        ("scene dimension", "scene_vectors", torch.zeros((1, 1, 3))),
        ("skill feature dimension", "candidate_skill_features", torch.zeros((1, 2, 2))),
        ("candidate legal mask", "candidate_legal_mask", torch.ones((1, 1), dtype=torch.bool)),
        ("label index", "label_index", torch.tensor([2])),
    ]
    for message, key, value in invalid_batches:
        batch = _encoder_batch()
        batch[key] = value
        with pytest.raises(ValueError, match=message):
            encoder._validate_batch(batch)


def test_input_encoder_materializes_compact_history_to_dense_semantics():
    spec = _encoder_spec()
    encoder = CandidateInputEncoder(
        spec,
        ModelConfig(d_model=8, pair_embedding_dim=4, n_layers=1, n_heads=2, ff_dim=16),
        vocab_size=4,
    )
    dense = _encoder_batch()
    dense["history_state_null_mask"] = torch.tensor([[[False, True, False]]])
    dense["history_skill_features"] = torch.tensor([[[2.0]]])
    dense["history_state_vectors"] = torch.tensor([[[3.0, 4.0, 5.0]]])

    compact = {
        key: value
        for key, value in dense.items()
        if not key.startswith("history_")
    }
    compact.update(
        {
            "history_lengths": torch.tensor([1], dtype=torch.int64),
            "history_ends": torch.tensor([2], dtype=torch.int64),
            "history_mask": torch.ones((1, 1), dtype=torch.bool),
            "history_bank_skill_ids": torch.tensor([0, 1], dtype=torch.long),
            "history_bank_skill_features": torch.tensor([[0.0], [2.0]]),
            "history_bank_state_vectors": torch.tensor(
                [[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]]
            ),
            "history_bank_state_null_mask": torch.tensor(
                [[False, False, False], [False, True, False]]
            ),
        }
    )

    encoder._materialize_compact_history(compact)
    for key in (
        "history_skill_ids",
        "history_skill_features",
        "history_state_vectors",
        "history_state_null_mask",
    ):
        assert torch.equal(compact[key], dense[key]), key


def test_model_and_trace_helpers_cover_error_and_norm_paths():
    from common.policy.model import CandidateTransformerModel

    with pytest.raises(ValueError, match="divisible"):
        CandidateTransformerModel(
            _encoder_spec(),
            ModelConfig(d_model=7, pair_embedding_dim=4, n_layers=1, n_heads=2, ff_dim=16),
            vocab_size=4,
        )
    with pytest.raises(ValueError, match="missing model_config"):
        CandidateTransformerModel.checkpoint_model_config({})
    with pytest.raises(ValueError, match="missing model_config"):
        CandidateTransformerModel.checkpoint_model_config({"model_config": []})
    with pytest.raises(ValueError, match="removed scorer_use_raw_projection"):
        CandidateTransformerModel.checkpoint_model_config(
            {"model_config": {"scorer_use_raw_projection": True}}
        )
    migrated_config = CandidateTransformerModel.checkpoint_model_config(
        {
            "model_config": {
                "history_capacity": 128,
                "scorer_use_raw_projection": False,
                "scorer_use_candidate_hidden": True,
            }
        }
    )
    assert migrated_config.history_capacity == 128
    assert migrated_config.num_kv_heads == migrated_config.n_heads
    assert "scorer_use_candidate_hidden" not in asdict(migrated_config)
    with pytest.raises(ValueError, match="removed candidate-shared RoPE"):
        CandidateTransformerModel.checkpoint_model_config(
            {
                "model_config": {
                    "history_capacity": 128,
                    "position_id_semantics": "candidate_block_shared",
                }
            }
        )
    with pytest.raises(ValueError, match="scorer_use_candidate_hidden=false"):
        CandidateTransformerModel.checkpoint_model_config(
            {
                "model_config": {
                    "history_capacity": 128,
                    "scorer_use_candidate_hidden": False,
                }
            }
        )

    tokens = torch.zeros((1, 3, 4))

    encoder = _attach_rope(nn.TransformerEncoder(
        TraceableTransformerEncoderLayer(
            d_model=4,
            nhead=2,
            dim_feedforward=8,
            dropout=0.0,
            batch_first=True,
        ),
        1,
        norm=nn.LayerNorm(4),
    ).eval())
    encoded = {
        "tokens": tokens,
        "prefix_length": 1,
        "candidate_count": 1,
        "prefix_valid": torch.ones((1, 1), dtype=torch.bool),
        "candidate_valid": torch.ones((1, 1), dtype=torch.bool),
        "cls_valid": torch.ones((1, 1), dtype=torch.bool),
    }
    trace = trace_encoder(encoder, encoded)
    assert trace.hidden.shape == tokens.shape
    assert len(trace.layer_hidden) == 1
    assert len(trace.attentions) == 1
    assert trace.attentions[0].shape[-2:] == (3, 3)


def test_model_defaults_to_pre_ln_gelu_with_final_layer_norm():
    from torch.nn import functional as F

    from common.policy.model import CandidateTransformerModel

    model = CandidateTransformerModel(
        _encoder_spec(),
        ModelConfig(d_model=8, pair_embedding_dim=4, n_layers=2, n_heads=2, ff_dim=16),
        vocab_size=4,
    )

    layer = model.encoder.layers[0]
    assert layer.norm_first is True
    assert layer.activation is F.gelu
    assert isinstance(model.encoder.norm, nn.LayerNorm)
    assert layer.activation_checkpoint_ffn is False
    assert layer.activation_checkpoint_attention is False


@pytest.mark.parametrize("activation", ("gelu", "swiglu"))
def test_ffn_activation_checkpoint_preserves_forward_and_gradients(activation):
    from unittest.mock import patch

    torch.manual_seed(7)
    layer_kwargs = {
        "d_model": 8,
        "nhead": 2,
        "dim_feedforward": 16,
        "dropout": 0.1,
        "activation": activation,
        "batch_first": True,
        "norm_first": True,
    }
    eager = TraceableTransformerEncoderLayer(**layer_kwargs)
    checkpointed = TraceableTransformerEncoderLayer(**layer_kwargs)
    checkpointed.load_state_dict(eager.state_dict())
    checkpointed.set_activation_checkpoint_ffn(True)
    eager.train()
    checkpointed.train()

    eager_input = torch.randn(2, 5, 8, requires_grad=True)
    checkpointed_input = eager_input.detach().clone().requires_grad_(True)
    with patch.object(
        eager,
        "_ff_block_without_checkpoint",
        wraps=eager._ff_block_without_checkpoint,
    ) as eager_ffn, patch.object(
        checkpointed,
        "_ff_block_without_checkpoint",
        wraps=checkpointed._ff_block_without_checkpoint,
    ) as checkpointed_ffn:
        torch.manual_seed(123)
        eager_output = eager(eager_input)
        torch.manual_seed(123)
        checkpointed_output = checkpointed(checkpointed_input)
        eager_output.square().sum().backward()
        checkpointed_output.square().sum().backward()
        assert eager_ffn.call_count == 1
        assert checkpointed_ffn.call_count == 2

    assert torch.allclose(eager_output, checkpointed_output)
    assert torch.allclose(eager_input.grad, checkpointed_input.grad)
    for eager_parameter, checkpointed_parameter in zip(
        eager.parameters(), checkpointed.parameters()
    ):
        assert torch.allclose(eager_parameter.grad, checkpointed_parameter.grad)


@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("activation", ("gelu", "swiglu"))
def test_split_finish_layer_applies_ffn_dropout_once(norm_first, activation):
    from unittest.mock import patch

    from common.policy.model.split_encoder import finish_layer

    layer = TraceableTransformerEncoderLayer(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        dropout=0.25,
        activation=activation,
        batch_first=True,
        norm_first=norm_first,
    ).train()
    hidden = torch.randn(2, 3, 8)
    attended = torch.randn(2, 3, 8)

    with patch.object(
        layer.dropout2,
        "forward",
        wraps=layer.dropout2.forward,
    ) as dropout2:
        torch.manual_seed(123)
        actual = finish_layer(layer, hidden, attended)

    assert dropout2.call_count == 1

    torch.manual_seed(123)
    if norm_first:
        expected_hidden = hidden + layer.dropout1(attended)
        expected = expected_hidden + layer._ff_block(
            layer.norm2(expected_hidden)
        )
    else:
        expected_hidden = layer.norm1(hidden + layer.dropout1(attended))
        expected = layer.norm2(expected_hidden + layer._ff_block(expected_hidden))
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("num_kv_heads", (1, 2, 4))
def test_attention_activation_checkpoint_preserves_forward_and_gradients(num_kv_heads):
    from unittest.mock import patch
    from common.policy.model.split_encoder import run_split_encoder

    torch.manual_seed(7)
    layer_kwargs = {
        "d_model": 8,
        "nhead": 4,
        "num_kv_heads": num_kv_heads,
        "dim_feedforward": 16,
        "dropout": 0.1,
        "activation": "gelu",
        "batch_first": True,
        "norm_first": True,
    }
    eager_layer = TraceableTransformerEncoderLayer(**layer_kwargs)
    checkpointed_layer = TraceableTransformerEncoderLayer(**layer_kwargs)
    checkpointed_layer.load_state_dict(eager_layer.state_dict())
    checkpointed_layer.set_activation_checkpoint_attention(True)
    eager = _attach_rope(nn.TransformerEncoder(eager_layer, 1, norm=nn.LayerNorm(8)).train())
    checkpointed = _attach_rope(nn.TransformerEncoder(
        checkpointed_layer,
        1,
        norm=nn.LayerNorm(8),
    ).train())
    checkpointed.norm.load_state_dict(eager.norm.state_dict())

    eager_input = torch.randn(2, 5, 8, requires_grad=True)
    checkpointed_input = eager_input.detach().clone().requires_grad_(True)
    encoded = {
        "tokens": eager_input,
        "prefix_length": 2,
        "candidate_count": 2,
        "prefix_valid": torch.ones((2, 2), dtype=torch.bool),
        "candidate_valid": torch.ones((2, 2), dtype=torch.bool),
        "cls_valid": torch.ones((2, 1), dtype=torch.bool),
    }
    checkpointed_encoded = {**encoded, "tokens": checkpointed_input}
    with patch(
        "common.policy.model.attention_variants.checkpoint",
        wraps=__import__("torch.utils.checkpoint", fromlist=["checkpoint"]).checkpoint,
    ) as attention_checkpoint, patch(
        "torch.nn.functional.scaled_dot_product_attention",
        wraps=torch.nn.functional.scaled_dot_product_attention,
    ) as sdpa:
        torch.manual_seed(123)
        eager_output = run_split_encoder(eager, encoded)
        sdpa.reset_mock()
        torch.manual_seed(123)
        checkpointed_output = run_split_encoder(checkpointed, checkpointed_encoded)
        forward_call_count = 3 * (num_kv_heads if num_kv_heads < 4 else 1)
        assert sdpa.call_count == forward_call_count
        sum(value.square().sum() for value in eager_output[:3]).backward()
        sum(value.square().sum() for value in checkpointed_output[:3]).backward()
        assert sdpa.call_count == 2 * forward_call_count

    assert attention_checkpoint.call_count == 3
    for call in attention_checkpoint.call_args_list:
        # checkpoint 在广播之前保存输入，K/V 激活仍是压缩的头数。
        assert call.args[2].shape[1] == call.args[3].shape[1] == num_kv_heads
    for eager_value, checkpointed_value in zip(eager_output[:3], checkpointed_output[:3]):
        assert torch.allclose(eager_value, checkpointed_value)
    assert torch.allclose(eager_input.grad, checkpointed_input.grad)
    for eager_parameter, checkpointed_parameter in zip(
        eager.parameters(), checkpointed.parameters()
    ):
        assert torch.allclose(eager_parameter.grad, checkpointed_parameter.grad)


@pytest.mark.parametrize("activation", ("gelu", "swiglu"))
def test_full_attention_residual_checkpoint_recomputes_source_path_and_preserves_gradients(activation):
    from unittest.mock import patch
    from common.policy.model.split_encoder import run_split_encoder

    torch.manual_seed(7)
    layer_kwargs = {
        "d_model": 8,
        "nhead": 2,
        "dim_feedforward": 16,
        "dropout": 0.1,
        "activation": activation,
        "batch_first": True,
        "norm_first": True,
    }
    eager_layer = TraceableTransformerEncoderLayer(**layer_kwargs)
    checkpointed_layer = TraceableTransformerEncoderLayer(**layer_kwargs)
    eager = _attach_rope(nn.TransformerEncoder(eager_layer, 2, norm=nn.LayerNorm(8)).train())
    checkpointed = _attach_rope(nn.TransformerEncoder(
        checkpointed_layer,
        2,
        norm=nn.LayerNorm(8),
    ).train())
    eager.attention_residual = FullAttentionResidual(d_model=8, num_queries=5)
    checkpointed.attention_residual = FullAttentionResidual(d_model=8, num_queries=5)
    checkpointed.load_state_dict(eager.state_dict())
    for layer in checkpointed.layers:
        layer.set_activation_checkpoint_ffn(True)
        layer.set_activation_checkpoint_attention(True)

    eager_input = torch.randn(2, 5, 8, requires_grad=True)
    checkpointed_input = eager_input.detach().clone().requires_grad_(True)
    encoded = {
        "tokens": eager_input,
        "prefix_length": 2,
        "candidate_count": 2,
        "prefix_valid": torch.ones((2, 2), dtype=torch.bool),
        "candidate_valid": torch.ones((2, 2), dtype=torch.bool),
        "cls_valid": torch.ones((2, 1), dtype=torch.bool),
    }
    checkpointed_encoded = {**encoded, "tokens": checkpointed_input}

    with patch(
        "common.policy.model.split_encoder.checkpoint",
        wraps=__import__("torch.utils.checkpoint", fromlist=["checkpoint"]).checkpoint,
    ) as residual_checkpoint, patch(
        "common.policy.model.trace.checkpoint",
        wraps=__import__("torch.utils.checkpoint", fromlist=["checkpoint"]).checkpoint,
    ) as ffn_checkpoint, patch(
        "torch.nn.functional.scaled_dot_product_attention",
        wraps=torch.nn.functional.scaled_dot_product_attention,
    ) as sdpa:
        torch.manual_seed(123)
        eager_output = run_split_encoder(eager, encoded)
        sdpa.reset_mock()
        torch.manual_seed(123)
        checkpointed_output = run_split_encoder(checkpointed, checkpointed_encoded)
        sum(value.square().sum() for value in eager_output[:3]).backward()
        sum(value.square().sum() for value in checkpointed_output[:3]).backward()

    assert residual_checkpoint.call_count == 1
    assert ffn_checkpoint.call_count == 0
    assert sdpa.call_count == 12
    for eager_value, checkpointed_value in zip(eager_output[:3], checkpointed_output[:3]):
        assert torch.allclose(eager_value, checkpointed_value)
    assert torch.allclose(eager_input.grad, checkpointed_input.grad)
    for eager_parameter, checkpointed_parameter in zip(
        eager.parameters(), checkpointed.parameters()
    ):
        assert torch.allclose(eager_parameter.grad, checkpointed_parameter.grad)


def test_full_attention_residual_checkpoint_preserves_partial_layer_granularity():
    from unittest.mock import patch
    from common.policy.model.split_encoder import run_split_encoder

    layer_kwargs = {
        "d_model": 8,
        "nhead": 2,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "activation": "gelu",
        "batch_first": True,
        "norm_first": True,
    }
    encoder = _attach_rope(nn.TransformerEncoder(
        TraceableTransformerEncoderLayer(**layer_kwargs),
        2,
        norm=nn.LayerNorm(8),
    ).train())
    encoder.attention_residual = FullAttentionResidual(d_model=8, num_queries=5)
    encoder.layers[0].set_activation_checkpoint_ffn(True)

    tokens = torch.randn(2, 5, 8, requires_grad=True)
    encoded = {
        "tokens": tokens,
        "prefix_length": 2,
        "candidate_count": 2,
        "prefix_valid": torch.ones((2, 2), dtype=torch.bool),
        "candidate_valid": torch.ones((2, 2), dtype=torch.bool),
        "cls_valid": torch.ones((2, 1), dtype=torch.bool),
    }
    with patch(
        "common.policy.model.split_encoder.checkpoint",
        wraps=__import__("torch.utils.checkpoint", fromlist=["checkpoint"]).checkpoint,
    ) as residual_checkpoint, patch(
        "common.policy.model.trace.checkpoint",
        wraps=__import__("torch.utils.checkpoint", fromlist=["checkpoint"]).checkpoint,
    ) as ffn_checkpoint:
        output = run_split_encoder(encoder, encoded)
        sum(value.square().sum() for value in output[:3]).backward()

    assert residual_checkpoint.call_count == 0
    assert ffn_checkpoint.call_count == 1


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        count = batch["label_index"].shape[0]
        logits = torch.stack(
            (self.bias.expand(count), torch.zeros_like(self.bias).expand(count)),
            dim=1,
        )
        loss = nn.functional.cross_entropy(logits, batch["label_index"])
        predictions = logits.argmax(dim=-1)
        top_indices = logits.topk(2, dim=-1).indices
        return {
            "logits": logits,
            "loss": loss,
            "top1_accuracy": (predictions == batch["label_index"]).float().mean(),
            "top3_accuracy": (
                (top_indices == batch["label_index"].unsqueeze(-1)).any(dim=-1).float().mean()
            ),
        }


def test_training_helpers_and_epoch_metrics(tmp_path, monkeypatch):
    direct_raw = tmp_path / "direct.json"
    direct_raw.touch()
    assert cache_module.select_training_raw_paths(tmp_path, max_files=1) == [direct_raw]
    assert cache_module.select_training_raw_paths(tmp_path) == [direct_raw]
    assert cache_module.select_training_raw_paths(tmp_path, max_files=0) == [direct_raw]

    config = RunConfig(raw_data_dir=tmp_path, output_dir=tmp_path, job_tag="black_mage")
    assert training_module._dataloader_options(config) == {"num_workers": 0, "pin_memory": True}
    worker_config = RunConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        job_tag="black_mage",
        num_workers=2,
        prefetch_factor=3,
        persistent_workers=False,
    )
    assert training_module._dataloader_options(worker_config) == {
        "num_workers": 2,
        "pin_memory": True,
        "prefetch_factor": 3,
        "persistent_workers": False,
    }

    assert model_dtype("float32") is torch.float32
    assert model_dtype("float16") is torch.float16
    assert model_dtype("bf16") is torch.bfloat16
    with pytest.raises(ValueError, match="unsupported precision"):
        model_dtype("int8")
    with autocast_context(torch.device("cpu"), "float32"):
        pass
    with pytest.raises(ValueError, match="requires a CUDA device"):
        autocast_context(torch.device("cpu"), "float16")
    assert training_module._lr_lambda(0, 2, 10) == pytest.approx(0.5)
    assert 0.0 <= training_module._lr_lambda(5, 2, 10) <= 1.0

    batch = {"tensor": torch.ones(1), "metadata": "keep"}
    moved = move_batch(batch, torch.device("cpu"))
    assert moved["metadata"] == "keep"
    assert torch.equal(moved["tensor"], batch["tensor"])

    class EmptyShards:
        def compiled_shard_indices(self):
            return []

    assert training_module._build_batch_sampler(
        EmptyShards(), batch_size=2, seed=1, shuffle=True
    ) is None

    class HasShards:
        def compiled_shard_indices(self):
            return [(0, 1)]

    monkeypatch.setattr(dataloaders_module, "ShardBatchSampler", lambda *args, **kwargs: ("plain", args, kwargs))
    monkeypatch.setattr(dataloaders_module, "WeightedShardBatchSampler", lambda *args, **kwargs: ("weighted", args, kwargs))
    assert training_module._build_batch_sampler(
        HasShards(), batch_size=2, seed=1, shuffle=False
    )[0] == "plain"
    assert training_module._build_batch_sampler(
        HasShards(), batch_size=2, seed=1, shuffle=True, sample_weights=(1, 2)
    )[0] == "weighted"

    output = {
        "loss": torch.tensor(1.0, requires_grad=True),
        "logits": torch.zeros((1, 2), requires_grad=True),
    }
    loss, value_loss = training_module._resolve_training_loss(
        output,
        {
            "label_index": torch.tensor([0]),
            "candidate_values": torch.tensor([[2.0, 1.0]]),
            "candidate_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        },
        value_preference=ValuePreferenceConfig(enabled=False),
    )
    assert loss is output["loss"]
    assert value_loss.item() == 0.0

    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    train_batch = {"label_index": torch.tensor([0, 0])}
    train_metrics = training_module.train_epoch(
        model,
        [train_batch],
        optimizer,
        scheduler,
        torch.device("cpu"),
    )
    validation_metrics = training_module.validate(
        model,
        [train_batch],
        torch.device("cpu"),
    )
    assert train_metrics["top1_accuracy"] == pytest.approx(1.0)
    assert validation_metrics["top3_accuracy"] == pytest.approx(1.0)

    checkpoint = tmp_path / "checkpoint.pt"
    spec = DataSpec(
        job_tag="black_mage",
        num_candidates=2,
        state_dim=3,
        scene_dim=3,
        skill_feature_dim=1,
        num_scene_types=1,
        candidate_action_keys=("a", "b"),
        skill_feature_names=("potency",),
    )
    schema = TrainingSchema(
        serialization_format="test",
        sample_schema_version=1,
        context_schema_version=1,
        scene_context_mode="absolute",
        scene_windows=(
            SceneWindowSchema.from_feature_keys(
                context_key="targetable_window_context",
                feature_keys=(
                    "start_offset_seconds",
                    "end_offset_seconds",
                    "duration_seconds",
                ),
                scene_type_id=0,
            ),
        ),
        state_group_feature_keys={"player_state": ("a", "b", "c")},
        candidate_skill_fields=("potency",),
        skill_history_fields=(),
    )
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    input_contract = ModelInputContract.from_training(
        data_spec=spec,
        schema=schema,
        normalizer=normalizer,
    )
    training_module._save_checkpoint(
        checkpoint,
        model,
        optimizer,
        1,
        replace(config, model_variant="artzip"),
        spec,
        validation_metrics,
        input_contract=input_contract,
    )
    saved = safe_torch_load(checkpoint)
    assert saved["epoch"] == 1
    assert saved["job_tag"] == "black_mage"
    assert saved["model_variant"] == "artzip"
    assert saved["data_spec"]["num_candidates"] == 2
    assert saved["input_contract"]["normalizer"]["config"]["fight_time_max"] == 1800.0


def test_select_training_raw_paths_covers_grouped_selection_branches(tmp_path):
    first = tmp_path / "FRU"
    second = tmp_path / "M12s"
    first.mkdir()
    second.mkdir()
    for index in range(3):
        (first / f"fight-{index}.json").touch()
    (second / "fight-0.json").touch()

    assert len(cache_module.select_training_raw_paths(tmp_path, max_files=None)) == 4
    selected = cache_module.select_training_raw_paths(tmp_path, max_files=3)
    assert len(selected) == 3
    assert sum(path.parent.name == "FRU" for path in selected) == 2
    assert sum(path.parent.name == "M12s" for path in selected) == 1


def test_select_training_raw_paths_reports_internal_allocation_mismatch(tmp_path, monkeypatch):
    group = tmp_path / "FRU"
    other_group = tmp_path / "M12s"
    group.mkdir()
    other_group.mkdir()
    (group / "fight.json").touch()
    (other_group / "fight.json").touch()
    monkeypatch.setattr(cache_paths_module, "zip", lambda *_args: iter(()), raising=False)

    with pytest.raises(RuntimeError, match="proportional raw selection mismatch"):
        cache_module.select_training_raw_paths(tmp_path, max_files=1)


def _fake_training_dataset(job_tag: str = "black_mage", *, actions=("a", "b")):
    class FakeDataset:
        def __init__(self):
            self.job_tag = job_tag
            self.num_candidates = len(actions)
            self.state_dim = 3
            self.scene_dim = 1
            self.num_scene_types = 1
            self.candidate_action_keys = tuple(actions)
            self.skill_feature_names = ("potency",)

        def __len__(self):
            return 2

        def compiled_shard_indices(self):
            return []

    return FakeDataset()


def test_build_dataloaders_covers_single_file_empty_shard_and_value_paths(
    tmp_path, monkeypatch
):
    dataset = _fake_training_dataset()
    collators = []

    monkeypatch.setattr(dataloaders_module, "TrainingDataset", lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr(dataloaders_module, "load_sequence_oversampler", lambda *_args: None)
    monkeypatch.setattr(dataloaders_module, "build_sample_weights", lambda *_args: None)
    monkeypatch.setattr(dataloaders_module, "load_skill_values", lambda _job: {"a": 1.0, "b": 2.0})

    class FakeCollator:
        def __init__(self, **kwargs):
            collators.append(kwargs)

    monkeypatch.setattr(dataloaders_module, "TrainingCollator", FakeCollator)
    config = RunConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        job_tag="black_mage",
        value_preference=ValuePreferenceConfig(enabled=True, loss_weight=0.1),
    )
    train_loader, val_loader, train_dataset, val_dataset = training_module.build_dataloaders(
        [tmp_path / "one.json"],
        config,
        int_dtype=torch.int32,
        float_dtype=torch.float32,
    )

    assert train_loader.batch_size == config.batch_size
    assert val_loader.batch_size == config.batch_size
    assert train_dataset is dataset
    assert val_dataset is dataset
    assert collators[0]["skill_values"] == {"a": 1.0, "b": 2.0}
    assert collators[1]["skill_values"] == {"a": 1.0, "b": 2.0}


def test_training_raw_quota_fills_failed_paths_with_same_directory_candidates(
    tmp_path, monkeypatch
):
    fru_paths = tuple(tmp_path / "FRU" / f"fight-{index}.json" for index in range(3))
    m12s_paths = tuple(tmp_path / "M12s" / f"fight-{index}.json" for index in range(3))
    groups = (
        cache_module.RawTrainingPathGroup("FRU", 2, fru_paths),
        cache_module.RawTrainingPathGroup("M12s", 2, m12s_paths),
    )
    failed_paths = {fru_paths[0], m12s_paths[0]}
    calls = []

    def fake_precompile(paths, **_kwargs):
        paths = list(paths)
        calls.append(paths)
        return [path for path in paths if path not in failed_paths]

    monkeypatch.setattr(cache_compile_module, "select_training_raw_path_groups", lambda *_args: groups)
    monkeypatch.setattr(cache_compile_module, "precompile_raw_training_caches", fake_precompile)

    valid_paths = cache_module.prepare_training_caches(
        tmp_path,
        max_files=4,
        job_tag="black_mage",
        int_dtype=torch.int32,
        float_dtype=torch.float32,
        cache_dir=tmp_path / ".cache",
    )

    assert valid_paths == [fru_paths[1], fru_paths[2], m12s_paths[1], m12s_paths[2]]
    assert calls == [
        [fru_paths[0], fru_paths[1], m12s_paths[0], m12s_paths[1]],
        [fru_paths[2], m12s_paths[2]],
    ]


def test_training_raw_quota_reports_unfillable_directory_shortage(tmp_path, monkeypatch):
    paths = tuple(tmp_path / "M11s" / f"fight-{index}.json" for index in range(2))
    groups = (cache_module.RawTrainingPathGroup("M11s", 3, paths),)
    monkeypatch.setattr(cache_compile_module, "select_training_raw_path_groups", lambda *_args: groups)
    monkeypatch.setattr(
        cache_compile_module,
        "precompile_raw_training_caches",
        lambda _paths, **_kwargs: [],
    )

    with pytest.raises(ValueError, match="M11s: required=3 valid=0 missing=3"):
        cache_module.prepare_training_caches(
            tmp_path,
            max_files=3,
            job_tag="black_mage",
            int_dtype=torch.int32,
            float_dtype=torch.float32,
            cache_dir=tmp_path / ".cache",
        )


def test_build_dataloaders_rejects_empty_inputs_and_missing_skill_values(tmp_path, monkeypatch):
    config = RunConfig(raw_data_dir=tmp_path, output_dir=tmp_path, job_tag="black_mage")
    with pytest.raises(ValueError, match="no raw JSON files found"):
        training_module.build_dataloaders(
            [], config, int_dtype=torch.int32, float_dtype=torch.float32
        )

    dataset = _fake_training_dataset(actions=("a", "missing"))
    monkeypatch.setattr(dataloaders_module, "TrainingDataset", lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr(dataloaders_module, "load_sequence_oversampler", lambda *_args: None)
    monkeypatch.setattr(dataloaders_module, "build_sample_weights", lambda *_args: None)
    monkeypatch.setattr(dataloaders_module, "load_skill_values", lambda _job: {"a": 1.0})

    config = RunConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        job_tag="black_mage",
        value_preference=ValuePreferenceConfig(enabled=True, loss_weight=0.1),
    )
    with pytest.raises(ValueError, match="missing value for candidate actions: missing"):
        training_module.build_dataloaders(
            [tmp_path / "one.json"],
            config,
            int_dtype=torch.int32,
            float_dtype=torch.float32,
        )


def test_resolve_training_loss_and_autocast_success_paths(monkeypatch):
    output = {
        "loss": torch.tensor(1.0, requires_grad=True),
        "logits": torch.tensor([[2.0, 0.0]], requires_grad=True),
    }
    batch = {
        "label_index": torch.tensor([0]),
        "candidate_values": torch.tensor([[2.0, 1.0]]),
        "candidate_legal_mask": torch.ones((1, 2), dtype=torch.bool),
    }
    total_loss, value_loss = training_module._resolve_training_loss(
        output,
        batch,
        value_preference=ValuePreferenceConfig(enabled=True, loss_weight=0.5),
    )
    assert value_loss.item() > 0.0
    assert total_loss.item() > output["loss"].item()

    called = {}

    def fake_autocast(**kwargs):
        called.update(kwargs)
        return nullcontext()

    monkeypatch.setattr(training_module.torch, "autocast", fake_autocast)
    with autocast_context(torch.device("cuda"), "bf16"):
        pass
    assert called == {"device_type": "cuda", "dtype": torch.bfloat16}


def test_run_training_rejects_device_and_data_contract_errors(tmp_path, monkeypatch):
    config = RunConfig(raw_data_dir=tmp_path, output_dir=tmp_path, job_tag="black_mage")
    monkeypatch.setattr(training_module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        training_module.run_training(config, raw_paths=[tmp_path / "one.json"], device_name="cuda")

    dataset = _fake_training_dataset(job_tag="machinist")
    monkeypatch.setattr(training_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        training_module,
        "build_dataloaders",
        lambda *_args, **_kwargs: ([0], [0], dataset, dataset),
    )
    with pytest.raises(ValueError, match="config job_tag 'black_mage' != cache job_tag 'machinist'"):
        training_module.run_training(
            config,
            raw_paths=[tmp_path / "one.json"],
            device_name="cpu",
        )

    unknown_dataset = _fake_training_dataset(job_tag="unknown")
    monkeypatch.setattr(
        training_module,
        "build_dataloaders",
        lambda *_args, **_kwargs: ([0], [0], unknown_dataset, unknown_dataset),
    )
    monkeypatch.setattr(training_module, "registered_job_tags", lambda: ("black_mage",))
    with pytest.raises(ValueError, match="has no registered combat state-machine route"):
        training_module.run_training(
            replace(config, job_tag=None),
            raw_paths=[tmp_path / "one.json"],
            device_name="cpu",
        )


def test_run_training_orchestrates_checkpoint_saving(tmp_path, monkeypatch):
    schema = TrainingSchema(
        serialization_format="test",
        sample_schema_version=1,
        context_schema_version=1,
        scene_context_mode="absolute",
        scene_windows=(
            SceneWindowSchema.from_feature_keys(
                context_key="targetable_window_context",
                feature_keys=(
                    "start_offset_seconds",
                    "end_offset_seconds",
                    "duration_seconds",
                ),
                scene_type_id=0,
            ),
        ),
        state_group_feature_keys={"player_state": ("a", "b", "c")},
        candidate_skill_fields=("potency",),
        skill_history_fields=(),
    )
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    dataset = SimpleNamespace(
        job_tag="black_mage",
        num_candidates=2,
        state_dim=3,
        scene_dim=3,
        num_scene_types=1,
        candidate_action_keys=("a", "b"),
        skill_feature_names=("potency",),
        schema=schema,
        normalizer=normalizer,
    )
    calls: dict[str, object] = {"validation_count": 0, "checkpoints": []}

    class FakeVocab:
        @classmethod
        def build_from_job_tag(cls, job_tag):
            calls["vocab_job_tag"] = job_tag
            return cls()

        def size(self):
            return 4

    class FakeModel(_TinyModel):
        def __init__(self, data_spec, model_config, *, vocab_size, repetition):
            super().__init__()
            calls["model_args"] = (data_spec, model_config, vocab_size, repetition)

    def fake_build_dataloaders(*args, **kwargs):
        calls["dataloader_args"] = (args, kwargs)
        return ([0], [0], dataset, dataset)

    def fake_train_epoch(*args, **kwargs):
        return {
            "loss": 1.0,
            "cross_entropy_loss": 1.0,
            "value_preference_loss": 0.0,
            "top1_accuracy": 0.7,
            "top3_accuracy": 0.9,
        }

    def fake_validate(*args, **kwargs):
        calls["validation_count"] += 1
        top1 = 0.8 if calls["validation_count"] == 1 else 0.6
        return {
            "loss": 1.0,
            "cross_entropy_loss": 1.0,
            "value_preference_loss": 0.0,
            "top1_accuracy": top1,
            "top3_accuracy": 0.9,
        }

    def fake_ppg(**kwargs):
        del kwargs
        ppg = 700.0 if calls["validation_count"] == 1 else 900.0
        return {
            "val_ppg": ppg,
            "val_ppg_normalized": ppg / 1000.0,
            "none_ppg": ppg - 100.0,
            "none_ppg_normalized": (ppg - 100.0) / 1000.0,
        }

    def fake_save_checkpoint(
        path,
        _model,
        _optimizer,
        epoch,
        _config,
        _spec,
        metrics,
        *,
        input_contract,
        scheduler=None,
        best_key=None,
        best_val_metrics=None,
    ):
        assert input_contract.job_tag == "black_mage"
        assert scheduler is not None
        assert best_key is not None
        assert best_val_metrics is not None
        calls["checkpoints"].append((Path(path), epoch, metrics["top1_accuracy"]))

    monkeypatch.setattr(training_module, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(training_module, "resolve_policy_cache_dir", lambda job: tmp_path / "cache")
    monkeypatch.setattr(training_module, "registered_job_tags", lambda: ("black_mage",))
    monkeypatch.setattr(training_module, "SkillVocab", FakeVocab)
    monkeypatch.setattr(training_module, "CandidateTransformerModel", FakeModel)
    monkeypatch.setattr(training_module, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(training_module, "validate", fake_validate)
    monkeypatch.setattr(training_module, "_save_checkpoint", fake_save_checkpoint)

    config = RunConfig(
        raw_data_dir=tmp_path / "configured-data",
        output_dir=tmp_path / "configured-output",
        job_tag="black_mage",
        max_epochs=2,
        model=ModelConfig(d_model=8, pair_embedding_dim=4, n_layers=1, n_heads=2, ff_dim=16),
    )
    result = training_module.run_training(
        config,
        raw_paths=[Path("sample.json")],
        output_dir=tmp_path / "override-output",
        max_epochs=2,
        batch_size=4,
        learning_rate=0.01,
        device_name="cpu",
        validation_metrics_callback=fake_ppg,
    )

    assert result["best_val_top1_accuracy"] == pytest.approx(0.8)
    assert result["best_val_ppg"] == pytest.approx(700.0)
    assert result["best_val_score"] == pytest.approx(0.75)
    assert result["last_val_metrics"]["top1_accuracy"] == pytest.approx(0.6)
    assert result["last_val_metrics"]["val_ppg"] == pytest.approx(900.0)
    assert result["output_dir"] == tmp_path / "override-output"
    assert calls["vocab_job_tag"] == "black_mage"
    assert [entry[0].name for entry in calls["checkpoints"]] == [
        "epoch_001_val_ppg_700.00.pt",
        "best.pt",
        "epoch_002_val_ppg_900.00.pt",
        "final.pt",
    ]
    assert calls["checkpoints"][0][1] == 1
    assert calls["checkpoints"][1][1] == 1
    assert calls["checkpoints"][2][1] == 2
    assert calls["checkpoints"][3][1] == 2

    resume_model = FakeModel(
        DataSpec.from_dataset(dataset),
        config.model,
        vocab_size=4,
        repetition=config.repetition,
    )
    resume_optimizer = torch.optim.AdamW(resume_model.parameters(), lr=0.01)
    resume_input_contract = ModelInputContract.from_training(
        data_spec=DataSpec.from_dataset(dataset),
        schema=schema,
        normalizer=normalizer,
    )
    resume_checkpoint = tmp_path / "epoch_001_val_ppg_600.00.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": resume_model.state_dict(),
            "optimizer_state_dict": resume_optimizer.state_dict(),
            "model_config": asdict(config.model),
            "data_spec": asdict(DataSpec.from_dataset(dataset)),
            "input_contract": resume_input_contract.to_dict(),
            "training_precision": config.precision,
            "metrics": {
                "loss": 1.0,
                "cross_entropy_loss": 1.0,
                "value_preference_loss": 0.0,
                "top1_accuracy": 0.8,
                "top3_accuracy": 0.9,
                "val_ppg": 700.0,
                "val_ppg_normalized": 0.7,
                "none_ppg": 600.0,
                "none_ppg_normalized": 0.6,
                "top1_val_ppg_average": 0.75,
            },
        },
        resume_checkpoint,
    )
    calls["checkpoints"] = []
    resumed = training_module.run_training(
        config,
        raw_paths=[Path("sample.json")],
        output_dir=tmp_path / "override-output",
        max_epochs=2,
        batch_size=4,
        learning_rate=0.01,
        device_name="cpu",
        validation_metrics_callback=fake_ppg,
        resume_path=resume_checkpoint,
    )

    assert resumed["best_val_top1_accuracy"] == pytest.approx(0.8)
    assert resumed["best_val_ppg"] == pytest.approx(700.0)
    assert resumed["last_val_metrics"]["top1_accuracy"] == pytest.approx(0.6)
    assert [entry[0].name for entry in calls["checkpoints"]] == [
        "epoch_002_val_ppg_900.00.pt",
        "final.pt",
    ]


def test_best_metric_key_uses_requested_tie_break_order():
    base = {
        "top1_accuracy": 0.8,
        "top3_accuracy": 0.9,
        "value_preference_loss": 0.3,
        "top1_val_ppg_average": 0.75,
    }
    better_top3 = {**base, "top3_accuracy": 0.95}
    better_value_loss = {**better_top3, "value_preference_loss": 0.1}

    assert training_module._best_metric_key(
        better_top3,
        ppg_enabled=True,
    ) > training_module._best_metric_key(base, ppg_enabled=True)
    assert training_module._best_metric_key(
        better_value_loss,
        ppg_enabled=True,
    ) > training_module._best_metric_key(better_top3, ppg_enabled=True)


def test_run_training_rejects_empty_prepared_paths(tmp_path):
    config = RunConfig(raw_data_dir=tmp_path, output_dir=tmp_path, job_tag=None)

    with pytest.raises(FileNotFoundError, match="no prepared raw JSON"):
        training_module.run_training(config, raw_paths=[], device_name="cpu")


def _resume_validation_context(tmp_path: Path, *, max_epochs: int = 3):
    schema = TrainingSchema(
        serialization_format="test",
        sample_schema_version=1,
        context_schema_version=1,
        scene_context_mode="absolute",
        scene_windows=(),
        state_group_feature_keys={"player_state": ("a", "b", "c")},
        candidate_skill_fields=("potency",),
        skill_history_fields=(),
    )
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    dataset = SimpleNamespace(
        job_tag="black_mage",
        num_candidates=2,
        state_dim=3,
        scene_dim=0,
        num_scene_types=0,
        candidate_action_keys=("a", "b"),
        skill_feature_names=("potency",),
        schema=schema,
        normalizer=normalizer,
    )
    data_spec = DataSpec.from_dataset(dataset)
    config = RunConfig(
        raw_data_dir=tmp_path / "raw",
        output_dir=tmp_path / "output",
        job_tag="black_mage",
        max_epochs=max_epochs,
        model=ModelConfig(d_model=8, pair_embedding_dim=4, n_layers=1, n_heads=2, ff_dim=16),
    )
    input_contract = ModelInputContract.from_training(
        data_spec=data_spec,
        schema=schema,
        normalizer=normalizer,
    )
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(config.model),
        "data_spec": asdict(data_spec),
        "input_contract": input_contract.to_dict(),
        "training_precision": config.precision,
        "metrics": {
            "loss": 1.0,
            "cross_entropy_loss": 1.0,
            "value_preference_loss": 0.2,
            "top1_accuracy": 0.7,
            "top3_accuracy": 0.9,
        },
    }
    return SimpleNamespace(
        config=config,
        dataset=dataset,
        data_spec=data_spec,
        input_contract=input_contract,
        checkpoint=checkpoint,
    )


def test_load_resume_checkpoint_rejects_missing_and_non_mapping(tmp_path):
    with pytest.raises(FileNotFoundError, match="resume checkpoint not found"):
        training_module._load_resume_checkpoint(tmp_path / "missing.pt")

    invalid = tmp_path / "invalid.pt"
    torch.save([], invalid)
    with pytest.raises(ValueError, match="resume checkpoint must be a mapping"):
        training_module._load_resume_checkpoint(invalid)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("data_spec", {"job_tag": "black_mage", "state_dim": 999}, "training data spec mismatch"),
        ("model_config", {"d_model": 999}, "resume checkpoint model config mismatch"),
        ("model_config", {"transformer_activation": "swiglu"}, "resume checkpoint model config mismatch"),
        ("training_precision", "bf16", "resume checkpoint training precision mismatch"),
    ],
)
def test_validate_resume_checkpoint_rejects_contract_mismatches(
    tmp_path,
    field,
    value,
    message,
):
    context = _resume_validation_context(tmp_path)
    checkpoint = dict(context.checkpoint)
    if field == "data_spec":
        checkpoint[field] = {**checkpoint[field], **value}
    elif field == "model_config":
        checkpoint[field] = {**checkpoint[field], **value}
    else:
        checkpoint[field] = value

    with pytest.raises(ValueError, match=message):
        training_module._validate_resume_checkpoint(
            checkpoint,
            data_spec=context.data_spec,
            dataset=context.dataset,
            config=context.config,
            input_contract=context.input_contract,
        )


def test_validate_resume_checkpoint_migrates_legacy_false_raw_projection(tmp_path):
    context = _resume_validation_context(tmp_path)
    checkpoint = dict(context.checkpoint)
    checkpoint["model_config"] = {
        **checkpoint["model_config"],
        "scorer_use_raw_projection": False,
        "scorer_use_candidate_hidden": True,
    }

    training_module._validate_resume_checkpoint(
        checkpoint,
        data_spec=context.data_spec,
        dataset=context.dataset,
        config=context.config,
        input_contract=context.input_contract,
    )


def test_validate_resume_checkpoint_rejects_removed_candidate_shared_semantics(tmp_path):
    context = _resume_validation_context(tmp_path)
    checkpoint = dict(context.checkpoint)
    checkpoint["model_config"] = dict(checkpoint["model_config"])
    checkpoint["model_config"]["position_id_semantics"] = "candidate_block_shared"

    with pytest.raises(ValueError, match="removed candidate-shared RoPE"):
        training_module._validate_resume_checkpoint(
            checkpoint,
            data_spec=context.data_spec,
            dataset=context.dataset,
            config=context.config,
            input_contract=context.input_contract,
        )


@pytest.mark.parametrize(
    ("epoch", "message"),
    [
        (None, "resume checkpoint epoch must be an integer"),
        ("not-an-int", "resume checkpoint epoch must be an integer"),
        (True, "resume checkpoint epoch must be an integer"),
        (0, "resume checkpoint epoch must be >= 1"),
    ],
)
def test_validate_resume_checkpoint_rejects_invalid_epoch(tmp_path, epoch, message):
    context = _resume_validation_context(tmp_path)
    checkpoint = dict(context.checkpoint)
    checkpoint["epoch"] = epoch

    with pytest.raises(ValueError, match=message):
        training_module._validate_resume_checkpoint(
            checkpoint,
            data_spec=context.data_spec,
            dataset=context.dataset,
            config=context.config,
            input_contract=context.input_contract,
        )


def test_run_training_rejects_resume_at_configured_epoch_limit(tmp_path, monkeypatch):
    context = _resume_validation_context(tmp_path, max_epochs=1)
    context.checkpoint["epoch"] = 1
    checkpoint_path = tmp_path / "resume.pt"
    torch.save(context.checkpoint, checkpoint_path)
    monkeypatch.setattr(
        training_module,
        "build_dataloaders",
        lambda *_args, **_kwargs: ([0], [0], context.dataset, context.dataset),
    )

    with pytest.raises(ValueError, match="already reached epoch 1"):
        training_module.run_training(
            context.config,
            raw_paths=[tmp_path / "prepared.json"],
            device_name="cpu",
            resume_path=checkpoint_path,
        )


def test_restore_best_state_uses_checkpoint_metadata_and_best_fallback(tmp_path):
    context = _resume_validation_context(tmp_path)
    best_metrics = {
        **context.checkpoint["metrics"],
        "top1_accuracy": 0.8,
        "top3_accuracy": 0.95,
    }
    checkpoint = {
        **context.checkpoint,
        "best_key": (0.8, 0.8, 0.95, -0.2),
        "best_val_metrics": best_metrics,
    }
    key, metrics = training_module._restore_best_state(
        checkpoint,
        tmp_path / "resume.pt",
        ppg_enabled=False,
    )
    assert key == (0.8, 0.8, 0.95, -0.2)
    assert metrics == best_metrics

    resume_path = tmp_path / "resume.pt"
    best_path = tmp_path / "best.pt"
    torch.save(context.checkpoint, resume_path)
    torch.save({"metrics": best_metrics}, best_path)
    key, metrics = training_module._restore_best_state(
        context.checkpoint,
        resume_path,
        ppg_enabled=False,
    )
    assert key == training_module._best_metric_key(best_metrics, ppg_enabled=False)
    assert metrics == best_metrics


def test_restore_scheduler_state_loads_saved_state():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))
    optimizer.step()
    scheduler.step()
    saved_state = scheduler.state_dict()

    restored_model = nn.Linear(1, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer,
        lambda step: 1.0 / (step + 1),
    )
    training_module._restore_scheduler_state(
        restored_scheduler,
        {"scheduler_state_dict": saved_state},
        completed_steps=99,
    )

    assert restored_scheduler.state_dict() == saved_state


def test_restore_rng_state_restores_matching_cuda_state_and_rejects_count_mismatch(monkeypatch):
    cuda_states = [torch.tensor([1]), torch.tensor([2])]
    calls = []
    monkeypatch.setattr(training_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(training_module.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        training_module.torch.cuda,
        "set_rng_state_all",
        lambda state: calls.append(state),
    )

    training_module._restore_rng_state({"rng_state": {"cuda": cuda_states}})
    assert calls == [cuda_states]

    monkeypatch.setattr(training_module.torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="CUDA RNG state device count mismatch"):
        training_module._restore_rng_state({"rng_state": {"cuda": cuda_states}})

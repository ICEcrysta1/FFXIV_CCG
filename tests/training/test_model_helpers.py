"""训练模型公共层的边界、配置和训练流程测试。"""

from __future__ import annotations

import logging
import random
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

import common.configuration.environment as project_config_module
import training.config as config_module
import training.loop as training_module
import training.loop.checkpoint as checkpoint_module
import training.loop.dataloaders as dataloaders_module
import training.loop.training_loop as training_loop_impl
from common.policy import config as policy_config_module
from common.policy.config import ModelConfig
from common.policy.data import DataSpec, ModelInputContract, Normalizer, SkillVocab
from common.policy.data.cache import source_selection as cache_paths_module
from common.policy.data.schema import SceneWindowSchema, TrainingSchema
from common.policy.model import RepetitionConfig, build_causal_attention_mask
from common.policy.model.attention.position_encoding import RotaryPositionEncoding
from common.policy.model.execution.trace import trace_encoder
from common.policy.model.layers.activation import activation_hidden
from common.policy.model.layers.attention_residual import FullAttentionResidual
from common.policy.model.layers.input_encoder import CausalInputEncoder
from common.policy.model.layers.transformer import (
    RMSNorm,
    TraceableTransformerEncoderLayer,
)
from common.policy.model.repetition import (
    apply_repetition_penalty,
    parse_repetition_config,
    repetition_config_from_checkpoint,
)
from common.pytorch.runtime import autocast_context, model_dtype, move_batch
from common.pytorch.serialization import safe_torch_load
from common.training.tensorboard import TensorBoardConfig
from scripts.convert_fflogs import cache as cache_module
from scripts.convert_fflogs.cache import cache_compile as cache_compile_module
from tests.training._causal_fixtures import (
    make_batch,
    make_checkpoint,
    make_data_spec,
    make_input_contract,
    make_state_groups,
)
from training.config import ActionQualityLossConfig, RunConfig, ValuePreferenceConfig
from training.loop.loss import compose_training_loss, configured_auxiliary_losses


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


def test_model_capacity_uses_one_token_budget():
    config = ModelConfig()
    assert config.context_capacity == 1024
    assert config.history_reset_keep == 8
    assert not hasattr(config, "max_history_actions")
    assert not hasattr(config, "scene_capacity")
    assert not hasattr(config, "history_capacity")
    assert ModelConfig.from_mapping(asdict(config)) == config


@pytest.mark.parametrize("seed", (0, 42, -7))
def test_seed_everything_seeds_python_cpu_and_available_cuda(seed):
    """默认生成器与同种子的独立生成器一致，覆盖所有可用 GPU。"""
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    devices = [torch.device("cpu"), *(torch.device("cuda", index) for index in cuda_devices)]
    python_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            training_loop_impl._seed_everything(seed)
            assert random.random() == random.Random(seed).random()
            for device in devices:
                expected = torch.rand(8, device=device, generator=torch.Generator(device=device).manual_seed(seed))
                assert torch.equal(torch.rand(8, device=device), expected)
    finally:
        random.setstate(python_state)


@pytest.mark.parametrize("fail", (False, True))
def test_compile_warmup_preserves_rng_parameters_and_existing_gradients(tmp_path, monkeypatch, fail):
    from common.policy.model import CausalPolicyModel

    config = RunConfig(
        raw_data_dir=tmp_path, output_dir=tmp_path, job_tag="black_mage", precision="float32", batch_size=3,
        model=ModelConfig(
            d_model=8, n_layers=1, n_heads=2, num_kv_heads=1, ff_dim=24,
            dropout=0.2, transformer_activation="swiglu", hc_sinkhorn_iters=1, context_capacity=17,
        ),
    )
    model = CausalPolicyModel(make_data_spec(), config.model, vocab_size=3)
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    gradients = [(parameter.grad, parameter.grad.clone()) for parameter in model.parameters()]
    rng = torch.get_rng_state().clone()
    calls = []
    original = training_loop_impl.run_causal_encoder

    def record(encoder, encoded):
        assert encoder is model.encoder
        calls.append(tuple(encoded["tokens"].shape[:2]))
        result = original(encoder, encoded)
        if fail:
            raise RuntimeError("预热失败")
        return result

    monkeypatch.setattr(training_loop_impl, "run_causal_encoder", record)
    if fail:
        with pytest.raises(RuntimeError, match="预热失败"):
            training_loop_impl._warmup_compiled_training(model, config)
    else:
        training_loop_impl._warmup_compiled_training(model, config)
        assert calls == [(3, 16), (3, 1), (1, 16), (1, 1)]
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    for parameter, (gradient, saved) in zip(model.parameters(), gradients, strict=True):
        assert parameter.grad is gradient
        torch.testing.assert_close(parameter.grad, saved, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (["invalid"], "training config must be a mapping"),
        ({"model": [1]}, "model and training config sections must be mappings"),
        ({"training": [1]}, "model and training config sections must be mappings"),
        ({"model": {"state_dim": 1}}, "cache-derived dimensions"),
        ({"model": {"pair_embedding_dim": 0}}, "model.pair_embedding_dim is removed"),
        ({"model": {"pair_embedding_dim": 192}}, "model.pair_embedding_dim is removed"),
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
            "requires transformer_norm_first=true",
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
            {"model": {"context_capacity": -1}},
            "model.context_capacity must be a non-negative integer",
        ),
        ({"model": {"context_capacity": True}}, "model.context_capacity must be a non-negative integer"),
        ({"model": {"context_capacity": 16}}, "model.context_capacity must be >= 2"),
        ({"model": {"history_reset_keep": 0}}, "model.history_reset_keep must be >= 1"),
        ({"model": {"history_capacity": 300}}, "removed model capacity options: history_capacity"),
        ({"model": {"scene_capacity": 200}}, "removed model capacity options: scene_capacity"),
        ({"model": {"max_history_actions": 128}}, "model.max_history_actions is removed"),
        (
            {"model": {"scorer_use_raw_projection": True}},
            "removed model options",
        ),
        (
            {"training": {"max_history": 128}},
            "training.max_history is removed",
        ),
        ({"training": {"num_workers": -1}}, "num_workers must be >= 0"),
        ({"training": {"prefetch_factor": 0}}, "prefetch_factor must be >= 1"),
        ({"training": {"compiled_cache_shard_size": 0}}, "compiled_cache_shard_size must be >= 1"),
        ({"training": {"compiled_cache_max_shards": 0}}, "compiled_cache_max_shards must be >= 1"),
        ({"training": {"compiled_cache_workers": 4}}, "CONVERT_FFLOGS_WORKERS"),
        ({"training": {"history_truncation": [1]}}, "history_truncation must be a mapping"),
        ({"training": {"history_truncation": {"probability": 1.1}}}, "probability must be between 0 and 1"),
        ({"training": {"history_truncation": {"min_recent": 0}}}, "history_min_recent must be >= 1"),
        ({"training": {"candidate_shuffle": [1]}}, "removed training options"),
        ({"training": {"candidate_shuffle": {"probability": -0.1}}}, "removed training options"),
        ({"training": {"value_preference": [1]}}, "value_preference must be a mapping"),
        ({"training": {"value_preference": {"loss_weight": -1}}}, "loss_weight must be >= 0"),
        ({"training": {"value_preference": {"margin_scale": 0}}}, "margin_scale must be > 0"),
        ({"training": {"ppg": [1]}}, "training.ppg must be a mapping"),
        ({"training": {"ppg": {"gcd_count": 0}}}, "gcd_count must be >= 1"),
        ({"training": {"ppg": {"normalization": 0}}}, "normalization must be > 0"),
        ({"training": {"max_files": 0}}, "max_files must be >= 1 or null"),
        ({"training": {"max_files": -3}}, "max_files must be >= 1 or null"),
        (
            {"training": {"max_files": True}},
            "max_files must be a positive integer or null",
        ),
        (
            {"training": {"max_files": "many"}},
            "max_files must be a positive integer or null",
        ),
        (
            {"training": {"max_files": 1.9}},
            "max_files must be a positive integer or null",
        ),
        (
            {"training": {"max_files": 1.0}},
            "max_files must be a positive integer or null",
        ),
        (
            {"training": {"max_files": "12"}},
            "max_files must be a positive integer or null",
        ),
        ({"training": {"validation_files": 0}}, "validation_files must be a positive integer"),
        ({"training": {"validation_files": True}}, "validation_files must be a positive integer"),
        ({"training": {"validation_files": 1.5}}, "validation_files must be a positive integer"),
        ({"training": {"precision": "int8"}}, "precision must be one of"),
    ],
)
def test_load_run_config_rejects_invalid_values(tmp_path, payload, message):
    error_type = TypeError if message in {
        "training config must be a mapping",
        "model and training config sections must be mappings",
        "history_truncation must be a mapping",
        "value_preference must be a mapping",
        "training.ppg must be a mapping",
    } else ValueError
    with pytest.raises(error_type, match=message):
        config_module.load_run_config(_write_config(tmp_path, payload))


def test_load_run_config_parses_max_files(tmp_path: Path):
    """training.max_files 只限制训练文件数，验证文件数单独配置。"""
    unlimited = config_module.load_run_config(
        _write_config(tmp_path, {"training": {"max_files": None}})
    )
    assert unlimited.max_files is None

    limited = config_module.load_run_config(
        _write_config(tmp_path, {"training": {"max_files": 12}})
    )
    assert limited.max_files == 12
    assert limited.validation_files == 100
    fixed_validation = config_module.load_run_config(
        _write_config(tmp_path, {"training": {"max_files": 12, "validation_files": 20}})
    )
    assert fixed_validation.max_files == 12
    assert fixed_validation.validation_files == 20


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


def test_resolve_policy_checkpoint_path_env_override_and_default(monkeypatch, tmp_path):
    """默认回退 `best.pt`；显式 env 覆盖仍生效，且显式参数优先于 env。"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"output_dir": "artifacts/checkpoints/run"}),
        encoding="utf-8",
    )

    monkeypatch.delenv(policy_config_module.POLICY_MODEL_CHECKPOINT_ENV, raising=False)
    assert policy_config_module.resolve_policy_checkpoint_path(config_path).name == "best.pt"

    monkeypatch.setenv(policy_config_module.POLICY_MODEL_CHECKPOINT_ENV, "final.pt")
    assert policy_config_module.resolve_policy_checkpoint_path(config_path).name == "final.pt"
    assert policy_config_module.resolve_policy_checkpoint_path(
        config_path,
        "explicit.pt",
    ).name == "explicit.pt"


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


def test_load_run_config_parses_ppg_settings(tmp_path, monkeypatch):
    monkeypatch.delenv(config_module.VAL_PPG_USE_KV_CACHE_ENV, raising=False)
    monkeypatch.setattr(config_module, "load_root_dotenv", lambda _root: None)
    config = config_module.load_run_config(
        _write_config(
            tmp_path,
            {"training": {"ppg": {"enabled": False, "gcd_count": 64, "normalization": 800}}},
        )
    )

    assert config.ppg.enabled is False
    assert config.ppg.gcd_count == 64
    assert config.ppg.normalization == pytest.approx(800.0)
    assert config.ppg.use_kv_cache is False


def test_load_run_config_reads_val_ppg_kv_cache_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv(config_module.VAL_PPG_USE_KV_CACHE_ENV, raising=False)
    (tmp_path / ".env").write_text(
        f"{config_module.VAL_PPG_USE_KV_CACHE_ENV}=true\n", encoding="utf-8", newline="\n"
    )
    path = _write_config(tmp_path, {})

    assert config_module.load_run_config(path).ppg.use_kv_cache is True

    monkeypatch.setenv(config_module.VAL_PPG_USE_KV_CACHE_ENV, "false")
    assert config_module.load_run_config(path).ppg.use_kv_cache is False
    monkeypatch.delenv(config_module.VAL_PPG_USE_KV_CACHE_ENV)


@pytest.mark.parametrize("value", ["TRUE", "yes", "1", "false", "No", "0"])
def test_load_run_config_parses_val_ppg_kv_cache_env(tmp_path, monkeypatch, value):
    monkeypatch.setattr(config_module, "load_root_dotenv", lambda _root: None)
    monkeypatch.setenv(config_module.VAL_PPG_USE_KV_CACHE_ENV, value)
    config = config_module.load_run_config(_write_config(tmp_path, {}))
    assert config.ppg.use_kv_cache is (value.lower() in {"true", "yes", "1"})


def test_load_run_config_rejects_invalid_val_ppg_kv_cache_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "load_root_dotenv", lambda _root: None)
    monkeypatch.setenv(config_module.VAL_PPG_USE_KV_CACHE_ENV, "maybe")
    with pytest.raises(ValueError, match=config_module.VAL_PPG_USE_KV_CACHE_ENV):
        config_module.load_run_config(_write_config(tmp_path, {}))


def test_attention_rejects_negative_token_lengths():
    with pytest.raises(ValueError, match="non-negative"):
        build_causal_attention_mask(
            token_count=-1,
            device=torch.device("cpu"),
        )


def test_data_spec_from_dict_and_mismatch_error():
    payload = {
        "job_tag": "black_mage",
        "num_actions": 2,
        "base_state_dim": 3,
        "state_dim": 7,
        "scene_dim": 1,
        "skill_feature_dim": 4,
        "num_scene_types": 1,
        "action_keys": ["fire_iii", "fire_iv"],
        "action_to_vocab_id": [1, 2],
        "action_is_gcd": [True, True],
        "skill_feature_names": ["potency", "cast_time.seconds", "gcd_window.seconds", "value"],
    }
    spec = DataSpec.from_dict(payload)
    assert spec.action_keys == ("fire_iii", "fire_iv")
    assert spec.skill_feature_names[-1] == "value"

    other = DataSpec.from_dict({**payload, "state_dim": 8})
    with pytest.raises(ValueError, match=r"training data spec mismatch: state_dim: 7 != 8"):
        spec.assert_compatible_with(other)


@pytest.mark.parametrize(
    ("raw", "message", "error_type"),
    [
        ("invalid", "must be a mapping", TypeError),
        ({"mode": "other"}, "mode must be none", ValueError),
        ({"skills": "fire_iv"}, "skills must be a list", TypeError),
        ({"penalty": -1}, "penalty must be >= 0", ValueError),
        ({"mode": "whitelist", "skills": []}, "skills must not be empty", ValueError),
    ],
)
def test_repetition_config_rejects_invalid_values(raw, message, error_type):
    with pytest.raises(error_type, match=message):
        parse_repetition_config(raw)


def test_repetition_config_checkpoint_and_disabled_mask_paths():
    assert repetition_config_from_checkpoint({"run_config": []}) == RepetitionConfig()
    assert repetition_config_from_checkpoint({}) == RepetitionConfig()

    batch = {
        "action_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "history_action_keys": "not-a-batch",
        "action_keys": [["fire_iv", "fire_iii"]],
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
                "action_legal_mask": torch.ones((1, 1), dtype=torch.bool),
                "history_action_keys": [["fire_iv"], ["fire_iv"]],
                "action_keys": [["fire_iv"]],
            },
            config,
        )


def _encoder_spec() -> DataSpec:
    return make_data_spec(
        base_state_dim=3, scene_dim=2, skill_feature_dim=1,
        action_keys=("fire_iii", "fire_iv"), skill_feature_names=("potency",),
    )


def test_input_encoder_handles_null_state_and_rejects_bad_shapes():
    spec = _encoder_spec()
    encoder = CausalInputEncoder(
        spec,
        ModelConfig(d_model=8, n_layers=1, n_heads=2, ff_dim=16),
        vocab_size=4,
    )
    assert encoder._embed_state(
        torch.zeros((1, spec.state_dim)),
        torch.ones((1, spec.base_state_dim), dtype=torch.bool),
        torch.zeros((1, spec.base_state_dim), dtype=torch.bool),
    ).shape == (1, 8)

    with pytest.raises(ValueError, match="non-empty scene"):
        CausalInputEncoder(
            DataSpec(**{**spec.__dict__, "scene_dim": 0}),
            ModelConfig(d_model=8, n_layers=1, n_heads=2, ff_dim=16),
            vocab_size=4,
        )

    invalid_batches = [
        ("history_state_vectors must have shape", "history_state_vectors", torch.zeros((1, 2, 7))),
        ("current_state_vectors must have shape", "current_state_vectors", torch.zeros((1, 1, 3))),
        ("current_state_vectors must have shape", "current_state_vectors", torch.zeros((1, 4))),
        ("history_state_vectors must have shape", "history_state_vectors", torch.zeros((1, 1, 4))),
        ("scene_vectors must have shape", "scene_vectors", torch.zeros((1, 1, 3))),
        ("history_skill_features must have shape", "history_skill_features", torch.zeros((1, 1, 2))),
        ("action legal mask", "action_legal_mask", torch.ones((1, 1), dtype=torch.bool)),
    ]
    for message, key, value in invalid_batches:
        batch = make_batch(spec, history_length=1)
        batch["history_state_reset_mask"] = torch.zeros_like(batch["history_state_reset_mask"])
        batch[key] = value
        with pytest.raises(ValueError, match=message):
            encoder._validate_batch(batch)


def test_model_and_trace_helpers_cover_error_and_norm_paths():
    from common.policy.model import CausalPolicyModel

    with pytest.raises(ValueError, match="divisible"):
        CausalPolicyModel(
            _encoder_spec(),
            ModelConfig(d_model=7, n_layers=1, n_heads=2, ff_dim=16),
            vocab_size=4,
        )
    with pytest.raises(TypeError, match="missing input_contract"):
        CausalPolicyModel.checkpoint_model_config({})
    checkpoint = make_checkpoint()
    checkpoint["model_config"] = []
    with pytest.raises(TypeError, match="missing model_config"):
        CausalPolicyModel.checkpoint_model_config(checkpoint)
    for removed_key, raw_value in (
        ("scorer_use_raw_projection", True),
        ("scorer_use_raw_projection", False),
        ("scorer_use_candidate_hidden", True),
        ("scorer_use_candidate_hidden", False),
        ("position_id_semantics", "candidate_block_shared"),
        ("pair_embedding_dim", 192),
    ):
        checkpoint = make_checkpoint()
        checkpoint["model_config"][removed_key] = raw_value
        message = "model.pair_embedding_dim is removed" if removed_key == "pair_embedding_dim" else "removed model options"
        with pytest.raises(ValueError, match=message):
            CausalPolicyModel.checkpoint_model_config(checkpoint)
    restored_config = CausalPolicyModel.checkpoint_model_config(make_checkpoint())
    assert restored_config.num_kv_heads == 1
    assert "scorer_use_candidate_hidden" not in asdict(restored_config)

    tokens = torch.zeros((1, 2, 4))

    encoder = _attach_rope(nn.TransformerEncoder(
        TraceableTransformerEncoderLayer(
            d_model=4,
            nhead=2,
            dim_feedforward=8,
            dropout=0.0,
            batch_first=True,
        ),
        1,
        norm=RMSNorm(4),
    ).eval())
    encoded = {
        "tokens": tokens,

        "valid": torch.cat((torch.ones((1, 1), dtype=torch.bool), torch.ones((1, 1), dtype=torch.bool)), dim=1),

    }
    trace = trace_encoder(encoder, encoded)
    assert trace.hidden.shape == tokens.shape
    assert len(trace.layer_hidden) == 1
    assert len(trace.attentions) == 1
    assert trace.attentions[0].shape[-2:] == (2, 2)


def test_model_defaults_to_pre_norm_gelu_with_trainable_rms():
    from torch.nn import functional as F

    from common.policy.model import CausalPolicyModel

    model = CausalPolicyModel(
        make_data_spec(),
        ModelConfig(d_model=8, n_layers=6, n_heads=2, ff_dim=16),
        vocab_size=4,
    )

    layer = model.encoder.layers[0]
    assert layer.norm_first is True
    assert layer.activation is F.gelu
    norms = [module for module in model.encoder.modules() if isinstance(module, RMSNorm)]
    assert len(norms) == 13
    for norm in norms:
        assert norm.eps == 1e-20
        assert norm.elementwise_affine is True
        assert norm.weight.requires_grad and norm.weight.shape == (8,)
        torch.testing.assert_close(norm.weight, torch.ones(8), rtol=0, atol=0)
    norm_keys = {f"encoder.layers.{index}.norm{number}.weight"
                 for index in range(6) for number in (1, 2)} | {"encoder.norm.weight"}
    checkpoint = make_checkpoint(model.data_spec, model.config, model_state_dict=model.state_dict())
    assert norm_keys <= checkpoint["model_state_dict"].keys()
    assert all(checkpoint["model_state_dict"][key].shape == (8,) for key in norm_keys)
    assert layer.activation_checkpoint_ffn is False
    assert layer.activation_checkpoint_attention is False


@pytest.mark.parametrize("dtype,weight_dtype,amp", (
    (torch.float32, torch.float32, False), (torch.float32, torch.bfloat16, True),
    (torch.float16, torch.float16, False), (torch.float16, torch.float32, True),
    (torch.bfloat16, torch.bfloat16, False), (torch.bfloat16, torch.float32, True),
))
def test_trainable_rms_preserves_input_dtype_and_matches_independent_gradients(dtype, weight_dtype, amp):
    norm = RMSNorm(4, dtype=weight_dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, -1.5, 0.25, 2.0]))
    # 零值、eps 主导区和大输入分别锁定除零、低精度下溢与平方溢出边界。
    values = torch.tensor([
        [0., 0., 0., 0.], [1e-12, -2e-12, 3e-12, -4e-12],
        [1e-10, -2e-10, 0., 4e-10], [1., -2., 3., -4.],
        [2000., -4000., 8000., -16000.],
    ], dtype=dtype, requires_grad=True)
    upstream = torch.tensor([
        [1e-6, -2e-6, 3e-6, -1e-6], [2e-6, 1e-6, -1e-6, 3e-6],
        [1e-6, -1e-6, 2e-6, -2e-6], [0.1, -0.2, 0.3, -0.4],
        [-0.2, 0.4, 0.1, -0.3],
    ], dtype=dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        actual = norm(values)
    actual.backward(upstream)

    # 用 FP64 的闭式雅可比核对输入和 gamma 梯度，不复用被测模块的前向。
    source = values.detach().double()
    weight = norm.weight.detach().double()
    inverse = (source.square().mean(-1, keepdim=True) + 1e-20).rsqrt()
    normalized = source * inverse
    weighted_gradient = upstream.double() * weight
    expected_grad = inverse * (
        weighted_gradient - normalized * (weighted_gradient * normalized).mean(-1, keepdim=True)
    )
    expected_weight_grad = (upstream.double() * normalized).sum(0)
    assert actual.dtype == dtype
    assert values.grad.dtype == dtype and norm.weight.grad.dtype == weight_dtype
    assert torch.isfinite(actual).all() and torch.isfinite(values.grad).all()
    assert torch.isfinite(norm.weight.grad).all()
    tolerance = 8e-3 if torch.bfloat16 in (dtype, weight_dtype) else 2e-3 if dtype == torch.float16 else 3e-6
    torch.testing.assert_close(actual, (normalized * weight).to(dtype), rtol=tolerance, atol=0)
    torch.testing.assert_close(values.grad, expected_grad.to(dtype), rtol=tolerance, atol=0)
    torch.testing.assert_close(norm.weight.grad, expected_weight_grad.to(weight_dtype), rtol=tolerance, atol=0)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_swiglu_limit_clamps_up_both_sides_and_gate_only_above_in_fp32(dtype):
    from torch.nn import functional as F

    gate = torch.tensor([-12., -2., 2., 12.], dtype=dtype, requires_grad=True)
    up = torch.tensor([-12., 12., -12., 12.], dtype=dtype, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = activation_hidden(F.silu, up, gate=gate, limit=3.0)
    expected_gate = torch.tensor([-12., -2., 2., 3.])
    expected = (F.silu(expected_gate) * torch.tensor([-3., 3., -3., 3.])).to(dtype)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().sum().backward()
    assert torch.equal(up.grad, torch.zeros_like(up))
    assert gate.grad[0] != 0 and gate.grad[-1] == 0
    assert torch.isfinite(gate.grad).all()


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

    from common.policy.model.execution.causal_encoder import finish_layer

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


def test_attention_block_checkpoint_matches_sdpa_checkpoint():
    """整块 attention checkpoint 与只重算 SDPA 的前向输出、梯度必须一致。"""
    from common.policy.model.execution.causal_encoder import run_causal_encoder

    layer_kwargs = {
        "d_model": 8,
        "nhead": 4,
        "num_kv_heads": 1,
        "dim_feedforward": 16,
        "dropout": 0.1,
        "activation": "gelu",
        "batch_first": True,
        "norm_first": True,
    }

    def build_encoder(block: bool):
        torch.manual_seed(7)
        layer = TraceableTransformerEncoderLayer(**layer_kwargs)
        layer.set_activation_checkpoint_attention(True)
        layer.set_activation_checkpoint_attention_block(block)
        return _attach_rope(
            nn.TransformerEncoder(
                layer, 1, norm=RMSNorm(8),
            ).train()
        )

    base_tokens = torch.randn(2, 4, 8)
    encoded_template = {

        "valid": torch.cat((torch.ones((2, 2), dtype=torch.bool), torch.ones((2, 2), dtype=torch.bool)), dim=1),

    }

    def run(encoder):
        # 固定 attention dropout 的 RNG 起点，两种重算粒度必须走出同一套掩码。
        torch.manual_seed(29)
        tokens = base_tokens.detach().clone().requires_grad_(True)
        encoded = {**encoded_template, "tokens": tokens}
        hidden, layer_hidden, attentions = run_causal_encoder(encoder, encoded)
        assert layer_hidden == attentions == ()
        loss = hidden.square().mean()
        loss.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in encoder.named_parameters()
            if parameter.grad is not None
        }
        return loss.detach(), tokens.grad.detach().clone(), gradients

    sdpa_loss, sdpa_input_grad, sdpa_grads = run(build_encoder(block=False))
    block_loss, block_input_grad, block_grads = run(build_encoder(block=True))

    torch.testing.assert_close(block_loss, sdpa_loss, atol=0, rtol=0)
    torch.testing.assert_close(block_input_grad, sdpa_input_grad, atol=0, rtol=0)
    assert set(block_grads) == set(sdpa_grads)
    for name, tensor in block_grads.items():
        torch.testing.assert_close(tensor, sdpa_grads[name], atol=0, rtol=0)


@pytest.mark.parametrize("num_kv_heads", (1, 2, 4))
def test_attention_activation_checkpoint_preserves_forward_and_gradients(num_kv_heads):
    from unittest.mock import patch

    from common.policy.model.execution.causal_encoder import run_causal_encoder

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
    eager = _attach_rope(nn.TransformerEncoder(
        eager_layer, 1, norm=RMSNorm(8),
    ).train())
    checkpointed = _attach_rope(nn.TransformerEncoder(
        checkpointed_layer,
        1,
        norm=RMSNorm(8),
    ).train())
    checkpointed.norm.load_state_dict(eager.norm.state_dict())

    eager_input = torch.randn(2, 4, 8, requires_grad=True)
    checkpointed_input = eager_input.detach().clone().requires_grad_(True)
    encoded = {
        "tokens": eager_input,

        "valid": torch.cat((torch.ones((2, 2), dtype=torch.bool), torch.ones((2, 2), dtype=torch.bool)), dim=1),

    }
    checkpointed_encoded = {**encoded, "tokens": checkpointed_input}
    with patch(
        "common.policy.model.attention.attention_variants.checkpoint",
        wraps=__import__("torch.utils.checkpoint", fromlist=["checkpoint"]).checkpoint,
    ) as attention_checkpoint, patch(
        "torch.nn.functional.scaled_dot_product_attention",
        wraps=torch.nn.functional.scaled_dot_product_attention,
    ) as sdpa:
        torch.manual_seed(123)
        eager_output = run_causal_encoder(eager, encoded)
        sdpa.reset_mock()
        torch.manual_seed(123)
        checkpointed_output = run_causal_encoder(checkpointed, checkpointed_encoded)
        forward_call_count = num_kv_heads if num_kv_heads < 4 else 1
        assert sdpa.call_count == forward_call_count
        eager_output[0].square().sum().backward()
        checkpointed_output[0].square().sum().backward()
        assert sdpa.call_count == 2 * forward_call_count

    assert attention_checkpoint.call_count == 1
    for call in attention_checkpoint.call_args_list:
        # checkpoint 在广播之前保存输入，K/V 激活仍是压缩的头数。
        assert call.args[2].shape[1] == call.args[3].shape[1] == num_kv_heads
    assert torch.allclose(eager_output[0], checkpointed_output[0])
    assert torch.allclose(eager_input.grad, checkpointed_input.grad)
    for eager_parameter, checkpointed_parameter in zip(
        eager.parameters(), checkpointed.parameters()
    ):
        assert torch.allclose(eager_parameter.grad, checkpointed_parameter.grad)


@pytest.mark.parametrize("activation", ("gelu", "swiglu"))
@pytest.mark.parametrize("checkpoint_ffn,checkpoint_attention", ((True, False), (False, True), (True, True)))
def test_full_attention_residual_preserves_independent_sublayer_checkpoints(
    activation, checkpoint_ffn, checkpoint_attention,
):
    """局部深度聚合重算不改变 FFN/attention 的独立开关或 dropout 梯度。"""
    from unittest.mock import patch

    from common.policy.model.execution.causal_encoder import run_causal_encoder

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
    eager = _attach_rope(nn.TransformerEncoder(
        eager_layer, 2, norm=RMSNorm(8),
    ).train())
    checkpointed = _attach_rope(nn.TransformerEncoder(
        checkpointed_layer,
        2,
        norm=RMSNorm(8),
    ).train())
    eager.attention_residual = FullAttentionResidual(d_model=8, num_queries=5)
    checkpointed.attention_residual = FullAttentionResidual(d_model=8, num_queries=5)
    with torch.no_grad():
        eager.attention_residual.pseudo_queries.normal_(std=0.15)
        for norm in eager.attention_residual.key_norms:
            norm.weight.uniform_(0.8, 1.2)
    checkpointed.load_state_dict(eager.state_dict())
    for layer in checkpointed.layers:
        layer.set_activation_checkpoint_ffn(checkpoint_ffn)
        layer.set_activation_checkpoint_attention(checkpoint_attention)

    eager_input = torch.randn(2, 4, 8, requires_grad=True)
    checkpointed_input = eager_input.detach().clone().requires_grad_(True)
    encoded = {
        "tokens": eager_input,

        "valid": torch.cat((torch.ones((2, 2), dtype=torch.bool), torch.ones((2, 2), dtype=torch.bool)), dim=1),

    }
    checkpointed_encoded = {**encoded, "tokens": checkpointed_input}

    # 参考只关闭深度聚合的重算，保留相同训练模式与 dropout。
    with patch(
        "common.policy.model.layers.attention_residual.checkpoint",
        side_effect=lambda function, *args, **_kwargs: function(*args),
    ):
        torch.manual_seed(123)
        eager_output = run_causal_encoder(eager, encoded)

    with patch(
        "common.policy.model.execution.causal_encoder.checkpoint",
        wraps=__import__("torch.utils.checkpoint", fromlist=["checkpoint"]).checkpoint,
    ) as block_checkpoint, patch(
        "common.policy.model.layers.transformer.checkpoint",
        wraps=__import__("torch.utils.checkpoint", fromlist=["checkpoint"]).checkpoint,
    ) as ffn_checkpoint, patch(
        "common.policy.model.attention.attention_variants.checkpoint",
        wraps=__import__("torch.utils.checkpoint", fromlist=["checkpoint"]).checkpoint,
    ) as attention_checkpoint:
        torch.manual_seed(123)
        checkpointed_output = run_causal_encoder(checkpointed, checkpointed_encoded)
        probe = torch.randn_like(eager_output[0])
        (eager_output[0] * probe).sum().backward()
        (checkpointed_output[0] * probe).sum().backward()

    assert block_checkpoint.call_count == 0
    assert sum(call.args[0].__name__ == "_ff_block_without_checkpoint"
               for call in ffn_checkpoint.call_args_list) == 2 * checkpoint_ffn
    assert attention_checkpoint.call_count == 2 * checkpoint_attention
    torch.testing.assert_close(eager_output[0], checkpointed_output[0])
    torch.testing.assert_close(eager_input.grad, checkpointed_input.grad)
    for eager_parameter, checkpointed_parameter in zip(
        eager.parameters(), checkpointed.parameters()
    ):
        torch.testing.assert_close(eager_parameter.grad, checkpointed_parameter.grad)


def test_full_attention_residual_checkpoint_preserves_partial_layer_granularity():
    """不同层分别启用 FFN 与整块 attention，仍与无重算参考一致。"""
    from copy import deepcopy
    from unittest.mock import patch

    from common.policy.model.execution.causal_encoder import run_causal_encoder

    layer_kwargs = {
        "d_model": 8,
        "nhead": 2,
        "dim_feedforward": 16,
        "dropout": 0.1,
        "activation": "gelu",
        "batch_first": True,
        "norm_first": True,
    }
    encoder = _attach_rope(nn.TransformerEncoder(
        TraceableTransformerEncoderLayer(**layer_kwargs),
        2,
        norm=RMSNorm(8),
    ).train())
    encoder.attention_residual = FullAttentionResidual(d_model=8, num_queries=5)
    with torch.no_grad():
        encoder.attention_residual.pseudo_queries.normal_(std=0.15)
    reference = deepcopy(encoder)
    encoder.layers[0].set_activation_checkpoint_ffn(True)
    encoder.layers[1].set_activation_checkpoint_attention(True)
    encoder.layers[1].set_activation_checkpoint_attention_block(True)

    tokens = torch.randn(2, 4, 8, requires_grad=True)
    reference_tokens = tokens.detach().clone().requires_grad_(True)
    encoded = {
        "tokens": tokens,

        "valid": torch.cat((torch.ones((2, 2), dtype=torch.bool), torch.ones((2, 2), dtype=torch.bool)), dim=1),

    }
    with patch(
        "common.policy.model.layers.attention_residual.checkpoint",
        side_effect=lambda function, *args, **_kwargs: function(*args),
    ):
        torch.manual_seed(29)
        expected = run_causal_encoder(reference, {**encoded, "tokens": reference_tokens})[0]

    with patch(
        "common.policy.model.execution.causal_encoder.checkpoint",
        wraps=__import__("torch.utils.checkpoint", fromlist=["checkpoint"]).checkpoint,
    ) as block_checkpoint, patch(
        "common.policy.model.layers.transformer.checkpoint",
        wraps=__import__("torch.utils.checkpoint", fromlist=["checkpoint"]).checkpoint,
    ) as ffn_checkpoint, patch(
        "common.policy.model.attention.attention_variants.checkpoint",
        wraps=__import__("torch.utils.checkpoint", fromlist=["checkpoint"]).checkpoint,
    ) as attention_checkpoint:
        torch.manual_seed(29)
        output = run_causal_encoder(encoder, encoded)
        probe = torch.randn_like(output[0])
        (expected * probe).sum().backward()
        (output[0] * probe).sum().backward()

    assert block_checkpoint.call_count == 1
    assert ffn_checkpoint.call_count == 1
    assert attention_checkpoint.call_count == 0
    torch.testing.assert_close(output[0], expected)
    torch.testing.assert_close(tokens.grad, reference_tokens.grad)
    for actual_parameter, expected_parameter in zip(encoder.parameters(), reference.parameters()):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(1.0))

    def set_compute_dtype(self, dtype):
        self.encoder.to(dtype=dtype)
        self.output_head.to(dtype=dtype)

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
    direct_raw = tmp_path / "direct.json.br"
    direct_raw.touch()
    assert cache_module.select_training_raw_paths(tmp_path, max_files=1) == [direct_raw]
    assert cache_module.select_training_raw_paths(tmp_path) == [direct_raw]
    assert cache_module.select_training_raw_paths(tmp_path, max_files=0) == [direct_raw]

    config = RunConfig(raw_data_dir=tmp_path, output_dir=tmp_path, job_tag="black_mage")
    assert dataloaders_module._dataloader_options(config) == {"num_workers": 0, "pin_memory": True}
    worker_config = RunConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        job_tag="black_mage",
        num_workers=2,
        prefetch_factor=3,
        persistent_workers=False,
    )
    assert dataloaders_module._dataloader_options(worker_config) == {
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
    assert training_loop_impl._lr_lambda(0, 2, 10) == pytest.approx(0.5)
    assert 0.0 <= training_loop_impl._lr_lambda(5, 2, 10) <= 1.0

    batch = {"tensor": torch.ones(1), "metadata": "keep"}
    moved = move_batch(batch, torch.device("cpu"))
    assert moved["metadata"] == "keep"
    assert torch.equal(moved["tensor"], batch["tensor"])

    class EmptyShards:
        def compiled_shard_indices(self):
            return []

    assert dataloaders_module._build_batch_sampler(
        EmptyShards(), batch_size=2, seed=1, shuffle=True
    ) is None

    class HasShards:
        def compiled_shard_indices(self):
            return [(0, 1)]

    monkeypatch.setattr(dataloaders_module, "ShardBatchSampler", lambda *args, **kwargs: ("plain", args, kwargs))
    monkeypatch.setattr(dataloaders_module, "WeightedShardBatchSampler", lambda *args, **kwargs: ("weighted", args, kwargs))
    assert dataloaders_module._build_batch_sampler(
        HasShards(), batch_size=2, seed=1, shuffle=False
    )[0] == "plain"
    assert dataloaders_module._build_batch_sampler(
        HasShards(), batch_size=2, seed=1, shuffle=True, sample_weights=(1, 2)
    )[0] == "weighted"

    output = {
        "logits": torch.zeros((1, 2), requires_grad=True),
    }
    losses = compose_training_loss(
        output,
        {
            "label_index": torch.tensor([0]),
            "action_values": torch.tensor([[2.0, 1.0]]),
            "action_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        },
        configured_auxiliary_losses(value_preference=ValuePreferenceConfig(enabled=False)),
    )
    assert losses.total is losses.primary
    assert losses.total.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert losses.auxiliary["value_preference_loss"].item() == 0.0

    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    train_batch = {"label_index": torch.tensor([0, 0])}
    class ScalarRecorder:
        def __init__(self):
            self.values = []

        def add_scalar(self, tag, value, step):
            self.values.append((tag, value, step))

    tensorboard_writer = ScalarRecorder()
    train_metrics = training_module.train_epoch(
        model,
        [train_batch],
        optimizer,
        scheduler,
        torch.device("cpu"),
        tensorboard_writer=tensorboard_writer,
        tensorboard_log_every_steps=1,
        global_step_offset=6,
    )
    validation_metrics = training_module.validate(
        model,
        [train_batch],
        torch.device("cpu"),
    )
    assert train_metrics["top1_accuracy"] == pytest.approx(1.0)
    assert validation_metrics["top3_accuracy"] == pytest.approx(1.0)
    assert any(
        tag == "训练/记录区间/总损失" and step == 7
        for tag, _value, step in tensorboard_writer.values
    )

    checkpoint = tmp_path / "checkpoint.pt"
    spec = DataSpec(
        job_tag="black_mage",
        num_actions=2,
        base_state_dim=3, state_dim=make_data_spec(base_state_dim=3).state_dim,
        scene_dim=3,
        skill_feature_dim=1,
        num_scene_types=1,
        action_to_vocab_id=(1, 2),
        action_is_gcd=(True, True),
        action_keys=("a", "b"),
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
        state_groups=make_state_groups({"player_state": ("request_state.time_seconds", "b", "c")}, ("a", "b")), state_snapshots=("previous_action_after", "request_state"),

        skill_history_fields=(),
    )
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    normalizer = Normalizer.from_contract({
        **normalizer.normalization_contract, "skill_charge_limits": {"a": 2, "b": 2},
    })
    input_contract = ModelInputContract.from_training(
        data_spec=spec,
        skill_vocab=SkillVocab.from_entries([(1001, 1), (1002, 2), (1003, 3)]),
        schema=schema,
        normalizer=normalizer,
    )
    checkpoint_module._save_checkpoint(
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
    assert saved["data_spec"]["num_actions"] == 2
    assert saved["input_contract"]["normalizer"]["config"]["time_scale"] == 270.0


def test_checkpoint_save_rejects_missing_model_variant(tmp_path):
    config = RunConfig(raw_data_dir=tmp_path, output_dir=tmp_path, job_tag=None)

    with pytest.raises(ValueError, match="checkpoint model_variant must be configured"):
        checkpoint_module._save_checkpoint(
            tmp_path / "checkpoint.pt",
            None,
            None,
            1,
            config,
            None,
            {},
            input_contract=None,
        )


def test_select_training_raw_paths_covers_grouped_selection_branches(tmp_path):
    first = tmp_path / "FRU"
    second = tmp_path / "M12s"
    first.mkdir()
    second.mkdir()
    for index in range(3):
        (first / f"fight-{index}.json.br").touch()
    (second / "fight-0.json.br").touch()

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
    (group / "fight.json.br").touch()
    (other_group / "fight.json.br").touch()
    monkeypatch.setattr(cache_paths_module, "zip", lambda *_args: iter(()), raising=False)

    with pytest.raises(RuntimeError, match="proportional raw selection mismatch"):
        cache_module.select_training_raw_paths(tmp_path, max_files=1)


def _fake_training_dataset(job_tag: str = "black_mage", *, actions=("a", "b")):
    class FakeDataset:
        def __init__(self):
            spec = make_data_spec(job_tag=job_tag, num_actions=len(actions), base_state_dim=3,
                                  scene_dim=3, skill_feature_dim=1, action_keys=tuple(actions),
                                  action_to_vocab_id=tuple(range(1, len(actions) + 1)),
                                  action_is_gcd=(True,) * len(actions), skill_feature_names=("potency",))
            self.__dict__.update(asdict(spec))
            self.schema = make_input_contract(spec).schema
            self.state_layout = self.schema.state_layout(spec.action_keys)

        def __len__(self):
            return 2

        def compiled_shard_indices(self):
            return []

    return FakeDataset()


@pytest.mark.parametrize("quality_enabled", [False, True])
def test_build_dataloaders_covers_single_file_empty_shard_and_value_paths(
    tmp_path, monkeypatch, quality_enabled,
):
    dataset = _fake_training_dataset()
    collators = []

    monkeypatch.setattr(dataloaders_module, "TrainingDataset", lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr(dataloaders_module, "load_sequence_oversampler", lambda *_args: None)
    monkeypatch.setattr(dataloaders_module, "build_sample_weights", lambda *_args: None)
    monkeypatch.setattr(dataloaders_module, "_load_skill_values", lambda _job: {"a": 1.0, "b": 2.0})

    class FakeCollator:
        def __init__(self, **kwargs):
            collators.append(kwargs)

    monkeypatch.setattr(dataloaders_module, "TrainingCollator", FakeCollator)
    config = RunConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        job_tag="black_mage",
        action_quality_loss=ActionQualityLossConfig(enabled=quality_enabled),
        value_preference=ValuePreferenceConfig(enabled=True, loss_weight=0.1),
    )
    train_loader, val_loader, train_dataset, val_dataset = training_module.build_dataloaders(
        [tmp_path / "one.json"],
        config,
        validation_paths=[tmp_path / "val.json"],
        int_dtype=torch.int32,
        float_dtype=torch.float32,
    )

    assert train_loader.batch_size == config.batch_size
    assert val_loader.batch_size == config.batch_size
    assert train_dataset is dataset
    assert val_dataset is dataset
    assert "skill_values" not in collators[0]
    assert "skill_values" not in collators[1]
    assert collators[0]["require_quality_percentile"] is quality_enabled
    assert collators[1]["require_quality_percentile"] is False


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
    monkeypatch.setattr(cache_compile_module, "cached_candidates_for_group", lambda *_args, **_kwargs: [])
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


def test_training_quota_counts_existing_cache_before_compiling(tmp_path, monkeypatch):
    """首选 JSON 没有缓存时，先用同组已编译 PT 抵扣配额。"""
    paths = tuple(tmp_path / "FRU" / f"fight-{index}.json" for index in range(4))
    groups = (cache_module.RawTrainingPathGroup("FRU", 2, paths),)
    calls = []
    monkeypatch.setattr(cache_compile_module, "select_training_raw_path_groups", lambda *_args: groups)
    monkeypatch.setattr(
        cache_compile_module, "cached_candidates_for_group",
        lambda *_args, **_kwargs: [paths[3]],
    )

    def fake_precompile(candidates, **_kwargs):
        calls.append(list(candidates))
        return [path for path in candidates if path == paths[1]]

    monkeypatch.setattr(cache_compile_module, "precompile_raw_training_caches", fake_precompile)
    selected = cache_module.prepare_training_caches(
        tmp_path, max_files=2, job_tag="black_mage",
        int_dtype=torch.int32, float_dtype=torch.float32,
        cache_dir=tmp_path / ".cache",
    )
    assert selected == [paths[1], paths[3]]
    assert calls == [[paths[0]], [paths[1]]]

    monkeypatch.setattr(
        cache_compile_module, "cached_candidates_for_group",
        lambda *_args, **_kwargs: [paths[2], paths[3]],
    )
    calls.clear()
    assert cache_module.prepare_training_caches(
        tmp_path, max_files=2, job_tag="black_mage",
        int_dtype=torch.int32, float_dtype=torch.float32,
        cache_dir=tmp_path / ".cache",
    ) == [paths[2], paths[3]]
    assert calls == []


def test_training_quota_does_not_fill_failed_bucket_from_another_bucket(tmp_path, monkeypatch):
    for bucket in ("90-100", "00-10"):
        directory = tmp_path / "FRU" / bucket
        directory.mkdir(parents=True)
        for index in range(3):
            (directory / f"fight-{index}.json.br").write_text("{}", encoding="utf-8")

    calls = []
    monkeypatch.setattr(cache_compile_module, "cached_candidates_for_group", lambda *_args, **_kwargs: [])

    def fake_precompile(paths, **_kwargs):
        calls.extend(paths)
        return [path for path in paths if path.parent.name == "90-100"]

    monkeypatch.setattr(cache_compile_module, "precompile_raw_training_caches", fake_precompile)
    with pytest.raises(ValueError, match="FRU/00-10: required=2 valid=0 missing=2"):
        cache_module.prepare_training_caches(
            tmp_path, max_files=4, job_tag="black_mage",
            int_dtype=torch.int32, float_dtype=torch.float32,
            cache_dir=tmp_path / ".cache",
        )
    assert sum(path.parent.name == "00-10" for path in calls) == 3
    assert sum(path.parent.name == "90-100" for path in calls) == 2


def test_training_raw_quota_reports_unfillable_directory_shortage(tmp_path, monkeypatch):
    paths = tuple(tmp_path / "M11s" / f"fight-{index}.json" for index in range(2))
    groups = (cache_module.RawTrainingPathGroup("M11s", 3, paths),)
    monkeypatch.setattr(cache_compile_module, "select_training_raw_path_groups", lambda *_args: groups)
    monkeypatch.setattr(cache_compile_module, "cached_candidates_for_group", lambda *_args, **_kwargs: [])
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
            [], config, validation_paths=[tmp_path / "val.json"],
            int_dtype=torch.int32, float_dtype=torch.float32
        )
    with pytest.raises(ValueError, match="no VAL compiled caches found"):
        training_module.build_dataloaders(
            [tmp_path / "train.json"], config, validation_paths=[],
            int_dtype=torch.int32, float_dtype=torch.float32,
        )
    with pytest.raises(ValueError, match="training and VAL sources must not overlap"):
        training_module.build_dataloaders(
            [tmp_path / "same.json"], config,
            validation_paths=[tmp_path / "same.json"],
            int_dtype=torch.int32, float_dtype=torch.float32,
        )

    dataset = _fake_training_dataset(actions=("a", "missing"))
    monkeypatch.setattr(dataloaders_module, "TrainingDataset", lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr(dataloaders_module, "load_sequence_oversampler", lambda *_args: None)
    monkeypatch.setattr(dataloaders_module, "build_sample_weights", lambda *_args: None)
    monkeypatch.setattr(dataloaders_module, "_load_skill_values", lambda _job: {"a": 1.0})

    config = RunConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        job_tag="black_mage",
        value_preference=ValuePreferenceConfig(enabled=True, loss_weight=0.1),
    )
    with pytest.raises(ValueError, match="missing value for output actions: missing"):
        training_module.build_dataloaders(
            [tmp_path / "one.json"],
            config,
            validation_paths=[tmp_path / "val.json"],
            int_dtype=torch.int32,
            float_dtype=torch.float32,
        )


def test_compose_training_loss_and_autocast_success_paths(monkeypatch):
    output = {
        "logits": torch.tensor([[2.0, 0.0]], requires_grad=True),
    }
    batch = {
        "label_index": torch.tensor([0]),
        "action_values": torch.tensor([[2.0, 1.0]]),
        "action_legal_mask": torch.ones((1, 2), dtype=torch.bool),
    }
    losses = compose_training_loss(
        output,
        batch,
        configured_auxiliary_losses(
            value_preference=ValuePreferenceConfig(enabled=True, loss_weight=0.5),
        ),
    )
    assert losses.auxiliary["value_preference_loss"].item() > 0.0
    assert losses.total.item() > losses.primary.item()

    called = {}

    def fake_autocast(**kwargs):
        called.update(kwargs)
        return nullcontext()

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    with autocast_context(torch.device("cuda"), "bf16"):
        pass
    assert called == {"device_type": "cuda", "dtype": torch.bfloat16}


def test_run_training_rejects_device_and_data_contract_errors(tmp_path, monkeypatch):
    config = RunConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        job_tag="black_mage",
        model_variant="artzip",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        training_module.run_training(config, raw_paths=[tmp_path / "one.json"], validation_paths=[tmp_path / "val.json"], device_name="cuda")

    dataset = _fake_training_dataset(job_tag="machinist")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        training_loop_impl,
        "build_dataloaders",
        lambda *_args, **_kwargs: ([0], [0], dataset, dataset),
    )
    with pytest.raises(ValueError, match="config job_tag 'black_mage' != cache job_tag 'machinist'"):
        training_module.run_training(
            config,
            raw_paths=[tmp_path / "one.json"],
            validation_paths=[tmp_path / "val.json"],
            device_name="cpu",
        )

    unknown_dataset = _fake_training_dataset(job_tag="unknown")
    monkeypatch.setattr(
        training_loop_impl,
        "build_dataloaders",
        lambda *_args, **_kwargs: ([0], [0], unknown_dataset, unknown_dataset),
    )
    monkeypatch.setattr(training_loop_impl, "resolve_registered_job_tags", lambda _root: ("black_mage",))
    with pytest.raises(ValueError, match="has no registered combat state-machine route"):
        training_module.run_training(
            replace(config, job_tag=None),
            raw_paths=[tmp_path / "one.json"],
            validation_paths=[tmp_path / "val.json"],
            device_name="cpu",
        )


@pytest.mark.parametrize("model_variant", [None, "  "])
def test_run_training_rejects_missing_model_variant_before_loading_data(
    tmp_path,
    model_variant,
):
    config = RunConfig(
        raw_data_dir=tmp_path,
        output_dir=tmp_path,
        job_tag="black_mage",
        model_variant=model_variant,
    )

    with pytest.raises(
        ValueError,
        match="training model_variant must be configured before initialization",
    ):
        training_module.run_training(
            config,
            raw_paths=[tmp_path / "prepared.json"],
            validation_paths=[tmp_path / "val.json"],
            device_name="cpu",
        )


@pytest.mark.parametrize("saved_scheduler_step", (None, 7))
@pytest.mark.parametrize(("precision", "fp32_gradients"), [
    ("float32", False), ("bf16", False), ("bf16", True),
    ("float16", False), ("float16", True),
])
def test_run_training_orchestrates_checkpoint_saving(
    tmp_path, monkeypatch, caplog, saved_scheduler_step, precision, fp32_gradients,
):
    caplog.set_level(logging.INFO)
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
        state_groups=make_state_groups({"player_state": ("request_state.time_seconds", "b", "c")}, ("a", "b")), state_snapshots=("previous_action_after", "request_state"),

        skill_history_fields=(),
    )
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    normalizer = Normalizer.from_contract({
        **normalizer.normalization_contract, "skill_charge_limits": {"a": 2, "b": 2},
    })
    dataset = SimpleNamespace(
        job_tag="black_mage",
        num_actions=2,
        base_state_dim=3, state_dim=make_data_spec(base_state_dim=3).state_dim,
        scene_dim=3,
        num_scene_types=1,
        action_keys=("a", "b"),
        action_to_vocab_id=(1, 2),
        action_is_gcd=(True, True),
        skill_feature_names=("potency",),
        schema=schema,
        normalizer=normalizer,
    )
    calls: dict[str, object] = {"validation_count": 0, "checkpoints": [], "train_step_offsets": []}

    class ScalarRecorder:
        log_dir = str(tmp_path / "tensorboard" / "fake-run")

        def __init__(self):
            self.values = []
            self.closed = False

        def add_scalar(self, tag, value, step):
            self.values.append((tag, value, step))

        def close(self):
            self.closed = True

    tensorboard_writer = ScalarRecorder()
    tensorboard_output_dirs: list[Path] = []

    dataset.skill_vocab = SkillVocab.from_entries([(1001, 1), (1002, 2), (1003, 3)])

    class FakeModel(_TinyModel):
        def __init__(self, data_spec, model_config, *, vocab_size, repetition):
            super().__init__()
            self.input_encoder = nn.Linear(1, 1, bias=False)
            self.encoder = nn.Linear(1, 1, bias=False)
            self.output_head = nn.Linear(1, 1, bias=False)
            calls["model_args"] = (data_spec, model_config, vocab_size, repetition)

    def fake_build_dataloaders(*args, **kwargs):
        calls["dataloader_args"] = (args, kwargs)
        return ([0], [0], dataset, dataset)

    def fake_train_epoch(*args, **kwargs):
        assert kwargs["tensorboard_monitor"] is not None
        expected_dtype = torch.float32 if fp32_gradients else model_dtype(precision)
        assert all(parameter.dtype == (
            expected_dtype if name.startswith(('encoder.', 'output_head.')) else torch.float32
        ) for name, parameter in args[0].named_parameters())
        assert ("master_weights" in args[2].state_dict()) == (expected_dtype != torch.float32)
        owners = {parameter_name: group for group in args[2].param_groups
                  for parameter_name in group["param_names"]}
        assert owners["output_head.weight"]["weight_decay"] == 0.0
        assert owners["input_encoder.weight"]["weight_decay"] == config.weight_decay
        assert owners["encoder.weight"]["weight_decay"] == config.weight_decay
        if "resume_optimizer" in calls:
            assert all(group["lr"] == 0.02 for group in args[2].param_groups)
            for saved_state, current_state in zip(
                calls["resume_optimizer"].state.values(), args[2].state.values(), strict=True,
            ):
                assert current_state.keys() == saved_state.keys()
                for name, value in saved_state.items():
                    torch.testing.assert_close(current_state[name], value, rtol=0, atol=0)
        calls["train_step_offsets"].append(kwargs["global_step_offset"])
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

    monkeypatch.setattr(training_loop_impl, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(training_loop_impl, "resolve_policy_cache_dir", lambda job: tmp_path / "cache")
    monkeypatch.setattr(training_loop_impl, "resolve_registered_job_tags", lambda _root: ("black_mage",))
    monkeypatch.setattr(training_loop_impl, "CausalPolicyModel", FakeModel)
    monkeypatch.setattr(training_loop_impl, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(training_loop_impl, "validate", fake_validate)
    monkeypatch.setattr(training_loop_impl, "_save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(
        training_loop_impl,
        "create_tensorboard_writer",
        lambda _config, output_dir, **_kwargs: (
            tensorboard_output_dirs.append(Path(output_dir)) or tensorboard_writer
        ),
    )

    config = RunConfig(
        raw_data_dir=tmp_path / "configured-data",
        output_dir=tmp_path / "configured-output",
        job_tag="black_mage",
        model_variant="artzip",
        max_epochs=2,
        precision=precision,
        fp32_gradients=fp32_gradients,
        tensorboard=TensorBoardConfig(enabled=True, log_every_steps=1),
        model=ModelConfig(d_model=8, n_layers=1, n_heads=2, ff_dim=16),
    )
    result = training_module.run_training(
        config,
        raw_paths=[Path("sample.json")],
        validation_paths=[Path("val.json")],
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
    assert calls["train_step_offsets"] == [0, 1]
    assert tensorboard_output_dirs == [config.output_dir]
    assert tensorboard_writer.closed
    assert any(
        tag == "验证/真实副本每GCD平均威力（val_ppg）" and step == 1
        for tag, _value, step in tensorboard_writer.values
    )
    assert calls["model_args"][2] == dataset.skill_vocab.size()
    model_log = next(
        record.getMessage() for record in caplog.records
        if "模型: job=black_mage model_variant=artzip" in record.getMessage()
    )
    expected_parameter_dtype = torch.float32 if fp32_gradients else model_dtype(precision)
    assert f"precision={precision}" in model_log
    assert "input_encoder_parameter_dtype=torch.float32" in model_log
    assert f"encoder_parameter_dtype={expected_parameter_dtype}" in model_log
    assert f"output_head_parameter_dtype={expected_parameter_dtype}" in model_log
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
    resume_dtype = torch.float32 if fp32_gradients else model_dtype(precision)
    resume_model.encoder.to(dtype=resume_dtype)
    resume_model.output_head.to(dtype=resume_dtype)
    resume_optimizer = training_loop_impl.build_optimizer(
        resume_model, config.optimizer, learning_rate=0.02, weight_decay=config.weight_decay,
    )
    for parameter in resume_model.parameters():
        parameter.grad = torch.full_like(parameter, 0.1)
    resume_optimizer.step()
    resume_scheduler = torch.optim.lr_scheduler.LambdaLR(resume_optimizer, lambda _: 1.0)
    resume_scheduler_state = resume_scheduler.state_dict()
    # 保存的历史 batch 数与当前 loader 长度不同，用于验证续训不会倒退日志步号。
    resume_scheduler_state["last_epoch"] = saved_scheduler_step
    resume_scheduler_state["_step_count"] = (saved_scheduler_step or 0) + 1
    resume_input_contract = ModelInputContract.from_training(
        skill_vocab=dataset.skill_vocab,
        data_spec=DataSpec.from_dataset(dataset),
        schema=schema,
        normalizer=normalizer,
    )
    resume_checkpoint = tmp_path / "epoch_001_val_ppg_600.00.pt"
    resume_payload = {
        "epoch": 1,
        "model_state_dict": resume_model.state_dict(),
        "optimizer_state_dict": resume_optimizer.state_dict(),
        **({"scheduler_state_dict": resume_scheduler_state} if saved_scheduler_step is not None else {}),
        "model_config": asdict(config.model),
        "data_spec": asdict(DataSpec.from_dataset(dataset)),
        "model_variant": "artzip",
        "input_contract": resume_input_contract.to_dict(),
        "training_precision": config.precision,
        "run_config": {
            "max_files": None, "validation_files": config.validation_files,
            "fp32_gradients": fp32_gradients,
        },
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
    }
    torch.save(
        resume_payload,
        resume_checkpoint,
    )
    calls["checkpoints"] = []
    calls["train_step_offsets"] = []
    calls["resume_optimizer"] = resume_optimizer
    config = replace(config, weight_decay=0.5)
    resumed = training_module.run_training(
        config,
        raw_paths=[Path("sample.json")],
        validation_paths=[Path("val.json")],
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
    assert calls["train_step_offsets"] == [saved_scheduler_step if saved_scheduler_step is not None else 1]
    assert [entry[0].name for entry in calls["checkpoints"]] == [
        "epoch_002_val_ppg_900.00.pt",
        "final.pt",
    ]

    forced_checkpoint = tmp_path / "epoch_001_val_ppg_500.00.pt"
    forced_payload = {
        **resume_payload,
        "run_config": {"max_files": 1280, "fp32_gradients": fp32_gradients},
        "best_key": (999.0, 999.0, 999.0, 0.0),
        "best_val_metrics": {
            **resume_payload["metrics"],
            "top1_accuracy": 0.99,
            "top3_accuracy": 0.99,
            "val_ppg": 9999.0,
            "val_ppg_normalized": 9.999,
            "top1_val_ppg_average": 5.494,
        },
    }
    torch.save(forced_payload, forced_checkpoint)
    calls["checkpoints"] = []
    calls["train_step_offsets"] = []
    forced = training_module.run_training(
        config,
        raw_paths=[Path("sample.json")],
        validation_paths=[Path("val.json")],
        output_dir=tmp_path / "override-output",
        max_epochs=2,
        batch_size=4,
        learning_rate=0.01,
        device_name="cpu",
        validation_metrics_callback=fake_ppg,
        resume_path=forced_checkpoint,
        force_resume_data_mismatch=True,
    )

    assert forced["best_val_ppg"] == pytest.approx(900.0)
    assert calls["train_step_offsets"] == [saved_scheduler_step if saved_scheduler_step is not None else 1]
    assert [entry[0].name for entry in calls["checkpoints"]] == [
        "epoch_002_val_ppg_900.00.pt",
        "best.pt",
        "final.pt",
    ]


@pytest.mark.parametrize("failure_stage", ("validation_callback", "hparams"))
def test_run_training_closes_tensorboard_writer_when_monitoring_or_validation_raises(
    tmp_path,
    monkeypatch,
    failure_stage,
):
    schema = TrainingSchema(
        serialization_format="test",
        sample_schema_version=1,
        context_schema_version=1,
        scene_context_mode="absolute",
        scene_windows=(SceneWindowSchema.from_feature_keys(
            context_key="targetable_window_context",
            feature_keys=("start_offset_seconds", "end_offset_seconds", "duration_seconds"),
            scene_type_id=0,
        ),),
        state_groups=make_state_groups({"player_state": ("request_state.time_seconds", "b", "c")}, ("a", "b")), state_snapshots=("previous_action_after", "request_state"),

        skill_history_fields=(),
    )
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    normalizer = Normalizer.from_contract({
        **normalizer.normalization_contract, "skill_charge_limits": {"a": 2, "b": 2},
    })
    dataset = SimpleNamespace(
        job_tag="black_mage",
        num_actions=2,
        base_state_dim=3, state_dim=make_data_spec(base_state_dim=3).state_dim,
        scene_dim=3,
        num_scene_types=1,
        action_keys=("a", "b"),
        action_to_vocab_id=(1, 2),
        action_is_gcd=(True, True),
        skill_feature_names=("potency",),
        schema=schema,
        normalizer=normalizer,
    )

    class FakeWriter:
        log_dir = str(tmp_path / "artifacts" / "tensorboard" / "failed-run")

        def __init__(self):
            self.closed = False
            self.hparams = []

        def add_scalar(self, *_args):
            pass

        def add_hparams(self, config, metrics, **kwargs):
            self.hparams.append((config, metrics, kwargs))
            if failure_stage == "hparams":
                raise RuntimeError("hparams failed")

        def close(self):
            self.closed = True

    dataset.skill_vocab = SkillVocab.from_entries([(1001, 1), (1002, 2)])

    class FakeModel(_TinyModel):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.input_encoder = nn.Linear(1, 1, bias=False)
            self.encoder = nn.Linear(1, 1, bias=False)
            self.output_head = nn.Linear(1, 1, bias=False)

    writer = FakeWriter()
    monkeypatch.setattr(
        training_loop_impl,
        "create_tensorboard_writer",
        lambda *_args, **_kwargs: writer,
    )

    def fail_validation_callback(**_kwargs):
        if failure_stage == "validation_callback":
            raise RuntimeError("validation callback failed")
        return {"val_ppg": 1.0, "val_ppg_normalized": 0.1,
                "none_ppg": 1.0, "none_ppg_normalized": 0.1}

    config = RunConfig(
        raw_data_dir=tmp_path / "raw",
        output_dir=tmp_path / "output",
        job_tag="black_mage",
        model_variant="artzip",
        max_epochs=1,
        tensorboard=TensorBoardConfig(enabled=True),
        model=ModelConfig(
            d_model=8,
            n_layers=1,
            n_heads=2,
            ff_dim=16,
        ),
    )
    metrics = {
        "loss": 1.0,
        "cross_entropy_loss": 1.0,
        "value_preference_loss": 0.0,
        "top1_accuracy": 0.7,
        "top3_accuracy": 0.9,
    }

    monkeypatch.setattr(
        training_loop_impl, "build_dataloaders",
        lambda *_args, **_kwargs: ([0], [0], dataset, dataset),
    )
    monkeypatch.setattr(training_loop_impl, "train_epoch", lambda *_args, **_kwargs: metrics)
    monkeypatch.setattr(training_loop_impl, "validate", lambda *_args, **_kwargs: metrics)
    monkeypatch.setattr(training_loop_impl, "resolve_policy_cache_dir", lambda _job: tmp_path / "cache")
    monkeypatch.setattr(training_loop_impl, "resolve_registered_job_tags", lambda _root: ("black_mage",))
    monkeypatch.setattr(training_loop_impl, "CausalPolicyModel", FakeModel)
    monkeypatch.setattr(training_loop_impl, "_save_checkpoint", lambda *_args, **_kwargs: None)

    expected_error = "validation callback failed" if failure_stage == "validation_callback" else "hparams failed"
    with pytest.raises(RuntimeError, match=expected_error):
        training_module.run_training(
            config,
            raw_paths=[tmp_path / "scene.json"],
            validation_paths=[tmp_path / "val.json"],
            device_name="cpu",
            validation_metrics_callback=fail_validation_callback,
        )

    assert writer.closed
    assert len(writer.hparams) == 1
    assert writer.hparams[0][1]["验证/总损失"] == pytest.approx(1.0)
    assert writer.hparams[0][1]["训练/每轮/总损失"] == pytest.approx(1.0)
    assert writer.hparams[0][2]["global_step"] == 1


def test_best_metric_key_uses_requested_tie_break_order():
    base = {
        "top1_accuracy": 0.8,
        "top3_accuracy": 0.9,
        "value_preference_loss": 0.3,
        "top1_val_ppg_average": 0.75,
    }
    better_top3 = {**base, "top3_accuracy": 0.95}
    better_value_loss = {**better_top3, "value_preference_loss": 0.1}

    assert checkpoint_module._best_metric_key(
        better_top3,
        ppg_enabled=True,
    ) > checkpoint_module._best_metric_key(base, ppg_enabled=True)
    assert checkpoint_module._best_metric_key(
        better_value_loss,
        ppg_enabled=True,
    ) > checkpoint_module._best_metric_key(better_top3, ppg_enabled=True)


def test_run_training_rejects_empty_prepared_paths(tmp_path):
    config = RunConfig(raw_data_dir=tmp_path, output_dir=tmp_path, job_tag=None)

    with pytest.raises(FileNotFoundError, match="no prepared raw JSON"):
        training_module.run_training(config, raw_paths=[], validation_paths=[tmp_path / "val.json"], device_name="cpu")


def _resume_validation_context(tmp_path: Path, *, max_epochs: int = 3):
    schema = TrainingSchema(
        serialization_format="test",
        sample_schema_version=1,
        context_schema_version=1,
        scene_context_mode="absolute",
        scene_windows=(),
        state_groups=make_state_groups({"player_state": ("request_state.time_seconds", "b", "c")}, ("a", "b")), state_snapshots=("previous_action_after", "request_state"),

        skill_history_fields=(),
    )
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    normalizer = Normalizer.from_contract({
        **normalizer.normalization_contract, "skill_charge_limits": {"a": 2, "b": 2},
    })
    dataset = SimpleNamespace(
        job_tag="black_mage",
        num_actions=2,
        base_state_dim=3, state_dim=make_data_spec(base_state_dim=3).state_dim,
        scene_dim=0,
        num_scene_types=0,
        action_keys=("a", "b"),
        action_to_vocab_id=(1, 2),
        action_is_gcd=(True, True),
        skill_feature_names=("potency",),
        schema=schema,
        normalizer=normalizer,
    )
    data_spec = DataSpec.from_dataset(dataset)
    config = RunConfig(
        raw_data_dir=tmp_path / "raw",
        output_dir=tmp_path / "output",
        job_tag="black_mage",
        model_variant="artzip",
        max_epochs=max_epochs,
        model=ModelConfig(d_model=8, n_layers=1, n_heads=2, ff_dim=16),
    )
    input_contract = ModelInputContract.from_training(
        data_spec=data_spec,
        skill_vocab=SkillVocab.from_entries([(1001, 1), (1002, 2), (1003, 3)]),
        schema=schema,
        normalizer=normalizer,
    )
    dataset.skill_vocab = input_contract.create_skill_vocab()
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(config.model),
        "data_spec": asdict(data_spec),
        "model_variant": "artzip",
        "input_contract": input_contract.to_dict(),
        "training_precision": config.precision,
        "run_config": asdict(config),
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
        checkpoint_module._load_resume_checkpoint(tmp_path / "missing.pt")

    invalid = tmp_path / "invalid.pt"
    torch.save([], invalid)
    with pytest.raises(TypeError, match="resume checkpoint must be a mapping"):
        checkpoint_module._load_resume_checkpoint(invalid)


@pytest.mark.parametrize(
    ("field", "value", "message", "error_type"),
    [
        ("data_spec", {"job_tag": "black_mage", "state_dim": 999}, "training data spec mismatch", ValueError),
        ("model_config", {"d_model": 999}, "resume checkpoint model config mismatch", ValueError),
        ("model_config", {"transformer_activation": "swiglu"}, "resume checkpoint model config mismatch", ValueError),
        ("training_precision", "bf16", "resume checkpoint training precision mismatch", ValueError),
        ("model_variant", None, "resume checkpoint missing model_variant", ValueError),
        ("model_variant", "other_variant", "resume checkpoint model variant mismatch", ValueError),
        ("model_config", {"scorer_use_raw_projection": False, "scorer_use_candidate_hidden": True}, "removed model options", ValueError),
        ("model_config", {"position_id_semantics": "candidate_block_shared"}, "removed model options", ValueError),
        ("epoch", None, "resume checkpoint epoch must be an integer", ValueError),
        ("epoch", "not-an-int", "resume checkpoint epoch must be an integer", ValueError),
        ("epoch", True, "resume checkpoint epoch must be an integer", TypeError),
        ("epoch", 0, "resume checkpoint epoch must be >= 1", ValueError),
    ],
)
def test_validate_resume_checkpoint_rejects_contract_mismatches(
    tmp_path, field, value, message, error_type,
):
    context = _resume_validation_context(tmp_path)
    checkpoint = dict(context.checkpoint)
    if field == "model_variant" and value is None:
        checkpoint.pop(field)
    elif isinstance(value, dict):
        checkpoint[field] = {**checkpoint[field], **value}
    else:
        checkpoint[field] = value

    with pytest.raises(error_type, match=message):
        checkpoint_module._validate_resume_checkpoint(
            checkpoint,
            data_spec=context.data_spec,
            dataset=context.dataset,
            config=context.config,
            input_contract=context.input_contract,
        )


@pytest.mark.parametrize(
    ("run_config", "max_files", "force", "field"),
    [
        pytest.param({"max_files": 1280}, None, False, "max_files", id="limited-to-unlimited"),
        pytest.param({"max_files": None}, 1280, False, "max_files", id="unlimited-to-limited"),
        pytest.param(None, None, False, "max_files", id="legacy-unknown"),
        pytest.param({"validation_files": 101}, None, False, "validation_files", id="validation-count"),
        pytest.param({"max_files": 1280}, None, True, "max_files", id="force-limited-to-unlimited"),
        pytest.param(None, None, True, "max_files", id="force-legacy-unknown"),
    ],
)
def test_validate_resume_checkpoint_checks_saved_data_counts(
    tmp_path, caplog, run_config, max_files, force, field,
):
    context = _resume_validation_context(tmp_path)
    checkpoint = dict(context.checkpoint)
    if run_config is None:
        checkpoint.pop("run_config")
    else:
        checkpoint["run_config"] = {**checkpoint["run_config"], **run_config}
    config = replace(context.config, max_files=max_files)
    message = f"resume checkpoint {field} mismatch"

    with caplog.at_level(logging.WARNING), (nullcontext() if force else pytest.raises(ValueError, match=message)):
        checkpoint_module._validate_resume_checkpoint(
            checkpoint,
            data_spec=context.data_spec,
            dataset=context.dataset,
            config=config,
            input_contract=context.input_contract,
            force_resume_data_mismatch=force,
        )

    if force:
        assert any(message in record.getMessage() for record in caplog.records)


def test_validate_resume_checkpoint_accepts_explicit_unlimited_max_files(tmp_path):
    context = _resume_validation_context(tmp_path)
    checkpoint = dict(context.checkpoint)
    checkpoint["run_config"] = {
        "max_files": None,
        "validation_files": context.config.validation_files,
    }

    checkpoint_module._validate_resume_checkpoint(
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
        training_loop_impl,
        "build_dataloaders",
        lambda *_args, **_kwargs: ([0], [0], context.dataset, context.dataset),
    )

    with pytest.raises(ValueError, match="already reached epoch 1"):
        training_module.run_training(
            context.config,
            raw_paths=[tmp_path / "prepared.json"],
            validation_paths=[tmp_path / "val.json"],
            device_name="cpu",
            resume_path=checkpoint_path,
        )


def test_collect_checkpoint_candidates_uses_payload_epoch(tmp_path):
    """续训候选项按 checkpoint 载荷里的真实 epoch 判定，而不是文件名或命名约定。"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    # best.pt 恰好在当前 max_epochs 生成时不可续训。
    torch.save({"epoch": 12}, output_dir / "best.pt")
    # final.pt 的 epoch 是 8，把 max_epochs 提高到 12 后即可合法续训。
    torch.save({"epoch": 8}, output_dir / "final.pt")
    # 文件名看起来是第 5 轮，载荷却是 20 轮，应按载荷判定。
    torch.save({"epoch": 20}, output_dir / "epoch_005_val_ppg_500.00.pt")
    # 正常中间 checkpoint 仍然可续训。
    torch.save({"epoch": 3}, output_dir / "epoch_003_val_ppg_480.00.pt")

    resumable, rejected, skipped = training_module.collect_checkpoint_candidates(
        output_dir,
        max_epochs=12,
    )

    assert skipped == ()
    assert {item.path.name: item.epoch for item in resumable} == {
        "epoch_003_val_ppg_480.00.pt": 3,
        "final.pt": 8,
    }
    assert {item.path.name: item.epoch for item in rejected} == {
        "best.pt": 12,
        "epoch_005_val_ppg_500.00.pt": 20,
    }
    assert all(
        item.max_files is checkpoint_module.UNKNOWN_MAX_FILES
        for item in (*resumable, *rejected)
    )
    assert all(item.validation_files is None for item in (*resumable, *rejected))


def test_collect_checkpoint_candidates_exposes_saved_data_counts(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    torch.save(
        {"epoch": 2, "run_config": {"max_files": 1280, "validation_files": 100}},
        output_dir / "epoch_002.pt",
    )

    resumable, rejected, skipped = training_module.collect_checkpoint_candidates(
        output_dir,
        max_epochs=3,
    )

    assert skipped == ()
    assert len(rejected) == 0
    assert resumable[0].max_files == 1280
    assert resumable[0].validation_files == 100


def test_collect_checkpoint_candidates_distinguishes_unknown_and_explicit_unlimited(
    tmp_path,
):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    torch.save({"epoch": 1}, output_dir / "legacy.pt")
    torch.save(
        {"epoch": 2, "run_config": {"max_files": None}},
        output_dir / "unlimited.pt",
    )

    resumable, rejected, skipped = training_module.collect_checkpoint_candidates(
        output_dir,
        max_epochs=3,
    )

    assert skipped == ()
    assert len(rejected) == 0
    assert resumable[0].max_files is checkpoint_module.UNKNOWN_MAX_FILES
    assert resumable[1].max_files is None
    assert resumable[0].validation_files is None
    assert resumable[1].validation_files is None


def test_read_checkpoint_epoch_uses_mmap_without_loading_storages(tmp_path, monkeypatch):
    calls = {}

    def fake_safe_torch_load(path, **kwargs):
        calls["path"] = path
        calls.update(kwargs)
        return {"epoch": 4}

    monkeypatch.setattr(checkpoint_module, "safe_torch_load", fake_safe_torch_load)

    checkpoint_path = tmp_path / "epoch_004.pt"
    assert training_module.read_checkpoint_epoch(checkpoint_path) == 4
    assert calls["path"] == checkpoint_path
    assert calls["mmap"] is True
    assert calls["map_location"] == "meta"


def test_collect_checkpoint_candidates_rejects_invalid_inputs(tmp_path):
    """非法 checkpoint 载荷不中断扫描，非法轮数上限要直接报错。"""
    broken = tmp_path / "broken.pt"
    torch.save({"epoch": 0}, broken)

    with pytest.raises(ValueError, match="epoch must be >= 1"):
        training_module.read_checkpoint_epoch(broken)

    resumable, rejected, skipped = training_module.collect_checkpoint_candidates(
        tmp_path,
        max_epochs=2,
    )

    # 单个不可读 `.pt` 只登记为跳过，菜单仍可继续列出其余候选项。
    assert resumable == ()
    assert rejected == ()
    assert [item.path.name for item in skipped] == ["broken.pt"]
    assert "epoch must be >= 1" in skipped[0].error

    with pytest.raises(ValueError, match="max_epochs must be >= 1"):
        training_module.collect_checkpoint_candidates(tmp_path, max_epochs=0)


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
    key, metrics = checkpoint_module._restore_best_state(
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
    key, metrics = checkpoint_module._restore_best_state(
        context.checkpoint,
        resume_path,
        ppg_enabled=False,
    )
    assert key == checkpoint_module._best_metric_key(best_metrics, ppg_enabled=False)
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
    checkpoint_module._restore_scheduler_state(
        restored_scheduler,
        {"scheduler_state_dict": saved_state},
        completed_steps=99,
    )

    assert restored_scheduler.state_dict() == saved_state


def test_restore_rng_state_restores_matching_cuda_state_and_rejects_count_mismatch(monkeypatch):
    cuda_states = [torch.tensor([1]), torch.tensor([2])]
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda state: calls.append(state),
    )

    checkpoint_module._restore_rng_state({"rng_state": {"cuda": cuda_states}})
    assert calls == [cuda_states]

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="CUDA RNG state device count mismatch"):
        checkpoint_module._restore_rng_state({"rng_state": {"cuda": cuda_states}})

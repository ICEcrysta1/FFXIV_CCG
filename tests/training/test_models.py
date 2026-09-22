"""公共训练模型测试。"""

from __future__ import annotations

from collections import Counter
import random
from pathlib import Path

import pytest

from common.torch_runtime import autocast_context, model_dtype, move_batch
from scripts.convert_fflogs.cache import select_training_raw_paths
from tests.training._common_fixtures import make_dataset as _make_dataset
from tests.training._common_fixtures import make_demo_pt as _make_demo_pt
from common.policy.config import ModelConfig
from common.policy.data import DataSpec, SkillVocab
from common.policy.model import (
    CandidateTransformerModel,
    RepetitionConfig,
    build_split_attention_mask,
)
from training import TrainingCollator
from training.config import (
    ValuePreferenceConfig,
)
from common.policy.config import resolve_policy_cache_dir
from training.config import load_run_config
from training.loop.value_preference import (
    compute_value_preference_loss,
    load_skill_values,
)
from common.policy.model.input_encoder import build_position_ids
from common.policy.model.repetition import apply_repetition_penalty


def test_select_training_raw_paths_uses_directory_proportions(tmp_path):
    directory_counts = {"FRU": 4, "M5s": 6, "M11s": 10, "M12s": 8}
    for directory_name, count in directory_counts.items():
        directory = tmp_path / directory_name
        directory.mkdir()
        for index in range(count):
            (directory / f"sample_{index:03d}.json").touch()

    selected = select_training_raw_paths(tmp_path, max_files=14)

    assert len(selected) == 14
    assert Counter(path.parent.name for path in selected) == {
        "FRU": 2,
        "M5s": 3,
        "M11s": 5,
        "M12s": 4,
    }


def test_common_model_uses_pt_dimensions_and_job_route(tmp_path):
    torch = pytest.importorskip("torch")
    pt_path = _make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="model_demo")
    dataset = _make_dataset([pt_path])
    data_spec = DataSpec.from_dataset(dataset)

    model = CandidateTransformerModel(
        data_spec,
        ModelConfig(d_model=32, n_layers=1, n_heads=4, ff_dim=64, dropout=0.0),
        vocab_size=SkillVocab.build_from_job_tag(data_spec.job_tag).size(),
    ).eval()
    batch = TrainingCollator()([dataset[0], dataset[1]])

    output = model(batch)
    assert data_spec.job_tag == "black_mage"
    assert output["logits"].shape == (2, data_spec.num_candidates)
    assert output["top3_accuracy"].ndim == 0

    inference_batch = {key: value for key, value in batch.items() if key != "label_index"}
    predictions, logits = model.predict(inference_batch)
    assert predictions.shape == (2,)
    assert logits.shape == (2, data_spec.num_candidates)


@pytest.mark.parametrize("precision", ("float32", "bf16"))
def test_swiglu_training_updates_all_ffn_projections_with_checkpointing(tmp_path, precision):
    torch = pytest.importorskip("torch")
    if precision == "bf16" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("CUDA BF16 is unavailable")
    device = torch.device("cuda" if precision == "bf16" else "cpu")
    torch.manual_seed(53)
    pt_path = _make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="swiglu_train")
    dataset = _make_dataset([pt_path])
    data_spec = DataSpec.from_dataset(dataset)
    model = CandidateTransformerModel(
        data_spec,
        ModelConfig(
            d_model=32, n_layers=2, n_heads=4, num_kv_heads=1,
            ff_dim=96, dropout=0.1, transformer_activation="swiglu",
        ),
        vocab_size=SkillVocab.build_from_job_tag(data_spec.job_tag).size(),
    ).to(device=device, dtype=model_dtype(precision)).train()
    model.enable_activation_checkpoint_attention()
    model.enable_activation_checkpoint_ffn()
    batch = move_batch(TrainingCollator()([dataset[0], dataset[1]]), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    weights_before = {
        (index, name): getattr(layer, name).weight.detach().clone()
        for index, layer in enumerate(model.encoder.layers)
        for name in ("linear1", "gate_proj", "linear2")
    }
    with autocast_context(device, precision):
        loss = model(batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    for layer in model.encoder.layers:
        for name in ("linear1", "gate_proj", "linear2"):
            gradient = getattr(layer, name).weight.grad
            assert gradient is not None
            assert torch.isfinite(gradient).all()
            assert torch.count_nonzero(gradient) > 0
    optimizer.step()
    for (index, name), before in weights_before.items():
        after = getattr(model.encoder.layers[index], name).weight
        assert torch.isfinite(after).all()
        assert not torch.equal(before, after)


def test_position_ids_are_logical_sequential_when_all_tokens_are_valid():
    torch = pytest.importorskip("torch")
    position_ids = build_position_ids(
        batch_size=1,
        scene_length=3,
        history_length=2,
        candidate_count=4,
        device=torch.device("cpu"),
    )[0].tolist()

    assert position_ids == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert position_ids[-1] == 9


def test_position_ids_ignore_right_padding_per_sample():
    torch = pytest.importorskip("torch")
    position_ids = build_position_ids(
        batch_size=2,
        scene_length=3,
        history_length=4,
        candidate_count=2,
        device=torch.device("cpu"),
        scene_mask=torch.tensor([[True, True, False], [True, True, True]]),
        history_mask=torch.tensor(
            [[True, False, False, False], [True, True, True, False]]
        ),
    )

    assert position_ids.tolist() == [
        [0, 1, 0, 2, 0, 0, 0, 3, 4, 5],
        [0, 1, 2, 3, 4, 5, 0, 6, 7, 8],
    ]


def test_position_ids_count_valid_tokens_without_right_padding():
    torch = pytest.importorskip("torch")
    position_ids = build_position_ids(
        batch_size=1,
        scene_length=3,
        history_length=4,
        candidate_count=2,
        device=torch.device("cpu"),
        scene_mask=torch.tensor([[False, True, True]]),
        history_mask=torch.tensor([[False, True, False, True]]),
    )

    assert position_ids.tolist() == [[0, 0, 1, 0, 2, 0, 3, 4, 5, 6]]


def test_candidate_rope_positions_are_indexed_and_cls_is_after_candidate_block():
    torch = pytest.importorskip("torch")
    position_ids = build_position_ids(
        batch_size=1,
        scene_length=2,
        history_length=2,
        candidate_count=3,
        device=torch.device("cpu"),
    )

    assert position_ids.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_rope_logits_are_invariant_to_other_samples_right_padding():
    torch = pytest.importorskip("torch")
    data_spec = DataSpec(
        job_tag="black_mage",
        num_candidates=2,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=1,
        num_scene_types=1,
        candidate_action_keys=("fire_iii", "fire_iv"),
        skill_feature_names=("potency",),
    )
    model = CandidateTransformerModel(
        data_spec,
        ModelConfig(
            d_model=8,
            pair_embedding_dim=6,
            n_layers=1,
            n_heads=2,
            ff_dim=16,
            dropout=0.0,
        ),
        vocab_size=4,
    ).eval()

    short_batch = {
        "history_skill_ids": torch.tensor([[1]]),
        "history_skill_features": torch.tensor([[[0.25]]]),
        "history_state_vectors": torch.tensor([[[0.1, 0.2, 0.3]]]),
        "history_state_null_mask": torch.zeros((1, 1, 3), dtype=torch.bool),
        "history_mask": torch.ones((1, 1), dtype=torch.bool),
        "candidate_skill_ids": torch.tensor([[1, 2]]),
        "candidate_skill_features": torch.tensor([[[0.4], [0.5]]]),
        "candidate_state_vectors": torch.tensor(
            [[[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]]
        ),
        "candidate_state_null_mask": torch.zeros((1, 2, 3), dtype=torch.bool),
        "candidate_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]),
        "scene_types": torch.zeros((1, 2), dtype=torch.long),
        "scene_mask": torch.ones((1, 2), dtype=torch.bool),
    }
    padded_batch = {
        key: torch.cat((value, value), dim=0)
        for key, value in short_batch.items()
    }
    padded_batch["scene_vectors"] = torch.cat(
        (
            torch.cat((short_batch["scene_vectors"], torch.zeros((1, 1, 2))), dim=1),
            torch.tensor([[[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]]),
        ),
        dim=0,
    )
    padded_batch["scene_types"] = torch.zeros((2, 3), dtype=torch.long)
    padded_batch["scene_mask"] = torch.tensor(
        [[True, True, False], [True, True, True]]
    )
    padded_batch["history_skill_ids"] = torch.tensor([[1, 0], [2, 3]])
    padded_batch["history_skill_features"] = torch.tensor(
        [[[0.25], [0.0]], [[0.6], [0.7]]]
    )
    padded_batch["history_state_vectors"] = torch.tensor(
        [
            [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]],
            [[0.8, 0.9, 1.0], [1.1, 1.2, 1.3]],
        ]
    )
    padded_batch["history_state_null_mask"] = torch.tensor(
        [
            [[False, False, False], [True, True, True]],
            [[False, False, False], [False, False, False]],
        ]
    )
    padded_batch["history_mask"] = torch.tensor(
        [[True, False], [True, True]]
    )

    with torch.no_grad():
        standalone_logits = model(short_batch)["logits"]
        mixed_logits = model(padded_batch)["logits"]

    torch.testing.assert_close(
        standalone_logits[0], mixed_logits[0], rtol=1e-5, atol=1e-6
    )


def test_split_attention_mask_keeps_prefix_causal_and_candidates_bidirectional():
    torch = pytest.importorskip("torch")
    mask = build_split_attention_mask(
        prefix_length=5,
        candidate_count=2,
        device=torch.device("cpu"),
    )

    assert mask is not None
    assert mask.shape == (8, 8)
    assert mask[0, :1].tolist() == [False]
    assert mask[0, 1:5].all().item() is True
    assert mask[4, :5].all().item() is False
    assert mask[4, 5:].all().item() is True
    assert mask[5, :7].all().item() is False
    assert mask[5, 7].item() is True
    assert mask[6, :7].all().item() is False
    assert mask[6, 7].item() is True
    assert not mask[7].any().item()


def test_split_attention_model_uses_single_candidate_block():
    torch = pytest.importorskip("torch")
    data_spec = DataSpec(
        job_tag="black_mage",
        num_candidates=2,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=1,
        num_scene_types=1,
        candidate_action_keys=("fire_iii", "fire_iv"),
        skill_feature_names=("cast_time.seconds",),
    )
    model = CandidateTransformerModel(
        data_spec,
        ModelConfig(
            d_model=8,
            pair_embedding_dim=6,
            n_layers=1,
            n_heads=2,
            ff_dim=16,
            dropout=0.0,
        ),
        vocab_size=4,
    ).eval()
    batch = {
        "history_skill_ids": torch.ones((1, 2), dtype=torch.int64),
        "history_skill_features": torch.zeros((1, 2, 1)),
        "history_state_vectors": torch.zeros((1, 2, 3)),
        "history_state_null_mask": torch.zeros((1, 2, 3), dtype=torch.bool),
        "history_mask": torch.ones((1, 2), dtype=torch.bool),
        "candidate_skill_ids": torch.ones((1, 2), dtype=torch.int64),
        "candidate_skill_features": torch.zeros((1, 2, 1)),
        "candidate_state_vectors": torch.zeros((1, 2, 3)),
        "candidate_state_null_mask": torch.zeros((1, 2, 3), dtype=torch.bool),
        "candidate_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.zeros((1, 1, 2)),
        "scene_types": torch.zeros((1, 1), dtype=torch.int64),
        "scene_mask": torch.ones((1, 1), dtype=torch.bool),
    }

    output = model(batch)
    assert output["logits"].shape == (1, 2)
    encoded = model.input_encoder(batch)
    assert "attention_mask" not in encoded
    assert encoded["prefix_length"] == 3
    assert encoded["candidate_count"] == 2
    assert encoded["candidate_positions"].tolist() == [3, 4]


def test_input_encoder_derives_context_capacity_from_context_blocks():
    torch = pytest.importorskip("torch")
    data_spec = DataSpec(
        job_tag="black_mage",
        num_candidates=2,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=1,
        num_scene_types=1,
        candidate_action_keys=("fire_iii", "fire_iv"),
        skill_feature_names=("cast_time.seconds",),
    )
    model = CandidateTransformerModel(
        data_spec,
        ModelConfig(
            d_model=8,
            pair_embedding_dim=6,
            n_layers=1,
            n_heads=2,
            ff_dim=16,
            dropout=0.0,
            scene_capacity=4,
            history_capacity=2,
        ),
        vocab_size=4,
    ).eval()
    # scene 4 + history 2 + candidate 2 + CLS 1 = 9，由各块容量自动换算。
    assert model.input_encoder.max_token_count == 9
    batch = {
        "history_skill_ids": torch.ones((1, 2), dtype=torch.int64),
        "history_skill_features": torch.zeros((1, 2, 1)),
        "history_state_vectors": torch.zeros((1, 2, 3)),
        "history_state_null_mask": torch.zeros((1, 2, 3), dtype=torch.bool),
        "history_mask": torch.ones((1, 2), dtype=torch.bool),
        "candidate_skill_ids": torch.ones((1, 2), dtype=torch.int64),
        "candidate_skill_features": torch.zeros((1, 2, 1)),
        "candidate_state_vectors": torch.zeros((1, 2, 3)),
        "candidate_state_null_mask": torch.zeros((1, 2, 3), dtype=torch.bool),
        "candidate_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.zeros((1, 4, 2)),
        "scene_types": torch.zeros((1, 4), dtype=torch.int64),
        "scene_mask": torch.ones((1, 4), dtype=torch.bool),
    }

    with torch.no_grad():
        output = model(batch)
    assert output["logits"].shape == (1, 2)

    batch["scene_vectors"] = torch.zeros((1, 5, 2))
    batch["scene_types"] = torch.zeros((1, 5), dtype=torch.int64)
    batch["scene_mask"] = torch.ones((1, 5), dtype=torch.bool)
    with pytest.raises(ValueError, match="model.scene_capacity"):
        model(batch)


def test_model_encode_with_attention_returns_per_head_weights():
    torch = pytest.importorskip("torch")
    data_spec = DataSpec(
        job_tag="black_mage",
        num_candidates=2,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=1,
        num_scene_types=1,
        candidate_action_keys=("fire_iii", "fire_iv"),
        skill_feature_names=("cast_time.seconds",),
    )
    model = CandidateTransformerModel(
        data_spec,
        ModelConfig(
            d_model=8,
            pair_embedding_dim=6,
            n_layers=2,
            n_heads=2,
            ff_dim=16,
            dropout=0.0,
        ),
        vocab_size=4,
    ).eval()
    batch = {
        "history_action_keys": [["fire_iii"]],
        "candidate_action_keys": [["fire_iii", "fire_iv"]],
        "history_skill_ids": torch.ones((1, 1), dtype=torch.int64),
        "history_skill_features": torch.zeros((1, 1, 1)),
        "history_state_vectors": torch.zeros((1, 1, 3)),
        "history_state_null_mask": torch.zeros((1, 1, 3), dtype=torch.bool),
        "history_mask": torch.ones((1, 1), dtype=torch.bool),
        "candidate_skill_ids": torch.ones((1, 2), dtype=torch.int64),
        "candidate_skill_features": torch.zeros((1, 2, 1)),
        "candidate_state_vectors": torch.zeros((1, 2, 3)),
        "candidate_state_null_mask": torch.zeros((1, 2, 3), dtype=torch.bool),
        "candidate_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.zeros((1, 1, 2)),
        "scene_types": torch.zeros((1, 1), dtype=torch.int64),
        "scene_mask": torch.ones((1, 1), dtype=torch.bool),
    }

    encoded, hidden, attentions = model.encode_with_attention(batch)

    assert hidden.shape == encoded["tokens"].shape
    assert len(attentions) == 2
    assert attentions[0].shape == (1, 2, encoded["tokens"].shape[1], encoded["tokens"].shape[1])
    assert encoded["candidate_positions"].tolist() == [2, 3]


def test_pair_embedding_reduces_history_and_candidate_to_one_token():
    torch = pytest.importorskip("torch")
    data_spec = DataSpec(
        job_tag="black_mage",
        num_candidates=2,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=1,
        num_scene_types=1,
        candidate_action_keys=("fire_iii", "fire_iv"),
        skill_feature_names=("cast_time.seconds",),
    )
    model = CandidateTransformerModel(
        data_spec,
        ModelConfig(
            d_model=8,
            pair_embedding_dim=6,
            n_layers=1,
            n_heads=2,
            ff_dim=16,
            dropout=0.0,
        ),
        vocab_size=4,
    ).eval()
    batch = {
        "history_skill_ids": torch.ones((1, 2), dtype=torch.int64),
        "history_skill_features": torch.zeros((1, 2, 1)),
        "history_state_vectors": torch.zeros((1, 2, 3)),
        "history_state_null_mask": torch.zeros((1, 2, 3), dtype=torch.bool),
        "history_mask": torch.ones((1, 2), dtype=torch.bool),
        "candidate_skill_ids": torch.ones((1, 2), dtype=torch.int64),
        "candidate_skill_features": torch.zeros((1, 2, 1)),
        "candidate_state_vectors": torch.zeros((1, 2, 3)),
        "candidate_state_null_mask": torch.zeros((1, 2, 3), dtype=torch.bool),
        "candidate_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.zeros((1, 1, 2)),
        "scene_types": torch.zeros((1, 1), dtype=torch.int64),
        "scene_mask": torch.ones((1, 1), dtype=torch.bool),
    }

    encoded = model.input_encoder(batch)

    assert encoded["tokens"].shape[1] == 1 + 2 + 2 + 1
    assert "raw_candidate_pair" not in encoded
    assert encoded["candidate_positions"].tolist() == [3, 4]


def test_input_encoder_routes_all_token_sources_through_shared_embedding():
    torch = pytest.importorskip("torch")
    data_spec = DataSpec(
        job_tag="black_mage",
        num_candidates=2,
        state_dim=3,
        scene_dim=2,
        skill_feature_dim=1,
        num_scene_types=1,
        candidate_action_keys=("fire_iii", "fire_iv"),
        skill_feature_names=("cast_time.seconds",),
    )
    model = CandidateTransformerModel(
        data_spec,
        ModelConfig(
            d_model=8,
            pair_embedding_dim=6,
            n_layers=1,
            n_heads=2,
            ff_dim=16,
            dropout=0.0,
        ),
        vocab_size=4,
    ).eval()
    batch = {
        "history_skill_ids": torch.ones((1, 2), dtype=torch.int64),
        "history_skill_features": torch.zeros((1, 2, 1)),
        "history_state_vectors": torch.zeros((1, 2, 3)),
        "history_state_null_mask": torch.zeros((1, 2, 3), dtype=torch.bool),
        "history_mask": torch.ones((1, 2), dtype=torch.bool),
        "candidate_skill_ids": torch.ones((1, 2), dtype=torch.int64),
        "candidate_skill_features": torch.zeros((1, 2, 1)),
        "candidate_state_vectors": torch.zeros((1, 2, 3)),
        "candidate_state_null_mask": torch.zeros((1, 2, 3), dtype=torch.bool),
        "candidate_legal_mask": torch.ones((1, 2), dtype=torch.bool),
        "scene_vectors": torch.zeros((1, 1, 2)),
        "scene_types": torch.zeros((1, 1), dtype=torch.int64),
        "scene_mask": torch.ones((1, 1), dtype=torch.bool),
    }
    captured = {}

    def capture(_module, inputs, _output):
        captured["shape"] = tuple(inputs[0].shape)

    handle = model.input_encoder.token_embedding.register_forward_hook(capture)
    try:
        model.input_encoder(batch)
    finally:
        handle.remove()

    assert captured["shape"] == (1, 6, 6)
    assert model.input_encoder.scene_proj[0].out_features == 6
    assert model.input_encoder.cls_token.shape[-1] == 6


def test_job_model_config_loads_training_precision(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "output_dir: artifacts/checkpoints/test\n"
        "model:\n"
        "  pair_embedding_dim: 192\n"
        "  full_attention_residuals: true\n"
        "training:\n"
        "  precision: bf16\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.precision == "bf16"
    assert config.model.pair_embedding_dim == 192
    assert config.model.full_attention_residuals is True
    assert config.model.transformer_norm_first is True
    assert config.model.transformer_activation == "gelu"


def test_job_model_config_loads_ffn_activation_checkpoint_switch(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  activation_checkpoint_ffn: true\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.activation_checkpoint_ffn is True


def test_job_model_config_loads_attention_activation_checkpoint_switch(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  activation_checkpoint_attention: true\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.activation_checkpoint_attention is True
    assert config.activation_checkpoint_attention_block is False


def test_job_model_config_loads_attention_checkpoint_block_switch(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  activation_checkpoint_attention: true\n"
        "  activation_checkpoint_attention_block: true\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.activation_checkpoint_attention_block is True


def test_job_model_config_loads_runtime_debug_switch(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  runtime_debug:\n"
        "    enabled: true\n"
        "    max_steps: 3\n"
        "    synchronize: false\n"
        "    output_filename: memory-debug.jsonl\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.runtime_debug.enabled is True
    assert config.runtime_debug.max_steps == 3
    assert config.runtime_debug.synchronize is False
    assert config.runtime_debug.output_filename == "memory-debug.jsonl"


def test_black_mage_artzip_uses_current_mainline_architecture():
    config = load_run_config(Path("config/models/black_mage/artzip/config.yaml"))

    assert config.model.d_model == 768
    assert config.model.n_layers == 12
    assert config.model.n_heads == 12
    assert config.model.num_kv_heads == 1
    assert config.model.ff_dim == 3072
    assert config.model.transformer_activation == "swiglu"
    assert config.model.history_capacity == 384
    assert config.model.full_attention_residuals is False
    assert config.candidate_shuffle_enabled is False
    assert config.candidate_shuffle_probability == 1.0


def test_job_model_config_loads_history_truncation_switch(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  history_truncation:\n"
        "    enabled: true\n"
        "    probability: 0.25\n"
        "    min_recent: 2\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.history_truncation_enabled is True
    assert config.history_truncation_probability == 0.25
    assert config.history_min_recent == 2


def test_job_model_config_loads_candidate_shuffle_switch(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  candidate_shuffle:\n"
        "    enabled: true\n"
        "    probability: 0.75\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.candidate_shuffle_enabled is True
    assert config.candidate_shuffle_probability == 0.75


def test_job_model_config_loads_candidate_order_file(tmp_path):
    order_path = tmp_path / "candidate_order.yaml"
    order_path.write_text(
        "candidate_order:\n"
        "  1: fire_iv\n"
        "  2: ogcd_wait\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  candidate_order_file: candidate_order.yaml\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.candidate_order_file == order_path.resolve()


def test_training_collator_builds_runtime_candidate_values_without_skill_features(tmp_path):
    torch = pytest.importorskip("torch")
    pt_path = _make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="value_batch_demo")
    dataset = _make_dataset([pt_path])
    sample = dataset[0]
    sample.pop("candidate_values")
    values = {
        action_key: float(index + 1)
        for index, action_key in enumerate(sample["candidate_action_keys"])
    }

    batch = TrainingCollator(skill_values=values)([sample])

    assert "value" not in dataset.skill_feature_names
    assert batch["candidate_values"].shape == (1, len(sample["candidate_action_keys"]))
    assert batch["candidate_values"].tolist()[0] == [values[key] for key in sample["candidate_action_keys"]]


def test_training_collator_prefers_dynamic_candidate_values_from_state(tmp_path):
    torch = pytest.importorskip("torch")
    pt_path = _make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="dynamic_value_batch_demo")
    sample = _make_dataset([pt_path])[0]
    runtime_values = torch.tensor([2.0, 1.0], dtype=sample["candidate_values"].dtype)
    sample["candidate_values"] = runtime_values

    batch = TrainingCollator(
        skill_values={key: 99.0 for key in sample["candidate_action_keys"]}
    )([sample])

    assert batch["candidate_values"].tolist()[0] == [2.0, 1.0]


def test_job_model_config_loads_value_preference_switch(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  value_preference:\n"
        "    enabled: true\n"
        "    loss_weight: 0.1\n"
        "    margin_scale: 0.5\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.value_preference.enabled is True
    assert config.value_preference.loss_weight == 0.1
    assert config.value_preference.margin_scale == 0.5


def test_value_preference_only_pushes_high_value_label_against_lower_value_legal_candidates():
    torch = pytest.importorskip("torch")
    config = ValuePreferenceConfig(enabled=True, loss_weight=0.05, margin_scale=0.25)
    batch = {
        "candidate_values": torch.tensor([[2.0, 1.0, 1.0]]),
        "candidate_legal_mask": torch.tensor([[True, True, True]]),
        "label_index": torch.tensor([0]),
    }

    unranked = compute_value_preference_loss(
        torch.zeros((1, 3)), batch, config
    )
    ranked = compute_value_preference_loss(
        torch.tensor([[1.0, 0.0, 0.0]]), batch, config
    )

    assert ranked < unranked

    batch["label_index"] = torch.tensor([1])
    assert compute_value_preference_loss(
        torch.zeros((1, 3)), batch, config
    ).item() == pytest.approx(0.0)


def test_value_preference_requires_runtime_job_values():
    torch = pytest.importorskip("torch")
    config = ValuePreferenceConfig(enabled=True, loss_weight=0.05, margin_scale=0.25)

    with pytest.raises(ValueError, match="runtime candidate values"):
        compute_value_preference_loss(
            torch.zeros((1, 1)),
            {"label_index": torch.tensor([0]), "candidate_legal_mask": torch.ones((1, 1), dtype=torch.bool)},
            config,
        )


def test_value_preference_loads_values_from_job_yaml():
    values = load_skill_values("machinist")

    assert values["full_metal_field"] == pytest.approx(2.0)
    assert values["heated_split_shot"] == pytest.approx(1.0)
    assert values["ogcd_wait"] == pytest.approx(1.0)


def test_training_collator_truncates_only_early_history_and_preserves_scene(tmp_path):
    torch = pytest.importorskip("torch")
    pt_path = _make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="history_demo")
    sample = dict(_make_dataset([pt_path])[1])
    state_width = sample["history_bank_state_vectors"].shape[-1]
    feature_width = sample["history_bank_skill_features"].shape[-1]
    sample["history_action_keys"] = ["old", "middle", "latest"]
    sample["history_end"] = 4
    sample["history_length"] = 3
    sample["history_bank_action_keys"] = ("", "old", "middle", "latest")
    sample["history_bank_skill_ids"] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    sample["history_bank_skill_features"] = torch.zeros((4, feature_width))
    sample["history_bank_state_vectors"] = torch.zeros((4, state_width))
    sample["history_bank_state_vectors"][:, 0] = torch.tensor([0.0, 10.0, 20.0, 30.0])
    sample["history_bank_state_null_mask"] = torch.zeros((4, state_width), dtype=torch.bool)
    scene_before = sample["scene_vectors"].clone()
    candidate_before = {
        key: sample[key].clone()
        for key in (
            "candidate_skill_ids",
            "candidate_skill_features",
            "candidate_state_vectors",
            "candidate_state_null_mask",
            "candidate_legal_mask",
        )
    }

    batch = TrainingCollator(
        history_truncation_enabled=True,
        history_truncation_probability=1.0,
        history_min_recent=1,
        rng=random.Random(0),
    )([sample])

    history_length = int(batch["history_lengths"][0].item())
    assert 1 <= history_length < 3
    assert batch["history_action_keys"][0][-1] == "latest"
    assert batch["history_bank_state_vectors"][batch["history_ends"][0] - 1, 0].item() == 30.0
    assert torch.equal(batch["scene_vectors"][0, : scene_before.shape[0]], scene_before)
    for key, expected in candidate_before.items():
        assert torch.equal(batch[key][0], expected)


def test_training_collator_shuffles_candidates_as_aligned_records(tmp_path):
    torch = pytest.importorskip("torch")
    pt_path = _make_demo_pt(tmp_path, ["fire_iii", "fire_iv"], fight_id="candidate_demo")
    sample = dict(_make_dataset([pt_path])[1])
    candidate_keys_before = list(sample["candidate_action_keys"])
    candidate_tensors_before = {
        key: sample[key].clone()
        for key in (
            "candidate_skill_ids",
            "candidate_skill_features",
            "candidate_values",
            "candidate_state_vectors",
            "candidate_state_null_mask",
            "candidate_legal_mask",
        )
    }
    label_key = sample["label_action_key"]
    skill_values = {
        action_key: float(index + 1)
        for index, action_key in enumerate(candidate_keys_before)
    }

    batch = TrainingCollator(
        candidate_shuffle_enabled=True,
        candidate_shuffle_probability=1.0,
        skill_values=skill_values,
        rng=random.Random(0),
    )([sample])

    shuffled_keys = batch["candidate_action_keys"][0]
    assert sorted(shuffled_keys) == sorted(candidate_keys_before)
    assert shuffled_keys[batch["label_index"][0].item()] == label_key
    for shuffled_index, key in enumerate(shuffled_keys):
        original_index = candidate_keys_before.index(key)
        assert batch["candidate_values"][0, shuffled_index].item() == pytest.approx(
            sample["candidate_values"][original_index].item()
        )
        for tensor_key, expected in candidate_tensors_before.items():
            assert torch.equal(batch[tensor_key][0, shuffled_index], expected[original_index])


def test_repetition_whitelist_only_penalizes_non_whitelisted_repeat():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 4))
    batch = {
        "candidate_skill_ids": torch.zeros((1, 4), dtype=torch.int64),
        "history_action_keys": [["ogcd_wait", "fire_iii"]],
        "candidate_action_keys": [["fire_iii", "fire_iv", "xenoglossy", "flare"]],
    }

    adjusted = apply_repetition_penalty(
        logits,
        batch,
        RepetitionConfig(
            mode="whitelist",
            skills=("fire_iv", "xenoglossy", "flare"),
            penalty=1.0,
        ),
    )

    assert adjusted.tolist() == [[-1.0, 0.0, 0.0, 0.0]]


def test_repetition_blacklist_only_penalizes_listed_repeat():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 2))
    batch = {
        "candidate_skill_ids": torch.zeros((1, 2), dtype=torch.int64),
        "history_action_keys": [["fire_iv"]],
        "candidate_action_keys": [["fire_iv", "fire_iii"]],
    }

    adjusted = apply_repetition_penalty(
        logits,
        batch,
        RepetitionConfig(mode="blacklist", skills=("fire_iv",), penalty=0.75),
    )

    assert adjusted.tolist() == [[-0.75, 0.0]]


def test_job_model_config_loads_repetition_policy(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  repetition:\n"
        "    mode: whitelist\n"
        "    skills: [fire_iv, flare]\n"
        "    penalty: 1.25\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.repetition.mode == "whitelist"
    assert config.repetition.skills == ("fire_iv", "flare")
    assert config.repetition.penalty == 1.25


def test_low_precision_autocast_requires_cuda():
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="requires a CUDA device"):
        autocast_context(torch.device("cpu"), "bf16")


def _obsolete_job_model_config_loads_bounded_sample_cache(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  sample_cache_size: 64\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.sample_cache_size == 64


def test_job_model_config_loads_bounded_compiled_cache(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  compiled_cache_shard_size: 128\n"
        "  compiled_cache_max_shards: 3\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.compiled_cache_shard_size == 128
    assert config.compiled_cache_max_shards == 3


def test_job_model_config_loads_compiled_cache_workers(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "raw_data_dir: data/human/job/black_mage/raw/FRU\n"
        "training:\n"
        "  compiled_cache_workers: 4\n",
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.compiled_cache_workers == 4


def test_training_cache_dir_is_scoped_by_job(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAINING_CACHE_ROOT", str(tmp_path))

    assert resolve_policy_cache_dir("black_mage") == tmp_path / "black_mage" / ".cache"

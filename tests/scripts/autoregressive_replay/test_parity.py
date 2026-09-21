"""PyTorch / ORT parity 双跑测试。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from common.policy.model import RepetitionConfig
from scripts.autoregressive_replay import main as replay_main_module
from scripts.autoregressive_replay import parity as parity_module
from scripts.autoregressive_replay.backends import compare_backend_logits
from scripts.autoregressive_replay.config import AutoregressiveReplayConfig


def test_backend_parity_failure_reports_candidate_logits_and_actions():
    class FakeBackend:
        def __init__(self, logits):
            self.logits = logits

        def raw_logits(self, _batch, _candidate_keys):
            return self.logits

    with pytest.raises(AssertionError) as error:
        compare_backend_logits(
            FakeBackend(torch.tensor([[1.0, 2.0, 3.0]])),
            FakeBackend(torch.tensor([[4.0, 2.0, 1.0]])),
            {},
            ("fire_iii", "fire_iv", "blizzard_iii"),
        )
    message = str(error.value)
    assert "fire_iii" in message
    assert "reference_logit=1.00000000" in message
    assert "candidate_logit=4.00000000" in message
    assert "reference_top1='blizzard_iii'" in message
    assert "candidate_top1='fire_iii'" in message


def test_parity_fixed_capacity_batch_stays_on_reference_device(tmp_path):
    """固定容量 batch 必须与 PT reference 同 device，避免 CUDA device mismatch。"""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from tests.scripts.onnx_export.test_exporter import _write_small_checkpoint
    from scripts.autoregressive_replay.backends import (
        ParityPolicyBackend,
        PyTorchPolicyBackend,
        _require_tensor,
    )
    from scripts.onnx_export import CapacityContract, TENSOR_INPUT_NAMES
    from scripts.onnx_export.contracts.contract import make_inputs, slice_dynamic_inputs
    from scripts.onnx_export.contracts.deployment_contract import TensorSpec

    checkpoint = tmp_path / "model.pt"
    data_spec, _input_contract = _write_small_checkpoint(checkpoint)
    reference = PyTorchPolicyBackend(
        checkpoint,
        device="cuda",
        use_kv_cache=False,
    )
    contract = CapacityContract(3, 4, data_spec.num_candidates)
    dynamic_inputs = slice_dynamic_inputs(
        make_inputs(
            data_spec,
            contract,
            vocab_size=8,
            scene_valid=2,
            history_valid=3,
            dtype=torch.float32,
            seed=4242,
        ),
        scene_valid=2,
        history_valid=3,
    )
    live_batch = dict(zip(TENSOR_INPUT_NAMES, dynamic_inputs, strict=True))
    live_batch.update(
        {
            "candidate_legal_mask": torch.ones(
                (1, data_spec.num_candidates), dtype=torch.bool
            ),
            "candidate_action_keys": [list(data_spec.candidate_action_keys)],
            "history_action_keys": [["fire_iii", "fire_iv", "fire_iv"]],
        }
    )

    b, s, h, c = 1, 3, 4, data_spec.num_candidates
    sd, fd, xd = data_spec.state_dim, data_spec.skill_feature_dim, data_spec.scene_dim

    def tensor_inputs():
        return (
            TensorSpec("scene_vectors", "tensor(float)", (b, s, xd), ""),
            TensorSpec("scene_types", "tensor(int64)", (b, s), ""),
            TensorSpec("scene_mask", "tensor(bool)", (b, s), ""),
            TensorSpec("history_skill_ids", "tensor(int64)", (b, h), ""),
            TensorSpec("history_skill_features", "tensor(float)", (b, h, fd), ""),
            TensorSpec("history_state_vectors", "tensor(float)", (b, h, sd), ""),
            TensorSpec("history_state_null_mask", "tensor(bool)", (b, h, sd), ""),
            TensorSpec("history_mask", "tensor(bool)", (b, h), ""),
            TensorSpec("candidate_skill_ids", "tensor(int64)", (b, c), ""),
            TensorSpec("candidate_skill_features", "tensor(float)", (b, c, fd), ""),
            TensorSpec("candidate_state_vectors", "tensor(float)", (b, c, sd), ""),
            TensorSpec("candidate_state_null_mask", "tensor(bool)", (b, c, sd), ""),
        )

    fake_contract = SimpleNamespace(
        capacity=contract,
        tensor_inputs=tensor_inputs,
        validate_tensor_inputs=lambda _inputs: None,
    )
    seen_device = {}

    def fake_raw_logits(batch, candidate_action_keys):
        seen_device["device"] = _require_tensor(batch, "scene_vectors").device
        return torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)

    candidate = SimpleNamespace(
        name="fake-ort",
        source_path=tmp_path / "fake.onnx",
        input_device=torch.device("cpu"),
        data_spec=reference.data_spec,
        input_contract=reference.input_contract,
        repetition=reference.repetition,
        vocab_entries=reference.vocab_entries,
        execution_provider="CPUExecutionProvider",
        contract=fake_contract,
        raw_logits=fake_raw_logits,
        configure_cache=lambda enabled: None,
        metrics=lambda: None,
    )
    parity = ParityPolicyBackend(reference, candidate, tolerance=1e-4)
    logits = parity.raw_logits(live_batch, data_spec.candidate_action_keys)

    assert logits.shape == (1, data_spec.num_candidates)
    assert seen_device["device"].type == reference.input_device.type


def test_parity_failure_writes_auditable_partial_report(monkeypatch, tmp_path):
    class FakeBackend:
        def __init__(self, name, logits):
            self.name = name
            self.logits = logits
            self.source_path = tmp_path / f"{name}.model"
            self.input_device = torch.device("cpu")
            self.data_spec = SimpleNamespace(job_tag="black_mage")
            self.input_contract = SimpleNamespace(to_dict=lambda: {"version": 1})
            self.repetition = RepetitionConfig()
            self.vocab_entries = ((100, 1), (200, 2), (300, 3))
            self.execution_provider = name
            self.contract = SimpleNamespace(precision="float32")

        def raw_logits(self, _batch, _candidate_keys):
            return self.logits

        def configure_cache(self, _enabled):
            return None

        @staticmethod
        def metrics():
            return SimpleNamespace(to_dict=lambda: {"calls": 1})

    reference = FakeBackend("pytorch", torch.tensor([[1.0, 2.0, 3.0]]))
    candidate = FakeBackend("onnxruntime", torch.tensor([[4.0, 2.0, 1.0]]))
    candidate.package_dir = tmp_path / "deployment"
    candidate.manifest = SimpleNamespace(payload={"runtime_targets": {}})

    class FakeReplay:
        def __init__(self, _config):
            self.backend = reference
            self.data_spec = reference.data_spec

        def _configure_kv_cache(self, _enabled):
            return None

        def run(self):
            self.backend.raw_logits(
                {"candidate_legal_mask": torch.tensor([[True, True, True]])},
                ("fire_iii", "fire_iv", "blizzard_iii"),
            )
            raise AssertionError("unreachable")

    monkeypatch.setattr(parity_module, "OrtPolicyBackend", lambda *_args, **_kwargs: candidate)
    monkeypatch.setattr(parity_module, "AutoregressiveReplay", FakeReplay)
    monkeypatch.setattr(
        parity_module,
        "parity_artifact_bindings",
        lambda **_kwargs: {
            "manifest": {"filename": "manifest.json", "sha256": "a" * 64},
            "model": {"filename": "model.onnx", "sha256": "b" * 64},
            "checkpoint": {"filename": "checkpoint.pt", "sha256": "c" * 64},
            "deployment_contract_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        parity_module,
        "package_runtime_targets",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        parity_module,
        "record_parity_result",
        lambda **_kwargs: pytest.fail("ad-hoc parity must not mutate release state"),
    )
    config = AutoregressiveReplayConfig(
        checkpoint_path=tmp_path / "checkpoint.pt",
        output_path=tmp_path / "rollout.md",
        scene_json_path=tmp_path / "scene.json",
        cache_dir=tmp_path / "cache",
        model_history_capacity=8,
        cache_shard_size=4,
        cache_max_shards=2,
        max_history=8,
        device="cpu",
        job_tag="black_mage",
        use_kv_cache=False,
    )
    output = tmp_path / "parity.json"

    with pytest.raises(AssertionError, match="audit report written"):
        parity_module.run_rollout_parity(
            config,
            onnx_package_path=tmp_path / "deployment",
            provider="CPUExecutionProvider",
            output_path=output,
        )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["release_gate"] == {"enabled": False, "version": 1}
    assert report["error"]["type"] == "AssertionError"
    assert report["parity"]["passed"] is False
    assert report["parity"]["first_divergence"]["decision_index"] == 0
    assert report["parity"]["decisions"][0]["max_diff_candidate"] == "fire_iii"
    assert report["rollout"]["action_sequence_match"] is False

    recorded = []
    monkeypatch.setattr(
        parity_module,
        "record_parity_result",
        lambda **kwargs: recorded.append(kwargs),
    )
    formal_output = tmp_path / "formal-parity.json"
    with pytest.raises(AssertionError, match="audit report written"):
        parity_module.run_rollout_parity(
            config,
            onnx_package_path=tmp_path / "deployment",
            provider="CPUExecutionProvider",
            output_path=formal_output,
            release_gate=True,
        )
    assert recorded == [
        {
            "package_dir": candidate.package_dir,
            "parity_report_path": formal_output.resolve(),
        }
    ]
    formal_report = json.loads(formal_output.read_text(encoding="utf-8"))
    assert formal_report["release_gate"] == {"enabled": True, "version": 1}

    class EmptyFailReplay(FakeReplay):
        def run(self):
            raise RuntimeError("scene compilation failed before first decision")

    monkeypatch.setattr(parity_module, "AutoregressiveReplay", EmptyFailReplay)
    empty_output = tmp_path / "empty-parity.json"
    with pytest.raises(AssertionError, match="audit report written"):
        parity_module.run_rollout_parity(
            config,
            onnx_package_path=tmp_path / "deployment",
            provider="CPUExecutionProvider",
            output_path=empty_output,
        )

    empty_report = json.loads(empty_output.read_text(encoding="utf-8"))
    assert empty_report["status"] == "failed"
    assert empty_report["parity"]["passed"] is False
    assert empty_report["parity"]["decision_count"] == 0
    assert empty_report["rollout"]["action_sequence_match"] is False


@pytest.mark.parametrize(
    "tolerance",
    (float("nan"), float("inf"), -float("inf"), -1.0),
)
def test_parity_rejects_non_finite_or_negative_tolerance(
    monkeypatch,
    tmp_path,
    tolerance,
):
    candidate = SimpleNamespace(contract=SimpleNamespace(precision="float32"))
    monkeypatch.setattr(
        parity_module,
        "OrtPolicyBackend",
        lambda *_args, **_kwargs: candidate,
    )
    config = AutoregressiveReplayConfig(
        checkpoint_path=tmp_path / "checkpoint.pt",
        output_path=tmp_path / "rollout.md",
        scene_json_path=tmp_path / "scene.json",
        cache_dir=tmp_path / "cache",
        model_history_capacity=8,
        cache_shard_size=4,
        cache_max_shards=2,
    )

    with pytest.raises(ValueError, match="finite and >= 0"):
        parity_module.run_rollout_parity(
            config,
            onnx_package_path=tmp_path / "deployment",
            provider="CPUExecutionProvider",
            output_path=tmp_path / "parity.json",
            tolerance=tolerance,
        )


def test_parity_cli_uses_resolved_env_provider(monkeypatch, tmp_path):
    config = SimpleNamespace(
        output_path=tmp_path / "rollout.md",
        ort_provider="CPUExecutionProvider",
    )
    calls = {}
    monkeypatch.setattr(
        replay_main_module,
        "load_replay_config",
        lambda **_kwargs: config,
    )

    def fake_parity(_config, **kwargs):
        calls.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(replay_main_module, "run_rollout_parity", fake_parity)
    monkeypatch.setattr(
        "sys.argv",
        [
            "autoregressive_replay",
            "--parity-onnx-package",
            "deployment",
        ],
    )

    replay_main_module.main()

    assert calls["provider"] == "CPUExecutionProvider"


def test_parity_cli_forwards_precision_and_tolerance(monkeypatch, tmp_path):
    config = SimpleNamespace(
        output_path=tmp_path / "rollout.md",
        ort_provider="CUDAExecutionProvider",
    )
    config_kwargs = {}
    parity_kwargs = {}

    def fake_load_replay_config(**kwargs):
        config_kwargs.update(kwargs)
        return config

    def fake_parity(_config, **kwargs):
        parity_kwargs.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr(replay_main_module, "load_replay_config", fake_load_replay_config)
    monkeypatch.setattr(replay_main_module, "run_rollout_parity", fake_parity)
    monkeypatch.setattr(
        "sys.argv",
        [
            "autoregressive_replay",
            "--backend",
            "pytorch",
            "--precision",
            "float32",
            "--parity-onnx-package",
            "deployment",
            "--parity-tolerance",
            "0.0002",
        ],
    )

    replay_main_module.main()

    assert config_kwargs["policy_precision"] == "float32"
    assert parity_kwargs["tolerance"] == pytest.approx(2e-4)

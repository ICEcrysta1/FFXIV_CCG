"""PyTorch / ORT parity 双跑测试。"""

from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from common.policy.config import ModelConfig
from common.policy.data import ActionSpace, DataSpec, SkillVocab
from common.policy.model import CausalPolicyModel, RepetitionConfig
from common.policy.model import repetition as repetition_module
from scripts.autoregressive_replay import backends as backends_module
from scripts.autoregressive_replay import main as replay_main_module
from scripts.autoregressive_replay import parity as parity_module
from scripts.autoregressive_replay.backends import (
    ParityPolicyBackend,
    PyTorchPolicyBackend,
    compare_backend_logits,
)
from scripts.autoregressive_replay.config import AutoregressiveReplayConfig
from tests.training._causal_fixtures import make_data_spec


def test_fp16_parity_rounds_real_fp32_softcap_before_host_policy(monkeypatch):
    """真实半精度动作头经 FP32 cap 后，接口舍入会把两个不同分数变为平局。"""
    spec = make_data_spec()
    model = CausalPolicyModel(
        spec,
        ModelConfig(d_model=8, n_layers=1, n_heads=2, num_kv_heads=1,
                    ff_dim=16, dropout=0.0, logit_softcap=15.0),
        vocab_size=3,
    ).half()
    with torch.no_grad():
        model.output_head.weight.zero_()
        model.output_head.weight[:, 0].copy_(torch.tensor([8.9296875, 8.9375]))
    hidden = torch.zeros((1, 8), dtype=torch.float16)
    hidden[:, 0] = 1.0
    # CPU 测试只复用真实动作读出，避免为设备选择与完整主干启动 CUDA。
    reference = PyTorchPolicyBackend.__new__(PyTorchPolicyBackend)
    reference.precision = "float16"
    reference.input_device = torch.device("cpu")
    reference.execution_provider = "PyTorch:cpu"
    reference._bf16_float_compute = False
    reference._fp16_output_quantization = False
    reference.model = lambda _batch: {"logits": model.compute_action_logits(hidden)}
    reference._start_measurement = lambda: 0.0
    reference._finish_measurement = lambda _started: None
    reference.data_spec = spec
    reference.input_contract = SimpleNamespace(to_dict=lambda: {"version": 1})
    reference.vocab_entries = ((100, 1), (200, 2))
    reference.repetition = RepetitionConfig(mode="blacklist", skills=("second",), penalty=.003)
    monkeypatch.setattr(backends_module, "autocast_context", lambda *_args: nullcontext())
    batch = {"action_legal_mask": torch.ones((1, 2), dtype=torch.bool),
             "action_keys": [list(spec.action_keys)], "history_action_keys": [["second"]]}
    ordinary_logits = reference.raw_logits(batch, spec.action_keys)
    expected = 15.0 * torch.tanh(torch.tensor([[8.9296875, 8.9375]]) / 15.0)
    assert ordinary_logits.dtype == torch.float32
    torch.testing.assert_close(ordinary_logits, expected, rtol=0, atol=0)
    rounded = ordinary_logits.half().float()
    assert ordinary_logits.argmax().item() == 1
    assert rounded.argmax().item() == 0
    compared = SimpleNamespace(
        data_spec=spec, input_contract=reference.input_contract,
        repetition=reference.repetition, vocab_entries=reference.vocab_entries,
        source_path="fake.onnx", input_device=torch.device("cpu"),
        execution_provider="CPUExecutionProvider",
        raw_logits=lambda *_args: rounded,
    )
    with pytest.raises(AssertionError, match="reference_top1='second'.*compared_top1='first'"):
        compare_backend_logits(reference, compared, batch, spec.action_keys, tolerance=.005)
    reference.enable_fp16_output_quantization()
    comparison = compare_backend_logits(reference, compared, batch, spec.action_keys, tolerance=0.0)
    assert comparison["top1_match"] and comparison["top3_set_match"]
    assert comparison["max_abs_diff"] == 0.0
    penalty_inputs = []
    apply_penalty = repetition_module.apply_repetition_penalty

    def record_penalty(logits, values, config):
        penalty_inputs.append(logits.detach().clone())
        return apply_penalty(logits, values, config)

    monkeypatch.setattr(repetition_module, "apply_repetition_penalty", record_penalty)
    parity = ParityPolicyBackend(reference, compared, tolerance=0.0)
    torch.testing.assert_close(parity.raw_logits(batch, spec.action_keys), rounded, rtol=0, atol=0)
    assert parity.rows[0]["passed"] and parity.rows[0]["final_selection_match"]
    assert parity.rows[0]["reference_final_action"] == "first"
    assert len(penalty_inputs) == 2  # 双方各在量化后执行一次宿主惩罚。
    for values in penalty_inputs:
        torch.testing.assert_close(values, rounded, rtol=0, atol=0)
    # 量化只属于参考后端，没有改动正式模型的 FP32 softcap。
    torch.testing.assert_close(model.compute_action_logits(hidden), ordinary_logits, rtol=0, atol=0)
    with pytest.raises(ValueError, match="fresh FP16 reference"):
        reference.enable_fp16_output_quantization()


@pytest.mark.parametrize("precision", ("float32", "bf16"))
def test_fp16_output_quantization_rejects_other_reference_precisions(precision):
    reference = PyTorchPolicyBackend.__new__(PyTorchPolicyBackend)
    reference.precision = precision
    reference._fp16_output_quantization = False
    with pytest.raises(ValueError, match="fresh FP16 reference"):
        reference.enable_fp16_output_quantization()
    assert reference._fp16_output_quantization is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bf16_float_compute_reference_keeps_quantized_checkpoint_weights(tmp_path):
    from scripts.autoregressive_replay.backends import PyTorchPolicyBackend
    from scripts.onnx_export.contracts.contract import (
        TENSOR_INPUT_NAMES,
        CapacityContract,
        make_inputs,
    )
    from tests.scripts.onnx_export.test_exporter import _write_small_checkpoint

    checkpoint = tmp_path / "model.pt"
    data_spec, _ = _write_small_checkpoint(checkpoint)
    reference = PyTorchPolicyBackend(
        checkpoint,
        device="cuda",
        use_kv_cache=False,
        precision="bf16",
    )
    original_bf16 = next(reference.model.parameters()).detach().clone()
    reference.enable_bf16_float_compute()
    promoted = next(reference.model.parameters()).detach()
    assert promoted.dtype == torch.float32
    assert torch.equal(promoted, original_bf16.float())

    inputs = make_inputs(
        data_spec,
        CapacityContract(3, 4),
        vocab_size=8,
        scene_valid=1,
        history_valid=1,
        dtype=torch.bfloat16,
        seed=7,
    )
    batch = {
        name: value.cuda()
        for name, value in zip(TENSOR_INPUT_NAMES, inputs, strict=True)
    }
    batch["action_legal_mask"] = torch.ones(
        (1, data_spec.num_actions), dtype=torch.bool, device="cuda"
    )
    logits = reference.raw_logits(batch, data_spec.action_keys)
    assert logits.dtype == torch.float32
    assert torch.equal(logits, logits.bfloat16().float())
    with pytest.raises(ValueError, match="fresh BF16 reference"):
        reference.enable_bf16_float_compute()


def test_backend_parity_failure_reports_compared_logits_and_actions():
    class FakeBackend:
        def __init__(self, logits):
            self.logits = logits

        def raw_logits(self, _batch, _action_keys):
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
    assert "compared_logit=4.00000000" in message
    assert "reference_top1='blizzard_iii'" in message
    assert "compared_top1='fire_iii'" in message


def test_parity_fixed_capacity_batch_stays_on_reference_device(tmp_path):
    """固定容量 batch 必须与 PT reference 同 device，避免 CUDA device mismatch。"""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from scripts.autoregressive_replay.backends import (
        ParityPolicyBackend,
        PyTorchPolicyBackend,
        _require_tensor,
    )
    from scripts.onnx_export import TENSOR_INPUT_NAMES, CapacityContract
    from scripts.onnx_export.contracts.contract import make_inputs, slice_dynamic_inputs
    from scripts.onnx_export.contracts.deployment_contract import TensorSpec
    from tests.scripts.onnx_export.test_exporter import _write_small_checkpoint

    checkpoint = tmp_path / "model.pt"
    data_spec, _input_contract = _write_small_checkpoint(checkpoint)
    reference = PyTorchPolicyBackend(
        checkpoint,
        device="cuda",
        use_kv_cache=False,
    )
    contract = CapacityContract(3, 4)
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
            "action_legal_mask": torch.ones(
                (1, data_spec.num_actions), dtype=torch.bool
            ),
            "action_keys": [list(data_spec.action_keys)],
            "history_action_keys": [["fire_iii", "fire_iv", "fire_iv"]],
        }
    )

    b, s, h, c = 1, 3, 4, data_spec.num_actions
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
            TensorSpec("current_state_vectors", "tensor(float)", (b, sd), ""),
            TensorSpec("current_state_null_mask", "tensor(bool)", (b, sd), ""),
        )

    fake_contract = SimpleNamespace(
        capacity=contract,
        tensor_inputs=tensor_inputs,
        validate_tensor_inputs=lambda _inputs: None,
    )
    seen_device = {}

    def fake_raw_logits(batch, action_keys):
        seen_device["device"] = _require_tensor(batch, "scene_vectors").device
        return torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)

    compared = SimpleNamespace(
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
    parity = ParityPolicyBackend(reference, compared, tolerance=1e-4)
    logits = parity.raw_logits(live_batch, data_spec.action_keys)

    assert logits.shape == (1, data_spec.num_actions)
    assert seen_device["device"].type == reference.input_device.type


@pytest.mark.parametrize(
    ("precision", "compute_precision", "reference_mode"),
    (("float32", "float32", None), ("float16", "float16", "fp16_output"),
     ("bf16", "bf16", None), ("bf16", "float32", "bf16_float_compute")),
)
def test_parity_failure_writes_auditable_partial_report(
    monkeypatch, tmp_path, precision, compute_precision, reference_mode,
):
    class FakeBackend:
        def __init__(self, name, logits):
            self.name = name
            self.logits = logits
            self.source_path = tmp_path / f"{name}.model"
            self.input_device = torch.device("cpu")
            self.data_spec = DataSpec(
                "black_mage", 3, 1, 1, 1, 1, ("fire_iii", "fire_iv", "blizzard_iii"),
                ("kind",), (1, 2, 3), (True, True, True),
            )
            self.input_contract = SimpleNamespace(
                to_dict=lambda: {"version": 1}, create_normalizer=lambda: object(),
                create_skill_vocab=lambda: SkillVocab.from_entries(self.vocab_entries),
            )
            self.repetition = RepetitionConfig()
            self.vocab_entries = ((100, 1), (200, 2), (300, 3))
            self.execution_provider = name
            self.contract = SimpleNamespace(precision=precision)
            self.compute_precision = compute_precision
            self.reference_modes = []

        def enable_fp16_output_quantization(self):
            self.reference_modes.append("fp16_output")

        def enable_bf16_float_compute(self):
            self.reference_modes.append("bf16_float_compute")

        def raw_logits(self, _batch, _action_keys):
            return self.logits

        def configure_cache(self, _enabled):
            return None

        @staticmethod
        def metrics():
            return SimpleNamespace(to_dict=lambda: {"calls": 1})

    reference = FakeBackend("pytorch", torch.tensor([[1.0, 2.0, 3.0]]))
    compared = FakeBackend("onnxruntime", torch.tensor([[4.0, 2.0, 1.0]]))
    compared.package_dir = tmp_path / "deployment"
    compared.manifest = SimpleNamespace(payload={"runtime_targets": {}})

    class FakeReplay:
        def __init__(self, _config, *, session):
            self.backend = session.backend
            self.data_spec = reference.data_spec

        def run(self):
            self.backend.raw_logits(
                {"action_legal_mask": torch.tensor([[True, True, True]])},
                ("fire_iii", "fire_iv", "blizzard_iii"),
            )
            raise AssertionError("unreachable")

    monkeypatch.setattr(parity_module, "OrtPolicyBackend", lambda *_args, **_kwargs: compared)
    monkeypatch.setattr(parity_module, "PyTorchPolicyBackend", lambda *_args, **_kwargs: reference)
    prepared_actions = []
    monkeypatch.setattr(
        parity_module.ReplayCacheStore, "prepare",
        lambda *_args, **kwargs: prepared_actions.append(kwargs["expected_action_space"]),
    )

    closed = []
    class FakeSession:
        def __init__(self, _config, **kwargs):
            self.backend = kwargs["backend"]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            closed.append(self)

    monkeypatch.setattr(parity_module, "AutoregressiveReplaySession", FakeSession)
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
        parity_module.run_rollout_parities(
            [config],
            onnx_package_path=tmp_path / "deployment",
            provider="CPUExecutionProvider",
            output_paths=[output],
        )

    report = json.loads(output.read_text(encoding="utf-8"))
    expected_modes = [] if reference_mode is None else [reference_mode]
    assert reference.reference_modes == expected_modes
    assert prepared_actions == [ActionSpace.from_data_spec(reference.data_spec)]
    assert report["status"] == "failed"
    assert report["release_gate"] == {"enabled": False, "version": 1}
    assert report["error"]["type"] == "AssertionError"
    assert report["parity"]["passed"] is False
    assert report["parity"]["first_divergence"]["decision_index"] == 0
    assert report["parity"]["decisions"][0]["max_diff_action"] == "fire_iii"
    assert report["rollout"]["action_sequence_match"] is False

    recorded = []
    monkeypatch.setattr(
        parity_module,
        "record_parity_result",
        lambda **kwargs: recorded.append(kwargs),
    )
    formal_output = tmp_path / "formal-parity.json"
    with pytest.raises(AssertionError, match="audit report written"):
        parity_module.run_rollout_parities(
            [config],
            onnx_package_path=tmp_path / "deployment",
            provider="CPUExecutionProvider",
            output_paths=[formal_output],
            release_gate=True,
        )
    assert recorded == [
        {
            "package_dir": compared.package_dir,
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
        parity_module.run_rollout_parities(
            [config],
            onnx_package_path=tmp_path / "deployment",
            provider="CPUExecutionProvider",
            output_paths=[empty_output],
        )

    empty_report = json.loads(empty_output.read_text(encoding="utf-8"))
    assert empty_report["status"] == "failed"
    assert empty_report["parity"]["passed"] is False
    assert empty_report["parity"]["decision_count"] == 0
    assert empty_report["rollout"]["action_sequence_match"] is False
    assert len(closed) == 3
    assert reference.reference_modes == expected_modes * 3


@pytest.mark.parametrize(
    "tolerance",
    (float("nan"), float("inf"), -float("inf"), -1.0),
)
def test_parity_rejects_non_finite_or_negative_tolerance(
    monkeypatch,
    tmp_path,
    tolerance,
):
    compared = SimpleNamespace(contract=SimpleNamespace(precision="float32"))
    monkeypatch.setattr(
        parity_module,
        "OrtPolicyBackend",
        lambda *_args, **_kwargs: compared,
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
        parity_module.run_rollout_parities(
            [config],
            onnx_package_path=tmp_path / "deployment",
            provider="CPUExecutionProvider",
            output_paths=[tmp_path / "parity.json"],
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
        return kwargs["output_paths"]

    monkeypatch.setattr(replay_main_module, "run_rollout_parities", fake_parity)
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
        return kwargs["output_paths"]

    monkeypatch.setattr(replay_main_module, "load_replay_config", fake_load_replay_config)
    monkeypatch.setattr(replay_main_module, "run_rollout_parities", fake_parity)
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

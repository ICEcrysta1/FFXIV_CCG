"""ONNX 导出包、固定容量契约和原子发布测试。"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from common.policy.config import ModelConfig
from common.policy.data import DataSpec, ModelInputContract, Normalizer, SkillVocab
from common.policy.data.schema import SceneWindowSchema, TrainingSchema, TRAINING_SAMPLE_SCHEMA_VERSION
from common.policy.data.input_contract import INPUT_CONTRACT_VERSION, TOKEN_ENCODING_CONTRACT
from common.output_context_schema import CANONICAL_CONTEXT_SCHEMA_VERSION
from common.policy.model import CausalPolicyModel
from common.torch_serialization import safe_torch_load
from scripts.autoregressive_replay.backends import (
    OrtPolicyBackend,
    PyTorchPolicyBackend,
    compare_backend_logits,
)
from scripts.onnx_export import TENSOR_INPUT_NAMES, CapacityContract, DeploymentManifest
from scripts.onnx_export import export as export_module
from scripts.onnx_export.contracts.contract import make_inputs, slice_dynamic_inputs
from scripts.onnx_export.contracts import contract as tensor_contract_module
from common.policy.data.context_fields import MODEL_INPUT_FIELDS
from tests.training._causal_fixtures import make_state_groups
from scripts.onnx_export.contracts.deployment_profile import DeploymentProfile
from scripts.onnx_export.contracts.deployment_contract import (
    DEPLOYMENT_CONTRACT_VERSION,
    DEPLOYMENT_MANIFEST_VERSION,
    DeploymentContract,
)
from scripts.onnx_export.export import environment as environment_module
from scripts.onnx_export.export import export_package
from scripts.onnx_export.export import publish as publish_module
from scripts.onnx_export.io.artifact_io import (
    normalize_torch_reports,
    write_deterministic_npz,
)
from scripts.onnx_export.release import release as release_module
from scripts.onnx_export.release.policy import minimum_empty_action_budget
from scripts.onnx_export.release.release import (
    parity_artifact_bindings,
    record_parity_result,
    record_successful_parity,
    verify_release,
)
from scripts.onnx_export.runtime.ort_runtime import (
    ORT_DISABLE_CPU_FALLBACK_KEY,
    create_ort_session,
    ort_session_options,
    ort_session_providers,
    resolve_ort_providers,
)
from scripts.onnx_export.runtime.precision import (
    onnx_torch_dtype,
    parity_max_abs_tolerance,
    precision_onnx_data_type,
    precision_tolerances,
)
from scripts.onnx_export.runtime.runtime_targets import (
    BF16_TARGET_ONNX_VERSION,
    BF16_TARGET_ONNXSCRIPT_VERSION,
    BF16_TARGET_ORT_VERSION,
)
from scripts.onnx_export.runtime.tensor_runtime import (
    GOLDEN_BF16_ENCODING,
    run_ort_tensors,
    tensor_to_golden_array,
)


def test_exception_note_falls_back_to_stderr(capsys):
    class LegacyException:
        pass

    release_module._attach_exception_note(
        LegacyException(),
        "cleanup evidence failed",
    )

    assert "cleanup evidence failed" in capsys.readouterr().err


def test_exception_note_uses_supported_add_note(capsys):
    class ModernException:
        def __init__(self):
            self.notes: list[str] = []

        def add_note(self, note: str) -> None:
            self.notes.append(note)

    error = ModernException()
    release_module._attach_exception_note(error, "cleanup evidence failed")

    assert error.notes == ["cleanup evidence failed"]
    assert capsys.readouterr().err == ""


def test_capacity_contract_rejects_invalid_or_oversized_layout():
    with pytest.raises(ValueError, match="scene_capacity"):
        CapacityContract(0, 1).validate()
    contract = CapacityContract(3, 8)
    contract.validate()
    assert contract.total_token_count == 20
    assert CapacityContract(200, 300).total_token_count == 801
    assert CapacityContract(200, 384).total_token_count == 969
    assert CapacityContract.from_dict(contract.to_dict()) == contract


@pytest.mark.parametrize(("field", "value", "message"), (
    ("history_tokens_per_action", 1, "two tokens per action"),
    ("history_capacity_unit", "tokens", "two tokens per action"),
    ("token_order", "scene, (skill_i, state_i)*H, current_state", "token order"),
    ("total_token_count", 12, "total token count"),
))
def test_capacity_contract_rejects_old_fusion_or_wrong_token_semantics(field, value, message):
    payload = CapacityContract(3, 8).to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        CapacityContract.from_dict(payload)


@pytest.mark.parametrize(("field", "value"), (
    ("history_tokens_per_action", 1),
    ("history_capacity_unit", "tokens"),
    ("token_order", "scene, fused_pair_i, current_state"),
    ("token_order", "scene, (skill_i, state_i)*H, current_state"),
    ("effective_sequence_length", "scene_valid + history_valid + 1"),
    ("unsupported_capacity", 10),
))
def test_manifest_schema_strictly_rejects_fused_capacity_metadata(field, value):
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(export_module.__file__).parents[1] / "manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    capacity_schema = schema["$defs"]["contract"]["properties"]["capacity"]
    validator = jsonschema.Draft202012Validator(capacity_schema)
    payload = CapacityContract(3, 8).to_dict()
    validator.validate(payload)
    payload[field] = value
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(payload)


@pytest.mark.parametrize("previous_version", (18, 19, 20, 21, 22))
def test_manifest_schema_rejects_previous_deployment_versions(previous_version):
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(export_module.__file__).parents[1] / "manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    version_schema = schema["$defs"]["contract"]["properties"]["contract_version"]
    validator = jsonschema.Draft202012Validator(version_schema)
    validator.validate(DEPLOYMENT_CONTRACT_VERSION)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(previous_version)


def test_manifest_schema_and_loader_reject_previous_manifest_version(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(export_module.__file__).parents[1] / "manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema["properties"]["manifest_version"])
    validator.validate(DEPLOYMENT_MANIFEST_VERSION)
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(DEPLOYMENT_MANIFEST_VERSION - 1)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"manifest_version": DEPLOYMENT_MANIFEST_VERSION - 1}), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="unsupported deployment manifest version"):
        DeploymentManifest.load(path, verify_files=False)


def test_manifest_schema_uses_authoritative_model_input_descriptor():
    schema_path = Path(export_module.__file__).parents[1] / "manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    input_schema = schema["$defs"]["contract"]["properties"]["model_input_contract"]
    assert input_schema["properties"]["version"]["const"] == INPUT_CONTRACT_VERSION
    assert input_schema["properties"]["token_encoding"]["const"] == TOKEN_ENCODING_CONTRACT


def test_failed_export_keeps_previous_valid_directory(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "deployment"
    output.mkdir()
    sentinel = output / "manifest.json"
    sentinel.write_text("previous", encoding="utf-8")

    def fail(**_kwargs):
        raise RuntimeError("validation failed")

    monkeypatch.setattr("scripts.onnx_export.export.api._build_package", fail)
    with pytest.raises(RuntimeError, match="validation failed"):
        export_package(
            checkpoint_path=checkpoint,
            output_dir=output,
            opset=18,
            precision="float32",
            ort_provider="CPUExecutionProvider",
            validation_devices=("cpu",),
            overwrite=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob(".deployment.export-*"))


def test_publish_directory_retries_transient_windows_access_denied(monkeypatch):
    attempts = {"count": 0}

    class FlakyPath:
        def rename(self, _destination):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise PermissionError(5, "access denied")

    monkeypatch.setattr(publish_module.time, "sleep", lambda _seconds: None)
    publish_module.rename_with_retry(
        FlakyPath(),
        object(),
        operation="test rename",
    )

    assert attempts["count"] == 3


def test_golden_npz_is_byte_reproducible(tmp_path):
    arrays = {"b": np.array([2.0]), "a": np.array([1], dtype=np.int64)}
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    write_deterministic_npz(first, arrays)
    write_deterministic_npz(second, arrays)

    assert first.read_bytes() == second.read_bytes()


def test_bf16_golden_uses_exact_uint16_bit_encoding():
    source = torch.tensor([1.0, -2.5, 0.0], dtype=torch.bfloat16)

    encoded = tensor_to_golden_array(source)
    restored = torch.from_numpy(encoded.copy()).view(torch.bfloat16)

    assert encoded.dtype == np.uint16
    assert torch.equal(restored, source)
    assert onnx_torch_dtype("tensor(bfloat16)") == torch.bfloat16


def test_precision_specific_parity_tolerances_are_stable():
    assert parity_max_abs_tolerance("float32") == 1e-5
    assert parity_max_abs_tolerance("float16") == 5e-3
    assert parity_max_abs_tolerance("bf16") == 2.5e-1


def test_precision_uses_onnx_tensor_proto_constants():
    onnx = pytest.importorskip("onnx")

    assert precision_onnx_data_type("float32", onnx.TensorProto) == (
        onnx.TensorProto.FLOAT
    )
    assert precision_onnx_data_type("float16", onnx.TensorProto) == (
        onnx.TensorProto.FLOAT16
    )
    assert precision_onnx_data_type("bf16", onnx.TensorProto) == (
        onnx.TensorProto.BFLOAT16
    )


def test_zero_padding_fill_clears_all_padding_dtypes():
    data_spec = SimpleNamespace(
        scene_dim=2,
        num_scene_types=4,
        skill_feature_dim=3,
        base_state_dim=5,
        state_dim=9,
        num_actions=2,
    )
    contract = CapacityContract(3, 4)
    values = make_inputs(
        data_spec,
        contract,
        vocab_size=8,
        scene_valid=1,
        history_valid=2,
        dtype=torch.float32,
        seed=17,
        padding_fill="zero",
    )

    assert torch.count_nonzero(values[0][:, 1:]) == 0
    assert torch.count_nonzero(values[1][:, 1:]) == 0
    assert torch.count_nonzero(values[3][:, 2:]) == 0
    assert torch.count_nonzero(values[4][:, 2:]) == 0
    assert torch.count_nonzero(values[5][:, 2:]) == 0
    assert values[6][:, 2:].all()
    assert not values[10][:, 2:].any()
    assert torch.count_nonzero(values[0][:, :1]) > 0
    assert torch.count_nonzero(values[3][:, :2]) > 0


def test_dynamic_slice_and_padding_follow_fields_when_input_order_changes(monkeypatch):
    """输入次序变化时，历史 reset 等序列字段仍按各自轴裁剪和补位。"""
    spec = SimpleNamespace(scene_dim=2, num_scene_types=4, skill_feature_dim=3, base_state_dim=5, state_dim=9, num_actions=2)
    inputs = make_inputs(spec, CapacityContract(3, 4), vocab_size=8,
                         scene_valid=1, history_valid=2, dtype=torch.float32, seed=17)
    fields = tuple(reversed(MODEL_INPUT_FIELDS))
    monkeypatch.setattr(tensor_contract_module, "MODEL_INPUT_FIELDS", fields)
    inputs = tuple(reversed(inputs))
    sliced = dict(zip((field.name for field in fields), slice_dynamic_inputs(
        inputs, scene_valid=1, history_valid=2,
    ), strict=True))
    assert sliced["scene_vectors"].shape == (1, 1, 2)
    assert sliced["history_state_vectors"].shape == (1, 2, 9)
    assert sliced["history_state_reset_mask"].shape == (1, 2, 5)
    assert sliced["current_state_vectors"].shape == (1, 9)
    padded = dict(zip((field.name for field in fields), tensor_contract_module.fill_padding_values(
        inputs, scene_valid=1, history_valid=2, value=7.0,
    ), strict=True))
    assert (padded["scene_vectors"][:, 1:] == 7).all()
    assert (padded["history_state_vectors"][:, 2:] == 7).all()
    assert padded["history_state_null_mask"][:, 2:].all()
    assert not padded["history_state_reset_mask"][:, 2:].any()
    assert not padded["history_mask"][:, 2:].any()
    assert not padded["scene_mask"][:, 1:].any()


def test_default_deployment_profile_rejects_unsafe_job_tag():
    with pytest.raises(ValueError, match="unsafe"):
        DeploymentProfile.default_path("../black_mage")
    with pytest.raises(ValueError, match="unsafe"):
        DeploymentProfile.default_path(r"..\black_mage")


def test_default_deployment_profile_path_targets_profile_directory():
    path = DeploymentProfile.default_path("black_mage")

    assert path == (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "onnx_export"
        / "profiles"
        / "black_mage.json"
    )
    assert path.is_file()


def test_multiple_torch_reports_are_combined(tmp_path):
    (tmp_path / "onnx_export_a.md").write_text("first", encoding="utf-8")
    (tmp_path / "onnx_export_b.md").write_text("second", encoding="utf-8")

    normalize_torch_reports(tmp_path)

    report = (tmp_path / "torch_export_report.md").read_text(encoding="utf-8")
    assert "onnx_export_a.md" in report
    assert "first" in report
    assert "onnx_export_b.md" in report
    assert "second" in report
    assert not list(tmp_path.glob("onnx_export_*.md"))


def test_float16_export_rejects_cpu_ort_provider(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-read")

    with pytest.raises(ValueError, match="float16.*non-CPU"):
        export_package(
            checkpoint_path=checkpoint,
            output_dir=tmp_path / "deployment",
            opset=18,
            precision="float16",
            ort_provider="CPUExecutionProvider",
            validation_devices=("cpu",),
            overwrite=False,
        )


def test_bf16_export_rejects_fallback_provider(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-read")

    with pytest.raises(ValueError, match="bf16.*explicit CUDAExecutionProvider"):
        export_package(
            checkpoint_path=checkpoint,
            output_dir=tmp_path / "deployment",
            opset=18,
            precision="bf16",
            ort_provider="auto",
            validation_devices=("cuda",),
            overwrite=False,
        )


def test_direct_bf16_export_rejects_unpinned_exporter_versions(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-read")
    monkeypatch.setattr(
        environment_module,
        "import_onnx_dependencies",
        lambda: (
            SimpleNamespace(__version__=BF16_TARGET_ONNX_VERSION),
            SimpleNamespace(__version__=BF16_TARGET_ORT_VERSION),
            SimpleNamespace(__version__="0.8.0"),
        ),
    )

    with pytest.raises(RuntimeError, match="onnxscript=0.8.0"):
        export_package(
            checkpoint_path=checkpoint,
            output_dir=tmp_path / "deployment",
            opset=18,
            precision="bf16",
            ort_provider="CUDAExecutionProvider",
            validation_devices=("cuda",),
            overwrite=False,
        )

    assert not (tmp_path / "deployment").exists()
    assert not list(tmp_path.glob(".deployment.export-*"))


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        (
            ("CUDAExecutionProvider", "CPUExecutionProvider"),
            ("CUDAExecutionProvider", "CPUExecutionProvider"),
        ),
        (("CPUExecutionProvider",), ("CPUExecutionProvider",)),
    ],
)
def test_ort_auto_provider_prefers_cuda_and_falls_back_to_cpu(available, expected):
    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return available

    assert resolve_ort_providers(FakeOrt, "auto") == expected


def test_explicit_ort_provider_must_be_available():
    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ("CPUExecutionProvider",)

    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_ort_providers(FakeOrt, "CUDAExecutionProvider")


def test_explicit_ort_provider_has_no_silent_fallback():
    class FakeOrt:
        @staticmethod
        def get_available_providers():
            return ("CUDAExecutionProvider", "CPUExecutionProvider")

    assert resolve_ort_providers(FakeOrt, "CUDAExecutionProvider") == (
        "CUDAExecutionProvider",
    )


def test_cuda_session_provider_disables_tf32_for_fp32_parity():
    assert ort_session_providers(
        ("CUDAExecutionProvider", "CPUExecutionProvider")
    ) == [
        ("CUDAExecutionProvider", {"use_tf32": "0"}),
        "CPUExecutionProvider",
    ]


def test_explicit_provider_disables_ort_cpu_fallback():
    class FakeSessionOptions:
        def __init__(self):
            self.entries = {}

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    class FakeOrt:
        SessionOptions = FakeSessionOptions

    explicit = ort_session_options(FakeOrt, "CUDAExecutionProvider")
    cpu = ort_session_options(FakeOrt, "CPUExecutionProvider")
    automatic = ort_session_options(FakeOrt, "auto")

    assert explicit.entries == {ORT_DISABLE_CPU_FALLBACK_KEY: "1"}
    assert cpu.entries == {}
    assert automatic.entries == {}


def test_ort_session_failure_reports_dependency_reinstall(tmp_path):
    class FakeSessionOptions:
        def add_session_config_entry(self, _key, _value):
            return None

    class FakeOrt:
        __version__ = "99.0"
        SessionOptions = FakeSessionOptions

        @staticmethod
        def get_available_providers():
            return ("CUDAExecutionProvider",)

        @staticmethod
        def InferenceSession(*_args, **_kwargs):
            raise OSError("missing CUDA DLL")

    with pytest.raises(RuntimeError, match="requirements-onnx-gpu.txt"):
        create_ort_session(
            FakeOrt,
            tmp_path / "model.onnx",
            "CUDAExecutionProvider",
        )


@pytest.mark.parametrize("activation", ("gelu", "swiglu"))
def test_small_model_exports_checker_and_ort_validated_package(tmp_path, activation):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    checkpoint = tmp_path / "checkpoint.pt"
    data_spec, _input_contract = _write_small_checkpoint(
        checkpoint,
        activation=activation,
        qk_norm_scale=1.7,
        logit_softcap=7.5,
    )
    profile = tmp_path / "deployment-profile.json"
    _write_small_profile(profile)
    output = tmp_path / "deployment"

    export_package(
        checkpoint_path=checkpoint,
        output_dir=output,
        deployment_profile_path=profile,
        opset=18,
        precision="float32",
        ort_provider="CPUExecutionProvider",
        validation_devices=("cpu",),
        overwrite=False,
    )

    assert {
        "export_report.json",
        "capacity_report.json",
        "golden_inputs.npz",
        "golden_outputs.npz",
        "manifest.json",
        "manifest.schema.json",
        "model.onnx",
        "torch_export_report.md",
    } <= {path.name for path in output.iterdir()}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == DEPLOYMENT_MANIFEST_VERSION
    assert manifest["contract"]["contract_version"] == DEPLOYMENT_CONTRACT_VERSION
    assert manifest["contract"]["model_input_contract"] == json.loads(json.dumps(_input_contract.to_dict()))
    assert manifest["contract"]["capacity"]["padding_direction"] == "right"
    assert manifest["contract"]["capacity"]["history_capacity"] == 4
    assert manifest["contract"]["capacity"]["history_capacity_unit"] == "actions"
    assert manifest["contract"]["capacity"]["history_tokens_per_action"] == 2
    assert manifest["contract"]["capacity"]["total_token_count"] == 12
    assert manifest["contract"]["capacity"]["token_order"] == "scene, (state_i, skill_i)*H, current_state"
    provenance = manifest["contract"]["capacity_provenance"]
    assert provenance["history_capacity_source"] == (
        "checkpoint.model_config.history_capacity"
    )
    assert provenance["scene_capacity_source"] == "checkpoint.model_config.scene_capacity"
    assert manifest["contract"]["model_config"]["d_model"] == 16
    assert manifest["contract"]["model_config"]["transformer_activation"] == activation
    assert manifest["contract"]["model_config"]["qk_norm_scale"] == 1.7
    assert manifest["contract"]["model_config"]["logit_softcap"] == 7.5
    assert manifest["contract"]["model_input_contract"]["token_encoding"]["output_projection"] == (
        TOKEN_ENCODING_CONTRACT["output_projection"]
    )
    assert manifest["contract"]["model_input_contract"]["token_encoding"]["logit_softcap"] == (
        TOKEN_ENCODING_CONTRACT["logit_softcap"]
    )
    residual = manifest["contract"]["residual_composition"]
    assert residual["type"] == "learned_residual_mix"
    assert residual["initialization"] == {
        "method": "linspace", "r": {"start": 1.15, "end": 1.05},
        "a": {"start": .20, "end": .05},
    }
    assert manifest["model"]["external_data"] is False
    # 导出常量折叠可能合并初始化值，因此仍以保留绝大多数模型权重作为验收。
    assert manifest["model"]["retained_float_initializer_ratio"] >= 0.98
    assert manifest["model"]["onnx_other_float_initializer_max_elements"] <= 1
    exported_model = onnx.load(output / "model.onnx")
    assert any(node.op_type == "Tanh" for node in exported_model.graph.node)
    assert not any(node.op_type == "LayerNormalization" for node in exported_model.graph.node)
    assert not any(".norm" in tensor.name for tensor in exported_model.graph.initializer)
    assert [
        item["name"] for item in manifest["contract"]["tensor_outputs"]
    ] == ["raw_logits"]
    loaded = DeploymentManifest.load(output / "manifest.json")
    assert loaded.contract.data_spec == data_spec

    # 机制、初始化元数据及版本均由 checkpoint 权威配置决定。
    for missing_field in ("residual_mix_r_start", "residual_mix_r_end", "residual_mix_a_start", "residual_mix_a_end"):
        incomplete = deepcopy(manifest["contract"])
        incomplete["model_config"].pop(missing_field)
        with pytest.raises(ValueError, match="missing residual mix initialization fields"):
            DeploymentContract.from_dict(incomplete)
    for field in ("formula", "x0", "attention_and_skip_input", "initialization"):
        forged = deepcopy(manifest["contract"])
        forged["residual_composition"][field] = "legacy"
        with pytest.raises(ValueError, match="authoritative layouts"):
            DeploymentContract.from_dict(forged)

    # 同宽旧普通残差部署图也必须拒绝。
    previous_layernorm_contract = deepcopy(manifest["contract"])
    previous_layernorm_contract["contract_version"] = DEPLOYMENT_CONTRACT_VERSION - 1
    with pytest.raises(ValueError, match="unsupported deployment contract version"):
        DeploymentContract.from_dict(previous_layernorm_contract)

    fixed_inputs = make_inputs(
        data_spec,
        loaded.contract.capacity,
        vocab_size=8,
        scene_valid=2,
        history_valid=3,
        dtype=torch.float32,
        seed=73,
    )
    dynamic_inputs = slice_dynamic_inputs(
        fixed_inputs,
        scene_valid=2,
        history_valid=3,
    )
    live_batch = dict(zip(TENSOR_INPUT_NAMES, dynamic_inputs, strict=True))
    live_batch.update(
        {
            "action_legal_mask": torch.tensor([[True, False, True]]),
            "action_keys": [list(data_spec.action_keys)],
            "history_action_keys": [["fire_iii", "fire_iv", "fire_iv"]],
        }
    )
    pytorch_backend = PyTorchPolicyBackend(
        checkpoint,
        device="cpu",
        use_kv_cache=False,
    )
    ort_backend = OrtPolicyBackend(output, provider="CPUExecutionProvider")
    parity = compare_backend_logits(
        pytorch_backend,
        ort_backend,
        live_batch,
        data_spec.action_keys,
    )
    assert parity["max_abs_diff"] <= 1e-4
    assert parity["top1_match"] is True
    assert parity["top3_set_match"] is True
    assert parity["reference_top3"] == parity["compared_top3"]

    # 已学习的 scalar 影响真实 logits；导出 parity 不能仅覆盖初始化 ramp。
    mix = pytorch_backend.model.encoder.residual_mix
    saved_r, saved_a = mix.r.detach().clone(), mix.a.detach().clone()
    with torch.no_grad():
        learned_logits = pytorch_backend.model(live_batch)["logits"]
        mix.r.copy_(torch.linspace(1.15, 1.05, 2))
        mix.a.copy_(torch.linspace(.20, .05, 2))
        initialized_logits = pytorch_backend.model(live_batch)["logits"]
        mix.r.copy_(saved_r)
        mix.a.copy_(saved_a)
    assert (learned_logits - initialized_logits).abs().max().item() > 1e-4

    tampered = deepcopy(manifest)
    tampered["contract"]["data_spec"]["action_keys"][0:2] = reversed(
        tampered["contract"]["data_spec"]["action_keys"][0:2]
    )
    tampered_path = output / "tampered-manifest.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch|differ"):
        DeploymentManifest.load(tampered_path, verify_files=False)
    for previous_manifest_version in (1, 11):
        legacy = deepcopy(manifest)
        legacy["manifest_version"] = previous_manifest_version
        legacy_path = output / "legacy-manifest.json"
        legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
        with pytest.raises(ValueError, match="re-export"):
            DeploymentManifest.load(legacy_path, verify_files=False)
    malformed = deepcopy(manifest)
    malformed["manifest_version"] = {"unexpected": 2}
    malformed_path = output / "malformed-manifest.json"
    malformed_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_version must be an integer"):
        DeploymentManifest.load(malformed_path, verify_files=False)
    export_report_path = output / "export_report.json"
    export_report_before = export_report_path.read_bytes()
    report = json.loads(export_report_path.read_text(encoding="utf-8"))
    assert report["status"] == "graph_validated"
    assert report["release_gate"] == "requires_rollout_parity"
    assert report["onnx_checker"] == "passed"
    assert report["shape_inference"] == "passed"
    assert report["ort_session"] == "passed"
    assert report["ort_cpu_fallback_disabled"] is False

    bindings = parity_artifact_bindings(
        manifest=loaded,
        manifest_path=output / "manifest.json",
        checkpoint_path=checkpoint,
    )

    def write_parity(
        name,
        *,
        scene_mode,
        decision_count,
        output_gcds,
        status="passed",
    ):
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "status": status,
                    "precision": "float32",
                    "tolerance": parity_max_abs_tolerance("float32"),
                    "release_gate": {
                        "enabled": True,
                        "version": 1,
                    },
                    "scene_mode": scene_mode,
                    "request": {
                        "max_steps": (
                            100
                            if scene_mode == "cache"
                            else minimum_empty_action_budget(128)
                        ),
                        "max_gcds": 128 if scene_mode == "empty" else None,
                    },
                    "artifacts": bindings,
                    "runtime_targets": report["runtime_targets"],
                    "rollout": {
                        "output_gcds": output_gcds,
                        "action_sequence_match": True,
                    },
                    "parity": {
                        "decision_count": decision_count,
                        "max_abs_diff": 0.0,
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    failed_path = write_parity(
        "failed-empty.json",
        scene_mode="empty",
        decision_count=140,
        output_gcds=128,
        status="failed",
    )
    failed_payload = json.loads(failed_path.read_text(encoding="utf-8"))
    failed_payload["artifacts"]["model"]["sha256"] = "0" * 64
    failed_path.write_text(json.dumps(failed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact bindings differ"):
        record_parity_result(
            package_dir=output,
            parity_report_path=failed_path,
        )
    failed_payload["artifacts"] = bindings
    failed_path.write_text(json.dumps(failed_payload), encoding="utf-8")
    ad_hoc_payload = deepcopy(failed_payload)
    ad_hoc_payload["release_gate"]["enabled"] = False
    ad_hoc_path = tmp_path / "ad-hoc-empty.json"
    ad_hoc_path.write_text(json.dumps(ad_hoc_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="formal release gate"):
        record_parity_result(
            package_dir=output,
            parity_report_path=ad_hoc_path,
        )
    loose_payload = deepcopy(failed_payload)
    loose_payload["tolerance"] = 1.0
    loose_path = tmp_path / "loose-empty.json"
    loose_path.write_text(json.dumps(loose_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest precision policy"):
        record_parity_result(
            package_dir=output,
            parity_report_path=loose_path,
        )
    scene_first_path = record_successful_parity(
        package_dir=output,
        parity_report_path=write_parity(
            "scene-first.json",
            scene_mode="cache",
            decision_count=100,
            output_gcds=80,
        ),
    )
    scene_first = json.loads(scene_first_path.read_text(encoding="utf-8"))
    assert scene_first["status"] == "parity_in_progress"
    assert scene_first["requirements"] == {
        "empty_scene_128_gcd": False,
        "scene_100_decisions": True,
    }
    failed_release_path = record_parity_result(
        package_dir=output,
        parity_report_path=failed_path,
    )
    failed_release = json.loads(
        failed_release_path.read_text(encoding="utf-8")
    )
    assert failed_release["status"] == "parity_failed"
    assert failed_release["release_report_version"] == 3
    assert failed_release["release_gate_version"] == 1
    assert failed_release["parity_reports"][0]["precision"] == "float32"
    assert failed_release["parity_reports"][0]["tolerance"] == pytest.approx(1e-5)

    release_path = record_successful_parity(
        package_dir=output,
        parity_report_path=write_parity(
            "empty.json",
            scene_mode="empty",
            decision_count=140,
            output_gcds=128,
        ),
    )
    released = json.loads(release_path.read_text(encoding="utf-8"))
    assert released["status"] == "release_validated"
    assert released["requirements"] == {
        "empty_scene_128_gcd": True,
        "scene_100_decisions": True,
    }
    assert all(
        evidence["sha256"] in evidence["filename"]
        for evidence in released["parity_reports"]
    )
    assert any(
        evidence["status"] == "failed"
        for evidence in released["superseded_parity_reports"]
    )
    verify_release(output)
    assert export_report_path.read_bytes() == export_report_before
    assert report["status"] == "graph_validated"
    assert report["release_gate"] == "requires_rollout_parity"
    release_before = (output / "release_report.json").read_bytes()
    v2_release = deepcopy(released)
    v2_release["release_report_version"] = 2
    (output / "release_report.json").write_text(
        json.dumps(v2_release),
        encoding="utf-8",
    )
    verified_v2 = verify_release(output, audit_history=True)
    assert verified_v2["release_report_version"] == 2
    assert verified_v2["status"] == "release_validated"
    assert verified_v2["superseded_parity_reports"] == released[
        "superseded_parity_reports"
    ]
    assert export_report_path.read_bytes() == export_report_before

    v2_failed_release = deepcopy(failed_release)
    v2_failed_release["release_report_version"] = 2
    v2_failed_superseded = next(
        evidence
        for evidence in released["parity_reports"]
        if evidence["scenario"] == "empty_128_gcd"
    )
    v2_failed_release["superseded_parity_reports"] = [
        v2_failed_superseded
    ]
    (output / "release_report.json").write_text(
        json.dumps(v2_failed_release),
        encoding="utf-8",
    )
    verified_failed_v2 = verify_release(
        output,
        require_validated=False,
        audit_history=True,
    )
    assert verified_failed_v2["release_report_version"] == 2
    assert verified_failed_v2["status"] == "parity_failed"
    assert verified_failed_v2["superseded_parity_reports"] == [
        v2_failed_superseded
    ]
    with pytest.raises(ValueError, match="has not passed"):
        verify_release(output)
    (output / "release_report.json").write_bytes(release_before)

    superseded = released["superseded_parity_reports"][0]
    superseded_path = output / superseded["filename"]
    superseded_before = superseded_path.read_bytes()
    superseded_path.write_bytes(b"tampered superseded evidence")
    runtime_release = verify_release(output)
    assert "superseded_parity_reports" not in runtime_release
    with pytest.raises(ValueError, match="superseded parity evidence hash mismatch"):
        verify_release(output, audit_history=True)
    superseded_path.write_bytes(superseded_before)

    active_evidence_before = {
        evidence["filename"]: (output / evidence["filename"]).read_bytes()
        for evidence in released["parity_reports"]
    }
    evidence_filenames_before = {
        path.name for path in output.glob("parity_*.json")
    }
    original_atomic_write = release_module.write_json_atomic

    def interrupt_release_commit(path, payload):
        if Path(path).name == "release_report.json":
            raise OSError("simulated release commit interruption")
        original_atomic_write(path, payload)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(release_module, "write_json_atomic", interrupt_release_commit)
        with pytest.raises(OSError, match="simulated release commit interruption"):
            record_successful_parity(
                package_dir=output,
                parity_report_path=write_parity(
                    "interrupted-empty.json",
                    scene_mode="empty",
                    decision_count=141,
                    output_gcds=128,
                ),
            )
        with pytest.raises(OSError, match="simulated release commit interruption"):
            record_successful_parity(
                package_dir=output,
                parity_report_path=write_parity(
                    "interrupted-existing-empty.json",
                    scene_mode="empty",
                    decision_count=140,
                    output_gcds=128,
                ),
            )
    assert (output / "release_report.json").read_bytes() == release_before
    assert {
        filename: (output / filename).read_bytes()
        for filename in active_evidence_before
    } == active_evidence_before
    assert {
        path.name for path in output.glob("parity_*.json")
    } == evidence_filenames_before
    verify_release(output)

    active_scene = next(
        evidence
        for evidence in released["parity_reports"]
        if evidence["scenario"] == "scene_100_decisions"
    )
    scene_path = output / active_scene["filename"]
    scene_before = scene_path.read_bytes()
    record_parity_result(
        package_dir=output,
        parity_report_path=write_parity(
            "failed-scene-after-release.json",
            scene_mode="cache",
            decision_count=100,
            output_gcds=80,
            status="failed",
        ),
    )
    assert (output / "release_report.json").read_bytes() == release_before
    assert scene_path.read_bytes() == scene_before
    legacy_release = json.loads(release_before)
    legacy_release["release_report_version"] = 1
    legacy_release.pop("release_gate_version")
    (output / "release_report.json").write_text(
        json.dumps(legacy_release),
        encoding="utf-8",
    )
    audited_legacy = verify_release(output, require_validated=False)
    assert audited_legacy["legacy_status"] == "release_validated"
    assert audited_legacy["status"] == "parity_in_progress"
    assert audited_legacy["requirements"] == {
        "empty_scene_128_gcd": False,
        "scene_100_decisions": False,
    }
    assert audited_legacy["parity_reports"] == []
    assert {
        evidence["filename"] for evidence in audited_legacy["legacy_parity_reports"]
    } >= {
        evidence["filename"] for evidence in legacy_release["parity_reports"]
    }
    with pytest.raises(ValueError, match="legacy deployment release evidence"):
        verify_release(output)

    valid_legacy_release = deepcopy(legacy_release)
    legacy_release["parity_reports"][0]["filename"] = "missing-parity.json"
    (output / "release_report.json").write_text(
        json.dumps(legacy_release),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="legacy deployment release evidence"):
        verify_release(output)
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        verify_release(output, require_validated=False)
    (output / "release_report.json").write_text(
        json.dumps(valid_legacy_release),
        encoding="utf-8",
    )
    migrated_path = record_successful_parity(
        package_dir=output,
        parity_report_path=write_parity(
            "migrated-empty.json",
            scene_mode="empty",
            decision_count=140,
            output_gcds=128,
        ),
    )
    migrated = json.loads(migrated_path.read_text(encoding="utf-8"))
    assert migrated["release_report_version"] == 3
    assert migrated["status"] == "parity_in_progress"
    assert len(migrated["parity_reports"]) == 1
    assert {
        evidence["filename"] for evidence in migrated["superseded_parity_reports"]
    } >= {
        evidence["filename"]
        for evidence in valid_legacy_release["parity_reports"]
        if evidence["scenario"] != "empty_128_gcd"
    }
    verify_release(output, require_validated=False)
    (output / "release_report.json").write_bytes(release_before)

    tampered_release = json.loads(release_before)
    tampered_release["parity_reports"][0]["tolerance"] = 1.0
    (output / "release_report.json").write_text(
        json.dumps(tampered_release),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requirements differ"):
        verify_release(output)
    (output / "release_report.json").write_bytes(release_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("activation", ("gelu", "swiglu"))
@pytest.mark.parametrize("logit_softcap", (torch.finfo(torch.float16).tiny, 7.5, torch.finfo(torch.float16).max))
def test_small_float16_model_exports_and_runs_on_strict_cuda(tmp_path, activation, logit_softcap):
    """FP32 softcap 后返回 FP16，真实图、manifest 与严格 CUDA ORT 必须一致。"""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("ORT CUDAExecutionProvider is unavailable")

    checkpoint = tmp_path / "checkpoint.pt"
    data_spec, input_contract = _write_small_checkpoint(
        checkpoint, activation=activation, logit_softcap=logit_softcap,
    )
    profile = tmp_path / "deployment-profile.json"
    _write_small_profile(profile)
    output = export_package(
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "deployment",
        deployment_profile_path=profile,
        opset=18,
        precision="float16",
        ort_provider="CUDAExecutionProvider",
        validation_devices=("cuda",),
        overwrite=False,
    )

    loaded = DeploymentManifest.load(output / "manifest.json", verify_files=True)
    manifest = loaded.payload
    assert manifest["contract"]["precision"] == "float16"
    assert manifest["contract"]["model_input_contract"] == json.loads(json.dumps(input_contract.to_dict()))
    assert manifest["contract"]["model_config"]["logit_softcap"] == logit_softcap
    assert manifest["contract"]["tensor_outputs"][0]["dtype"] == "tensor(float16)"
    assert manifest["model"]["compute_precision"] == "float16"

    exported_model = onnx.load(output / "model.onnx")
    assert exported_model.graph.output[0].name == "raw_logits"
    assert exported_model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    # cap 仍在 FP32 运算；输出的最终舍入不能把 Tanh 本身降成 FP16。
    inferred = onnx.shape_inference.infer_shapes(exported_model)
    value_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
    }
    tanh_nodes = [node for node in inferred.graph.node if node.op_type == "Tanh"]
    assert tanh_nodes
    for node in tanh_nodes:
        assert value_types[node.input[0]] == value_types[node.output[0]] == onnx.TensorProto.FLOAT

    report = json.loads((output / "export_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "graph_validated"
    assert report["ort_provider"] == "CUDAExecutionProvider"
    assert report["ort_cpu_fallback_disabled"] is True
    assert report["ort_padding_matrix"]
    assert all(row["argmax_match"] for row in report["ort_padding_matrix"])
    assert max(row["max_logit_abs_diff"] for row in report["ort_padding_matrix"]) <= parity_max_abs_tolerance("float16")

    session, _, active = create_ort_session(ort, output / "model.onnx", "CUDAExecutionProvider")
    assert active[0] == "CUDAExecutionProvider"
    assert session.get_outputs()[0].type == "tensor(float16)"
    policy, _, vocab_size, dtype, saved = export_module.load_policy(checkpoint, precision="float16")
    assert dtype == torch.float16
    torch.testing.assert_close(
        policy.model.output_head.weight,
        saved["model_state_dict"]["output_head.weight"].half(),
        rtol=0,
        atol=0,
    )
    inputs = make_inputs(
        data_spec, loaded.contract.capacity,
        vocab_size=vocab_size, scene_valid=2, history_valid=3,
        dtype=dtype, seed=73,
    )
    policy.to(device="cuda")
    with torch.no_grad():
        expected = policy(*(tensor.cuda() for tensor in inputs))
    actual = run_ort_tensors(session, dict(zip(TENSOR_INPUT_NAMES, inputs, strict=True)))[0]
    assert actual.dtype == expected.dtype == torch.float16
    assert torch.isfinite(actual).all()
    rtol, atol = precision_tolerances(torch.float16)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("activation", ("gelu", "swiglu"))
def test_small_bf16_model_exports_and_runs_on_strict_cuda(tmp_path, activation):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("ORT CUDAExecutionProvider is unavailable")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")

    checkpoint = tmp_path / "checkpoint.pt"
    _, input_contract = _write_small_checkpoint(checkpoint, activation=activation)
    profile = tmp_path / "deployment-profile.json"
    _write_small_profile(profile)
    output = export_package(
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "deployment",
        deployment_profile_path=profile,
        opset=18,
        precision="bf16",
        ort_provider="CUDAExecutionProvider",
        validation_devices=("cuda",),
        overwrite=False,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == DEPLOYMENT_MANIFEST_VERSION
    assert manifest["contract"]["contract_version"] == DEPLOYMENT_CONTRACT_VERSION
    assert manifest["contract"]["model_input_contract"] == json.loads(json.dumps(input_contract.to_dict()))
    assert manifest["contract"]["precision"] == "bf16"
    assert manifest["contract"]["model_config"]["logit_softcap"] == 15.0
    assert manifest["contract"]["residual_composition"]["type"] == "learned_residual_mix"
    assert manifest["model"]["compute_precision"] == "float32"
    assert manifest["exporter"]["onnxscript"] == BF16_TARGET_ONNXSCRIPT_VERSION
    assert manifest["contract"]["tensor_outputs"][0]["dtype"] == (
        "tensor(bfloat16)"
    )
    assert manifest["golden"]["float_encoding"] == GOLDEN_BF16_ENCODING
    assert manifest["model"]["onnx_other_float_initializer_max_elements"] <= 1
    assert manifest["model"]["onnx_float_initializer_dtype_counts"]["bfloat16"] > 0
    exported_model = onnx.load(output / "model.onnx")
    assert any(node.op_type == "Tanh" for node in exported_model.graph.node)
    assert not any(node.op_type == "LayerNormalization" for node in exported_model.graph.node)
    assert not any(".norm" in tensor.name for tensor in exported_model.graph.initializer)
    model_metadata = {item.key: item.value for item in exported_model.metadata_props}
    assert model_metadata["ffxiv.precision"] == "bf16"
    assert model_metadata["ffxiv.compute_precision"] == "float32"
    assert model_metadata["ffxiv.history_tokens_per_action"] == "2"
    assert model_metadata["ffxiv.total_token_count"] == "12"
    report = json.loads((output / "export_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "graph_validated"
    assert report["release_gate"] == "requires_rollout_parity"
    assert report["ort_provider"] == "CUDAExecutionProvider"
    assert report["ort_cpu_fallback_disabled"] is True
    assert report["runtime_targets"] == {
        "execution_provider": "CUDAExecutionProvider",
        "python": {
            "package": "onnxruntime-gpu",
            "version": BF16_TARGET_ORT_VERSION,
        },
        "dotnet": {
            "package": "Microsoft.ML.OnnxRuntime.Gpu",
            "version": BF16_TARGET_ORT_VERSION,
        },
    }
    assert max(
        row["max_logit_abs_diff"] for row in report["ort_padding_matrix"]
    ) <= parity_max_abs_tolerance("bf16")


def test_export_uses_checkpoint_vocab_instead_of_stale_profile(tmp_path):
    from scripts.onnx_export.export.checkpoint import load_policy_contracts

    checkpoint, profile = tmp_path / "checkpoint.pt", tmp_path / "profile.json"
    _, saved = _write_small_checkpoint(checkpoint)
    _write_small_profile(profile)
    payload = json.loads(profile.read_text(encoding="utf-8"))
    entries = payload["vocab_entries"]
    entries[0]["vocab_id"], entries[1]["vocab_id"] = entries[1]["vocab_id"], entries[0]["vocab_id"]
    profile.write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    contracts = load_policy_contracts(checkpoint_path=checkpoint, deployment_profile_path=profile, precision="float32")
    assert contracts.deployment_contract.vocab_entries == saved.skill_vocab_entries
    assert contracts.contract_payload["vocab"] == saved.create_skill_vocab().to_dict()
    assert contracts.capacity_report["vocab_entries"] == saved.create_skill_vocab().to_dict()["entries"]


@pytest.mark.parametrize("full_attention_residuals", [False, True])
def test_deployment_contract_preserves_actual_residual_path(tmp_path, full_attention_residuals):
    from scripts.onnx_export.export.checkpoint import load_policy_contracts
    jsonschema = pytest.importorskip("jsonschema")
    checkpoint, profile = tmp_path / "checkpoint.pt", tmp_path / "profile.json"
    _write_small_checkpoint(
        checkpoint, full_attention_residuals=full_attention_residuals,
        qk_norm_scale=1.7, logit_softcap=7.5,
    )
    _write_small_profile(profile)
    contracts = load_policy_contracts(checkpoint_path=checkpoint, deployment_profile_path=profile, precision="float32")
    payload = contracts.deployment_contract.to_dict()
    expected = "full_attention_residual" if full_attention_residuals else "learned_residual_mix"
    assert payload["residual_composition"]["type"] == expected
    assert payload["model_config"]["qk_norm_scale"] == 1.7
    assert payload["model_config"]["logit_softcap"] == 7.5
    assert payload["model_input_contract"]["token_encoding"]["attention_qk_normalization"] == (
        TOKEN_ENCODING_CONTRACT["attention_qk_normalization"]
    )
    assert payload["model_input_contract"]["token_encoding"]["output_projection"] == (
        TOKEN_ENCODING_CONTRACT["output_projection"]
    )
    assert payload["model_input_contract"]["token_encoding"]["logit_softcap"] == (
        TOKEN_ENCODING_CONTRACT["logit_softcap"]
    )
    restored = DeploymentContract.from_dict(payload)
    assert restored.to_dict() == payload
    schema_path = Path(export_module.__file__).parents[1] / "manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator({"$ref": "#/$defs/contract", "$defs": schema["$defs"]})
    validator.validate(payload)
    wrong = deepcopy(payload)
    wrong["model_config"]["full_attention_residuals"] = not full_attention_residuals
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(wrong)
    with pytest.raises(ValueError, match="authoritative layouts"):
        DeploymentContract.from_dict(wrong)


def test_deployment_state_masks_use_base_width_and_outer_layout_cannot_override_saved_schema(tmp_path):
    from scripts.onnx_export.export.checkpoint import load_policy_contracts

    checkpoint, profile = tmp_path / "checkpoint.pt", tmp_path / "profile.json"
    spec, _ = _write_small_checkpoint(checkpoint)
    _write_small_profile(profile)
    contract = load_policy_contracts(checkpoint_path=checkpoint, deployment_profile_path=profile,
                                     precision="float32").deployment_contract
    fields = {field.name: field for field in contract.tensor_inputs()}
    assert len(fields) == 12
    assert fields["current_state_vectors"].shape[-1] == spec.state_dim == 9
    assert fields["current_state_null_mask"].shape[-1] == spec.base_state_dim == 3
    assert fields["history_state_reset_mask"].shape[-1] == spec.base_state_dim
    inputs = make_inputs(spec, contract.capacity, vocab_size=8, scene_valid=1, history_valid=1,
                         dtype=torch.float32, seed=19)
    contract.validate_tensor_inputs(inputs)
    named = dict(zip(TENSOR_INPUT_NAMES, inputs, strict=True))
    named["current_state_vectors"][0, spec.base_state_dim] = .5
    with pytest.raises(ValueError, match="absolute binary"):
        contract.validate_tensor_inputs(inputs)
    payload = contract.to_dict()
    payload["state_layout"][0]["feature_keys"].reverse()
    with pytest.raises(ValueError, match="authoritative layouts"):
        DeploymentContract.from_dict(payload)


@pytest.mark.parametrize("full_attention_residuals", [False, True])
def test_load_policy_preserves_saved_readout_and_qk_scale_without_project_yaml(tmp_path, monkeypatch, full_attention_residuals):
    import common.config as config_module

    checkpoint = tmp_path / "checkpoint.pt"
    _write_small_checkpoint(
        checkpoint, full_attention_residuals=full_attention_residuals,
        qk_norm_scale=1.7, logit_softcap=7.5,
    )
    monkeypatch.setattr(
        config_module, "load_project_config",
        lambda **_kwargs: pytest.fail("checkpoint readout and QK scales must not load project YAML"),
    )
    policy, _, _, _, payload = export_module.load_policy(checkpoint, precision="float32")
    assert policy.model.config.qk_norm_scale == 1.7
    assert all(layer.qk_norm_scale == 1.7 for layer in policy.model.encoder.layers)
    assert policy.model.config.logit_softcap == 7.5
    assert policy.model.output_head.bias is None
    assert policy.model.output_head.weight.shape == (3, 16)
    saved_head = payload["model_state_dict"]["output_head.weight"]
    saved_embedding = payload["model_state_dict"]["input_encoder.skill_embed.weight"]
    assert not torch.equal(saved_head, saved_embedding[list(policy.model.data_spec.action_to_vocab_id)])
    assert hasattr(policy.model.encoder, "residual_mix") is (not full_attention_residuals)
    for name, tensor in policy.model.state_dict().items():
        torch.testing.assert_close(tensor, payload["model_state_dict"][name], atol=0, rtol=0)


@pytest.mark.parametrize("value", ["missing", True, False, 0, -1, float("nan"), float("inf"), float("-inf"), "invalid", None])
def test_checkpoint_and_deployment_reject_missing_or_invalid_qk_scale(tmp_path, value):
    from scripts.onnx_export.export.checkpoint import load_policy_contracts

    # 缺失元数据独立于默认值；错误数值也不能靠重算部署签名绕过模型校验。
    checkpoint, profile = tmp_path / "checkpoint.pt", tmp_path / "profile.json"
    _write_small_checkpoint(checkpoint)
    _write_small_profile(profile)
    contracts = load_policy_contracts(checkpoint_path=checkpoint, deployment_profile_path=profile, precision="float32")
    deployment = contracts.deployment_contract.to_dict()
    payload = safe_torch_load(checkpoint)
    for model_config in (payload["model_config"], deployment["model_config"]):
        if value == "missing":
            model_config.pop("qk_norm_scale")
        else:
            model_config["qk_norm_scale"] = value
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="qk_norm_scale"):
        export_module.load_policy(checkpoint, precision="float32")
    with pytest.raises(ValueError, match="qk_norm_scale"):
        DeploymentContract.from_dict(deployment)


@pytest.mark.parametrize("value", [None, True, False, "1.2", 0, -1])
def test_manifest_schema_rejects_missing_or_invalid_qk_scale(value):
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(export_module.__file__).parents[1] / "manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema["$defs"]["contract"]["properties"]["model_config"])
    payload = asdict(ModelConfig())
    validator.validate(payload)
    if value is None:
        payload.pop("qk_norm_scale")
    else:
        payload["qk_norm_scale"] = value
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(payload)


def test_deployment_qk_scale_is_bound_by_saved_model_config_signature(tmp_path):
    from scripts.onnx_export.export.checkpoint import load_policy_contracts

    checkpoint, profile = tmp_path / "checkpoint.pt", tmp_path / "profile.json"
    _write_small_checkpoint(checkpoint, qk_norm_scale=1.7)
    _write_small_profile(profile)
    contracts = load_policy_contracts(checkpoint_path=checkpoint, deployment_profile_path=profile, precision="float32")
    deployment = contracts.deployment_contract.to_dict()
    deployment["model_config"]["qk_norm_scale"] = 1.2
    with pytest.raises(ValueError, match="authoritative layouts"):
        DeploymentContract.from_dict(deployment)


@pytest.mark.parametrize("value", ["missing", True, False, 0, -1, 1e-46, 1e39, float("nan"), float("inf"), float("-inf"), "invalid", None])
def test_checkpoint_and_deployment_reject_missing_or_invalid_logit_softcap(tmp_path, value):
    from scripts.onnx_export.export.checkpoint import load_policy_contracts

    checkpoint, profile = tmp_path / "checkpoint.pt", tmp_path / "profile.json"
    _write_small_checkpoint(checkpoint)
    _write_small_profile(profile)
    contracts = load_policy_contracts(checkpoint_path=checkpoint, deployment_profile_path=profile, precision="float32")
    deployment = contracts.deployment_contract.to_dict()
    payload = safe_torch_load(checkpoint)
    for model_config in (payload["model_config"], deployment["model_config"]):
        if value == "missing":
            model_config.pop("logit_softcap")
        else:
            model_config["logit_softcap"] = value
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="logit_softcap"):
        export_module.load_policy(checkpoint, precision="float32")
    with pytest.raises(ValueError, match="logit_softcap"):
        DeploymentContract.from_dict(deployment)


@pytest.mark.parametrize("value", [None, True, False, "15.0", 0, -1, 1e-46, 1e39])
def test_manifest_schema_rejects_missing_or_invalid_logit_softcap(value):
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(export_module.__file__).parents[1] / "manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema["$defs"]["contract"]["properties"]["model_config"])
    payload = asdict(ModelConfig())
    validator.validate(payload)
    if value is None:
        payload.pop("logit_softcap")
    else:
        payload["logit_softcap"] = value
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(payload)


@pytest.mark.parametrize("cap", [torch.finfo(torch.float32).tiny, 1e-8, 2**-24, 2**-15, 65505.0, 65520.0, 1e10, 1e38])
def test_float16_export_rejects_softcap_outside_normal_output_range(tmp_path, cap):
    """模型允许的 FP32 尺度不能绕过导出加载、部署对象或 manifest 的 FP16 限制。"""
    from scripts.onnx_export.export.checkpoint import load_policy_contracts
    jsonschema = pytest.importorskip("jsonschema")

    checkpoint, profile = tmp_path / "checkpoint.pt", tmp_path / "profile.json"
    _write_small_checkpoint(checkpoint, logit_softcap=cap)
    _write_small_profile(profile)
    contracts = load_policy_contracts(checkpoint_path=checkpoint, deployment_profile_path=profile, precision="float32")
    saved = contracts.deployment_contract
    with pytest.raises(ValueError, match="float16.*logit_softcap.*FP16 normal range"):
        export_module.load_policy(checkpoint, precision="float16")
    with pytest.raises(ValueError, match="float16.*logit_softcap.*FP16 normal range"):
        DeploymentContract.create(
            precision="float16", capacity=saved.capacity, data_spec=saved.data_spec,
            input_contract=saved.input_contract, model_config=saved.model_config,
            repetition_config=saved.repetition_config, vocab_entries=saved.vocab_entries,
            capacity_report=contracts.capacity_report, embedding_vocab_size=contracts.vocab_size,
        )
    # 重新生成相符的签名和张量 dtype，仍必须由数值契约拒绝。
    payload = replace(saved, precision="float16").to_dict()
    with pytest.raises(ValueError, match="float16.*logit_softcap.*FP16 normal range"):
        DeploymentContract.from_dict(payload)
    schema = json.loads((Path(export_module.__file__).parents[1] / "manifest.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator({"$defs": schema["$defs"], "$ref": "#/$defs/contract"})
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(payload)


@pytest.mark.parametrize("cap", [torch.finfo(torch.float16).tiny, 0.001, 15.0, torch.finfo(torch.float16).max])
def test_float16_softcap_normal_range_includes_both_endpoints(tmp_path, cap):
    from scripts.onnx_export.export.checkpoint import load_policy_contracts
    jsonschema = pytest.importorskip("jsonschema")

    checkpoint, profile = tmp_path / "checkpoint.pt", tmp_path / "profile.json"
    _write_small_checkpoint(checkpoint, logit_softcap=cap)
    _write_small_profile(profile)
    contracts = load_policy_contracts(checkpoint_path=checkpoint, deployment_profile_path=profile, precision="float16")
    assert contracts.dtype == torch.float16
    assert contracts.policy.model.config.logit_softcap == cap
    payload = contracts.deployment_contract.to_dict()
    assert DeploymentContract.from_dict(payload).model_config["logit_softcap"] == cap
    schema = json.loads((Path(export_module.__file__).parents[1] / "manifest.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator({"$defs": schema["$defs"], "$ref": "#/$defs/contract"}).validate(payload)


@pytest.mark.parametrize("precision", ["float32", "bf16"])
@pytest.mark.parametrize("cap", [torch.finfo(torch.float32).tiny, 1e-8, 65520.0, 1e38])
def test_other_deployment_precisions_keep_existing_softcap_range(tmp_path, precision, cap):
    from scripts.onnx_export.export.checkpoint import load_policy_contracts
    jsonschema = pytest.importorskip("jsonschema")

    checkpoint, profile = tmp_path / "checkpoint.pt", tmp_path / "profile.json"
    _write_small_checkpoint(checkpoint, logit_softcap=cap)
    _write_small_profile(profile)
    contracts = load_policy_contracts(checkpoint_path=checkpoint, deployment_profile_path=profile, precision=precision)
    payload = contracts.deployment_contract.to_dict()
    assert DeploymentContract.from_dict(payload).model_config["logit_softcap"] == cap
    schema = json.loads((Path(export_module.__file__).parents[1] / "manifest.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator({"$defs": schema["$defs"], "$ref": "#/$defs/contract"}).validate(payload)


def test_deployment_logit_softcap_is_bound_by_saved_model_config_signature(tmp_path):
    from scripts.onnx_export.export.checkpoint import load_policy_contracts

    checkpoint, profile = tmp_path / "checkpoint.pt", tmp_path / "profile.json"
    _write_small_checkpoint(checkpoint, logit_softcap=7.5)
    _write_small_profile(profile)
    contracts = load_policy_contracts(checkpoint_path=checkpoint, deployment_profile_path=profile, precision="float32")
    deployment = contracts.deployment_contract.to_dict()
    deployment["model_config"]["logit_softcap"] = 15.0
    with pytest.raises(ValueError, match="authoritative layouts"):
        DeploymentContract.from_dict(deployment)


@pytest.mark.parametrize("legacy", ["version", "shared_projection", "missing_softcap", "wrong_softcap_dtype"])
def test_load_policy_rejects_previous_shared_or_uncapped_output(tmp_path, legacy):
    checkpoint = tmp_path / "checkpoint.pt"
    _write_small_checkpoint(checkpoint)
    payload = safe_torch_load(checkpoint)
    encoding = payload["input_contract"]["token_encoding"]
    if legacy == "version":
        payload["input_contract"]["version"] = 19
    elif legacy == "shared_projection":
        encoding["output_projection"] = "hidden @ E[action_to_vocab_id].T"
    elif legacy == "missing_softcap":
        encoding.pop("logit_softcap")
    else:
        encoding["logit_softcap"]["compute_dtype"] = "bfloat16"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="input contract version|token_encoding"):
        export_module.load_policy(checkpoint, precision="float32")


@pytest.mark.parametrize("legacy", ["version", "missing_descriptor", "wrong_descriptor"])
def test_load_policy_rejects_unnormalized_qk_checkpoint(tmp_path, legacy):
    checkpoint = tmp_path / "checkpoint.pt"
    _write_small_checkpoint(checkpoint)
    payload = safe_torch_load(checkpoint)
    if legacy == "version":
        payload["input_contract"]["version"] = 18
    elif legacy == "missing_descriptor":
        payload["input_contract"]["token_encoding"].pop("attention_qk_normalization")
    else:
        payload["input_contract"]["token_encoding"]["attention_qk_normalization"]["eps"] = 1e-5
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="input contract version|token_encoding"):
        export_module.load_policy(checkpoint, precision="float32")


@pytest.mark.parametrize("field", ["residual_mix_r_start", "residual_mix_r_end", "residual_mix_a_start", "residual_mix_a_end"])
def test_load_policy_requires_all_saved_residual_initialization_fields(tmp_path, field):
    checkpoint = tmp_path / "checkpoint.pt"
    _write_small_checkpoint(checkpoint)
    payload = safe_torch_load(checkpoint)
    payload["model_config"].pop(field)
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="residual.*initialization|residual_mix"):
        export_module.load_policy(checkpoint, precision="float32")


def test_load_policy_rejects_experiment_guard_even_with_current_contract(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    _write_small_checkpoint(checkpoint)
    payload = safe_torch_load(checkpoint)
    payload["model_state_dict"]["_learned_residual_mix_experiment_guard"] = torch.tensor(1)
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="Unexpected key.*_learned_residual_mix_experiment_guard"):
        export_module.load_policy(checkpoint, precision="float32")


def test_deployment_rejects_reassigned_vocab_rows_even_with_recomputed_signatures(tmp_path):
    from dataclasses import replace
    from scripts.onnx_export.export.checkpoint import load_policy_contracts

    checkpoint, profile = tmp_path / "checkpoint.pt", tmp_path / "profile.json"
    _write_small_checkpoint(checkpoint)
    _write_small_profile(profile)
    contracts = load_policy_contracts(checkpoint_path=checkpoint, deployment_profile_path=profile, precision="float32")
    entries = list(contracts.deployment_contract.vocab_entries)
    entries[0], entries[1] = (entries[0][0], entries[1][1]), (entries[1][0], entries[0][1])
    wrong = replace(contracts.deployment_contract, vocab_entries=tuple(entries))
    with pytest.raises(ValueError, match="deployment skill vocab mismatch"):
        wrong.validate(embedding_vocab_size=8)
    # 签名与内容一致也不能把错误映射变成 checkpoint 的权威词表。
    with pytest.raises(ValueError, match="deployment skill vocab mismatch"):
        DeploymentContract.from_dict(wrong.to_dict())


def test_load_policy_rejects_removed_candidate_shared_semantics(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    _write_small_checkpoint(checkpoint)
    payload = safe_torch_load(checkpoint)
    payload["model_config"] = dict(payload["model_config"])
    payload["model_config"]["position_id_semantics"] = "candidate_block_shared"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="removed model options"):
        export_module.load_policy(checkpoint, precision="float32")


def test_load_policy_rejects_stage1_fusion_config_and_input_contract(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    _write_small_checkpoint(checkpoint)
    payload = safe_torch_load(checkpoint)
    payload["model_config"]["pair_embedding_dim"] = 8
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="pair_embedding_dim is removed"):
        export_module.load_policy(checkpoint, precision="float32")
    payload["model_config"].pop("pair_embedding_dim")
    payload["input_contract"]["version"] = 10
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="input contract version"):
        export_module.load_policy(checkpoint, precision="float32")
    payload["input_contract"]["version"] = INPUT_CONTRACT_VERSION
    payload["input_contract"].pop("token_encoding")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="token_encoding"):
        export_module.load_policy(checkpoint, precision="float32")


@pytest.mark.parametrize("activation", ("gelu", "swiglu"))
def test_load_policy_preserves_ffn_weights_and_rejects_mislabeled_activation(tmp_path, activation):
    checkpoint = tmp_path / "checkpoint.pt"
    _write_small_checkpoint(checkpoint, activation=activation)
    policy, _, _, _, payload = export_module.load_policy(checkpoint, precision="float32")
    assert policy.model.config.transformer_activation == activation
    for name, tensor in policy.model.state_dict().items():
        torch.testing.assert_close(tensor, payload["model_state_dict"][name], atol=0, rtol=0)

    payload = dict(payload)
    payload["model_config"] = dict(payload["model_config"])
    payload["model_config"]["transformer_activation"] = (
        "gelu" if activation == "swiglu" else "swiglu"
    )
    mislabeled = tmp_path / "mislabeled.pt"
    torch.save(payload, mislabeled)
    with pytest.raises(RuntimeError, match=r"Error\(s\) in loading state_dict"):
        export_module.load_policy(mislabeled, precision="float32")


def _write_small_checkpoint(
    path: Path, *, activation: str = "gelu", full_attention_residuals: bool = False,
    qk_norm_scale: float = 1.2, logit_softcap: float = 15.0,
) -> tuple[DataSpec, ModelInputContract]:
    data_spec = DataSpec(
        job_tag="black_mage",
        num_actions=3,
        base_state_dim=3,
        state_dim=9,
        scene_dim=3,
        skill_feature_dim=2,
        num_scene_types=4,
        action_keys=("fire_iii", "fire_iv", "blizzard_iii"),
        action_to_vocab_id=(2, 4, 1),
        action_is_gcd=(True, True, True),
        skill_feature_names=("potency", "cast_time.seconds"),
    )
    config = ModelConfig(
        d_model=16,
        n_layers=2,
        n_heads=2,
        ff_dim=32,
        dropout=0.0,
        history_capacity=4,
        history_reset_keep=4,
        scene_capacity=3,
        transformer_activation=activation,
        full_attention_residuals=full_attention_residuals,
        qk_norm_scale=qk_norm_scale,
        logit_softcap=logit_softcap,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260813)
        model = CausalPolicyModel(data_spec, config, vocab_size=8).eval()
        # 模拟训练后的 scalar；真实导出必须使用权重，不能重建初始化 ramp。
        with torch.no_grad():
            # 保存独立头训练后的值，恢复时不得重新拷贝输入 embedding。
            model.output_head.weight[0].add_(0.125)
            if not full_attention_residuals:
                model.encoder.residual_mix.r.copy_(torch.tensor([.93, 1.27]))
                model.encoder.residual_mix.a.copy_(torch.tensor([-.07, .13]))
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    scene_windows = tuple(
        SceneWindowSchema.from_feature_keys(
            context_key=f"scene_{scene_type_id}",
            feature_keys=(
                "start_offset_seconds",
                "end_offset_seconds",
                "duration_seconds",
            ),
            scene_type_id=scene_type_id,
        )
        for scene_type_id in range(data_spec.num_scene_types)
    )
    input_contract = ModelInputContract.from_training(
        skill_vocab=SkillVocab.from_entries([(1000 + row, row) for row in range(1, 8)]),
        data_spec=data_spec,
        schema=TrainingSchema(
            serialization_format="test",
            sample_schema_version=TRAINING_SAMPLE_SCHEMA_VERSION,
            context_schema_version=CANONICAL_CONTEXT_SCHEMA_VERSION,
            scene_context_mode="absolute",
            scene_windows=scene_windows,
            state_groups=make_state_groups({
                "player_state": (
                    "previous_action_after.time_seconds",
                    "request_state.time_seconds",
                    "request_state.mp",
                )
            }, data_spec.action_keys),
            state_snapshots=("previous_action_after", "request_state"),
            skill_history_fields=("skill_key",),
        ),
        normalizer=normalizer,
    )
    torch.save(
        {
            "data_spec": asdict(data_spec),
            "input_contract": input_contract.to_dict(),
            "model_config": asdict(config),
            "model_variant": "artzip",
            "run_config": {},
            "model_state_dict": model.state_dict(),
            # 部署加载必须忽略这些训练字段。
            "optimizer_state_dict": {"sentinel": torch.tensor(1)},
            "scheduler_state_dict": {"sentinel": torch.tensor(2)},
        },
        path,
    )
    return data_spec, input_contract


def _write_small_profile(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "profile_version": 1,
                "job_tag": "black_mage",
                "vocab_entries": [
                    {"raw_skill_id": 1000 + index, "vocab_id": index}
                    for index in range(1, 8)
                ],
                "evidence": {
                    "method": "test_fixture",
                    "scene_length_max": 3,
                },
            }
        ),
        encoding="utf-8",
    )

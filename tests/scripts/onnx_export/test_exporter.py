"""ONNX 导出包、固定容量契约和原子发布测试。"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from common.policy.config import ModelConfig
from common.policy.data import DataSpec, ModelInputContract, Normalizer
from common.policy.data.schema import SceneWindowSchema, TrainingSchema
from common.policy.model import CandidateTransformerModel
from common.torch_serialization import safe_torch_load
from scripts.autoregressive_replay.backends import (
    OrtPolicyBackend,
    PyTorchPolicyBackend,
    compare_backend_logits,
)
from scripts.onnx_export import TENSOR_INPUT_NAMES, CapacityContract, DeploymentManifest
from scripts.onnx_export import export as export_module
from scripts.onnx_export.release import release as release_module
from scripts.onnx_export.io.artifact_io import (
    normalize_torch_reports,
    write_deterministic_npz,
)
from scripts.onnx_export.contracts.contract import make_inputs, slice_dynamic_inputs
from scripts.onnx_export.contracts.deployment_profile import DeploymentProfile
from scripts.onnx_export.export import export_package
from scripts.onnx_export.export import environment as environment_module
from scripts.onnx_export.export import publish as publish_module
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
)
from scripts.onnx_export.release.release import (
    parity_artifact_bindings,
    record_parity_result,
    record_successful_parity,
    verify_release,
)
from scripts.onnx_export.release.policy import minimum_empty_action_budget
from scripts.onnx_export.runtime.runtime_targets import (
    BF16_TARGET_ONNX_VERSION,
    BF16_TARGET_ONNXSCRIPT_VERSION,
    BF16_TARGET_ORT_VERSION,
)
from scripts.onnx_export.runtime.tensor_runtime import (
    GOLDEN_BF16_ENCODING,
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
        CapacityContract(0, 1, 3).validate()
    contract = CapacityContract(3, 8, 3)
    contract.validate()
    assert contract.total_token_count == 15


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
        state_dim=5,
        num_candidates=2,
    )
    contract = CapacityContract(3, 4, 2)
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
    assert torch.count_nonzero(values[6][:, 2:]) == 0
    assert torch.count_nonzero(values[0][:, :1]) > 0
    assert torch.count_nonzero(values[3][:, :2]) > 0


def test_default_deployment_profile_rejects_unsafe_job_tag():
    with pytest.raises(ValueError, match="unsafe"):
        DeploymentProfile.default_path("../black_mage")
    with pytest.raises(ValueError, match="unsafe"):
        DeploymentProfile.default_path(r"..\black_mage")


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
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    checkpoint = tmp_path / "checkpoint.pt"
    data_spec, _input_contract = _write_small_checkpoint(
        checkpoint,
        activation=activation,
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
    assert manifest["contract"]["capacity"]["padding_direction"] == "right"
    assert manifest["contract"]["capacity"]["history_capacity"] == 4
    provenance = manifest["contract"]["capacity_provenance"]
    assert provenance["history_capacity_source"] == (
        "checkpoint.model_config.history_capacity"
    )
    assert provenance["scene_capacity_source"] == "checkpoint.model_config.scene_capacity"
    assert manifest["contract"]["model_config"]["d_model"] == 16
    assert manifest["contract"]["model_config"]["transformer_activation"] == activation
    assert manifest["model"]["external_data"] is False
    # RoPE 删除了原先的大型 learned absolute position table；小 fixture 中
    # 重复的 LayerNorm 常量会被 ONNX exporter 去重，因此该统计值不再等价于
    # “所有 PyTorch 参数逐元素保留”，但仍需覆盖绝大多数模型权重。
    assert manifest["model"]["retained_float_initializer_ratio"] >= 0.98
    assert manifest["model"]["onnx_other_float_initializer_max_elements"] <= 1
    assert [
        item["name"] for item in manifest["contract"]["tensor_outputs"]
    ] == ["raw_logits"]
    loaded = DeploymentManifest.load(output / "manifest.json")
    assert loaded.contract.data_spec == data_spec

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
            "candidate_legal_mask": torch.tensor([[True, False, True]]),
            "candidate_action_keys": [list(data_spec.candidate_action_keys)],
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
        data_spec.candidate_action_keys,
    )
    assert parity["max_abs_diff"] <= 1e-4
    assert parity["top1_match"] is True
    assert parity["top3_set_match"] is True
    assert parity["reference_top3"] == parity["candidate_top3"]

    tampered = deepcopy(manifest)
    tampered["contract"]["data_spec"]["candidate_action_keys"][0:2] = reversed(
        tampered["contract"]["data_spec"]["candidate_action_keys"][0:2]
    )
    tampered_path = output / "tampered-manifest.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch|differ"):
        DeploymentManifest.load(tampered_path, verify_files=False)
    legacy = deepcopy(manifest)
    legacy["manifest_version"] = 1
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
def test_small_bf16_model_exports_and_runs_on_strict_cuda(tmp_path, activation):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("ORT CUDAExecutionProvider is unavailable")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")

    checkpoint = tmp_path / "checkpoint.pt"
    _write_small_checkpoint(checkpoint, activation=activation)
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
    assert manifest["manifest_version"] == 7
    assert manifest["contract"]["contract_version"] == 9
    assert manifest["contract"]["precision"] == "bf16"
    assert manifest["exporter"]["onnxscript"] == BF16_TARGET_ONNXSCRIPT_VERSION
    assert manifest["contract"]["tensor_outputs"][0]["dtype"] == (
        "tensor(bfloat16)"
    )
    assert manifest["golden"]["float_encoding"] == GOLDEN_BF16_ENCODING
    assert manifest["model"]["onnx_other_float_initializer_max_elements"] <= 1
    assert manifest["model"]["onnx_float_initializer_dtype_counts"]["bfloat16"] > 0
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


def test_load_policy_rejects_removed_candidate_shared_semantics(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    _write_small_checkpoint(checkpoint)
    payload = safe_torch_load(checkpoint)
    payload["model_config"] = dict(payload["model_config"])
    payload["model_config"]["position_id_semantics"] = "candidate_block_shared"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="removed candidate-shared RoPE"):
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
    path: Path, *, activation: str = "gelu",
) -> tuple[DataSpec, ModelInputContract]:
    data_spec = DataSpec(
        job_tag="black_mage",
        num_candidates=3,
        state_dim=3,
        scene_dim=3,
        skill_feature_dim=2,
        num_scene_types=4,
        candidate_action_keys=("fire_iii", "fire_iv", "blizzard_iii"),
        skill_feature_names=("potency", "cast_time.seconds"),
    )
    config = ModelConfig(
        d_model=16,
        pair_embedding_dim=8,
        n_layers=2,
        n_heads=2,
        ff_dim=32,
        dropout=0.0,
        history_capacity=4,
        scene_capacity=3,
        transformer_activation=activation,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260813)
        model = CandidateTransformerModel(data_spec, config, vocab_size=8).eval()
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
        data_spec=data_spec,
        schema=TrainingSchema(
            serialization_format="test",
            sample_schema_version=1,
            context_schema_version=1,
            scene_context_mode="absolute",
            scene_windows=scene_windows,
            state_group_feature_keys={
                "player_state": (
                    "before.time_seconds",
                    "before.current_gcd_seconds",
                    "before.mp",
                )
            },
            candidate_skill_fields=data_spec.skill_feature_names,
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

"""把通过的 rollout parity 证据绑定到不可变 ONNX 部署包。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from ..io.artifact_io import file_sha256, json_payload_sha256, write_json_atomic
from ..contracts.deployment_contract import DeploymentManifest
from ..runtime.ort_runtime import ORT_PROVIDER_CUDA
from ..runtime.precision import parity_max_abs_tolerance
from ..runtime.runtime_targets import runtime_targets, validate_runtime_targets
from .policy import validate_release_scenario_request


RELEASE_REPORT_FILENAME = "release_report.json"
RELEASE_REPORT_VERSION = 3
RELEASE_GATE_VERSION = 1
PARITY_EMPTY_STEM = "parity_empty_128gcd"
PARITY_SCENE_STEM = "parity_scene_100"


def parity_artifact_bindings(
    *,
    manifest: DeploymentManifest,
    manifest_path: Path,
    checkpoint_path: Path,
) -> dict[str, object]:
    """生成 parity 报告必须携带的 checkpoint/ONNX/契约指纹。"""
    manifest_path = Path(manifest_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = _mapping(manifest.payload["checkpoint"], "manifest.checkpoint")
    actual_checkpoint_sha256 = file_sha256(checkpoint_path)
    if actual_checkpoint_sha256 != checkpoint["sha256"]:
        raise ValueError(
            "parity checkpoint differs from the checkpoint used to export ONNX: "
            f"{actual_checkpoint_sha256} != {checkpoint['sha256']}"
        )
    model = _mapping(manifest.payload["model"], "manifest.model")
    contract = _mapping(manifest.payload["contract"], "manifest.contract")
    signatures = _mapping(contract["signatures"], "manifest.contract.signatures")
    return {
        "manifest": {
            "filename": manifest_path.name,
            "sha256": file_sha256(manifest_path),
        },
        "model": {
            "filename": str(model["filename"]),
            "sha256": str(model["sha256"]),
        },
        "checkpoint": {
            "filename": checkpoint_path.name,
            "sha256": actual_checkpoint_sha256,
        },
        "deployment_contract_sha256": str(
            signatures["deployment_contract_sha256"]
        ),
    }


def record_successful_parity(
    *,
    package_dir: Path,
    parity_report_path: Path,
) -> Path:
    """登记成功 parity；满足空场景和真实场景门槛后晋升发布状态。"""
    report = _load_json(Path(parity_report_path), "parity report")
    if report.get("status") != "passed":
        raise ValueError("only a passed parity report can be recorded for release")
    return record_parity_result(
        package_dir=package_dir,
        parity_report_path=parity_report_path,
    )


def record_parity_result(
    *,
    package_dir: Path,
    parity_report_path: Path,
) -> Path:
    """登记正式 parity 成功或失败结果，并原子刷新发布状态。"""
    package_dir = Path(package_dir).resolve()
    parity_report_path = Path(parity_report_path).resolve()
    manifest_path = package_dir / "manifest.json"
    manifest = DeploymentManifest.load(manifest_path, verify_files=True)
    report = _load_json(parity_report_path, "parity report")
    if report.get("status") not in {"passed", "failed"}:
        raise ValueError("parity report status must be passed or failed")
    if report.get("artifacts") != parity_artifact_bindings_from_package(
        manifest=manifest,
        manifest_path=manifest_path,
    ):
        raise ValueError("parity report artifact bindings differ from deployment package")
    if report.get("runtime_targets") != package_runtime_targets(
        package_dir=package_dir,
        manifest=manifest,
    ):
        raise ValueError("parity report runtime targets differ from deployment package")
    _validate_formal_release_report(report, manifest=manifest)

    scenario = _requested_release_scenario(report)
    target_stem = (
        PARITY_EMPTY_STEM
        if scenario == "empty_128_gcd"
        else PARITY_SCENE_STEM
    )
    target_name = f"{target_stem}.{json_payload_sha256(report)}.json"
    target_path = package_dir / target_name
    target_existed = target_path.is_file()
    evidence_by_scenario, superseded_evidence = _existing_evidence(package_dir)
    previous = evidence_by_scenario.get(scenario)
    if (
        previous is not None
        and _evidence_passed(previous, precision=manifest.contract.precision)
        and report.get("status") == "failed"
    ):
        return package_dir / RELEASE_REPORT_FILENAME
    write_json_atomic(target_path, report)
    current_evidence = _evidence_entry(
        scenario=scenario,
        report=report,
        path=target_path,
    )
    if previous is not None and previous.get("sha256") != current_evidence["sha256"]:
        _append_unique_evidence(superseded_evidence, previous)
    current_identity = (
        str(current_evidence.get("filename")),
        str(current_evidence.get("sha256")),
    )
    superseded_evidence = [
        evidence
        for evidence in superseded_evidence
        if (
            str(evidence.get("filename")),
            str(evidence.get("sha256")),
        )
        != current_identity
    ]
    evidence_by_scenario[scenario] = current_evidence
    empty_passed = _evidence_passed(
        evidence_by_scenario.get("empty_128_gcd"),
        precision=manifest.contract.precision,
    )
    scene_passed = _evidence_passed(
        evidence_by_scenario.get("scene_100_decisions"),
        precision=manifest.contract.precision,
    )
    release_validated = empty_passed and scene_passed
    any_failed = any(
        evidence.get("status") == "failed"
        for evidence in evidence_by_scenario.values()
    )
    status = (
        "release_validated"
        if release_validated
        else "parity_failed" if any_failed else "parity_in_progress"
    )
    release_report = {
        "release_report_version": RELEASE_REPORT_VERSION,
        "release_gate_version": RELEASE_GATE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "package": parity_artifact_bindings_from_package(
            manifest=manifest,
            manifest_path=manifest_path,
        ),
        "runtime_targets": package_runtime_targets(
            package_dir=package_dir,
            manifest=manifest,
        ),
        "requirements": {
            "empty_scene_128_gcd": empty_passed,
            "scene_100_decisions": scene_passed,
        },
        "parity_reports": [
            evidence_by_scenario[key]
            for key in sorted(evidence_by_scenario)
        ],
        "superseded_parity_reports": sorted(
            superseded_evidence,
            key=lambda evidence: (
                str(evidence.get("filename")),
                str(evidence.get("sha256")),
            ),
        ),
    }
    release_path = package_dir / RELEASE_REPORT_FILENAME
    try:
        write_json_atomic(release_path, release_report)
    except Exception as exc:
        if not target_existed:
            try:
                target_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                note = (
                    f"failed to remove uncommitted parity evidence {target_path}: "
                    f"{cleanup_exc}"
                )
                _attach_exception_note(exc, note)
        raise
    return release_path


def verify_release(
    package_dir: Path,
    *,
    require_validated: bool = True,
    audit_history: bool = False,
) -> dict[str, object]:
    """校验当前发布证明；正式审计可额外复核全部 superseded 证据。"""
    package_dir = Path(package_dir).resolve()
    manifest_path = package_dir / "manifest.json"
    manifest = DeploymentManifest.load(manifest_path, verify_files=True)
    release_path = package_dir / RELEASE_REPORT_FILENAME
    if not release_path.is_file():
        raise FileNotFoundError(f"deployment release report missing: {release_path}")
    release = _load_json(release_path, "release report")
    version = int(release.get("release_report_version", -1))
    if version not in {1, 2, RELEASE_REPORT_VERSION}:
        raise ValueError("unsupported deployment release report version")
    if version != 1 and int(
        release.get("release_gate_version", -1)
    ) != RELEASE_GATE_VERSION:
        raise ValueError("unsupported deployment release gate version")
    if (
        version != 1
        and require_validated
        and release.get("status") != "release_validated"
    ):
        raise ValueError("deployment package has not passed all rollout parity gates")
    expected_package = parity_artifact_bindings_from_package(
        manifest=manifest,
        manifest_path=manifest_path,
    )
    if release.get("package") != expected_package:
        raise ValueError("release report is bound to a different deployment package")
    if release.get("runtime_targets") != package_runtime_targets(
        package_dir=package_dir,
        manifest=manifest,
    ):
        raise ValueError("release report runtime targets differ from manifest")
    if version == 1:
        if require_validated:
            raise ValueError(
                "legacy deployment release evidence must be rerun with the explicit gate"
            )
        legacy_evidence: list[dict[str, object]] = []
        for key in ("parity_reports", "superseded_parity_reports"):
            for entry in release.get(key, []):
                evidence = _mapping(entry, f"release.{key}[]")
                path = package_dir / str(evidence["filename"])
                if not path.is_file() or file_sha256(path) != evidence["sha256"]:
                    raise ValueError(f"release parity evidence hash mismatch: {path}")
                _append_unique_evidence(legacy_evidence, evidence)
        legacy = dict(release)
        legacy["legacy_status"] = release.get("status")
        legacy["status"] = "parity_in_progress"
        legacy["requirements"] = {
            "empty_scene_128_gcd": False,
            "scene_100_decisions": False,
        }
        legacy["parity_reports"] = []
        legacy.pop("superseded_parity_reports", None)
        legacy["legacy_parity_reports"] = legacy_evidence
        return legacy
    evidence_by_scenario: dict[str, Mapping[str, object]] = {}
    referenced_filenames: set[str] = set()
    for entry in release.get("parity_reports", []):
        evidence = _mapping(entry, "release.parity_reports[]")
        filename = str(evidence["filename"])
        path = package_dir / filename
        if not path.is_file() or file_sha256(path) != evidence["sha256"]:
            raise ValueError(f"release parity evidence hash mismatch: {path}")
        if filename in referenced_filenames:
            raise ValueError(f"duplicate release parity filename: {filename}")
        referenced_filenames.add(filename)
        scenario = str(evidence.get("scenario"))
        if scenario in evidence_by_scenario:
            raise ValueError(f"duplicate release parity scenario: {scenario}")
        evidence_by_scenario[scenario] = evidence
    if audit_history:
        for entry in release.get("superseded_parity_reports", []):
            evidence = _mapping(entry, "release.superseded_parity_reports[]")
            filename = str(evidence["filename"])
            path = package_dir / filename
            if not path.is_file() or file_sha256(path) != evidence["sha256"]:
                raise ValueError(f"superseded parity evidence hash mismatch: {path}")
            if filename in referenced_filenames:
                raise ValueError(f"duplicate release parity filename: {filename}")
            referenced_filenames.add(filename)
    empty_passed = _evidence_passed(
        evidence_by_scenario.get("empty_128_gcd"),
        precision=manifest.contract.precision,
    )
    scene_passed = _evidence_passed(
        evidence_by_scenario.get("scene_100_decisions"),
        precision=manifest.contract.precision,
    )
    expected_requirements = {
        "empty_scene_128_gcd": empty_passed,
        "scene_100_decisions": scene_passed,
    }
    if release.get("requirements") != expected_requirements:
        raise ValueError("release requirements differ from parity evidence")
    any_failed = any(
        evidence.get("status") == "failed"
        for evidence in evidence_by_scenario.values()
    )
    expected_status = (
        "release_validated"
        if empty_passed and scene_passed
        else "parity_failed" if any_failed else "parity_in_progress"
    )
    if release.get("status") != expected_status:
        raise ValueError("release status differs from parity evidence")
    if audit_history:
        return release
    runtime_release = deepcopy(release)
    runtime_release.pop("superseded_parity_reports", None)
    return runtime_release


def parity_artifact_bindings_from_package(
    *,
    manifest: DeploymentManifest,
    manifest_path: Path,
) -> dict[str, object]:
    """从部署包恢复 bindings，不要求 checkpoint 文件随包分发。"""
    model = _mapping(manifest.payload["model"], "manifest.model")
    checkpoint = _mapping(manifest.payload["checkpoint"], "manifest.checkpoint")
    contract = _mapping(manifest.payload["contract"], "manifest.contract")
    signatures = _mapping(contract["signatures"], "manifest.contract.signatures")
    return {
        "manifest": {
            "filename": Path(manifest_path).name,
            "sha256": file_sha256(manifest_path),
        },
        "model": {
            "filename": str(model["filename"]),
            "sha256": str(model["sha256"]),
        },
        "checkpoint": {
            "filename": str(checkpoint["filename"]),
            "sha256": str(checkpoint["sha256"]),
        },
        "deployment_contract_sha256": str(
            signatures["deployment_contract_sha256"]
        ),
    }


def package_runtime_targets(
    *,
    package_dir: Path,
    manifest: DeploymentManifest,
) -> dict[str, object]:
    """从固定依赖和导出报告恢复当前部署包的目标 ORT 矩阵。"""
    report_path = Path(package_dir) / "export_report.json"
    report = _load_json(report_path, "export report")
    existing = report.get("runtime_targets")
    if isinstance(existing, Mapping):
        validate_runtime_targets(
            existing,
            precision=manifest.contract.precision,
        )
        return dict(existing)
    exporter = _mapping(manifest.payload["exporter"], "manifest.exporter")
    provider = (
        ORT_PROVIDER_CUDA
        if manifest.contract.precision == "bf16"
        else str(report["ort_provider"])
    )
    return runtime_targets(
        precision=manifest.contract.precision,
        ort_version=str(exporter["onnxruntime"]),
        provider=provider,
    )


def _requested_release_scenario(report: Mapping[str, object]) -> str:
    request = _mapping(report.get("request"), "parity.request")
    return validate_release_scenario_request(
        scene_mode=str(report.get("scene_mode")),
        max_steps=int(request.get("max_steps") or 0),
        max_gcds=(
            None
            if request.get("max_gcds") is None
            else int(request["max_gcds"])
        ),
    )


def _existing_evidence(
    package_dir: Path,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    release_path = package_dir / RELEASE_REPORT_FILENAME
    if not release_path.is_file():
        return {}, []
    existing = verify_release(
        package_dir,
        require_validated=False,
        audit_history=True,
    )
    result: dict[str, dict[str, object]] = {}
    for item in existing.get("parity_reports", []):
        entry = dict(_mapping(item, "release.parity_reports[]"))
        result[str(entry["scenario"])] = entry
    superseded: list[dict[str, object]] = []
    for key in ("superseded_parity_reports", "legacy_parity_reports"):
        for item in existing.get(key, []):
            _append_unique_evidence(
                superseded,
                dict(_mapping(item, f"release.{key}[]")),
            )
    return result, superseded


def _append_unique_evidence(
    target: list[dict[str, object]],
    evidence: Mapping[str, object],
) -> None:
    identity = (str(evidence.get("filename")), str(evidence.get("sha256")))
    if any(
        (str(item.get("filename")), str(item.get("sha256"))) == identity
        for item in target
    ):
        return
    target.append(dict(evidence))


def _attach_exception_note(exc: object, note: str) -> None:
    """兼容 Python 3.10：无法附加异常 note 时至少写入 stderr。"""
    add_note = getattr(exc, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    print(note, file=sys.stderr)


def _evidence_entry(
    *,
    scenario: str,
    report: Mapping[str, object],
    path: Path,
) -> dict[str, object]:
    rollout = _mapping(report["rollout"], "parity.rollout")
    parity = _mapping(report["parity"], "parity.parity")
    return {
        "scenario": scenario,
        "filename": path.name,
        "sha256": file_sha256(path),
        "status": report["status"],
        "release_gate_version": RELEASE_GATE_VERSION,
        "precision": report["precision"],
        "tolerance": report["tolerance"],
        "scene_mode": report["scene_mode"],
        "decision_count": int(parity["decision_count"]),
        "output_gcds": rollout["output_gcds"],
        "max_abs_diff": parity["max_abs_diff"],
        "action_sequence_match": rollout["action_sequence_match"],
    }


def _evidence_passed(
    evidence: Mapping[str, object] | None,
    *,
    precision: str,
) -> bool:
    if evidence is None or evidence.get("status") != "passed":
        return False
    if int(evidence.get("release_gate_version", -1)) != RELEASE_GATE_VERSION:
        return False
    if evidence.get("precision") != precision:
        return False
    try:
        tolerance = float(evidence.get("tolerance"))
    except (TypeError, ValueError):
        return False
    if tolerance != parity_max_abs_tolerance(precision):
        return False
    scenario = evidence.get("scenario")
    if scenario == "empty_128_gcd":
        return int(evidence.get("output_gcds") or 0) >= 128
    if scenario == "scene_100_decisions":
        return int(evidence.get("decision_count") or 0) >= 100
    return False


def _validate_formal_release_report(
    report: Mapping[str, object],
    *,
    manifest: DeploymentManifest,
) -> None:
    gate = _mapping(report.get("release_gate"), "parity.release_gate")
    if gate.get("enabled") is not True:
        raise ValueError("parity report was not produced by the formal release gate")
    if int(gate.get("version", -1)) != RELEASE_GATE_VERSION:
        raise ValueError("unsupported parity release gate version")
    precision = manifest.contract.precision
    if report.get("precision") != precision:
        raise ValueError("parity report precision differs from deployment manifest")
    try:
        tolerance = float(report.get("tolerance"))
    except (TypeError, ValueError) as exc:
        raise ValueError("parity report tolerance must be numeric") from exc
    required = parity_max_abs_tolerance(precision)
    if tolerance != required:
        raise ValueError(
            "formal parity tolerance differs from the manifest precision policy: "
            f"expected {required}, got {tolerance}"
        )


def _load_json(path: Path, label: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value

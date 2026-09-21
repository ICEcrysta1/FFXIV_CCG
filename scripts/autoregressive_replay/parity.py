"""真实状态机轨迹上的 PyTorch / ORT 慢速一致性验收。"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

from .backends import OrtPolicyBackend, ParityPolicyBackend
from .config import AutoregressiveReplayConfig
from .replay import AutoregressiveReplay
from scripts.onnx_export.release import (
    RELEASE_GATE_VERSION,
    parity_artifact_bindings,
    package_runtime_targets,
    record_parity_result,
)
from scripts.onnx_export.precision import parity_max_abs_tolerance


def run_rollout_parity(
    config: AutoregressiveReplayConfig,
    *,
    onnx_package_path: Path,
    provider: str,
    output_path: Path,
    tolerance: float | None = None,
    release_gate: bool = False,
) -> Path:
    """使用一份状态机轨迹双跑 backend，并生成可审计 JSON 报告。"""
    if config.checkpoint_path is None:
        raise ValueError("rollout parity requires a PyTorch checkpoint")
    candidate = OrtPolicyBackend(onnx_package_path, provider=provider)
    resolved_tolerance = (
        parity_max_abs_tolerance(candidate.contract.precision)
        if tolerance is None
        else float(tolerance)
    )
    if not math.isfinite(resolved_tolerance) or resolved_tolerance < 0:
        raise ValueError("parity tolerance must be finite and >= 0")
    artifacts = parity_artifact_bindings(
        manifest=candidate.manifest,
        manifest_path=candidate.package_dir / "manifest.json",
        checkpoint_path=config.checkpoint_path,
    )
    reference_config = replace(
        config,
        backend="pytorch",
        use_kv_cache=False,
        onnx_package_path=None,
        policy_precision=candidate.contract.precision,
    )
    replay = AutoregressiveReplay(reference_config)
    parity_backend = ParityPolicyBackend(
        replay.backend,
        candidate,
        tolerance=resolved_tolerance,
    )
    replay.backend = parity_backend
    replay._configure_kv_cache(False)
    output_path = Path(output_path).resolve()
    result = None
    failure: Exception | None = None
    try:
        result = replay.run()
    except Exception as exc:  # 报告必须先落盘，再保持 CLI 非零退出。
        failure = exc

    parity_report = parity_backend.report()
    parity_rows = parity_report["decisions"]
    if failure is None and not parity_rows:
        failure = RuntimeError("rollout parity produced no comparable decisions")
    if failure is None and not parity_report["passed"]:
        first = parity_report["first_divergence"]
        failure = AssertionError(
            "policy backend parity failed at decision "
            f"{first['decision_index']}: candidate={first['max_diff_candidate']!r}, "
            f"reference_logit={first['reference_logit']:.8f}, "
            f"candidate_logit={first['candidate_logit']:.8f}, "
            f"max_abs_diff={first['max_abs_diff']:.8f}, "
            f"reference_top1={first['reference_top1']!r}, "
            f"candidate_top1={first['candidate_top1']!r}, "
            f"top1_match={first['top1_match']}, "
            f"reference_top3={first['reference_top3']!r}, "
            f"candidate_top3={first['candidate_top3']!r}, "
            f"top3_set_match={first['top3_set_match']}, "
            f"reference_final_action={first['reference_final_action']!r}, "
            f"candidate_final_action={first['candidate_final_action']!r}, "
            f"final_selection_match={first['final_selection_match']}"
        )
    first_action_divergence = next(
        (
            row
            for row in parity_rows
            if not row["final_selection_match"]
        ),
        None,
    )
    reference_actions = [
        str(row["reference_final_action"])
        for row in parity_rows
    ]
    report = {
        "status": "failed" if failure is not None else "passed",
        "error": (
            None
            if failure is None
            else {
                "type": type(failure).__name__,
                "message": str(failure),
            }
        ),
        "tolerance": resolved_tolerance,
        "precision": candidate.contract.precision,
        "release_gate": {
            "enabled": bool(release_gate),
            "version": RELEASE_GATE_VERSION,
        },
        "checkpoint": str(config.checkpoint_path),
        "onnx_package": str(Path(onnx_package_path).resolve()),
        "artifacts": artifacts,
        "runtime_targets": package_runtime_targets(
            package_dir=candidate.package_dir,
            manifest=candidate.manifest,
        ),
        "job_tag": replay.data_spec.job_tag,
        "scene_mode": config.scene_mode,
        "scene_json": str(config.scene_json_path),
        "history_capacity": config.max_history,
        "request": {
            "max_steps": config.max_steps,
            "max_gcds": config.max_gcds,
        },
        "rollout": {
            "action_count": (
                len(result.rows) if result is not None else len(reference_actions)
            ),
            "actions": (
                [row.action_key for row in result.rows]
                if result is not None
                else reference_actions
            ),
            "output_gcds": None if result is None else result.output_gcds,
            "cumulative_potency": (
                None if result is None else result.cumulative_potency
            ),
            "cumulative_dot_potency": (
                None if result is None else result.cumulative_dot_potency
            ),
            "ppg": None if result is None else result.ppg,
            "action_sequence_match": bool(parity_rows) and first_action_divergence is None,
            "first_divergence": first_action_divergence,
        },
        "parity": parity_report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if release_gate:
        record_parity_result(
            package_dir=candidate.package_dir,
            parity_report_path=output_path,
        )
    if failure is not None:
        raise AssertionError(
            f"rollout parity failed; audit report written to {output_path}: {failure}"
        ) from failure
    return output_path

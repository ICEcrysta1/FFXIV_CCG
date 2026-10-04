"""真实状态机轨迹上的 PyTorch / ORT 慢速一致性验收。"""

from __future__ import annotations

import json
import math
import logging
from dataclasses import replace
from pathlib import Path

from common.policy.data import ActionSpace
from scripts.onnx_export.release.release import (
    RELEASE_GATE_VERSION,
    package_runtime_targets,
    parity_artifact_bindings,
    record_parity_result,
)
from scripts.onnx_export.runtime.precision import parity_max_abs_tolerance

from .backends import OrtPolicyBackend, ParityPolicyBackend, PyTorchPolicyBackend
from .parallel import ParallelRollouts
from .replay import AutoregressiveReplay, AutoregressiveReplaySession, ReplayCacheStore


def run_rollout_parities(
    configs, *, onnx_package_path: Path, provider: str, output_paths,
    tolerance: float | None = None, release_gate: bool = False, workers=None,
) -> tuple[Path, ...]:
    """多个验收共用 PT/ORT 模型和引擎；报告独立，发布状态只由调用线程写入。"""
    configs = tuple(configs)
    paths = tuple(Path(path).resolve() for path in output_paths)
    if not configs or len(configs) != len(paths) or len(set(paths)) != len(paths):
        raise ValueError("parity configs require distinct matching output paths")
    first = configs[0]
    if first.checkpoint_path is None:
        raise ValueError("rollout parity requires a PyTorch checkpoint")
    if any(config.checkpoint_path != first.checkpoint_path or config.device != first.device for config in configs):
        raise ValueError("parallel parity requires one checkpoint and device")
    compared = OrtPolicyBackend(onnx_package_path, provider=provider)
    resolved_tolerance = parity_max_abs_tolerance(compared.contract.precision) if tolerance is None else float(tolerance)
    if not math.isfinite(resolved_tolerance) or resolved_tolerance < 0:
        raise ValueError("parity tolerance must be finite and >= 0")
    artifacts = parity_artifact_bindings(
        manifest=compared.manifest, manifest_path=compared.package_dir / "manifest.json",
        checkpoint_path=first.checkpoint_path,
    )
    reference = PyTorchPolicyBackend(
        first.checkpoint_path, device=first.device, use_kv_cache=False,
        precision=compared.contract.precision,
    )
    if compared.contract.precision == "bf16" and compared.compute_precision == "float32":
        reference.enable_bf16_float_compute()
    tasks = [(replace(config, backend="pytorch", use_kv_cache=False, onnx_package_path=None,
                      policy_precision=compared.contract.precision), path)
             for config, path in zip(configs, paths, strict=True)]
    failures = []
    with ParallelRollouts(reference, job_tag=reference.data_spec.job_tag, workers=workers) as pool:
        cache = ReplayCacheStore(max_shards=first.cache_max_shards, max_readers=pool.workers)

        def prepare(items, engine):
            try:
                cache.prepare(
                    [config for config, _ in items], job_tag=reference.data_spec.job_tag,
                    normalizer=reference.input_contract.create_normalizer(), engine=engine, workers=pool.workers,
                    expected_action_space=ActionSpace.from_data_spec(reference.data_spec),
                    expected_skill_vocab=reference.input_contract.create_skill_vocab(),
                )
            except Exception:
                # 每个会话仍使用本引擎重试自己的来源，并将失败写入独立报告。
                logging.getLogger(__name__).warning("parity 批量缓存准备失败，继续逐场景验收", exc_info=True)

        def policy_factory(_item):
            # 每条队列独立记录对齐结果，底层模型只加载一份。
            return ParityPolicyBackend(reference, compared, tolerance=resolved_tolerance)

        def run(item, policy, engine):
            config, path = item
            result = None
            failure = None
            try:
                with AutoregressiveReplaySession(config, backend=policy, cache_store=cache, engine=engine) as session:
                    result = AutoregressiveReplay(config, session=session).run()
            except Exception as exc:
                failure = exc
            return _build_report(
                config, policy, result, failure, compared=compared,
                onnx_package_path=onnx_package_path, artifacts=artifacts,
                resolved_tolerance=resolved_tolerance, release_gate=release_gate,
            ), path

        for report, path in pool.map(run, tasks, prepare=prepare, policy_factory=policy_factory, isolate_errors=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
            if release_gate:
                try:
                    record_parity_result(package_dir=compared.package_dir, parity_report_path=path)
                except Exception as exc:
                    failures.append(f"{path}: release state update failed: {exc}")
            if report["status"] != "passed":
                failures.append(f"{path}: {report['error']['message']}")
    if failures:
        raise AssertionError("rollout parity failed; audit report written to " + "; ".join(failures))
    return paths


def _build_report(config, parity_backend, result, failure, *, compared,
                  onnx_package_path, artifacts, resolved_tolerance, release_gate):
    parity_report = parity_backend.report()
    parity_rows = parity_report["decisions"]
    if failure is None and not parity_rows:
        failure = RuntimeError("rollout parity produced no comparable decisions")
    if failure is None and not parity_report["passed"]:
        first = parity_report["first_divergence"]
        failure = AssertionError(
            "policy backend parity failed at decision "
            f"{first['decision_index']}: compared={first['max_diff_action']!r}, "
            f"reference_logit={first['reference_logit']:.8f}, "
            f"compared_logit={first['compared_logit']:.8f}, "
            f"max_abs_diff={first['max_abs_diff']:.8f}, "
            f"reference_top1={first['reference_top1']!r}, "
            f"compared_top1={first['compared_top1']!r}, "
            f"top1_match={first['top1_match']}, "
            f"reference_top3={first['reference_top3']!r}, "
            f"compared_top3={first['compared_top3']!r}, "
            f"top3_set_match={first['top3_set_match']}, "
            f"reference_final_action={first['reference_final_action']!r}, "
            f"compared_final_action={first['compared_final_action']!r}, "
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
        "precision": compared.contract.precision,
        "release_gate": {
            "enabled": bool(release_gate),
            "version": RELEASE_GATE_VERSION,
        },
        "checkpoint": str(config.checkpoint_path),
        "onnx_package": str(Path(onnx_package_path).resolve()),
        "artifacts": artifacts,
        "runtime_targets": package_runtime_targets(
            package_dir=compared.package_dir,
            manifest=compared.manifest,
        ),
        "job_tag": parity_backend.data_spec.job_tag,
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
    return report

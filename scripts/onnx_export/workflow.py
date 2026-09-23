"""由根目录 PowerShell 菜单调用的 ONNX 发布工作流。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.autoregressive_replay.config import load_replay_config
from scripts.autoregressive_replay.outputs import write_markdown
from scripts.autoregressive_replay.parity import run_rollout_parity
from scripts.autoregressive_replay.replay import AutoregressiveReplay

from .config.config import load_export_config, load_parity_config
from .contracts.deployment_contract import DeploymentManifest
from .export import export_from_config
from .release.policy import validate_release_scenario_request
from .release.release import verify_release
from .runtime.ort_runtime import ORT_PROVIDER_CUDA
from .runtime.precision import PRECISION_BF16, parity_max_abs_tolerance
from .runtime.runtime_targets import validate_bf16_export_environment_versions

WORKFLOW_ACTIONS = (
    "all",
    "env",
    "empty-parity",
    "scene-parity",
    "verify",
    "run",
)


def _load_export_config(checkpoint: Path | None):
    if checkpoint is None:
        return load_export_config()
    return load_export_config(checkpoint=checkpoint)


def main() -> int:
    parser = argparse.ArgumentParser(description="执行 ONNX 导出后的发布工作流")
    parser.add_argument("action", choices=WORKFLOW_ACTIONS)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="显式选择导出与 parity 使用的 checkpoint；默认读取 .env/YAML",
    )
    args = parser.parse_args()
    if args.action == "all":
        return _run_all(checkpoint=args.checkpoint)
    if args.action == "env":
        check_environment(checkpoint=args.checkpoint)
    elif args.action == "empty-parity":
        run_parity("empty", checkpoint=args.checkpoint)
    elif args.action == "scene-parity":
        run_parity("scene", checkpoint=args.checkpoint)
    elif args.action == "verify":
        print_release_status(checkpoint=args.checkpoint)
    elif args.action == "run":
        run_onnx_replay(checkpoint=args.checkpoint)
    return 0


def run_export(*, checkpoint: Path | None = None) -> None:
    """按 `.env` 导出图；与无参数 `python -m scripts.onnx_export` 等价。"""
    config = _load_export_config(checkpoint)
    output = export_from_config(config)
    print(f"ONNX 图验证完成，仍需执行 rollout parity：{output}")


def check_environment(*, checkpoint: Path | None = None) -> None:
    """按导出精度检查当前 PyTorch/ONNX Runtime 环境。"""
    import onnx
    import onnxruntime as ort
    import onnxscript
    import torch

    config = _load_export_config(checkpoint)
    providers = ort.get_available_providers()
    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"onnx={onnx.__version__}")
    print(f"onnxscript={onnxscript.__version__}")
    print(f"onnxruntime={ort.__version__}")
    print(f"providers={providers}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(
        "bf16_supported="
        f"{torch.cuda.is_available() and torch.cuda.is_bf16_supported()}"
    )
    if config.precision != PRECISION_BF16:
        return
    validate_bf16_export_environment_versions(
        torch_version=torch.__version__,
        onnx_version=onnx.__version__,
        onnxscript_version=onnxscript.__version__,
        ort_version=ort.__version__,
    )
    if config.ort_provider != ORT_PROVIDER_CUDA or ORT_PROVIDER_CUDA not in providers:
        raise RuntimeError("正式 BF16 流程要求 CUDAExecutionProvider")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA 当前不可用")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA 设备不支持原生 BF16")


def run_parity(scenario: str, *, checkpoint: Path | None = None) -> None:
    """从 `.env` 装配并执行指定的正式 parity 门禁。"""
    export_config = _load_export_config(checkpoint)
    parity_config = load_parity_config()
    if scenario == "empty":
        scene_mode = "empty"
        max_steps = parity_config.empty_max_steps
        max_gcds = parity_config.empty_max_gcds
        output_path = parity_config.empty_report_path
    elif scenario == "scene":
        scene_mode = "cache"
        max_steps = parity_config.scene_max_steps
        max_gcds = None
        output_path = parity_config.scene_report_path
    else:
        raise ValueError(f"unsupported parity scenario: {scenario}")

    validate_release_scenario_request(
        scene_mode=scene_mode,
        max_steps=max_steps,
        max_gcds=max_gcds,
    )

    manifest = DeploymentManifest.load(
        export_config.output_dir / "manifest.json",
        verify_files=True,
    )
    manifest_precision = manifest.contract.precision
    if export_config.precision != manifest_precision:
        raise ValueError(
            "configured export precision differs from the existing deployment package: "
            f"{export_config.precision} != {manifest_precision}"
        )
    required_tolerance = parity_max_abs_tolerance(manifest_precision)
    if (
        parity_config.tolerance is not None
        and parity_config.tolerance != required_tolerance
    ):
        raise ValueError(
            "formal release parity tolerance is fixed by manifest precision: "
            f"{manifest_precision} requires {required_tolerance}, got "
            f"{parity_config.tolerance}"
        )

    replay_config = load_replay_config(
        checkpoint=export_config.checkpoint_path,
        backend="pytorch",
        ort_provider=export_config.ort_provider,
        scene_mode=scene_mode,
        max_steps=max_steps,
        max_gcds=max_gcds,
        temperature=0.0,
        top_p=1.0,
        use_kv_cache=False,
    )
    report_path = run_rollout_parity(
        replay_config,
        onnx_package_path=export_config.output_dir,
        provider=export_config.ort_provider,
        output_path=output_path,
        tolerance=required_tolerance,
        release_gate=True,
    )
    print(report_path)


def print_release_status(*, checkpoint: Path | None = None) -> None:
    """校验当前包并显示发布状态；失败 parity 状态仍允许被审计。"""
    config = _load_export_config(checkpoint)
    report = verify_release(
        config.output_dir,
        require_validated=False,
        audit_history=True,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "requirements": report["requirements"],
                "runtime_targets": report["runtime_targets"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_onnx_replay(*, checkpoint: Path | None = None) -> None:
    """使用共享 `.env` 参数执行普通 ONNX Runtime 回放。"""
    export_config = _load_export_config(checkpoint)
    replay_config = load_replay_config(
        backend="onnxruntime",
        onnx_package=export_config.output_dir,
        ort_provider=export_config.ort_provider,
    )
    output = write_markdown(
        AutoregressiveReplay(replay_config).run(),
        replay_config.output_path,
    )
    print(output)


def _run_all(*, checkpoint: Path | None = None) -> int:
    """顺序执行完整发布流程；各步骤统一接收可选的显式 checkpoint。"""
    try:
        check_environment(checkpoint=checkpoint)
    except Exception as exc:
        print(f"ONNX 环境检查失败，已停止发布流程：{exc}")
        return 1
    try:
        run_export(checkpoint=checkpoint)
    except Exception as exc:
        print(f"ONNX 导出失败，已停止发布流程：{exc}")
        return 1
    failures: list[str] = []
    for scenario in ("empty", "scene"):
        try:
            run_parity(scenario, checkpoint=checkpoint)
        except Exception as exc:
            failures.append(f"{scenario}: {exc}")
            print(f"{scenario} parity 未通过，已继续下一项：{exc}")
    print_release_status(checkpoint=checkpoint)
    if failures:
        print()
        print("ONNX 图已经导出并保留，但发布门禁未通过。")
        print("部署包状态为 parity_failed，不能声明与 PyTorch 逐决策等价。")
        for failure in failures:
            print(f"- {failure}")
        return 2
    print("完整流程通过，部署包已经 release_validated。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

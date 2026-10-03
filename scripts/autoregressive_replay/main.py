"""自回归回放命令行入口。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from scripts.onnx_export.runtime.precision import SUPPORTED_PRECISIONS

from .config import load_replay_config
from .outputs import write_markdown
from .parity import run_rollout_parity
from .replay import AutoregressiveReplay
from .batch_replay import run_replays


def main() -> None:
    parser = argparse.ArgumentParser(description="使用状态机和 checkpoint 执行模型自回归回放")
    parser.add_argument(
        "--backend",
        choices=("pytorch", "onnxruntime"),
        default=None,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--onnx-package", type=Path, default=None)
    parser.add_argument("--ort-provider", default=None)
    parser.add_argument(
        "--parity-onnx-package",
        type=Path,
        default=None,
        help="用同一状态机轨迹双跑 PyTorch/ORT，并输出 parity JSON",
    )
    parser.add_argument("--parity-output", type=Path, default=None)
    parser.add_argument(
        "--parity-tolerance",
        type=float,
        default=None,
        help="raw logits 最大绝对误差；默认按 manifest 精度选择",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--scene-json", type=Path, default=None)
    parser.add_argument("--scenes", type=Path, nargs="+", help="多个场景共用状态机引擎和模型，按输入顺序生成报告")
    parser.add_argument("--workers", type=int, default=None, help="最大同时回放队列数，默认读取 .env")
    parser.add_argument("--scene-mode", choices=("cache", "empty"), default=None)
    parser.add_argument("--scene-sample", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-gcds", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-history", type=int, default=None)
    parser.add_argument(
        "--history-ablation",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="先生成完整轨迹，再在相同决策点分别只保留最近 N 条历史",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--precision",
        choices=SUPPORTED_PRECISIONS,
        default=None,
        help=(
            "PyTorch 回放权重精度；可用 --device cpu --precision float32 "
            "显式分析 BF16 checkpoint，ONNX 精度仍以 manifest 为准"
        ),
    )
    parser.add_argument(
        "--use-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="PyTorch 可开启模型内部 KV cache；ONNX Runtime v1 必须关闭",
    )
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("--workers 必须是正整数")
    if args.scenes and (args.scene_json or args.parity_onnx_package or args.history_ablation):
        parser.error("--scenes 不能与 --scene-json、parity 或历史消融一起使用")

    config = load_replay_config(
        checkpoint=args.checkpoint,
        backend=args.backend,
        onnx_package=args.onnx_package,
        ort_provider=args.ort_provider,
        output=args.output,
        scene_json=args.scenes[0] if args.scenes else args.scene_json,
        scene_mode=args.scene_mode,
        scene_sample_index=args.scene_sample,
        max_steps=args.max_steps,
        max_gcds=args.max_gcds,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_history=args.max_history,
        device=args.device,
        use_kv_cache=args.use_kv_cache,
        policy_precision=args.precision,
    )
    if args.parity_onnx_package is not None:
        output_path = args.parity_output or config.output_path.with_suffix(".parity.json")
        print(
            run_rollout_parity(
                config,
                onnx_package_path=args.parity_onnx_package,
                provider=config.ort_provider,
                output_path=output_path,
                tolerance=args.parity_tolerance,
            )
        )
        return
    if not args.history_ablation:
        configs = [config] if not args.scenes else [
            replace(config, scene_json_path=path, scene_duration_seconds=None,
                    output_path=config.output_path.with_name(
                        f"{config.output_path.stem}_{index:03d}_{path.stem}{config.output_path.suffix}"
                    ))
            for index, path in enumerate(args.scenes)
        ]
        for item, result in zip(configs, run_replays(configs, workers=args.workers), strict=True):
            print(write_markdown(result, item.output_path))
        return
    with AutoregressiveReplay(config) as replay:
        results = replay.run_history_ablation(tuple(args.history_ablation))
        output_paths = [config.output_path]
        write_markdown(results[0], config.output_path)
        for result in results[1:]:
            output_path = config.output_path.with_name(
                f"{config.output_path.stem}_history_{result.history_limit}"
                f"{config.output_path.suffix}"
            )
            write_markdown(result, output_path)
            output_paths.append(output_path)
        for output_path in output_paths:
            print(output_path)
        return

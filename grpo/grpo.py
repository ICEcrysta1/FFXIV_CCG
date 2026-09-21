"""独立 GRPO 后训练入口。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
from pathlib import Path
import sys


if __package__ in {None, ""}:
    # 允许从项目根目录直接执行 `python grpo/grpo.py`。
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import torch

from common.project_config import resolve_project_path
from common.policy.config import (
    PROJECT_ROOT,
    resolve_policy_cache_dir,
    resolve_policy_checkpoint_path,
    resolve_policy_device,
    resolve_policy_model_config_path,
    resolve_policy_model_job_tag,
)
from grpo.config import load_grpo_config, load_grpo_run_config
from grpo.trainer import run_grpo_training


def _prepare_grpo_scenes(config, max_files: int | None) -> list[Path]:
    """编译真实训练集场景 cache，并返回实际可用的 raw 场景文件。"""
    if config.job_tag is None:
        raise ValueError("GRPO job_tag is required for real training scenes")
    from scripts.convert_fflogs.cache import prepare_training_caches

    return prepare_training_caches(
        config.raw_data_dir,
        max_files=max_files,
        job_tag=config.job_tag,
        int_dtype=torch.int32,
        float_dtype=torch.float32,
        cache_dir=resolve_policy_cache_dir(config.job_tag),
        shard_size=config.compiled_cache_shard_size,
        max_workers=config.compiled_cache_workers,
        max_shards=config.compiled_cache_max_shards,
    )


def _optional_override(value, current):
    return current if value is None else value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用真实训练集场景 token 执行贪心基准保护的 GRPO 后训练"
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="初始 BC/GRPO checkpoint；缺省读取 TRAINING_MODEL_CHECKPOINT 或 best.pt",
    )
    parser.add_argument("--raw-data-dir", type=Path, default=None)
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="用于 GRPO 的真实训练场景文件数量；缺省使用配置目录下全部有效文件",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--precision", choices=("float32", "float16", "bf16"), default=None)
    parser.add_argument(
        "--samples-per-scene",
        "--group-size",
        dest="group_size",
        type=int,
        default=None,
        help="每个场景采样的轨迹条数（默认 16），不含贪心基准",
    )
    parser.add_argument("--prompt-batch-size", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=None,
        help="统一的自回归时间上限；缺省运行到每个场景自身的战斗结束时间",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--inner-updates", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--clip-low", type=float, default=None)
    parser.add_argument("--clip-high", type=float, default=None)
    parser.add_argument("--kl-coefficient", type=float, default=None)
    parser.add_argument("--greedy-guard-tolerance", type=float, default=None)
    parser.add_argument("--advantage-scale-floor-ratio", type=float, default=None)
    parser.add_argument("--advantage-clip", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be >= 1")

    config_path = resolve_policy_model_config_path(args.config)
    config = load_grpo_run_config(config_path)
    grpo_config = load_grpo_config(config_path)
    output_dir = (
        resolve_project_path(args.output_dir, project_root=PROJECT_ROOT)
        if args.output_dir is not None
        else None
    )
    config = replace(
        config,
        job_tag=resolve_policy_model_job_tag(config_path),
        **(
            {"raw_data_dir": resolve_project_path(args.raw_data_dir, project_root=PROJECT_ROOT)}
            if args.raw_data_dir is not None
            else {}
        ),
        **(
            {"output_dir": output_dir}
            if output_dir is not None
            else {}
        ),
    )
    grpo = replace(
        grpo_config,
        group_size=_optional_override(args.group_size, grpo_config.group_size),
        prompt_batch_size=_optional_override(
            args.prompt_batch_size,
            grpo_config.prompt_batch_size,
        ),
        max_iterations=_optional_override(args.iterations, grpo_config.max_iterations),
        max_duration_seconds=_optional_override(
            args.max_duration_seconds,
            grpo_config.max_duration_seconds,
        ),
        temperature=_optional_override(args.temperature, grpo_config.temperature),
        top_p=_optional_override(args.top_p, grpo_config.top_p),
        inner_updates=_optional_override(args.inner_updates, grpo_config.inner_updates),
        minibatch_size=_optional_override(args.minibatch_size, grpo_config.minibatch_size),
        learning_rate=_optional_override(args.lr, grpo_config.learning_rate),
        warmup_steps=_optional_override(args.warmup_steps, grpo_config.warmup_steps),
        clip_low=_optional_override(args.clip_low, grpo_config.clip_low),
        clip_high=_optional_override(args.clip_high, grpo_config.clip_high),
        kl_coefficient=_optional_override(
            args.kl_coefficient,
            grpo_config.kl_coefficient,
        ),
        greedy_guard_tolerance=_optional_override(
            args.greedy_guard_tolerance,
            grpo_config.greedy_guard_tolerance,
        ),
        advantage_scale_floor_ratio=_optional_override(
            args.advantage_scale_floor_ratio,
            grpo_config.advantage_scale_floor_ratio,
        ),
        advantage_clip=_optional_override(
            args.advantage_clip,
            grpo_config.advantage_clip,
        ),
    )

    checkpoint_path = (
        resolve_project_path(args.checkpoint, project_root=PROJECT_ROOT)
        if args.checkpoint is not None
        else resolve_policy_checkpoint_path(config_path)
    )
    raw_paths = _prepare_grpo_scenes(config, args.max_files)
    if not raw_paths:
        raise FileNotFoundError("--max-files selected no valid real training scenes")

    result = run_grpo_training(
        config,
        grpo=grpo,
        checkpoint_path=checkpoint_path,
        raw_paths=raw_paths,
        output_dir=output_dir,
        device_name=resolve_policy_device(args.device),
        precision=args.precision,
    )
    print(result["final_checkpoint"])


if __name__ == "__main__":
    main()

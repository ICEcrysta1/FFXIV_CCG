"""raw FFLogs JSON 直接编译为最终 compiled cache 的 CLI。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from common.policy.data import Normalizer
from common.policy.config import (
    resolve_policy_cache_dir,
    resolve_policy_model_config_path,
    resolve_policy_model_job_tag,
    resolve_policy_model_variant,
)
from training.config import load_run_config

from .cache import precompile_raw_training_caches
from .config import load_convert_fflogs_config, load_convert_fflogs_dotenv, resolve_convert_fflogs_job_tag
from .constants import DEFAULT_DOWNTIME_GAP_SECONDS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def main() -> None:
    """读取 raw JSON，并把最终训练样本写入职业 `.cache`。"""
    load_convert_fflogs_dotenv()
    parser = argparse.ArgumentParser(description="raw FFLogs JSON -> compiled cache")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="raw JSON 文件或目录；省略时读取训练 YAML 的 raw_data_dir",
    )
    parser.add_argument("--job-tag", default=None, help="用于解析动作的职业标签")
    parser.add_argument("--source", type=int, default=None, help="覆盖 JSON 中的 source_id")
    parser.add_argument("--encounter", default=None, help="覆盖 JSON 中的副本名称")
    parser.add_argument("--cache-root", type=Path, default=None, help="compiled cache 根目录")
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--downtime-gap",
        type=float,
        default=DEFAULT_DOWNTIME_GAP_SECONDS,
        help=f"downtime 判定的伤害间隔阈值，默认 {DEFAULT_DOWNTIME_GAP_SECONDS}",
    )
    args = parser.parse_args()

    model_config_path = resolve_policy_model_config_path()
    run_config = load_run_config(model_config_path)
    configured_job_tag = resolve_policy_model_job_tag(model_config_path)
    resolve_policy_model_variant(model_config_path)
    resolved_job_tag = resolve_convert_fflogs_job_tag(args.job_tag)
    if configured_job_tag != resolved_job_tag:
        raise ValueError(
            f"job_tag {resolved_job_tag!r} does not match training model job {configured_job_tag!r}"
        )

    raw_root = run_config.raw_data_dir
    input_paths = _resolve_input_files(args.inputs or [raw_root])
    if not input_paths:
        logger.warning("没有找到 raw JSON 输入文件: %s", raw_root)
        return

    cache_dir = (
        args.cache_root.resolve()
        if args.cache_root is not None
        else resolve_policy_cache_dir(resolved_job_tag)
    )
    shard_size = (
        run_config.compiled_cache_shard_size
        if args.shard_size is None
        else int(args.shard_size)
    )
    workers = (
        load_convert_fflogs_config().default_worker_count
        if args.workers is None
        else int(args.workers)
    )
    valid_paths = precompile_raw_training_caches(
        input_paths,
        job_tag=resolved_job_tag,
        source=args.source,
        encounter=args.encounter,
        downtime_gap_seconds=float(args.downtime_gap),
        normalizer=Normalizer(),
        int_dtype=torch.int32,
        float_dtype=torch.float32,
        cache_dir=cache_dir,
        shard_size=shard_size,
        max_workers=workers,
        max_shards=run_config.compiled_cache_max_shards,
    )
    logger.info("raw JSON 编译完成: %d 个文件 -> %s", len(valid_paths), cache_dir)


def _resolve_input_files(inputs: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw_input in inputs:
        path = Path(raw_input).resolve()
        if path.is_file():
            if path.suffix.lower() == ".json":
                files.append(path)
            continue
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json")))
            continue
        raise FileNotFoundError(f"raw JSON input not found: {path}")
    return sorted(set(files), key=lambda item: str(item).casefold())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()

"""FFLogs JSON（可含评估标签）编译为最终 compiled cache 的 CLI。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from common.config import load_precision_config
from common.dataset_layout import find_dataset_json_files
from scripts.common.json_io import is_json_file
from common.policy.config import (
    resolve_policy_cache_dir,
    resolve_policy_model_config_path,
    resolve_policy_model_job_tag,
    resolve_policy_model_variant,
)
from common.policy.data import Normalizer
from common.project_config import resolve_positive_worker_count
from training.config import load_run_config

from .cache import prepare_training_caches, precompile_raw_training_caches
from .config import (
    load_convert_fflogs_dotenv,
    resolve_convert_fflogs_job_tag,
)
from .config.constants import DEFAULT_DOWNTIME_GAP_SECONDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def main() -> None:
    """读取配置的数据阶段，并把最终训练样本写入职业 `.cache`。"""
    load_convert_fflogs_dotenv()
    parser = argparse.ArgumentParser(description="FFLogs JSON -> compiled cache")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="FFLogs JSON 文件或目录；省略时读取模型 YAML 的 raw_data_dir",
    )
    parser.add_argument("--job-tag", default=None, help="用于解析动作的职业标签")
    parser.add_argument("--source", type=int, default=None, help="覆盖 JSON 中的 source_id")
    parser.add_argument("--encounter", default=None, help="覆盖 JSON 中的副本名称")
    parser.add_argument("--cache-root", type=Path, default=None, help="compiled cache 根目录")
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--training-selection", action="store_true",
        help="按 training.max_files 的配额选择并编译训练文件，失败时按副本和区间补位",
    )
    parser.add_argument("--max-files", type=int, default=None, help="覆盖 training.max_files；仅用于 --training-selection")
    parser.add_argument(
        "--downtime-gap",
        type=float,
        default=DEFAULT_DOWNTIME_GAP_SECONDS,
        help=f"downtime 判定的伤害间隔阈值，默认 {DEFAULT_DOWNTIME_GAP_SECONDS}",
    )
    args = parser.parse_args()
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be >= 1")
    if args.max_files is not None and not args.training_selection:
        parser.error("--max-files requires --training-selection")
    if args.training_selection and (args.inputs or args.source is not None or args.encounter is not None):
        parser.error("--training-selection uses the configured input directory and source metadata")

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
        resolve_positive_worker_count(
            project_root=PROJECT_ROOT, env_name="CONVERT_FFLOGS_WORKERS",
        )
        if args.workers is None else int(args.workers)
    )
    if workers < 1:
        parser.error("--workers must be >= 1")
    precision = load_precision_config()
    if args.training_selection:
        valid_paths = prepare_training_caches(
            raw_root,
            max_files=run_config.max_files if args.max_files is None else args.max_files,
            job_tag=resolved_job_tag,
            int_dtype=precision.resolve_int_dtype(),
            float_dtype=precision.resolve_float_dtype(),
            cache_dir=cache_dir,
            shard_size=shard_size,
            max_workers=workers,
            max_shards=run_config.compiled_cache_max_shards,
            downtime_gap_seconds=float(args.downtime_gap),
        )
        if not valid_paths:
            raise FileNotFoundError(f"没有找到可编译的训练 JSON：{raw_root}")
        logger.info("训练文件转换完成: %d 个文件 -> %s", len(valid_paths), cache_dir)
        return

    input_paths = _resolve_input_files(args.inputs or [raw_root])
    if not input_paths:
        raise FileNotFoundError(f"没有找到 FFLogs JSON 输入文件: {raw_root}")
    valid_paths = precompile_raw_training_caches(
        input_paths,
        job_tag=resolved_job_tag,
        source=args.source,
        encounter=args.encounter,
        downtime_gap_seconds=float(args.downtime_gap),
        normalizer=Normalizer(),
        int_dtype=precision.resolve_int_dtype(),
        float_dtype=precision.resolve_float_dtype(),
        cache_dir=cache_dir,
        shard_size=shard_size,
        max_workers=workers,
        max_shards=run_config.compiled_cache_max_shards,
    )
    if not valid_paths:
        raise RuntimeError(f"{len(input_paths)} 份 FFLogs JSON 均未编译成功")
    logger.info("FFLogs JSON 编译完成: %d 个文件 -> %s", len(valid_paths), cache_dir)


def _resolve_input_files(inputs: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw_input in inputs:
        path = Path(raw_input).resolve()
        if path.is_file():
            if is_json_file(path):
                files.append(path)
            continue
        if path.is_dir():
            files.extend(find_dataset_json_files(path))
            continue
        raise FileNotFoundError(f"FFLogs JSON input not found: {path}")
    return sorted(set(files), key=lambda item: str(item).casefold())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()

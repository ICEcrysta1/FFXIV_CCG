"""公共行为克隆训练入口。"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
import sys


if __package__ in {None, ""}:
    # 允许从项目根目录直接执行 `python training/train.py`。
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from common.policy.config import (
    resolve_policy_cache_dir,
    resolve_policy_device,
    resolve_policy_model_config_path,
    resolve_policy_model_job_tag,
    resolve_policy_model_variant,
)
from training.config import load_run_config
from training.loop import run_training


def _prepare_training_caches(config, max_files: int | None) -> list[Path]:
    """调用脚本层入口准备训练所需的 compiled cache。"""
    if config.job_tag is None:
        raise ValueError("training job_tag is required for raw JSON conversion")
    import torch

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


def _evaluate_training_metrics(*, model, dataset, data_spec, vocab, config, device):
    """按训练约定依次计算验证集 PPG 与空场景 PPG。"""
    from scripts.autoregressive_replay.ppg import (
        evaluate_none_ppg,
        evaluate_validation_ppg,
    )

    # 验证 PPG 使用真实副本 scene 和初始状态执行模型自回归回放。
    metrics = evaluate_validation_ppg(
        model=model,
        dataset=dataset,
        data_spec=data_spec,
        vocab=vocab,
        config=config,
        device=device,
    )
    # 空场景 PPG 才执行模型的自回归状态机回放。
    metrics.update(
        evaluate_none_ppg(
            model=model,
            dataset=dataset,
            data_spec=data_spec,
            vocab=vocab,
            config=config,
            device=device,
        )
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="训练职业行为克隆模型")
    parser.add_argument("--config", type=Path, default=None, help="职业 YAML 配置；默认读取根目录 .env")
    parser.add_argument("--raw-data-dir", type=Path, default=None, help="覆盖 YAML 中的 raw JSON 目录")
    parser.add_argument("--output-dir", type=Path, default=None, help="覆盖 YAML 中的 checkpoint 目录")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 batch size")
    parser.add_argument("--lr", type=float, default=None, help="覆盖学习率")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="覆盖 training.max_files：按副本目录比例最多使用 N 个 raw JSON 文件；省略时读取 YAML",
    )
    parser.add_argument("--device", default=None, help="覆盖根目录 .env 中的训练设备：cuda 或 cpu")
    parser.add_argument("--resume", type=Path, default=None, help="从指定 checkpoint 继续训练")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config_path = resolve_policy_model_config_path(args.config)
    config = replace(
        load_run_config(config_path),
        job_tag=resolve_policy_model_job_tag(config_path),
        model_variant=resolve_policy_model_variant(config_path),
        **({"raw_data_dir": args.raw_data_dir} if args.raw_data_dir is not None else {}),
    )
    max_files = args.max_files if args.max_files is not None else config.max_files
    if max_files is not None and max_files < 1:
        raise ValueError("--max-files must be >= 1")
    logging.info(
        "raw JSON 数量上限: %s（来源: %s）",
        "全部有效文件" if max_files is None else max_files,
        "--max-files" if args.max_files is not None else "training.max_files",
    )
    raw_paths = _prepare_training_caches(config, max_files)
    training_kwargs = {
        "raw_paths": raw_paths,
        "output_dir": args.output_dir,
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device_name": resolve_policy_device(args.device),
        "resume_path": args.resume,
    }
    if getattr(config, "ppg", None) is not None and config.ppg.enabled:
        training_kwargs["validation_metrics_callback"] = _evaluate_training_metrics
    result = run_training(config, **training_kwargs)
    spec = result["data_spec"]
    logging.info(
        "训练完成: job=%s model_variant=%s candidates=%d state_dim=%d scene_dim=%d skill_dim=%d output=%s",
        spec.job_tag,
        config.model_variant,
        spec.num_candidates,
        spec.state_dim,
        spec.scene_dim,
        spec.skill_feature_dim,
        result["output_dir"],
    )


if __name__ == "__main__":
    main()

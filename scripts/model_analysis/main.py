"""科研模型分析中心入口。"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from common.policy.config import (
    resolve_policy_cache_dir,
    resolve_policy_checkpoint_path,
    resolve_policy_model_config_path,
    resolve_policy_model_job_tag,
    resolve_policy_model_variant,
    validate_policy_model_variant,
)
from common.torch_serialization import safe_torch_load
from training.config import load_run_config

from .common import (
    configure_matplotlib,
    load_analysis_context,
    load_loss_landscape_context,
)
from .outputs import (
    plot_hidden_statistics,
    plot_layer_pca,
    plot_loss_landscape,
    plot_opener_attention,
    plot_standard_attention_outputs,
    plot_pair_embedding,
    plot_skill_embedding,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成模型科研分析 PNG")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="模型 checkpoint；默认读取模型 YAML 输出目录中的 .env checkpoint",
    )
    parser.add_argument("--model-config", type=Path, default=None, help="模型 YAML；默认读取根目录 .env")
    parser.add_argument("--raw-json", type=Path, default=None, help="raw JSON；默认按训练 YAML 自动寻找")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/model_analysis")
    parser.add_argument("--max-samples", type=int, default=256, help="用于分析的样本数，<=0 表示整份 compiled cache")
    parser.add_argument("--max-tokens", type=int, default=20000, help="每层最多保留的有效 token 数")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--loss-landscape",
        action="store_true",
        help="导出逐 Transformer 层损失地图；计算量与层数和分辨率平方成正比",
    )
    parser.add_argument("--loss-landscape-only", action="store_true", help="仅导出损失地图，跳过其他分析图")
    parser.add_argument(
        "--loss-landscape-resolution",
        type=int,
        default=31,
        help="逐层损失地图单轴采样点数；必须是 >=3 的奇数，默认 31",
    )
    parser.add_argument(
        "--loss-landscape-radius",
        type=float,
        default=0.5,
        help="filter-normalized 参数方向的正负扰动半径，默认 0.5",
    )
    parser.add_argument(
        "--loss-landscape-max-samples",
        type=int,
        default=None,
        help="损失地图固定样本数；默认复用 --max-samples，0 表示完整数据集",
    )
    parser.add_argument(
        "--loss-landscape-seed",
        type=int,
        default=3407,
        help="逐层正交参数方向的可复现随机种子",
    )
    parser.add_argument(
        "--attention-steps",
        type=int,
        default=28,
        help="开场注意力图使用 compiled cache 中前 N 个真实决策样本；默认 28，和旧脚本开场长度一致",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    configure_matplotlib()
    model_config_path = resolve_policy_model_config_path(args.model_config)
    checkpoint_path = args.checkpoint or resolve_policy_checkpoint_path(model_config_path)
    run_config = load_run_config(model_config_path)
    job_tag = resolve_policy_model_job_tag(model_config_path)
    model_variant = resolve_policy_model_variant(model_config_path)
    checkpoint_payload = safe_torch_load(checkpoint_path)
    if not isinstance(checkpoint_payload, dict):
        raise ValueError(f"checkpoint must be a mapping: {checkpoint_path}")
    validate_policy_model_variant(
        checkpoint_payload,
        model_variant,
        artifact_name="checkpoint",
    )
    raw_root = run_config.raw_data_dir
    cache_dir = resolve_policy_cache_dir(job_tag)
    device_name = "cuda" if args.device == "auto" else args.device
    if args.loss_landscape_only:
        context = load_loss_landscape_context(
            checkpoint_path=checkpoint_path,
            source_path=args.raw_json,
            raw_root=raw_root,
            cache_dir=cache_dir,
            max_history=run_config.model.history_capacity,
            cache_shard_size=run_config.compiled_cache_shard_size,
            cache_max_shards=run_config.compiled_cache_max_shards,
            candidate_order_file=run_config.candidate_order_file,
            output_dir=args.output,
            device_name=device_name,
            precision=run_config.precision,
        )
        paths = plot_loss_landscape(
            context,
            resolution=args.loss_landscape_resolution,
            radius=args.loss_landscape_radius,
            max_samples=(args.max_samples if args.loss_landscape_max_samples is None
                         else args.loss_landscape_max_samples),
            batch_size=args.batch_size,
            seed=args.loss_landscape_seed,
        )
        for path in paths:
            print(path)
        return
    context = load_analysis_context(
        checkpoint_path=checkpoint_path,
        source_path=args.raw_json,
        raw_root=raw_root,
        cache_dir=cache_dir,
        max_history=run_config.model.history_capacity,
        cache_shard_size=run_config.compiled_cache_shard_size,
        cache_max_shards=run_config.compiled_cache_max_shards,
        candidate_order_file=run_config.candidate_order_file,
        output_dir=args.output,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        device_name=device_name,
        precision=run_config.precision,
    )
    outputs = []
    outputs.extend(plot_hidden_statistics(context))
    outputs.extend(plot_layer_pca(context))
    outputs.extend(
        plot_opener_attention(
            context,
            steps=args.attention_steps,
            batch_size=args.batch_size,
        )
    )
    outputs.extend(
        plot_standard_attention_outputs(
            context,
            steps=args.attention_steps,
        )
    )
    outputs.append(plot_skill_embedding(context))
    outputs.append(plot_pair_embedding(context, batch_size=args.batch_size))
    loss_landscape_max_samples = None
    metadata = {
        "checkpoint": str(context.checkpoint_path),
        "model_config": str(model_config_path),
        "raw_json": str(context.source_path),
        "job_tag": context.data_spec.job_tag,
        "device": str(context.device),
        "precision": context.precision,
        "num_samples": min(args.max_samples, len(context.dataset)) if args.max_samples > 0 else len(context.dataset),
        "num_candidates": context.data_spec.num_candidates,
        "state_dim": context.data_spec.state_dim,
        "scene_dim": context.data_spec.scene_dim,
        "skill_feature_dim": context.data_spec.skill_feature_dim,
        "layer_count": len(context.layer_vectors),
        "layer_token_counts": [int(len(values)) for values in context.layer_vectors],
    }
    if args.loss_landscape:
        loss_landscape_max_samples = (
            args.max_samples
            if args.loss_landscape_max_samples is None
            else args.loss_landscape_max_samples
        )
        source_path = context.source_path
        analysis_device = context.device
        # loss 仍属于同一次导出，但不应继承 hidden/PCA/attention 的重对象。
        # 先释放完整上下文，再以同一 YAML precision 加载轻量模型和 dataset。
        del context
        gc.collect()
        if analysis_device.type == "cuda":
            torch.cuda.empty_cache()
        loss_context = load_loss_landscape_context(
            checkpoint_path=checkpoint_path,
            source_path=source_path,
            raw_root=raw_root,
            cache_dir=cache_dir,
            max_history=run_config.model.history_capacity,
            cache_shard_size=run_config.compiled_cache_shard_size,
            cache_max_shards=run_config.compiled_cache_max_shards,
            candidate_order_file=run_config.candidate_order_file,
            output_dir=args.output,
            device_name=device_name,
            precision=run_config.precision,
        )
        outputs.extend(
            plot_loss_landscape(
                loss_context,
                resolution=args.loss_landscape_resolution,
                radius=args.loss_landscape_radius,
                max_samples=loss_landscape_max_samples,
                batch_size=args.batch_size,
                seed=args.loss_landscape_seed,
            )
        )
        del loss_context
        gc.collect()
        if analysis_device.type == "cuda":
            torch.cuda.empty_cache()
    metadata["loss_landscape"] = {
        "enabled": args.loss_landscape,
        "resolution": args.loss_landscape_resolution,
        "radius": args.loss_landscape_radius,
        "max_samples": loss_landscape_max_samples,
        "seed": args.loss_landscape_seed,
    }
    metadata["outputs"] = [str(path) for path in outputs]
    metadata_path = args.output / "analysis_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    for path in outputs:
        print(path)
    print(metadata_path)


if __name__ == "__main__":
    main()

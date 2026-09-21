"""独立 GRPO 后训练配置。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from common.policy.config import (
    PROJECT_ROOT,
    ModelConfig,
    load_model_config,
    load_policy_config,
)
from common.project_config import resolve_project_path


@dataclass(frozen=True)
class GrpoConfig:
    """控制 GRPO rollout、损失和优化参数。"""

    # 每个场景先完成 16 条无梯度自回归轨迹，再统一计算优势并更新策略。
    group_size: int = 16
    prompt_batch_size: int = 1
    max_iterations: int = 10
    # None 表示跟随每个 raw 场景自身的战斗结束时间；也可配置统一秒数上限。
    max_duration_seconds: float | None = None
    # 提高 rollout 的采样温度，避免候选分布过早塌缩到少数动作。
    temperature: float = 1.3
    top_p: float = 1.0
    inner_updates: int = 1
    minibatch_size: int = 64
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 20
    clip_low: float = 0.2
    clip_high: float = 0.2
    kl_coefficient: float = 0.001
    greedy_guard_tolerance: float = 0.0
    advantage_scale_floor_ratio: float = 0.1
    advantage_clip: float = 5.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object] | None = None) -> "GrpoConfig":
        """从拆分后的 `grpo` mapping 构造并校验配置。"""
        values = {} if raw is None else raw
        if not isinstance(values, Mapping):
            raise ValueError("grpo must be a mapping")
        return cls(
            group_size=int(values.get("group_size", cls.group_size)),
            prompt_batch_size=int(
                values.get("prompt_batch_size", cls.prompt_batch_size)
            ),
            max_iterations=int(values.get("max_iterations", cls.max_iterations)),
            max_duration_seconds=(
                None
                if values.get("max_duration_seconds") is None
                else float(values["max_duration_seconds"])
            ),
            temperature=float(values.get("temperature", cls.temperature)),
            top_p=float(values.get("top_p", cls.top_p)),
            inner_updates=int(values.get("inner_updates", cls.inner_updates)),
            minibatch_size=int(values.get("minibatch_size", cls.minibatch_size)),
            learning_rate=float(values.get("learning_rate", cls.learning_rate)),
            weight_decay=float(values.get("weight_decay", cls.weight_decay)),
            warmup_steps=int(values.get("warmup_steps", cls.warmup_steps)),
            clip_low=float(values.get("clip_low", cls.clip_low)),
            clip_high=float(values.get("clip_high", cls.clip_high)),
            kl_coefficient=float(
                values.get("kl_coefficient", cls.kl_coefficient)
            ),
            greedy_guard_tolerance=float(
                values.get("greedy_guard_tolerance", cls.greedy_guard_tolerance)
            ),
            advantage_scale_floor_ratio=float(
                values.get(
                    "advantage_scale_floor_ratio",
                    cls.advantage_scale_floor_ratio,
                )
            ),
            advantage_clip=float(values.get("advantage_clip", cls.advantage_clip)),
        )

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("grpo.group_size must be >= 2")
        if self.prompt_batch_size < 1:
            raise ValueError("grpo.prompt_batch_size must be >= 1")
        if self.max_iterations < 1:
            raise ValueError("grpo.max_iterations must be >= 1")
        if self.max_duration_seconds is not None and self.max_duration_seconds <= 0.0:
            raise ValueError("grpo.max_duration_seconds must be > 0")
        if self.temperature <= 0.0:
            raise ValueError("grpo.temperature must be > 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("grpo.top_p must be > 0 and <= 1")
        if self.inner_updates < 1:
            raise ValueError("grpo.inner_updates must be >= 1")
        if self.minibatch_size < 1:
            raise ValueError("grpo.minibatch_size must be >= 1")
        if self.learning_rate <= 0.0:
            raise ValueError("grpo.learning_rate must be > 0")
        if self.weight_decay < 0.0:
            raise ValueError("grpo.weight_decay must be >= 0")
        if self.warmup_steps < 0:
            raise ValueError("grpo.warmup_steps must be >= 0")
        if not 0.0 <= self.clip_low < 1.0:
            raise ValueError("grpo.clip_low must be >= 0 and < 1")
        if self.clip_high < 0.0:
            raise ValueError("grpo.clip_high must be >= 0")
        if self.kl_coefficient < 0.0:
            raise ValueError("grpo.kl_coefficient must be >= 0")
        if self.greedy_guard_tolerance < 0.0:
            raise ValueError("grpo.greedy_guard_tolerance must be >= 0")
        if self.advantage_scale_floor_ratio < 0.0:
            raise ValueError(
                "grpo.advantage_scale_floor_ratio must be >= 0"
            )
        if self.advantage_clip <= 0.0:
            raise ValueError("grpo.advantage_clip must be > 0")


def load_grpo_config(path: Path) -> GrpoConfig:
    """从职业模型 YAML 读取独立的 `grpo` 配置。

    旧单文件的 `training.grpo` 仍可读取，避免已有实验配置失效。
    """
    raw = load_policy_config(Path(path).resolve(), description="GRPO config")
    grpo_raw = raw.get("grpo")
    if grpo_raw is None:
        training_raw = raw.get("training", {}) or {}
        if not isinstance(training_raw, Mapping):
            raise ValueError("training config section must be a mapping")
        grpo_raw = training_raw.get("grpo", {})
    return GrpoConfig.from_mapping(grpo_raw or {})


@dataclass(frozen=True)
class GrpoRunConfig:
    """GRPO 运行所需的最小运行配置，不携带预训练 RunConfig。"""

    raw_data_dir: Path
    output_dir: Path
    job_tag: str | None
    model_variant: str | None = None
    seed: int = 42
    precision: str = "float32"
    compiled_cache_shard_size: int = 512
    compiled_cache_max_shards: int = 8
    compiled_cache_workers: int = 1
    model: ModelConfig = ModelConfig()
    config_path: Path | None = None


def load_grpo_run_config(path: Path) -> GrpoRunConfig:
    """只读取 GRPO 需要的运行字段与共享策略模型架构。"""
    resolved_path = Path(path).resolve()
    raw = load_policy_config(resolved_path, description="GRPO config")
    training_raw = raw.get("training", {}) or {}
    if not isinstance(training_raw, Mapping):
        raise ValueError("training config section must be a mapping")
    precision = str(
        raw.get("precision", training_raw.get("precision", "float32"))
    ).strip().lower()
    precision_aliases = {
        "fp32": "float32",
        "float32": "float32",
        "fp16": "float16",
        "float16": "float16",
        "bf16": "bf16",
        "bfloat16": "bf16",
    }
    try:
        normalized_precision = precision_aliases[precision]
    except KeyError as exc:
        raise ValueError(
            "model precision must be one of float32, float16, bf16"
        ) from exc
    shard_size = int(training_raw.get("compiled_cache_shard_size", 512))
    max_shards = int(training_raw.get("compiled_cache_max_shards", 8))
    workers = int(training_raw.get("compiled_cache_workers", 1))
    if shard_size < 1 or max_shards < 1 or workers < 1:
        raise ValueError("GRPO compiled cache settings must be positive")
    return GrpoRunConfig(
        raw_data_dir=resolve_project_path(
            raw.get("raw_data_dir", ""),
            project_root=PROJECT_ROOT,
        ),
        output_dir=resolve_project_path(
            raw.get("output_dir", "artifacts/checkpoints"),
            project_root=PROJECT_ROOT,
        ),
        job_tag=None if raw.get("job_tag") is None else str(raw["job_tag"]),
        seed=int(training_raw.get("seed", 42)),
        precision=normalized_precision,
        compiled_cache_shard_size=shard_size,
        compiled_cache_max_shards=max_shards,
        compiled_cache_workers=workers,
        model=load_model_config(resolved_path),
        config_path=resolved_path,
    )

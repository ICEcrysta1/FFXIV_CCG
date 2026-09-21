"""公共模型与训练配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.policy.config import ModelConfig as _ModelConfig
from common.policy.config import load_policy_config
from common.project_config import (
    resolve_project_path,
)
from common.policy.model.repetition import (
    RepetitionConfig as _RepetitionConfig,
    parse_repetition_config as _parse_repetition_config,
)
from ..runtime.runtime_debug import RuntimeDebugConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]

_TRAINING_PRECISION_ALIASES = {
    "float32": "float32",
    "fp32": "float32",
    "float16": "float16",
    "fp16": "float16",
    "bf16": "bf16",
    "bfloat16": "bf16",
}


@dataclass(frozen=True)
class ValuePreferenceConfig:
    """控制技能 value 辅助排序损失。"""

    enabled: bool = False
    loss_weight: float = 0.05
    margin_scale: float = 0.25


@dataclass(frozen=True)
class PpgConfig:
    """控制训练期间的验证集 PPG 与空场景自回归 PPG 评估。"""

    enabled: bool = True
    gcd_count: int = 128
    normalization: float = 1000.0


@dataclass(frozen=True)
class RunConfig:
    """职业配置文件对应的训练运行参数。"""

    raw_data_dir: Path
    output_dir: Path
    job_tag: str | None
    batch_size: int = 32
    max_epochs: int = 50
    history_truncation_enabled: bool = False
    history_truncation_probability: float = 0.0
    history_min_recent: int = 1
    candidate_order_file: Path | None = None
    candidate_shuffle_enabled: bool = False
    candidate_shuffle_probability: float = 0.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    train_split: float = 0.9
    seed: int = 42
    precision: str = "float32"
    activation_checkpoint_ffn: bool = False
    activation_checkpoint_attention: bool = False
    runtime_debug: RuntimeDebugConfig = RuntimeDebugConfig()
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    compiled_cache_shard_size: int = 512
    compiled_cache_max_shards: int = 8
    compiled_cache_workers: int = 1
    repetition: _RepetitionConfig = _RepetitionConfig()
    value_preference: ValuePreferenceConfig = ValuePreferenceConfig()
    ppg: PpgConfig = PpgConfig()
    config_path: Path | None = None

    model: _ModelConfig = _ModelConfig()


def load_run_config(path: Path) -> RunConfig:
    """读取职业 YAML，并把相对路径解析到项目根目录。"""
    path = Path(path).resolve()
    raw = load_policy_config(path, description="training config")

    model_raw = raw.get("model", {}) or {}
    training_raw = raw.get("training", {}) or {}
    if not isinstance(model_raw, dict) or not isinstance(training_raw, dict):
        raise ValueError("model and training config sections must be mappings")
    model = _ModelConfig.from_mapping(model_raw)

    num_workers = int(training_raw.get("num_workers", 0))
    prefetch_factor = int(training_raw.get("prefetch_factor", 2))
    if num_workers < 0:
        raise ValueError("training.num_workers must be >= 0")
    if prefetch_factor < 1:
        raise ValueError("training.prefetch_factor must be >= 1")
    compiled_cache_shard_size = int(training_raw.get("compiled_cache_shard_size", 512))
    compiled_cache_max_shards = int(training_raw.get("compiled_cache_max_shards", 8))
    compiled_cache_workers = int(training_raw.get("compiled_cache_workers", 1))
    if compiled_cache_shard_size < 1:
        raise ValueError("training.compiled_cache_shard_size must be >= 1")
    if compiled_cache_max_shards < 1:
        raise ValueError("training.compiled_cache_max_shards must be >= 1")
    if compiled_cache_workers < 1:
        raise ValueError("training.compiled_cache_workers must be >= 1")
    if "max_history" in training_raw:
        raise ValueError(
            "training.max_history is removed; configure model.history_capacity"
        )
    history_truncation_raw = training_raw.get("history_truncation", {}) or {}
    if not isinstance(history_truncation_raw, dict):
        raise ValueError("training.history_truncation must be a mapping")
    history_truncation_enabled = bool(history_truncation_raw.get("enabled", False))
    history_truncation_probability = float(history_truncation_raw.get("probability", 0.0))
    history_min_recent = int(history_truncation_raw.get("min_recent", 1))
    if not 0.0 <= history_truncation_probability <= 1.0:
        raise ValueError("training.history_truncation_probability must be between 0 and 1")
    if history_min_recent < 1:
        raise ValueError("training.history_min_recent must be >= 1")
    candidate_order_raw = training_raw.get("candidate_order_file")
    candidate_order_file = None
    if candidate_order_raw is not None:
        candidate_order_file = Path(str(candidate_order_raw))
        if not candidate_order_file.is_absolute():
            candidate_order_file = (path.parent / candidate_order_file).resolve()
        if not candidate_order_file.is_file():
            raise FileNotFoundError(
                f"training.candidate_order_file not found: {candidate_order_file}"
            )
    candidate_shuffle_raw = training_raw.get("candidate_shuffle", {}) or {}
    if not isinstance(candidate_shuffle_raw, dict):
        raise ValueError("training.candidate_shuffle must be a mapping")
    candidate_shuffle_enabled = bool(candidate_shuffle_raw.get("enabled", False))
    candidate_shuffle_probability = float(candidate_shuffle_raw.get("probability", 0.0))
    if not 0.0 <= candidate_shuffle_probability <= 1.0:
        raise ValueError("training.candidate_shuffle.probability must be between 0 and 1")
    value_preference_raw = training_raw.get("value_preference", {}) or {}
    if not isinstance(value_preference_raw, dict):
        raise ValueError("training.value_preference must be a mapping")
    value_preference = ValuePreferenceConfig(
        enabled=bool(value_preference_raw.get("enabled", ValuePreferenceConfig.enabled)),
        loss_weight=float(
            value_preference_raw.get("loss_weight", ValuePreferenceConfig.loss_weight)
        ),
        margin_scale=float(
            value_preference_raw.get("margin_scale", ValuePreferenceConfig.margin_scale)
        ),
    )
    if value_preference.loss_weight < 0.0:
        raise ValueError("training.value_preference.loss_weight must be >= 0")
    if value_preference.margin_scale <= 0.0:
        raise ValueError("training.value_preference.margin_scale must be > 0")

    ppg_raw = training_raw.get("ppg", {}) or {}
    if not isinstance(ppg_raw, dict):
        raise ValueError("training.ppg must be a mapping")
    ppg = PpgConfig(
        enabled=bool(ppg_raw.get("enabled", PpgConfig.enabled)),
        gcd_count=int(ppg_raw.get("gcd_count", PpgConfig.gcd_count)),
        normalization=float(
            ppg_raw.get("normalization", PpgConfig.normalization)
        ),
    )
    if ppg.gcd_count < 1:
        raise ValueError("training.ppg.gcd_count must be >= 1")
    if ppg.normalization <= 0.0:
        raise ValueError("training.ppg.normalization must be > 0")

    runtime_debug_raw = training_raw.get("runtime_debug", {}) or {}
    if not isinstance(runtime_debug_raw, dict):
        raise ValueError("training.runtime_debug must be a mapping")
    runtime_debug = RuntimeDebugConfig(
        enabled=bool(runtime_debug_raw.get("enabled", RuntimeDebugConfig.enabled)),
        max_steps=int(runtime_debug_raw.get("max_steps", RuntimeDebugConfig.max_steps)),
        synchronize=bool(
            runtime_debug_raw.get("synchronize", RuntimeDebugConfig.synchronize)
        ),
        output_filename=str(
            runtime_debug_raw.get(
                "output_filename",
                RuntimeDebugConfig.output_filename,
            )
        ).strip(),
    )
    if runtime_debug.max_steps < 1:
        raise ValueError("training.runtime_debug.max_steps must be >= 1")
    if not runtime_debug.output_filename:
        raise ValueError("training.runtime_debug.output_filename must not be empty")

    return RunConfig(
        raw_data_dir=resolve_project_path(
            raw.get("raw_data_dir", ""),
            project_root=PROJECT_ROOT,
        ),
        output_dir=resolve_project_path(
            raw.get("output_dir", "artifacts/checkpoints"),
            project_root=PROJECT_ROOT,
        ),
        job_tag=None if raw.get("job_tag") is None else str(raw["job_tag"]),
        batch_size=int(training_raw.get("batch_size", 32)),
        max_epochs=int(training_raw.get("max_epochs", 50)),
        history_truncation_enabled=history_truncation_enabled,
        history_truncation_probability=history_truncation_probability,
        history_min_recent=history_min_recent,
        candidate_order_file=candidate_order_file,
        candidate_shuffle_enabled=candidate_shuffle_enabled,
        candidate_shuffle_probability=candidate_shuffle_probability,
        learning_rate=float(training_raw.get("learning_rate", 1e-4)),
        weight_decay=float(training_raw.get("weight_decay", 0.01)),
        warmup_steps=int(training_raw.get("warmup_steps", 500)),
        train_split=float(training_raw.get("train_split", 0.9)),
        seed=int(training_raw.get("seed", 42)),
        # 精度属于模型运行契约；旧单文件仍兼容 training.precision。
        precision=_normalize_training_precision(
            raw.get("precision", training_raw.get("precision", "float32"))
        ),
        activation_checkpoint_ffn=bool(
            training_raw.get("activation_checkpoint_ffn", False)
        ),
        activation_checkpoint_attention=bool(
            training_raw.get("activation_checkpoint_attention", False)
        ),
        runtime_debug=runtime_debug,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=bool(training_raw.get("persistent_workers", True)),
        pin_memory=bool(training_raw.get("pin_memory", True)),
        compiled_cache_shard_size=compiled_cache_shard_size,
        compiled_cache_max_shards=compiled_cache_max_shards,
        compiled_cache_workers=compiled_cache_workers,
        repetition=_parse_repetition_config(training_raw.get("repetition", {})),
        value_preference=value_preference,
        ppg=ppg,
        config_path=path,
        model=model,
    )


def _normalize_training_precision(value: object) -> str:
    precision = str(value).strip().lower()
    normalized = _TRAINING_PRECISION_ALIASES.get(precision)
    if normalized is None:
        allowed = ", ".join(("float32", "float16", "bf16"))
        raise ValueError(f"training.precision must be one of {allowed}, got {value!r}")
    return normalized

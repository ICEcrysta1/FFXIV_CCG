"""策略模型架构配置与模型配置文件解析。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path

from common.project_config import (
    PROJECT_JOB_TAG_ENV,
    PROJECT_MODEL_VARIANT_ENV,
    load_root_dotenv,
    resolve_project_job_tag,
    resolve_project_model_variant,
    resolve_project_path,
)
from common.yaml_config import load_yaml_mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_MODEL_ROOT = PROJECT_ROOT / "config" / "models"
POLICY_MODEL_CHECKPOINT_ENV = "TRAINING_MODEL_CHECKPOINT"
POLICY_DEVICE_ENV = "TRAINING_DEVICE"
POLICY_CACHE_ROOT_ENV = "TRAINING_CACHE_ROOT"

# Transformer 主干 FFN 的激活选项；独立输入编码只使用线性投影与归一化。
TRANSFORMER_ACTIVATIONS = ("gelu", "relu", "swiglu")


_DATA_DERIVED_KEYS = {
    "num_actions",
    "state_dim",
    "scene_dim",
    "skill_feat_dim",
    "num_scene_types",
}
_TRUE_BOOLEAN_STRINGS = {"true", "yes", "1"}
_FALSE_BOOLEAN_STRINGS = {"false", "no", "0"}


def _parse_strict_bool(value: object, *, field_name: str) -> bool:
    """解析架构开关，拒绝 bool() 对字符串的静默转换。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_BOOLEAN_STRINGS:
            return True
        if normalized in _FALSE_BOOLEAN_STRINGS:
            return False
    raise ValueError(
        f"{field_name} must be a boolean (true/yes/1 or false/no/0), got {value!r}"
    )


def _parse_qk_norm_scale(value: object) -> float:
    """解析 Q/K 共用的正有限尺度，避免布尔值被 float() 静默接受。"""
    message = "model.qk_norm_scale must be finite and positive"
    if isinstance(value, bool):
        # 架构配置错误统一使用 ValueError，与 checkpoint/续训校验保持一致。
        raise ValueError(message)  # noqa: TRY004
    try:
        scale = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(message)
    return scale


def _parse_residual_mix_endpoint(value: object, *, field_name: str) -> float:
    """解析残差初始化端点，保留有限负值与零并拒绝布尔值。"""
    message = f"{field_name} must be finite"
    if isinstance(value, bool):
        # 必须在 float() 前检查，避免 true/false 被静默转换为 1/0。
        raise ValueError(f"{message}; boolean values are not allowed")  # noqa: TRY004
    try:
        endpoint = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(endpoint):
        raise ValueError(message)
    return endpoint


def _parse_logit_softcap(value: object) -> float:
    """解析输出分数的正有限 softcap，禁止布尔值隐式变成数值。"""
    message = "model.logit_softcap must be finite and positive within the FP32 normal range"
    if isinstance(value, bool):
        raise ValueError(message)  # noqa: TRY004
    try:
        cap = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    # softcap 固定以 FP32 执行；排除上溢、下溢及 GPU 可能冲零的非规格化尺度。
    minimum, maximum = float.fromhex("0x1p-126"), float.fromhex("0x1.fffffep+127")
    if not math.isfinite(cap) or not minimum <= cap <= maximum:
        raise ValueError(message)
    return cap


@dataclass(frozen=True)
class ModelConfig:
    """只保存策略模型架构超参数，数据维度由 DataSpec 提供。"""

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    num_kv_heads: int = 1
    ff_dim: int = 1024
    transformer_norm_first: bool = True
    transformer_activation: str = "gelu"
    full_attention_residuals: bool = False
    residual_mix_r_start: float = 1.15
    residual_mix_r_end: float = 1.05
    residual_mix_a_start: float = 0.20
    residual_mix_a_end: float = 0.05
    dropout: float = 0.1
    rope_theta: float = 10_000.0
    qk_norm_scale: float = 1.2
    logit_softcap: float = 15.0
    history_capacity: int = 128
    scene_capacity: int = 160

    def __post_init__(self) -> None:
        if self.ff_dim < 1:
            raise ValueError("model.ff_dim must be positive")
        if self.transformer_activation not in TRANSFORMER_ACTIVATIONS:
            raise ValueError("model.transformer_activation must be gelu, relu or swiglu")
        if self.history_capacity < 0:
            raise ValueError("model.history_capacity must be >= 0")
        if self.scene_capacity < 1:
            raise ValueError("model.scene_capacity must be >= 1")
        if self.n_heads <= 0:
            raise ValueError("model.n_heads must be positive")
        if self.num_kv_heads <= 0 or self.num_kv_heads > self.n_heads:
            raise ValueError("model.num_kv_heads must be between 1 and model.n_heads")
        if self.n_heads % self.num_kv_heads != 0:
            raise ValueError("model.n_heads must be divisible by model.num_kv_heads")
        if self.rope_theta <= 1.0:
            raise ValueError("model.rope_theta must be greater than 1")
        object.__setattr__(self, "qk_norm_scale", _parse_qk_norm_scale(self.qk_norm_scale))
        object.__setattr__(self, "logit_softcap", _parse_logit_softcap(self.logit_softcap))
        if self.full_attention_residuals and not self.transformer_norm_first:
            raise ValueError(
                "model.full_attention_residuals requires transformer_norm_first=true"
            )
        for name in ("residual_mix_r_start", "residual_mix_r_end",
                     "residual_mix_a_start", "residual_mix_a_end"):
            object.__setattr__(
                self, name,
                _parse_residual_mix_endpoint(getattr(self, name), field_name=f"model.{name}"),
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object] | None = None) -> "ModelConfig":
        """从 YAML 的 ``model`` mapping 解析共享架构配置。"""
        values = {} if raw is None else raw
        if not isinstance(values, Mapping):
            raise ValueError("model config section must be a mapping")
        if "position_encoding" in values:
            raise ValueError("model.position_encoding is removed; the model always uses RoPE")
        if "max_sequence_length" in values:
            raise ValueError(
                "model.max_sequence_length is removed; physical token capacity is derived "
                "from scene_capacity, history_capacity and the current state"
            )
        if "max_history" in values:
            raise ValueError("model.max_history is removed; configure model.history_capacity")
        if "pair_embedding_dim" in values:
            raise ValueError(
                "model.pair_embedding_dim is removed; skill/state tokens use model.d_model directly"
            )
        removed = {
            "num_candidates", "scorer_use_raw_projection",
            "scorer_use_candidate_hidden", "position_id_semantics",
        }.intersection(values)
        if removed:
            raise ValueError("removed model options: " + ", ".join(sorted(removed)))
        derived_keys = sorted(_DATA_DERIVED_KEYS.intersection(values))
        if derived_keys:
            raise ValueError(
                "model config must not declare cache-derived dimensions: "
                + ", ".join(derived_keys)
            )
        return cls(
            d_model=int(values.get("d_model", cls.d_model)),
            n_layers=int(values.get("n_layers", cls.n_layers)),
            n_heads=int(values.get("n_heads", cls.n_heads)),
            num_kv_heads=int(values.get("num_kv_heads", cls.num_kv_heads)),
            ff_dim=int(values.get("ff_dim", cls.ff_dim)),
            transformer_norm_first=_parse_strict_bool(
                values.get("transformer_norm_first", cls.transformer_norm_first),
                field_name="model.transformer_norm_first",
            ),
            transformer_activation=str(
                values.get("transformer_activation", cls.transformer_activation)
            ).strip().lower(),
            full_attention_residuals=_parse_strict_bool(
                values.get("full_attention_residuals", cls.full_attention_residuals),
                field_name="model.full_attention_residuals",
            ),
            residual_mix_r_start=_parse_residual_mix_endpoint(
                values.get("residual_mix_r_start", cls.residual_mix_r_start),
                field_name="model.residual_mix_r_start",
            ),
            residual_mix_r_end=_parse_residual_mix_endpoint(
                values.get("residual_mix_r_end", cls.residual_mix_r_end),
                field_name="model.residual_mix_r_end",
            ),
            residual_mix_a_start=_parse_residual_mix_endpoint(
                values.get("residual_mix_a_start", cls.residual_mix_a_start),
                field_name="model.residual_mix_a_start",
            ),
            residual_mix_a_end=_parse_residual_mix_endpoint(
                values.get("residual_mix_a_end", cls.residual_mix_a_end),
                field_name="model.residual_mix_a_end",
            ),
            dropout=float(values.get("dropout", cls.dropout)),
            rope_theta=float(values.get("rope_theta", cls.rope_theta)),
            qk_norm_scale=_parse_qk_norm_scale(values.get("qk_norm_scale", cls.qk_norm_scale)),
            logit_softcap=_parse_logit_softcap(values.get("logit_softcap", cls.logit_softcap)),
            history_capacity=int(values.get("history_capacity", cls.history_capacity)),
            scene_capacity=int(values.get("scene_capacity", cls.scene_capacity)),
        )


def load_model_config(path: Path) -> ModelConfig:
    """读取只包含策略模型架构部分的项目 YAML。"""
    raw = load_policy_config(path)
    model_raw = raw.get("model", {}) or {}
    return ModelConfig.from_mapping(model_raw)


def _merge_policy_config(
    base: Mapping[str, object],
    overlay: Mapping[str, object],
) -> dict[str, object]:
    """递归合并拆分后的模型、训练与 GRPO 配置。"""
    merged = dict(base)
    for key, value in overlay.items():
        previous = merged.get(key)
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_policy_config(previous, value)
        else:
            merged[key] = value
    return merged


def load_policy_config(
    path: Path,
    *,
    description: str = "policy model config",
) -> dict[str, object]:
    """加载兼容旧单文件与新拆分 manifest 的策略配置。

    `config.yaml` 可以声明 `model_config`、`training_config`、`optimizer_config`、
    `grpo_config` 和 `action_quality_config` 相对路径；各子文件仍可独立传入，便于
    调试和测试。返回值统一为旧调用方使用的合并视图。
    """
    resolved_path = Path(path).resolve()
    raw = load_yaml_mapping(resolved_path, description=description)
    references = (
        "model_config",
        "training_config",
        "optimizer_config",
        "grpo_config",
        "action_quality_config",
    )
    if not any(key in raw for key in references):
        return dict(raw)

    merged: dict[str, object] = {
        key: value
        for key, value in raw.items()
        if key not in set(references)
    }
    for reference_key in references:
        reference = raw.get(reference_key)
        if reference is None:
            continue
        if not isinstance(reference, (str, Path)) or not str(reference).strip():
            raise ValueError(f"{reference_key} must be a non-empty relative path")
        child_path = (resolved_path.parent / str(reference)).resolve()
        child = load_yaml_mapping(child_path, description=reference_key)
        merged = _merge_policy_config(merged, child)
    return merged


def load_action_quality_severity_weights(job_tag: str) -> dict[str, float]:
    """从所选职业模型配置读取动作质量等级权重；未配置时保留原始等级。"""
    if not os.environ.get(PROJECT_MODEL_VARIANT_ENV, "").strip():
        return {}
    variant = resolve_project_model_variant(project_root=PROJECT_ROOT)
    manifest = POLICY_MODEL_ROOT / job_tag / variant / "config.yaml"
    if not manifest.is_file():
        return {}
    quality = load_policy_config(manifest).get("action_quality", {})
    if not isinstance(quality, Mapping):
        raise TypeError(f"{manifest}: action_quality must be a mapping")
    weights = quality.get("severity_weights", {})
    if not isinstance(weights, Mapping):
        raise TypeError(f"{manifest}: severity_weights must be a mapping")
    if not weights:
        return {}
    result = {}
    for level in ("minor", "medium", "major"):
        value = weights.get(level)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"invalid action quality severity weight: {level}")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid action quality severity weight: {level}")
        result[level] = float(value)
    return result


def resolve_policy_model_config_path(explicit_path: Path | None = None) -> Path:
    """按 ``.env`` 的职业标签和模型变体解析策略模型 YAML。"""
    if explicit_path is not None:
        path = resolve_project_path(explicit_path, project_root=PROJECT_ROOT)
    else:
        load_root_dotenv(PROJECT_ROOT)
        job_tag = resolve_project_job_tag(project_root=PROJECT_ROOT)
        variant = resolve_project_model_variant(project_root=PROJECT_ROOT)
        path = POLICY_MODEL_ROOT / job_tag / variant / "config.yaml"
        if not path.is_file():
            job_root = POLICY_MODEL_ROOT / job_tag
            available = sorted(
                child.name
                for child in job_root.iterdir()
                if child.is_dir() and (child / "config.yaml").is_file()
            ) if job_root.is_dir() else []
            suffix = f"; available variants: {', '.join(available)}" if available else ""
            raise FileNotFoundError(
                f"no policy model config found for job {job_tag!r}, "
                f"variant {variant!r}: {path}{suffix}"
            )
    if not path.is_file():
        raise FileNotFoundError(f"policy model config not found: {path}")
    return path


def resolve_policy_model_job_tag(config_path: Path) -> str:
    """从策略模型 YAML 路径解析并校验职业标签。"""
    return _resolve_policy_model_identity(config_path)[0]


def resolve_policy_model_variant(config_path: Path) -> str:
    """从策略模型 YAML 路径解析并校验模型变体。"""
    return _resolve_policy_model_identity(config_path)[1]


def _resolve_policy_model_identity(config_path: Path) -> tuple[str, str]:
    """从策略模型路径解析并校验职业与模型变体。"""
    models_root = POLICY_MODEL_ROOT.resolve()
    resolved_path = Path(config_path).resolve()
    try:
        relative_path = resolved_path.relative_to(models_root)
    except ValueError as exc:
        raise ValueError(
            "policy model config must live under "
            f"{models_root}, got {resolved_path}"
        ) from exc
    if (
        len(relative_path.parts) < 3
        or not relative_path.parts[0]
        or not relative_path.parts[1]
    ):
        raise ValueError(
            "cannot derive policy model job tag and variant from model config: "
            f"{resolved_path}"
        )
    path_job_tag = relative_path.parts[0]
    path_variant = relative_path.parts[1]
    job_tag = resolve_project_job_tag(project_root=PROJECT_ROOT)
    if job_tag != path_job_tag:
        raise ValueError(
            f"{PROJECT_JOB_TAG_ENV}={job_tag!r} does not match policy model path job "
            f"{path_job_tag!r}: {resolved_path}"
        )
    variant = resolve_project_model_variant(project_root=PROJECT_ROOT)
    if variant != path_variant:
        raise ValueError(
            f"{PROJECT_MODEL_VARIANT_ENV}={variant!r} does not match policy model path "
            f"variant {path_variant!r}: {resolved_path}"
        )
    return job_tag, variant


def validate_policy_model_variant(
    payload: Mapping[str, object],
    expected_variant: str,
    *,
    artifact_name: str,
) -> None:
    """校验模型产物记录的变体，禁止同职业不同变体混用。"""
    actual_variant = payload.get("model_variant")
    if not isinstance(actual_variant, str) or not actual_variant.strip():
        raise ValueError(
            f"{artifact_name} missing model_variant; regenerate it with the current "
            "model configuration"
        )
    actual_variant = actual_variant.strip()
    if actual_variant != expected_variant:
        raise ValueError(
            f"{artifact_name} model_variant {actual_variant!r} does not match "
            f"configured model_variant {expected_variant!r}"
        )


def resolve_policy_device(explicit_device: str | None = None) -> str:
    """解析策略运行设备。"""
    load_root_dotenv(PROJECT_ROOT)
    device = (explicit_device or os.environ.get(POLICY_DEVICE_ENV, "cuda")).strip().lower()
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"{POLICY_DEVICE_ENV} must be cuda or cpu, got {device!r}")
    return device


def resolve_policy_cache_dir(job_tag: str) -> Path:
    """解析按职业隔离的策略输入缓存目录。"""
    load_root_dotenv(PROJECT_ROOT)
    raw_root = os.environ.get(POLICY_CACHE_ROOT_ENV, "data/human/job")
    return resolve_project_path(raw_root, project_root=PROJECT_ROOT) / str(job_tag) / ".cache"


def resolve_policy_grpo_dir(job_tag: str) -> Path:
    """解析按职业隔离的 GRPO rollout 持久化目录。"""
    return resolve_policy_cache_dir(job_tag).parent / "grpo"


def resolve_policy_checkpoint_path(
    model_config_path: Path,
    explicit_name: str | None = None,
) -> Path:
    """按策略模型 YAML 的输出目录解析初始化 checkpoint。"""
    load_root_dotenv(PROJECT_ROOT)
    checkpoint_name = explicit_name or os.environ.get(
        POLICY_MODEL_CHECKPOINT_ENV,
        "best.pt",
    )
    checkpoint_path = Path(checkpoint_name)
    if checkpoint_path.is_absolute():
        return checkpoint_path
    raw = load_policy_config(model_config_path)
    output_dir = resolve_project_path(
        raw.get("output_dir", "artifacts/checkpoints"),
        project_root=PROJECT_ROOT,
    )
    return output_dir / checkpoint_path

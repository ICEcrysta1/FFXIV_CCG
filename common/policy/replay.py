"""策略回放运行时共享配置契约。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AutoregressiveReplayConfig:
    """一次自回归回放运行所需的固定参数。"""

    checkpoint_path: Path | None
    output_path: Path
    scene_json_path: Path
    cache_dir: Path
    model_history_capacity: int
    cache_shard_size: int
    cache_max_shards: int
    scene_mode: str = "cache"
    scene_sample_index: int = 0
    # 普通回放仍可使用动作上限；GRPO 传 None，改由时间边界终止。
    max_steps: int | None = 100
    max_gcds: int | None = None
    max_duration_seconds: float | None = None
    # 从 raw scene 的 fights 元数据解析出的绝对战斗结束时刻（秒）。
    scene_duration_seconds: float | None = None
    top_k: int = 8
    top_p: float = 1.0
    temperature: float = 0.0
    max_history: int = 128
    device: str = "cuda"
    job_tag: str | None = None
    initial_action: str | None = "fire_iii"
    initial_time_seconds: float | None = None
    base_gcd: float | None = None
    use_kv_cache: bool = True
    backend: str = "pytorch"
    onnx_package_path: Path | None = None
    ort_provider: str = "CUDAExecutionProvider"
    policy_precision: str | None = None

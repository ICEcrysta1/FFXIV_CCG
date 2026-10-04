"""跨脚本复用的 raw JSON 到 compiled cache 调用入口。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from common.policy.data.action_space import ActionSpace
    from common.policy.data.normalizer import Normalizer
    from common.policy.data.skill_vocab import SkillVocab


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def compile_raw_training_caches(
    *, source_paths, cache_dir: Path, cache_shard_size: int, job_tag: str,
    engine=None, workers: int | None = None,
    expected_action_space: ActionSpace | None = None,
    expected_skill_vocab: SkillVocab | None = None,
    normalizer: Normalizer | None = None,
) -> None:
    """调用正式转换 API，多文件共享引擎；允许宿主提供现有引擎。"""
    # 延迟导入，避免公共配置加载时初始化转换器或 PyTorch。
    from common.config import load_precision_config
    from common.policy.data import Normalizer
    from common.project_config import resolve_positive_worker_count
    from scripts.convert_fflogs.cache import precompile_raw_training_caches

    if normalizer is None:
        normalizer = Normalizer()
    normalizer.ensure_job_resources(job_tag)
    precision = load_precision_config()
    paths = list(dict.fromkeys(Path(path).resolve() for path in source_paths))
    valid = precompile_raw_training_caches(
        paths, cache_dir=Path(cache_dir).resolve(), shard_size=cache_shard_size,
        job_tag=job_tag, normalizer=normalizer,
        expected_action_space=expected_action_space,
        expected_skill_vocab=expected_skill_vocab,
        int_dtype=precision.resolve_int_dtype(), float_dtype=precision.resolve_float_dtype(),
        max_workers=workers if workers is not None else resolve_positive_worker_count(
            project_root=PROJECT_ROOT, env_name="CONVERT_FFLOGS_WORKERS",
        ),
        engine=engine,
    )
    missing = set(paths) - set(valid)
    if missing:
        raise RuntimeError("failed to compile cache for raw sources: " + ", ".join(map(str, sorted(missing))))

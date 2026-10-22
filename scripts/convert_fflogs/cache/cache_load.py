"""raw JSON 对应的 compiled cache 校验与读取。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from common.policy.data.contracts.action_space import ActionSpace
from common.policy.data.cache.compiled_cache import (
    CACHE_FORMAT,
    DEFAULT_CACHE_MAX_SHARDS,
    DEFAULT_CACHE_SHARD_SIZE,
    DEFAULT_CONVERSION_VERSION,
    CompiledShardCache,
    build_cache_signature,
    cache_path_for_source,
    load_compiled_cache_for_source,
)
from common.policy.data.encoding.normalizer import Normalizer
from common.policy.data.contracts.skill_vocab import SkillVocab
from scripts.common.file_inventory import FileInventory

RAW_CONVERSION_VERSION = DEFAULT_CONVERSION_VERSION


class ConversionCacheInventory:
    """转换预检只缓存兼容性结论；正式读取仍走完整 CacheReader 校验。"""

    def __init__(self, *, job_tag: str, cache_dir: Path, **load_options):
        self.job_tag = job_tag
        self.cache_dir = Path(cache_dir)
        self.load_options = load_options
        actions = load_options["expected_action_space"]
        vocab = load_options.get("expected_skill_vocab")
        # 技能 bank 新增 FP32 校验，旧预检结论必须重新通过正式读取器。
        context = {
            "validator": 2, "format": CACHE_FORMAT, "conversion": RAW_CONVERSION_VERSION,
            "job": job_tag, "normalizer": load_options["normalizer"].cache_signature,
            "int_dtype": str(load_options["int_dtype"]), "float_dtype": str(load_options["float_dtype"]),
            "shard_size": load_options["shard_size"], "config": load_options.get("expected_config"),
            "actions": [actions.action_keys, actions.action_to_vocab_id, actions.action_is_gcd],
            "vocab": None if vocab is None else tuple(vocab),
        }
        self.context = hashlib.sha256(json.dumps(context, sort_keys=True).encode("utf-8")).hexdigest()
        self.inventory = FileInventory(self.cache_dir / ".conversion-inventory.json")

    def is_current(self, source: Path) -> bool:
        manifest = cache_path_for_source(self.cache_dir, source)

        def validate():
            cached = _load_cache(source, cache_dir=self.cache_dir, **self.load_options)
            valid = cached is not None and cached.num_samples > 0 and cached.job_tag == self.job_tag
            return valid, cached.shard_paths if valid else ()

        return self.inventory.check(
            manifest, context=self.context, paths=[source, manifest], validate=validate,
        ) is True

    def save(self) -> None:
        self.inventory.save()


def load_raw_compiled_cache(
    source_path: Path,
    *,
    cache_dir: Path,
    normalizer: Normalizer,
    expected_action_space: ActionSpace,
    int_dtype,
    float_dtype,
    shard_size: int = DEFAULT_CACHE_SHARD_SIZE,
    max_shards: int = DEFAULT_CACHE_MAX_SHARDS,
    shard_cache: CompiledShardCache | None = None,
    expected_skill_vocab: SkillVocab | None = None,
):
    """只读取 raw JSON 对应的最终 cache，不触发转换。"""
    return _load_cache(
        Path(source_path),
        cache_dir=Path(cache_dir),
        normalizer=normalizer,
        expected_action_space=expected_action_space,
        expected_skill_vocab=expected_skill_vocab,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
        shard_size=shard_size,
        shard_cache=(
            CompiledShardCache(max_shards)
            if shard_cache is None
            else shard_cache
        ),
    )


def _load_cache(
    source_path: Path,
    *,
    cache_dir: Path,
    normalizer: Normalizer,
    expected_action_space: ActionSpace,
    int_dtype,
    float_dtype,
    shard_size: int,
    shard_cache: CompiledShardCache,
    expected_skill_vocab: SkillVocab | None = None,
    expected_config: dict[str, object] | None = None,
):
    signature = build_cache_signature(
        source_path,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
        normalizer=normalizer,
        shard_size=shard_size,
        conversion_version=RAW_CONVERSION_VERSION,
    )
    return load_compiled_cache_for_source(
        Path(cache_dir),
        source_path,
        signature=signature,
        expected_action_space=expected_action_space,
        expected_skill_vocab=expected_skill_vocab,
        shard_cache=shard_cache,
        expected_config=expected_config,
    )

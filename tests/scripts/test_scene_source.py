"""参考场景只读选择与缓存有效性回归测试。"""

from pathlib import Path

import pytest
import torch

from common.config import load_precision_config
from common.policy.data import Normalizer
from common.policy.data.compiled_cache import (
    CACHE_FORMAT,
    build_cache_signature,
    cache_path_for_source,
)
from scripts.common.json_io import atomic_write_json
from scripts.common.scene_source import find_prepared_scene_source


@pytest.fixture
def scene_cache(tmp_path):
    """构造可由正式缓存 reader 校验的最小 manifest 和分片。"""
    root = tmp_path / "annotated"
    cache_dir = tmp_path / ".cache"
    normalizer = Normalizer()
    normalizer.ensure_job_resources("black_mage")
    precision = load_precision_config()

    def create(relative, *, invalid=None, legacy=False):
        source = root / relative
        atomic_write_json(source, {})
        if invalid == "missing":
            return source
        manifest = cache_path_for_source(cache_dir, source)
        if legacy:
            manifest = cache_dir / manifest.name
        manifest.parent.mkdir(parents=True, exist_ok=True)
        shard = manifest.with_suffix(".shard.pt")
        torch.save({"samples": [{}]}, shard)
        signature = build_cache_signature(
            source, normalizer=normalizer,
            int_dtype=precision.resolve_int_dtype(),
            float_dtype=precision.resolve_float_dtype(), shard_size=768,
        )
        bank = {key: torch.zeros(1) for key in (
            "skill_ids", "skill_features", "state_vectors", "state_null_mask",
            "skill_potencies", "cumulative_dot_potencies",
        )}
        bank["action_keys"] = [""]
        payload = {
            "cache_format": CACHE_FORMAT, "cache_signature": signature,
            "schema": {}, "job_tag": "black_mage", "num_samples": 1,
            "num_candidates": 1, "skill_feature_names": [],
            "candidate_action_keys": ["fire"], "shard_size": 768,
            "history_bank": bank, "shard_files": [shard.name],
        }
        if invalid == "stale":
            signature["source_size"] += 1
        elif invalid == "format":
            payload["cache_format"] = "old"
        elif invalid == "job":
            payload["job_tag"] = "machinist"
        elif invalid == "empty":
            payload["num_samples"] = 0
        elif invalid == "shard":
            payload["shard_files"] = ["missing.pt"]
        torch.save(payload, manifest)
        if invalid == "corrupt":
            manifest.write_bytes(b"broken")
        return source

    return root, cache_dir, create


def _select(root: Path, cache_dir: Path) -> Path:
    return find_prepared_scene_source(
        root, cache_dir=cache_dir, job_tag="black_mage", cache_shard_size=768,
    )


def test_scene_source_prefers_high_bucket_across_encounters(scene_cache):
    root, cache_dir, create = scene_cache
    create("AAA/00-10/a.json.br")
    create("AAA/80-90/a.json.br")
    create("VAL/AAA/a.json.br")
    create("ZZZ/90-100/b.json.br")
    expected = create("ZZZ/90-100/a.json.br")
    assert _select(root, cache_dir) == expected
    assert _select(root, cache_dir) == expected


@pytest.mark.parametrize("invalid", ["missing", "stale", "format", "job", "empty", "shard", "corrupt"])
def test_scene_source_skips_invalid_cache_without_writing(scene_cache, invalid):
    root, cache_dir, create = scene_cache
    create("AAA/90-100/a.json.br", invalid=invalid)
    expected = create("ZZZ/80-90/a.json.br")
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in cache_dir.rglob("*.pt")}
    assert _select(root, cache_dir) == expected
    after = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in cache_dir.rglob("*.pt")}
    assert before == after


def test_scene_source_accepts_legacy_flat_cache(scene_cache):
    root, cache_dir, create = scene_cache
    expected = create("FRU/90-100/a.json.br", legacy=True)
    assert _select(root, cache_dir) == expected


def test_scene_source_requires_existing_valid_cache(scene_cache):
    root, cache_dir, create = scene_cache
    create("FRU/90-100/a.json.br", invalid="missing")
    with pytest.raises(FileNotFoundError, match="先运行训练文件转换"):
        _select(root, cache_dir)
    assert not cache_dir.exists()

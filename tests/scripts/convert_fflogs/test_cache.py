"""raw JSON 直接编译最终 cache 的测试。"""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

import pytest

from common.policy.data import Normalizer
from common.policy.data.compiled_cache import cache_path_for_source
from scripts.convert_fflogs import build_training_samples
from scripts.convert_fflogs.cache import cache_compile as cache_compile_module
from scripts.convert_fflogs.cache import precompile_raw_training_caches
from scripts.convert_fflogs.cache.cache_load import load_raw_compiled_cache
from tests.helpers import build_test_scene_context, targetable_window_token
from training import TrainingDataset


@pytest.mark.parametrize("source_stage", ["raw", "annotated"])
@pytest.mark.parametrize("legacy_layout", [False, True])
def test_raw_cache_compiler_only_writes_compiled_cache(
    cs_backend, cs_skill_book, tmp_path, monkeypatch, source_stage, legacy_layout,
):
    torch = pytest.importorskip("torch")
    fight_payload = {
        "fight_id": "raw_cache_demo",
        "job_tag": "black_mage",
        "player": "Tester",
        "encounter": "Demo",
        "duration": 10.0,
        "scene_context": build_test_scene_context(
            targetable_tokens=[
                targetable_window_token(0.0, 10.0, targetable=True, segment_kind="combat"),
            ],
        ),
        "actions": [
            {
                "time_offset": 0.0,
                "time_gap": 0.0,
                "action_key": "fire_iii",
                "fight_remaining": 10.0,
                "anchor": "combat",
            }
        ],
    }
    training_payload = build_training_samples(cs_backend, cs_skill_book, fight_payload)
    raw_path = tmp_path / source_stage / "FRU" / "00-10" / "demo.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("{}", encoding="utf-8", newline="\n")
    raw_before = raw_path.read_bytes()

    monkeypatch.setattr(
        "scripts.convert_fflogs.cache.cache_compile.convert_raw_file",
        lambda *_args, **_kwargs: (training_payload, {}),
    )
    cache_dir = tmp_path / ".cache"
    valid_paths = precompile_raw_training_caches(
        [raw_path],
        job_tag="black_mage",
        normalizer=Normalizer(),
        int_dtype=torch.int32,
        float_dtype=torch.float32,
        cache_dir=cache_dir,
        shard_size=1,
        max_workers=1,
    )

    assert valid_paths == [raw_path]
    assert raw_path.read_bytes() == raw_before
    manifest = cache_path_for_source(cache_dir, raw_path)
    assert manifest.parent == cache_dir / "FRU" / "00-10"
    assert manifest.is_file()
    assert list(manifest.parent.glob("*.shard-*.pt"))
    assert not list(cache_dir.glob("*.compiled.pt"))
    if legacy_layout:
        for path in manifest.parent.glob("*.pt"):
            path.replace(cache_dir / path.name)
    monkeypatch.setattr(
        "scripts.convert_fflogs.cache.cache_compile.convert_raw_file",
        lambda *_args, **_kwargs: pytest.fail("有效缓存不应重新转换"),
    )
    assert precompile_raw_training_caches(
        [raw_path], job_tag="black_mage", normalizer=Normalizer(),
        int_dtype=torch.int32, float_dtype=torch.float32,
        cache_dir=cache_dir, shard_size=1, max_workers=1,
    ) == [raw_path]
    assert manifest.is_file() is not legacy_layout
    dataset = TrainingDataset(
        [raw_path],
        normalizer=Normalizer(),
        job_tag="black_mage",
        max_history=128,
        int_dtype=torch.int32,
        float_dtype=torch.float32,
        cache_dir=cache_dir,
        compiled_cache_shard_size=1,
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert float(sample["scene_vectors"][:, :3].max().item()) <= 1.0
    raw_path.write_text('{"changed": true}', encoding="utf-8", newline="\n")
    assert load_raw_compiled_cache(
        raw_path, cache_dir=cache_dir, normalizer=Normalizer(),
        int_dtype=torch.int32, float_dtype=torch.float32, shard_size=1,
    ) is None


def test_parallel_cache_compile_skips_failed_source_and_reports_it(tmp_path, monkeypatch, caplog):
    torch = pytest.importorskip("torch")
    bad_path = tmp_path / "bad.json"
    good_path = tmp_path / "good.json"
    bad_path.write_text("{}", encoding="utf-8")
    good_path.write_text("{}", encoding="utf-8")

    class ImmediateExecutor:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, function, task):
            future = Future()
            try:
                future.set_result(function(task))
            except Exception as exc:
                future.set_exception(exc)
            return future

    def fake_worker(task):
        source_path = Path(task[0])
        if source_path.name == "bad.json":
            raise ValueError("invalid raw JSON fixture")
        return str(source_path), 1, 1

    monkeypatch.setattr(cache_compile_module, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(cache_compile_module, "_compile_raw_source_worker", fake_worker)
    with caplog.at_level("ERROR"):
        valid_paths = precompile_raw_training_caches(
            [bad_path, good_path],
            job_tag="black_mage",
            normalizer=Normalizer(),
            int_dtype=torch.int32,
            float_dtype=torch.float32,
            cache_dir=tmp_path / ".cache",
            max_workers=2,
        )

    assert valid_paths == [good_path]
    assert "bad.json" in caplog.text
    assert "跳过并继续" in caplog.text

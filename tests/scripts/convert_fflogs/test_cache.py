"""raw JSON 直接编译最终 cache 的测试。"""

from __future__ import annotations

import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from common.policy.data import Normalizer
from common.policy.data.compiled_cache import cache_path_for_source
from common.policy.data import prepared_sources
from common.policy.data.prepared_sources import select_prepared_training_sources
from scripts.convert_fflogs import build_training_samples
from scripts.convert_fflogs import cli as convert_cli
from scripts.convert_fflogs.cache import cache_compile as cache_compile_module
from scripts.convert_fflogs.cache import prepare_training_caches, precompile_raw_training_caches
from scripts.convert_fflogs.cache.cache_load import load_raw_compiled_cache
from scripts.convert_fflogs.source import raw_source
from scripts.common.json_io import atomic_write_json
from tests.helpers import build_test_scene_context, targetable_window_token
from training import TrainingDataset


def test_convert_raw_file_reads_brotli_json(tmp_path, monkeypatch):
    source = tmp_path / "raw" / "FRU" / "00-10" / "fight.json.br"
    atomic_write_json(source, {
        "source_id": 2, "fight_id": 5, "report_code": "REPORT",
        "fights": [{"id": 5, "name": "FRU"}], "player_name": "Tester",
    })
    backend = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(raw_source, "build_backend", lambda **_kwargs: backend)
    monkeypatch.setattr(raw_source, "load_job_project_config", lambda _job: object())
    monkeypatch.setattr(raw_source, "build_skill_book", lambda _config: object())
    seen = {}

    def convert(payload, **kwargs):
        seen.update(payload=payload, kwargs=kwargs)
        return {"converted": True}, {}

    monkeypatch.setattr(raw_source, "convert_report_to_training_payload", convert)
    assert raw_source.convert_raw_file(source, job_tag="black_mage") == ({"converted": True}, {})
    assert seen["payload"]["fight_id"] == 5
    assert seen["kwargs"]["encounter_name"] == "FRU"


def test_training_source_selection_requires_precompiled_cache(tmp_path, monkeypatch):
    """训练侧只读取有效缓存，并按转换侧相同顺序从后备文件补位。"""
    source_root = tmp_path / "annotated" / "FRU" / "00-10"
    source_root.mkdir(parents=True)
    sources = [source_root / f"{index}.json.br" for index in range(3)]
    for source in sources:
        source.write_text("{}", encoding="utf-8", newline="\n")

    valid = {sources[1], sources[2]}
    checked: list[Path] = []
    monkeypatch.setattr(prepared_sources, "build_cache_signature", lambda path, **_kwargs: path)

    def read_cache(_cache_dir, source, **_kwargs):
        checked.append(source)
        return SimpleNamespace(num_samples=3, job_tag="black_mage") if source in valid else None

    monkeypatch.setattr(prepared_sources, "load_compiled_cache_for_source", read_cache)
    selected = prepared_sources.select_prepared_training_sources(
        tmp_path / "annotated", max_files=2, job_tag="black_mage",
        int_dtype="int32", float_dtype="float32", cache_dir=tmp_path / ".cache",
    )
    assert selected == [sources[2], sources[1]]
    assert checked == [sources[0], sources[2], sources[1]]

    valid.clear()
    with pytest.raises(FileNotFoundError, match="先运行训练文件转换"):
        prepared_sources.select_prepared_training_sources(
            tmp_path / "annotated", max_files=2, job_tag="black_mage",
            int_dtype="int32", float_dtype="float32", cache_dir=tmp_path / ".cache",
        )


def test_cli_training_selection_uses_model_quota(tmp_path, monkeypatch):
    """独立转换入口按训练配置选择文件，训练入口不参与编译。"""
    run_config = SimpleNamespace(
        raw_data_dir=tmp_path / "annotated",
        max_files=8,
        compiled_cache_shard_size=16,
        compiled_cache_max_shards=4,
    )
    calls = {}
    monkeypatch.setattr(sys, "argv", ["convert_fflogs", "--training-selection"])
    monkeypatch.setattr(convert_cli, "load_convert_fflogs_dotenv", lambda: None)
    monkeypatch.setenv("CONVERT_FFLOGS_WORKERS", "2")
    monkeypatch.setattr(convert_cli, "resolve_policy_model_config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setattr(convert_cli, "load_run_config", lambda _path: run_config)
    monkeypatch.setattr(convert_cli, "resolve_policy_model_job_tag", lambda _path: "black_mage")
    monkeypatch.setattr(convert_cli, "resolve_policy_model_variant", lambda _path: "artzip")
    monkeypatch.setattr(convert_cli, "resolve_convert_fflogs_job_tag", lambda _tag: "black_mage")
    monkeypatch.setattr(convert_cli, "resolve_policy_cache_dir", lambda _tag: tmp_path / ".cache")
    monkeypatch.setattr(convert_cli, "prepare_training_caches", lambda path, **kwargs: calls.update(path=path, **kwargs) or [tmp_path / "done.json.br"])
    convert_cli.main()
    assert calls["path"] == run_config.raw_data_dir
    assert calls["max_files"] == 8
    assert calls["max_workers"] == 2


def test_cli_fails_when_annotated_inputs_produce_no_compiled_cache(tmp_path, monkeypatch):
    source = tmp_path / "annotated" / "FRU" / "00-10" / "old_schema.json.br"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    run_config = SimpleNamespace(
        raw_data_dir=source.parent.parent.parent,
        compiled_cache_shard_size=16,
        compiled_cache_max_shards=16,
    )
    monkeypatch.setattr(sys, "argv", [
        "convert_fflogs", "--cache-root", str(tmp_path / ".cache"),
    ])
    monkeypatch.setattr(convert_cli, "load_convert_fflogs_dotenv", lambda: None)
    monkeypatch.setenv("CONVERT_FFLOGS_WORKERS", "3")
    monkeypatch.setattr(convert_cli, "resolve_policy_model_config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setattr(convert_cli, "load_run_config", lambda _path: run_config)
    monkeypatch.setattr(convert_cli, "resolve_policy_model_job_tag", lambda _path: "black_mage")
    monkeypatch.setattr(convert_cli, "resolve_policy_model_variant", lambda _path: "artzip")
    monkeypatch.setattr(convert_cli, "resolve_convert_fflogs_job_tag", lambda _tag: "black_mage")
    calls = {}
    monkeypatch.setattr(
        convert_cli, "precompile_raw_training_caches",
        lambda *_args, **kwargs: calls.update(kwargs) or [],
    )

    with pytest.raises(RuntimeError, match="均未编译成功"):
        convert_cli.main()
    assert calls["max_workers"] == 3
    source.unlink()
    with pytest.raises(FileNotFoundError, match="没有找到 FFLogs JSON"):
        convert_cli.main()


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
    raw_path = tmp_path / source_stage / "FRU" / "00-10" / "demo.json.br"
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
    # primary 候选 z.json.br 没有缓存；现有 demo PT 已满足配额，不应编译 z.json.br。
    (raw_path.parent / "z.json.br").write_text("{}", encoding="utf-8", newline="\n")
    assert prepare_training_caches(
        tmp_path / source_stage, max_files=1, job_tag="black_mage",
        int_dtype=torch.int32, float_dtype=torch.float32,
        cache_dir=cache_dir, shard_size=1, max_workers=1,
    ) == [raw_path]
    assert select_prepared_training_sources(
        tmp_path / source_stage,
        max_files=1, job_tag="black_mage",
        int_dtype=torch.int32, float_dtype=torch.float32,
        cache_dir=cache_dir, shard_size=1,
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
    bad_path = tmp_path / "bad.json.br"
    good_path = tmp_path / "good.json.br"
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
        if source_path.name == "bad.json.br":
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
    assert "bad.json.br" in caplog.text
    assert "跳过并继续" in caplog.text

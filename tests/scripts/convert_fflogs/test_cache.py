"""raw JSON 直接编译最终 cache 的测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from threading import Barrier, Lock, get_ident
from types import SimpleNamespace

import pytest

from common.policy.data import Normalizer, prepared_sources
from common.policy.data.compiled_cache import CACHE_FORMAT, cache_path_for_source
from common.policy.data.prepared_sources import select_prepared_training_sources
from scripts.common.json_io import atomic_write_json
from scripts.convert_fflogs import build_training_samples
from scripts.convert_fflogs import cli as convert_cli
from scripts.convert_fflogs.cache import cache_compile as cache_compile_module
from scripts.convert_fflogs.cache import (
    precompile_raw_training_caches,
    prepare_training_caches,
)
from scripts.convert_fflogs.cache.cache_load import load_raw_compiled_cache
from scripts.convert_fflogs.source import raw_source
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


@pytest.mark.parametrize("fails", [False, True])
def test_raw_source_releases_only_its_queue_and_keeps_full_history(tmp_path, monkeypatch, fails):
    source = tmp_path / "fight.json.br"
    atomic_write_json(source, {"source_id": 2})
    engine = object()
    closed = []
    backend = SimpleNamespace(close=lambda: closed.append(True))

    def create_backend(**kwargs):
        assert kwargs == {"job_tag": "black_mage", "max_history": None, "engine": engine}
        return backend

    def convert(_payload, **kwargs):
        assert kwargs["backend"] is backend
        if fails:
            raise ValueError("conversion failed")
        return {"converted": True}, {}

    monkeypatch.setattr(raw_source, "build_backend", create_backend)
    monkeypatch.setattr(raw_source, "load_job_project_config", lambda _job: object())
    monkeypatch.setattr(raw_source, "build_skill_book", lambda _config: object())
    monkeypatch.setattr(raw_source, "convert_report_to_training_payload", convert)
    if fails:
        with pytest.raises(ValueError, match="conversion failed"):
            raw_source.convert_raw_file(source, job_tag="black_mage", engine=engine)
    else:
        assert raw_source.convert_raw_file(source, job_tag="black_mage", engine=engine) == ({"converted": True}, {})
    assert closed == [True]


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


def test_validation_source_selection_requires_each_encounter_quota(tmp_path, monkeypatch):
    """验证只读 VAL 的 PT；一个副本缺额时，即使另一副本富余也拒绝。"""
    cache_dir = tmp_path / ".cache"
    source_root = tmp_path / "annotated"
    manifests = {}
    for encounter, count in (("FRU", 3), ("M5S", 3)):
        for index in range(count):
            source = source_root / "VAL" / encounter / f"{index}.json.br"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("{}", encoding="utf-8", newline="\n")
            manifest = cache_path_for_source(cache_dir, source)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.touch()
            manifests[manifest] = source
    raw_source = tmp_path / "raw" / "VAL" / "FRU" / "raw_only.json.br"
    raw_source.parent.mkdir(parents=True)
    raw_source.write_text("{}", encoding="utf-8", newline="\n")
    raw_manifest = cache_path_for_source(cache_dir, raw_source)
    raw_manifest.touch()
    manifests[raw_manifest] = raw_source
    monkeypatch.setattr(
        prepared_sources, "safe_torch_load",
        lambda path, **_kwargs: {"cache_format": CACHE_FORMAT, "source_path": str(manifests[path])},
    )
    monkeypatch.setattr(prepared_sources, "build_cache_signature", lambda path, **_kwargs: path)
    valid = set(manifests.values())
    monkeypatch.setattr(
        prepared_sources, "load_compiled_cache_for_source",
        lambda _cache_dir, source, **_kwargs: (
            SimpleNamespace(num_samples=3, job_tag="black_mage") if source in valid else None
        ),
    )
    selected = prepared_sources.select_prepared_validation_sources(
        cache_dir, data_dir=source_root, max_files=4, job_tag="black_mage",
        int_dtype="int32", float_dtype="float32",
    )
    assert len(selected) == 4
    assert [sum(path.parent.name == name for path in selected) for name in ("FRU", "M5S")] == [2, 2]
    assert all(path.parent.parent.parent == source_root for path in selected)

    valid.remove(source_root / "VAL" / "FRU" / "0.json.br")
    valid.remove(source_root / "VAL" / "FRU" / "1.json.br")
    with pytest.raises(FileNotFoundError, match="FRU: 1/2"):
        prepared_sources.select_prepared_validation_sources(
            cache_dir, data_dir=source_root, max_files=4, job_tag="black_mage",
            int_dtype="int32", float_dtype="float32",
        )


def test_validation_conversion_reuses_same_encounter_fallback(tmp_path, monkeypatch):
    """转换入口把固定副本配额交给现有编译/同组补位器。"""
    root = tmp_path / "annotated"
    for encounter in ("FRU", "M5S"):
        source = root / "VAL" / encounter / "one.json.br"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("{}", encoding="utf-8", newline="\n")
    captured = {}
    def compile_groups(groups, **kwargs):
        captured["groups"] = groups
        captured["kwargs"] = kwargs
        return []
    monkeypatch.setattr(cache_compile_module, "_compile_training_path_groups", compile_groups)
    cache_compile_module.prepare_validation_caches(
        root, max_files=10, job_tag="black_mage",
        int_dtype="int32", float_dtype="float32", cache_dir=tmp_path / ".cache",
    )
    assert [(group.directory_name, group.target_count) for group in captured["groups"]] == [
        ("FRU", 5), ("M5S", 5),
    ]
    assert all(
        all(source.parent.name == group.directory_name for source in group.candidates)
        for group in captured["groups"]
    )


def test_training_shortage_does_not_stop_validation_conversion(tmp_path, monkeypatch):
    """训练档位缺额时仍尝试 VAL 同副本的后备文件，再统一报缺额。"""
    root = tmp_path / "annotated"
    train_sources = [root / "FRU" / "00-10" / f"train_{index}.json.br" for index in range(2)]
    val_sources = [root / "VAL" / "FRU" / f"val_{index}.json.br" for index in range(2)]
    for source in train_sources + val_sources:
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()

    monkeypatch.setattr(
        cache_compile_module, "cached_candidates_for_group", lambda *_args, **_kwargs: [],
    )
    attempted = []

    def compile_round(paths, **_kwargs):
        attempted.extend(paths)
        return [path for path in paths if path == val_sources[1]]

    monkeypatch.setattr(cache_compile_module, "precompile_raw_training_caches", compile_round)
    with pytest.raises(ValueError, match=r"FRU/00-10: required=1 valid=0 missing=1") as error:
        cache_compile_module.prepare_training_and_validation_caches(
            root, max_files=1, validation_files=1, job_tag="black_mage",
            int_dtype="int32", float_dtype="float32", cache_dir=tmp_path / ".cache",
        )

    assert "VAL/FRU" not in str(error.value)
    assert set(attempted) == set(train_sources + val_sources)


def test_joint_conversion_reports_training_and_validation_shortages(tmp_path, monkeypatch):
    """两个数据集都缺额时，报告列出各自的组，不用另一组补位。"""
    root = tmp_path / "annotated"
    for source in (
        root / "FRU" / "00-10" / "train.json.br",
        root / "VAL" / "FRU" / "val.json.br",
    ):
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
    monkeypatch.setattr(
        cache_compile_module, "cached_candidates_for_group", lambda *_args, **_kwargs: [],
    )
    attempted = []

    def compile_round(paths, **_kwargs):
        attempted.extend(paths)
        return []

    monkeypatch.setattr(cache_compile_module, "precompile_raw_training_caches", compile_round)
    with pytest.raises(ValueError) as error:
        cache_compile_module.prepare_training_and_validation_caches(
            root, max_files=1, validation_files=1, job_tag="black_mage",
            int_dtype="int32", float_dtype="float32", cache_dir=tmp_path / ".cache",
        )

    assert "FRU/00-10: required=1 valid=0 missing=1" in str(error.value)
    assert "VAL/FRU: required=1 valid=0 missing=1" in str(error.value)
    assert len(attempted) == 2


def test_missing_validation_directory_is_reported_after_training_conversion(tmp_path, monkeypatch):
    """尚无 VAL 目录时也先编译训练文件，然后报告验证缺额。"""
    root = tmp_path / "annotated"
    source = root / "FRU" / "00-10" / "train.json.br"
    source.parent.mkdir(parents=True)
    source.touch()
    monkeypatch.setattr(
        cache_compile_module, "cached_candidates_for_group", lambda *_args, **_kwargs: [],
    )
    attempted = []

    def compile_round(paths, **_kwargs):
        attempted.extend(paths)
        return paths

    monkeypatch.setattr(cache_compile_module, "precompile_raw_training_caches", compile_round)
    with pytest.raises(ValueError, match=r"VAL: required=1 valid=0 missing=1"):
        cache_compile_module.prepare_training_and_validation_caches(
            root, max_files=1, validation_files=1, job_tag="black_mage",
            int_dtype="int32", float_dtype="float32", cache_dir=tmp_path / ".cache",
        )
    assert attempted == [source]


def test_training_selector_rejects_explicit_val_root(tmp_path):
    with pytest.raises(ValueError, match="训练输入不能指向验证目录"):
        select_prepared_training_sources(
            tmp_path / "annotated" / "VAL", max_files=1, job_tag="black_mage",
            int_dtype="int32", float_dtype="float32", cache_dir=tmp_path / ".cache",
        )


def test_cli_training_selection_uses_model_quota(tmp_path, monkeypatch):
    """独立转换入口按训练配置选择文件，训练入口不参与编译。"""
    run_config = SimpleNamespace(
        raw_data_dir=tmp_path / "annotated",
        max_files=8,
        validation_files=4,
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
    def prepare_both(path, **kwargs):
        calls.update(path=path, **kwargs)
        return [tmp_path / "done.json.br"], [tmp_path / "VAL" / "FRU" / "val.json.br"]
    monkeypatch.setattr(convert_cli, "prepare_training_and_validation_caches", prepare_both)
    convert_cli.main()
    assert calls["path"] == run_config.raw_data_dir
    assert calls["max_files"] == 8
    assert calls["max_workers"] == 2
    assert calls["validation_files"] == 4


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


def test_parallel_cache_compile_bounds_tasks_shares_engine_and_continues_after_failure(tmp_path, monkeypatch, caplog):
    torch = pytest.importorskip("torch")
    bad_path = tmp_path / "bad.json.br"
    good_path = tmp_path / "good.json.br"
    bad_path.write_text("{}", encoding="utf-8")
    good_path.write_text("{}", encoding="utf-8")
    later_path = tmp_path / "later.json.br"
    later_path.write_text("{}", encoding="utf-8")
    barrier = Barrier(2)
    lock = Lock()
    engines = []
    seen_threads = set()
    started = []
    waiting_counts = []

    class Engine:
        def __init__(self, job_tag, *, capacity):
            assert job_tag == "black_mage"
            assert capacity == 2
            self.closed = False
            engines.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

    def fake_worker(task, *, engine):
        assert engine is engines[0]
        assert not engine.closed
        source_path = Path(task[0])
        with lock:
            seen_threads.add(get_ident())
            started.append(source_path)
        if source_path != later_path:
            barrier.wait(timeout=10)
        if source_path.name == "bad.json.br":
            raise ValueError("invalid raw JSON fixture")
        return str(source_path), 1, 1

    actual_wait = cache_compile_module.wait

    def track_wait(futures, **kwargs):
        waiting_counts.append(len(futures))
        return actual_wait(futures, **kwargs)

    monkeypatch.setattr(cache_compile_module, "InProcessEngine", Engine)
    monkeypatch.setattr(cache_compile_module, "wait", track_wait)
    monkeypatch.setattr(cache_compile_module, "_compile_raw_source_worker", fake_worker)
    with caplog.at_level("ERROR"):
        valid_paths = precompile_raw_training_caches(
            [bad_path, good_path, later_path, good_path],
            job_tag="black_mage",
            normalizer=Normalizer(),
            int_dtype=torch.int32,
            float_dtype=torch.float32,
            cache_dir=tmp_path / ".cache",
            max_workers=2,
        )

    assert valid_paths == [good_path, later_path]
    assert len(engines) == 1 and engines[0].closed
    assert len(seen_threads) == 2
    assert sorted(started) == sorted([bad_path, good_path, later_path])
    assert max(waiting_counts) == 2
    assert "bad.json.br" in caplog.text
    assert "跳过并继续" in caplog.text


def test_valid_cache_does_not_start_engine(tmp_path, monkeypatch):
    source = tmp_path / "fight.json.br"
    monkeypatch.setattr(cache_compile_module, "_load_cache", lambda *_args, **_kwargs:
                        SimpleNamespace(num_samples=3, job_tag="black_mage"))

    def unexpected_engine(*_args, **_kwargs):
        pytest.fail("有效缓存不应启动引擎")

    monkeypatch.setattr(cache_compile_module, "InProcessEngine", unexpected_engine)
    assert precompile_raw_training_caches(
        [source], job_tag="black_mage", normalizer=Normalizer(),
        int_dtype="int32", float_dtype="float32", cache_dir=tmp_path / ".cache", max_workers=2,
    ) == [source]


def test_real_threaded_cache_compile_matches_serial_and_keeps_full_history(tmp_path, monkeypatch, cs_backend):
    """从压缩日志到最终 PT 走正式入口，逐项比较共享引擎的并发与串行结果。"""
    from dataclasses import asdict, is_dataclass
    from scripts.common.inprocess_backend import InProcessEngine

    torch = pytest.importorskip("torch")
    sources = []
    for index in range(4):
        source = tmp_path / "raw" / f"fight{index}.json.br"
        atomic_write_json(source, {
            "source_id": 10, "report_code": f"REPORT{index}", "fight_id": index + 1,
            "events": [
                {"type": "cast", "sourceID": 10, "timestamp": 1000 + step * (5000 - 100 * index),
                 "abilityGameID": 152 if step % 2 == 0 else 154}
                for step in range(6 + index * 2)
            ],
        })
        sources.append(source)
    engines = []
    queues = []

    class TrackedEngine(InProcessEngine):
        def __init__(self, job_tag, *, capacity):
            super().__init__(job_tag, capacity=capacity)
            engines.append(self)

        def __exit__(self, *args):
            assert self.active_count == 0
            return super().__exit__(*args)

    actual_build_backend = raw_source.build_backend

    def create_backend(**kwargs):
        assert kwargs["max_history"] is None
        assert kwargs["engine"] is engines[-1]
        backend = actual_build_backend(**kwargs)
        queues.append(backend)
        return backend

    monkeypatch.setattr(cache_compile_module, "InProcessEngine", TrackedEngine)
    monkeypatch.setattr(raw_source, "build_backend", create_backend)
    cache_dirs = [tmp_path / "serial", tmp_path / "parallel"]
    for workers, cache_dir in zip((1, 3), cache_dirs):
        assert precompile_raw_training_caches(
            sources, job_tag="black_mage", normalizer=Normalizer(),
            int_dtype=torch.int32, float_dtype=torch.float32, cache_dir=cache_dir,
            max_workers=workers, shard_size=4,
        ) == sources
    assert len(engines) == 2 and len(queues) == 8
    assert all(engine._engine is None for engine in engines)

    def assert_equal(left, right):
        if isinstance(left, torch.Tensor):
            assert left.dtype == right.dtype and torch.equal(left, right)
        elif is_dataclass(left):
            assert_equal(asdict(left), asdict(right))
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                # 缓存目录身份不同；完整历史、候选、标签及其他元数据必须逐值一致。
                if key != "history_bank_id":
                    assert_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                assert_equal(a, b)
        else:
            assert left == right

    for source in sources:
        manifests = [cache_path_for_source(root, source) for root in cache_dirs]
        payloads = [torch.load(path, weights_only=False) for path in manifests]
        assert payloads[0]["num_samples"] >= 6
        assert_equal(*payloads)
        for shard in payloads[0]["shard_files"]:
            assert_equal(*(torch.load(path.parent / shard, weights_only=False) for path in manifests))

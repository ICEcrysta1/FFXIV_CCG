"""训练公共层测试共享的 PT/cache 构造辅助。"""

from __future__ import annotations

from pathlib import Path

from common.models import ActionKind
from scripts.common.cs_backend import SidecarBackend
from scripts.convert_fflogs import build_training_samples
from scripts.convert_fflogs.cache.cache_writer import write_compiled_cache_stream
from scripts.convert_fflogs.source.source_reader import TrainingSourceReader
from scripts.convert_fflogs.training.history_bank import build_history_bank
from scripts.convert_fflogs.training.sample_builder import TrainingSampleBuilder
from scripts.convert_fflogs.utils import build_skill_book, load_job_project_config
from common.policy.data import Normalizer, NormalizerConfig, SkillVocab
from training import TrainingDataset
from common.policy.data.compiled_cache import build_cache_signature, cache_path_for_source
from common.policy.data.schema import SceneWindowSchema


_TEST_TRAINING_PAYLOADS: dict[Path, dict[str, object]] = {}


def _build_training_samples(fight_payload: dict[str, object]) -> dict[str, object]:
    """经 C# 状态机后端构建训练样本（转换链路已切后端）。

    SidecarHost 未构建时跳过依赖它的测试（与 tests/scripts/conftest.py
    的 cs_backend fixture 同一守卫，避免训练测试组无 dotnet 环境全红）。
    """
    from tests.scripts.conftest import _require_sidecar_host

    _require_sidecar_host()
    with SidecarBackend(job_tag="black_mage") as backend:
        return build_training_samples(
            backend,
            build_skill_book(load_job_project_config("black_mage")),
            fight_payload,
        )


def make_dataset(source_paths: list[Path], **kwargs) -> TrainingDataset:
    from common.config import load_precision_config

    precision = load_precision_config()
    kwargs.setdefault("int_dtype", precision.resolve_int_dtype())
    kwargs.setdefault("float_dtype", precision.resolve_float_dtype())
    kwargs.setdefault("normalizer", Normalizer())
    kwargs.setdefault("job_tag", "black_mage")
    if "cache_dir" not in kwargs:
        kwargs["cache_dir"] = source_paths[0].parents[1] / ".cache"
    for source_path in source_paths:
        payload = _TEST_TRAINING_PAYLOADS.get(source_path.resolve())
        if payload is not None:
            write_test_compiled_cache(source_path, payload, kwargs["cache_dir"], kwargs)
    return TrainingDataset(source_paths, **kwargs)


def enabled_black_mage_config(tmp_path: Path) -> Path:
    """为过采样行为测试复制一份显式开启过采样的职业配置。"""
    import yaml

    source = Path("config/models/black_mage/artzip/config.yaml")
    from common.policy.config import load_policy_config

    payload = load_policy_config(source)
    payload["oversampling"]["enabled"] = True
    target = tmp_path / "black_mage_oversampling.yaml"
    target.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return target


def make_demo_pt(
    tmp_path: Path,
    actions: list[str],
    *,
    fight_id: str,
) -> Path:
    fight_duration = max(len(actions) * 3.5, 10.0)
    from tests.scripts.conftest import _require_sidecar_host

    _require_sidecar_host()
    skill_book = build_skill_book(load_job_project_config("black_mage"))
    with SidecarBackend(job_tag="black_mage", fight_remaining=fight_duration) as backend:
        fight_payload = make_demo_payload(backend, skill_book, actions, fight_id=fight_id)
        # 动作生成会推进后端 history；构建训练样本前重新初始化到空历史
        backend.init(fight_remaining=fight_duration)
        payload = build_training_samples(backend, skill_book, fight_payload)
    source_path = tmp_path / "raw" / f"{fight_id}.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("{}", encoding="utf-8", newline="\n")
    _TEST_TRAINING_PAYLOADS[source_path.resolve()] = payload
    write_test_compiled_cache(source_path, payload, tmp_path / ".cache", {})
    return source_path


def make_illegal_candidate_pt(tmp_path: Path) -> Path:
    from tests.helpers import (
        build_test_scene_context,
        forced_movement_window_token,
        raid_buff_window_token,
        targetable_window_token,
    )

    fight_payload = {
        "fight_id": "illegal_candidate_demo",
        "job_tag": "black_mage",
        "player": "Tester",
        "encounter": "Demo",
        "duration": 12.0,
        "scene_context": build_test_scene_context(
            targetable_tokens=[
                targetable_window_token(0.0, 12.0, targetable=False, segment_kind="downtime"),
            ],
            forced_movement_tokens=[
                forced_movement_window_token(0.0, 2.0),
            ],
            raid_buff_tokens=[
                raid_buff_window_token(1.0, 5.0),
            ],
        ),
        "actions": [
            {
                "time_offset": 0.0,
                "request_time_offset": 0.0,
                "time_gap": 0.0,
                "action_key": "lucid_dreaming",
                "fight_remaining": 12.0,
                "anchor": "combat",
            }
        ],
    }
    payload = _build_training_samples(fight_payload)
    source_path = tmp_path / "raw" / "illegal_candidate_demo.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("{}", encoding="utf-8", newline="\n")
    _TEST_TRAINING_PAYLOADS[source_path.resolve()] = payload
    write_test_compiled_cache(source_path, payload, tmp_path / ".cache", {})
    return source_path


def make_demo_payload(
    backend,
    skill_book,
    actions: list[str],
    *,
    fight_id: str,
) -> dict[str, object]:
    from tests.helpers import (
        build_test_scene_context,
        forced_movement_window_token,
        raid_buff_window_token,
        targetable_window_token,
    )

    fight_duration = max(len(actions) * 3.5, 10.0)
    return {
        "fight_id": fight_id,
        "job_tag": "black_mage",
        "player": "Tester",
        "encounter": "Demo",
        "duration": fight_duration,
        "scene_context": build_test_scene_context(
            targetable_tokens=[
                targetable_window_token(0.0, 30.0, targetable=True, segment_kind="combat"),
            ],
            forced_movement_tokens=[
                forced_movement_window_token(1.0, 2.0),
            ],
            raid_buff_tokens=[
                raid_buff_window_token(3.0, 8.0),
            ],
        ),
        "actions": build_demo_actions(backend, skill_book, actions, fight_duration=fight_duration),
    }


def write_test_compiled_cache(
    source_path: Path,
    payload: dict[str, object],
    cache_dir: Path,
    dataset_kwargs: dict[str, object],
) -> None:
    import torch

    cache_dir = Path(cache_dir)
    cache_path = cache_path_for_source(cache_dir, source_path)
    if cache_path.is_file():
        return
    reader = TrainingSourceReader(payload)
    normalizer = dataset_kwargs.get("normalizer") or Normalizer()
    normalizer.configure_job_resources(reader.job_tag)
    normalizer.register_schema(reader.schema)
    int_dtype = dataset_kwargs.get("int_dtype", torch.int32)
    float_dtype = dataset_kwargs.get("float_dtype", torch.float32)
    shard_size = int(dataset_kwargs.get("compiled_cache_shard_size", 512))
    vocab = SkillVocab.build_from_job_tag(reader.job_tag)
    history_bank = build_history_bank(
        reader,
        torch=torch,
        normalizer=normalizer,
        skill_vocab=vocab,
        skill_feature_names=reader.skill_feature_names,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
    )
    builder = TrainingSampleBuilder(
        torch=torch,
        normalizer=normalizer,
        skill_vocab=vocab,
        skill_feature_names=reader.skill_feature_names,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
        num_candidates=reader.num_candidates,
        history_bank=history_bank,
    )
    signature = build_cache_signature(
        source_path,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
        normalizer=normalizer,
        shard_size=shard_size,
    )
    write_compiled_cache_stream(
        cache_path,
        source_path,
        signature=signature,
        reader=reader,
        sample_batches=(
            [builder.build(reader, index) for index in range(start, min(start + shard_size, reader.num_samples))]
            for start in range(0, reader.num_samples, shard_size)
        ),
        num_samples=reader.num_samples,
        vocab_signature=tuple(vocab),
        shard_size=shard_size,
        history_bank=history_bank,
    )


def build_demo_actions(
    backend,
    skill_book,
    actions: list[str],
    *,
    fight_duration: float,
) -> list[dict[str, object]]:
    """用绝对时间 Sidecar 协议生成合法的测试动作序列。"""
    request_time = 0.0
    previous_time_offset: float | None = None
    payload: list[dict[str, object]] = []

    for action_key in actions:
        skill = skill_book.get(action_key)
        result = backend.submit_action(request_time, action_key)
        assert result.accepted, (
            f"unexpected illegal demo action: {action_key} / {result.reason} "
            f"at {request_time:.4f}"
        )

        time_offset = round(request_time, 4)
        time_gap = 0.0 if previous_time_offset is None else round(time_offset - previous_time_offset, 4)
        payload.append(
            {
                "time_offset": time_offset,
                "request_time_offset": time_offset,
                "time_gap": time_gap,
                "action_key": action_key,
                "fight_remaining": round(max(0.0, fight_duration - time_offset), 4),
                "anchor": "combat",
            }
        )
        previous_time_offset = time_offset
        accepted_time = result.accepted_timestamp
        assert accepted_time is not None
        action_lock_seconds = max(
            float(skill.cast_time),
            float(skill.recast_time) if skill.kind is ActionKind.GCD else 0.0,
        )
        effect_time = (
            request_time
            if result.effect_timestamp is None
            else float(result.effect_timestamp)
        )
        request_time = round(
            max(
                request_time,
                effect_time,
                float(accepted_time) + action_lock_seconds,
            ),
            4,
        )

    return payload

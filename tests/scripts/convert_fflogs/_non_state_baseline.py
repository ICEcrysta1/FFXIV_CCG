"""从指定源码树采集非状态 token 基线；独立进程运行，不启动模型或训练。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def capture_cases():
    """只用两个小型固定轨迹，经过真实桥接、raw bank 和正式 live 编码器。"""
    import torch
    from common.policy.config import ModelConfig
    from common.policy.data import Normalizer, SkillVocab
    from scripts.autoregressive_replay.context import LiveBatchBuilder, SceneTemplateProvider
    from scripts.common.inprocess_backend import InProcessEngine
    from scripts.convert_fflogs import build_training_samples
    from scripts.convert_fflogs.source.source_reader import TrainingSourceReader
    from scripts.convert_fflogs.scene.scene_context import build_target_count_window_token as target_count_window_token
    from scripts.convert_fflogs.training.history_bank import build_history_bank
    from scripts.convert_fflogs.utils import build_skill_book, load_job_project_config
    from tests.helpers import (
        build_test_scene_context, forced_movement_window_token, raid_buff_window_token,
        targetable_window_token,
    )

    torch.set_num_threads(1)
    normalizer = Normalizer()
    normalizer.ensure_job_resources("black_mage")
    vocabulary = SkillVocab.build_from_job_tag("black_mage")
    skills = build_skill_book(load_job_project_config("black_mage"))
    scenes = {
        "empty_scene": build_test_scene_context(),
        "all_window_types": build_test_scene_context(
            targetable_tokens=[
                targetable_window_token(0, 30, targetable=True, segment_kind="combat"),
                targetable_window_token(30, 40, targetable=False, segment_kind="downtime"),
                targetable_window_token(40, 100, targetable=True, segment_kind="combat_final"),
            ],
            forced_movement_tokens=[
                forced_movement_window_token(60.004, 65.007),
                forced_movement_window_token(64.002, 67.006),
            ],
            raid_buff_tokens=[raid_buff_window_token(0, 8), raid_buff_window_token(50.004, 55.007)],
            target_count_tokens=[target_count_window_token(0, 2, 2), target_count_window_token(70.004, 80.006, 3)],
        ),
    }
    results = {}
    with InProcessEngine("black_mage", capacity=1) as engine:
        for name, scene in scenes.items():
            with engine.create_backend(max_history=None) as backend:
                payload = build_training_samples(backend, skills, {
                    "fight_id": name, "job_tag": "black_mage", "duration": 100.0,
                    "scene_context": scene,
                    "actions": [
                        {"time_offset": time, "request_time_offset": time, "time_gap": 6.0,
                         "action_key": key, "anchor": "combat"}
                        for time, key in [(0.0, "fire_iii"), (6.0, "high_thunder"),
                                          (12.0, "blizzard_iii"), (18.0, "fire_iii"), (24.0, "lucid_dreaming")]
                    ],
                })
                reader = TrainingSourceReader(payload)
                normalizer.register_schema(reader.schema)
                bank = build_history_bank(
                    reader, torch=torch, normalizer=normalizer, skill_vocab=vocabulary,
                    skill_feature_names=reader.skill_feature_names,
                    int_dtype=torch.int32, float_dtype=torch.float32,
                )
                scene_values, scene_types = reader.scene_tokens(0, float_dtype=torch.float32, int_dtype=torch.int32)
                provider = SceneTemplateProvider(reader, normalizer=normalizer, enabled=bool(len(scene_values)))
                builder = LiveBatchBuilder(
                    backend=backend, vocab=vocabulary, normalizer=normalizer, schema=reader.schema,
                    skill_feature_names=reader.skill_feature_names, scene_provider=provider,
                    device=torch.device("cpu"), max_history=5,
                    model_config=ModelConfig(history_capacity=5, history_reset_keep=2),
                    action_keys=reader.action_keys, action_is_gcd=reader.action_is_gcd,
                )
                encoded = []
                for sample in payload["samples"]:
                    for limit in (0, 2, 5):
                        batch, _ = builder.build_from_canonical(sample["context"], gcd_phase=True, max_history=limit)
                        encoded.append({"step": sample["step"], "max_history": limit, "inputs": {
                            field: _tensor_payload(batch[field]) for field in (
                                "history_skill_ids", "history_skill_features", "history_mask",
                                "scene_vectors", "scene_types", "scene_mask",
                            )
                        }})
                results[name] = {
                    "canonical": [{
                        "step": sample["step"], "history_cursor": sample["context"]["history_cursor"],
                        "skill_history_context": sample["context"]["skill_history_context"],
                        "scene_context": sample["context"]["scene_context"],
                        "label_action_key": sample["label"]["action_key"],
                        "label_action_index": sample["label"]["action_index"],
                    } for sample in payload["samples"]],
                    "raw": {
                        **{field: _tensor_payload(bank[field]) for field in (
                            "skill_ids", "skill_features", "skill_potencies", "cumulative_dot_potencies",
                        )},
                        "bank_action_keys": list(bank["action_keys"]),
                        "action_keys": list(reader.action_keys),
                        "action_to_vocab_id": list(reader.action_to_vocab_id),
                        "action_is_gcd": list(reader.action_is_gcd),
                        "scene_abs_values": _tensor_payload(scene_values),
                        "scene_types": _tensor_payload(scene_types),
                    },
                    "model": encoded,
                }
    return results


def _tensor_payload(value):
    return {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin-commit", required=True)
    args = parser.parse_args()
    repo, output = args.repo_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite frozen baseline: {output}")
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    source_paths = (
        "Combat.Sim/FightEngine/Outputs/TokenBuilders/SkillTokenBuilder.cs",
        "scripts/convert_fflogs/training/training.py",
        "scripts/autoregressive_replay/context.py",
        "common/policy/data/context_encoding.py",
        "common/policy/data/normalizer.py",
    )
    artifact = {
        "origin_commit": args.origin_commit,
        "source_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in source_paths},
        "description": "真实技能与 policy wait；空场景和四类窗口；非二进制端点；空历史和裁剪窗口；CPU FP32，无训练",
        "cases": capture_cases(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

"""GRPO 后训练核心数学、变长 live batch 和独立 CLI 测试。"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from grpo.config import GrpoConfig, GrpoRunConfig, load_grpo_config
from grpo.trainer import (
    GrpoDecision,
    _restore_grpo_rollback_state,
    _detach_batch_to_cpu,
    _update_policy,
    collate_grpo_decisions,
    compute_baseline_relative_advantages,
    compute_grpo_loss,
    run_grpo_training,
)
from grpo.storage import GrpoRolloutStore
from common.policy.model import repetition as repetition_module
from common.policy.config import resolve_policy_grpo_dir


def test_grpo_import_does_not_require_pretraining_package():
    """GRPO 的导入边界不能要求预训练包存在。"""
    project_root = Path(__file__).resolve().parents[2]
    script = (
        "import builtins\n"
        "real_import = builtins.__import__\n"
        "def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    if level == 0 and (name == 'training' or name.startswith('training.')):\n"
        "        raise AssertionError(f'unexpected GRPO import of {name}')\n"
        "    return real_import(name, globals, locals, fromlist, level)\n"
        "builtins.__import__ = guarded_import\n"
        "import grpo.grpo\n"
        "import grpo.trainer\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _decision(*, history_length: int, scene_length: int, action_index: int) -> GrpoDecision:
    candidate_keys = ["fire_iii", "blizzard_iii", "ogcd_wait"]
    batch = {
        "scene_vectors": torch.zeros((1, scene_length, 2), dtype=torch.float32),
        "scene_types": torch.zeros((1, scene_length), dtype=torch.int32),
        "scene_mask": torch.ones((1, scene_length), dtype=torch.bool),
        "history_skill_ids": torch.zeros((1, history_length), dtype=torch.int32),
        "history_skill_features": torch.zeros(
            (1, history_length, 2),
            dtype=torch.float32,
        ),
        "history_state_vectors": torch.zeros(
            (1, history_length, 3),
            dtype=torch.float32,
        ),
        "history_state_null_mask": torch.zeros(
            (1, history_length, 3),
            dtype=torch.bool,
        ),
        "history_mask": torch.ones((1, history_length), dtype=torch.bool),
        "candidate_skill_ids": torch.zeros((1, 3), dtype=torch.int32),
        "candidate_skill_features": torch.zeros((1, 3, 2), dtype=torch.float32),
        "candidate_state_vectors": torch.zeros((1, 3, 3), dtype=torch.float32),
        "candidate_state_null_mask": torch.zeros((1, 3, 3), dtype=torch.bool),
        "candidate_legal_mask": torch.ones((1, 3), dtype=torch.bool),
        "history_action_keys": [["fire_iii"] * history_length],
        "candidate_action_keys": [candidate_keys],
    }
    return GrpoDecision(
        batch=batch,
        candidate_keys=tuple(candidate_keys),
        action_index=action_index,
        old_logprob=-1.0986122886681098,
    )


def test_baseline_relative_advantages_preserve_greedy_zero_point():
    advantages = compute_baseline_relative_advantages(((-1.0, -2.0), (1.0, -1.0)))

    assert torch.all(advantages[0] < 0)
    assert advantages[1][0] > 0
    assert advantages[1][1] < 0


def test_baseline_relative_advantages_protect_low_variance_scale():
    advantages = compute_baseline_relative_advantages(
        ((1.0, 1.001, 0.999, 1.002), (-1.0, -1.001, -0.999, -1.002))
    )

    assert torch.isfinite(torch.cat(advantages)).all()
    assert torch.all(advantages[0] > 0)
    assert torch.all(advantages[1] < 0)
    assert float(torch.cat(advantages).abs().max()) <= 5.0


def test_baseline_relative_advantages_protect_identical_nonzero_group():
    advantages = compute_baseline_relative_advantages(((2.0, 2.0, 2.0, 2.0),))

    assert torch.equal(advantages[0], torch.full((4,), 5.0))


def test_grpo_rollback_restores_previous_disk_checkpoint(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    input_batch = torch.ones((2, 2))
    loss = model(input_batch).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    expected_model = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    expected_optimizer = deepcopy(optimizer.state_dict())
    expected_scheduler = deepcopy(scheduler.state_dict())
    checkpoint_path = tmp_path / "latest.pt"
    torch.save(
        {
            "model_state_dict": expected_model,
            "optimizer_state_dict": expected_optimizer,
            "scheduler_state_dict": expected_scheduler,
        },
        checkpoint_path,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    optimizer.zero_grad(set_to_none=True)

    _restore_grpo_rollback_state(
        checkpoint_path=checkpoint_path,
        initial_model_state={},
        initial_optimizer_state=None,
        initial_scheduler_state=None,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    for key, value in expected_model.items():
        assert torch.equal(model.state_dict()[key], value)
    actual_optimizer = optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    assert actual_optimizer["state"].keys() == expected_optimizer["state"].keys()
    for state_key, expected_state in expected_optimizer["state"].items():
        for field, expected_value in expected_state.items():
            actual_value = actual_optimizer["state"][state_key][field]
            if isinstance(expected_value, torch.Tensor):
                assert torch.equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value
    assert scheduler.state_dict() == expected_scheduler


def test_collate_grpo_decisions_right_pads_scene_and_history():
    batch = collate_grpo_decisions(
        (
            _decision(history_length=1, scene_length=2, action_index=0),
            _decision(history_length=3, scene_length=0, action_index=1),
        )
    )

    assert batch["scene_vectors"].shape == (2, 2, 2)
    assert batch["scene_mask"].tolist() == [[True, True], [False, False]]
    assert batch["history_skill_ids"].shape == (2, 3)
    assert batch["history_mask"].tolist() == [[True, False, False], [True, True, True]]
    assert batch["history_state_null_mask"].tolist() == [
        [[False, False, False], [True, True, True], [True, True, True]],
        [[False, False, False], [False, False, False], [False, False, False]],
    ]
    assert batch["candidate_legal_mask"].shape == (2, 3)


def test_rollout_decision_copy_leaves_inference_mode():
    with torch.inference_mode():
        source = {"values": torch.ones((1, 2))}
        copied = _detach_batch_to_cpu(source)

    assert copied["values"].is_inference() is False


def test_compute_grpo_loss_is_finite_and_differentiable(monkeypatch):
    class ToyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(3))

        def forward(self, batch):
            return {"logits": self.logits.unsqueeze(0).expand(batch["candidate_skill_ids"].shape[0], -1)}

    policy = ToyPolicy()
    batch = collate_grpo_decisions(
        (
            _decision(history_length=0, scene_length=0, action_index=0),
            _decision(history_length=1, scene_length=1, action_index=1),
        )
    )
    grpo = GrpoConfig(
        group_size=2,
        max_iterations=1,
        minibatch_size=2,
    )
    with monkeypatch.context() as guard:
        guard.setattr(
            torch.Tensor, "item",
            lambda *_args, **_kwargs: pytest.fail("GRPO loss must not synchronize scalar metrics"),
        )
        loss, metrics = compute_grpo_loss(
            policy,
            batch,
            action_indices=torch.tensor([0, 1]),
            old_logprobs=torch.tensor([-1.0986122886681098, -1.0986122886681098]),
            advantages=torch.tensor([1.0, -1.0]),
            grpo=grpo,
            repetition=SimpleNamespace(enabled=False),
            device=torch.device("cpu"),
            precision="float32",
        )

    assert torch.isfinite(loss)
    assert metrics["kl"] == pytest.approx(0.0, abs=1e-6)
    assert all(isinstance(value, torch.Tensor) and not value.requires_grad for value in metrics.values())
    loss.backward()
    assert policy.logits.grad is not None
    assert torch.isfinite(policy.logits.grad).all()


def test_collate_grpo_decisions_preserves_all_empty_sequences():
    batch = collate_grpo_decisions((
        _decision(history_length=0, scene_length=0, action_index=0),
        _decision(history_length=0, scene_length=0, action_index=1),
    ))
    assert batch["history_skill_ids"].shape == (2, 0)
    assert batch["history_state_null_mask"].shape == (2, 0, 3)
    assert batch["history_state_null_mask"].dtype == torch.bool
    assert batch["scene_vectors"].shape == (2, 0, 2)
    assert batch["scene_mask"].shape == (2, 0)


def test_grpo_config_rejects_nonpositive_time_horizon():
    with pytest.raises(ValueError, match="max_duration_seconds must be > 0"):
        GrpoConfig(max_duration_seconds=0.0)


def test_grpo_defaults_use_sixteen_samples_and_scene_time_horizon():
    config = GrpoConfig()
    assert config.group_size == 16
    assert config.max_duration_seconds is None


def test_grpo_temperature_default_is_higher_and_increases_entropy():
    assert GrpoConfig().temperature == pytest.approx(1.3)
    config = load_grpo_config(Path("config/models/black_mage/artzip/config.yaml"))
    assert config.group_size == 16
    assert config.max_duration_seconds == pytest.approx(1200.0)
    assert config.temperature == pytest.approx(1.3)

    class NonUniformPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([4.0, 1.0, -1.0]))

        def forward(self, batch):
            return {
                "logits": self.logits.unsqueeze(0).expand(
                    batch["candidate_skill_ids"].shape[0], -1
                )
            }

    batch = collate_grpo_decisions(
        (_decision(history_length=0, scene_length=0, action_index=0),)
    )
    policy = NonUniformPolicy()

    def entropy_for(temperature: float) -> float:
        _, metrics = compute_grpo_loss(
            policy,
            batch,
            action_indices=torch.tensor([0]),
            old_logprobs=torch.tensor([-0.1]),
            advantages=torch.tensor([0.0]),
            grpo=GrpoConfig(
                temperature=temperature,
                max_iterations=1,
            ),
            repetition=SimpleNamespace(enabled=False),
            device=torch.device("cpu"),
            precision="float32",
        )
        return float(metrics["entropy"])

    assert entropy_for(1.3) > entropy_for(1.0)


def test_grpo_training_closes_replay_session_on_outer_failure(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    scene_path = tmp_path / "scene.json"
    scene_path.write_text("{}", encoding="utf-8", newline="\n")
    config = GrpoRunConfig(
        raw_data_dir=tmp_path / "raw",
        output_dir=tmp_path / "output",
        job_tag="black_mage",
    )
    grpo = GrpoConfig(group_size=2, max_iterations=1)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = config.model

    class FakeBackend:
        def __init__(self):
            self.model = FakeModel()
            self.data_spec = SimpleNamespace(
                job_tag="black_mage",
                num_candidates=1,
                candidate_action_keys=("fire",),
            )
            self.input_contract = object()
            self.repetition = SimpleNamespace()
            self.checkpoint = {}

    backend = FakeBackend()

    class FakeSession:
        instance = None

        def __init__(self, _config, *, backend, cache_store):
            del _config, cache_store
            self.backend = backend
            self.close_calls = 0
            FakeSession.instance = self

        def close(self):
            self.close_calls += 1

    monkeypatch.setattr(
        "grpo.trainer.PyTorchPolicyBackend",
        lambda *_args, **_kwargs: backend,
    )
    monkeypatch.setattr("grpo.trainer.AutoregressiveReplaySession", FakeSession)
    monkeypatch.setattr(
        "grpo.trainer.resolve_policy_cache_dir",
        lambda _job_tag: tmp_path / "cache",
    )
    monkeypatch.setattr(
        "grpo.trainer.resolve_policy_grpo_dir",
        lambda _job_tag: tmp_path / "grpo",
    )
    monkeypatch.setattr(
        "grpo.trainer._run_scene_rollout",
        lambda *_args, **_kwargs: (
            SimpleNamespace(ppg=1.0),
            (_decision(history_length=0, scene_length=0, action_index=0),),
        ),
    )
    monkeypatch.setattr(
        "grpo.trainer.compute_baseline_relative_advantages",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("metric failure")
        ),
    )

    with pytest.raises(RuntimeError, match="metric failure"):
        run_grpo_training(
            config,
            grpo=grpo,
            checkpoint_path=checkpoint_path,
            raw_paths=(scene_path,),
            device_name="cpu",
        )

    assert FakeSession.instance is not None
    assert FakeSession.instance.close_calls == 1


@dataclass
class _StubConfig:
    raw_data_dir: Path
    output_dir: Path
    job_tag: str | None
    model_variant: str | None = None


def test_grpo_cli_forwards_max_files_and_overrides(monkeypatch, capsys, tmp_path):
    cli = importlib.import_module("grpo.grpo")
    config_path = tmp_path / "config.yaml"
    input_config = _StubConfig(
        raw_data_dir=tmp_path / "raw-default",
        output_dir=tmp_path / "bc",
        job_tag=None,
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(cli, "resolve_policy_model_config_path", lambda value: config_path)
    monkeypatch.setattr(cli, "load_grpo_run_config", lambda path: input_config)
    monkeypatch.setattr(cli, "load_grpo_config", lambda path: GrpoConfig())
    monkeypatch.setattr(cli, "resolve_policy_model_job_tag", lambda path: "black_mage")
    monkeypatch.setattr(cli, "resolve_policy_model_variant", lambda path: "artzip")
    monkeypatch.setattr(cli, "resolve_policy_device", lambda value: "cpu")
    monkeypatch.setattr(cli, "resolve_policy_checkpoint_path", lambda path: tmp_path / "best.pt")
    monkeypatch.setattr(
        cli,
        "_prepare_grpo_scenes",
        lambda config, max_files: calls.update(
            {"scene_config": config, "max_files": max_files}
        ) or [tmp_path / "scene.json"],
    )

    def fake_run(config, **kwargs):
        calls["config"] = config
        calls["kwargs"] = kwargs
        return {"final_checkpoint": tmp_path / "grpo" / "final.pt"}

    monkeypatch.setattr(cli, "run_grpo_training", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "grpo.py",
            "--config",
            str(config_path),
            "--max-files",
            "7",
            "--group-size",
            "5",
            "--iterations",
            "2",
            "--device",
            "cpu",
        ],
    )

    cli.main()

    assert calls["max_files"] == 7
    assert calls["scene_config"].job_tag == "black_mage"
    assert calls["kwargs"]["grpo"].group_size == 5
    assert calls["kwargs"]["grpo"].max_iterations == 2
    assert calls["kwargs"]["checkpoint_path"] == tmp_path / "best.pt"
    assert calls["kwargs"]["device_name"] == "cpu"
    assert "final.pt" in capsys.readouterr().out


def test_grpo_directory_follows_env_cache_root_and_job(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAINING_CACHE_ROOT", str(tmp_path / "data" / "human" / "job"))

    assert resolve_policy_grpo_dir("black_mage") == (
        tmp_path / "data" / "human" / "job" / "black_mage" / "grpo"
    ).resolve()


def test_grpo_rollout_store_persists_and_streams_cpu_decisions(tmp_path):
    decision = _decision(history_length=1, scene_length=2, action_index=1)
    store = GrpoRolloutStore(tmp_path / "grpo", iteration=1, run_id="test-run")

    entry = store.write_trajectory(
        scene_json_path=tmp_path / "scene.json",
        decisions=(decision,),
        ppg=12.5,
        greedy_ppg=10.0,
        reward=2.5,
    )
    store.set_advantages((1.25,))

    assert entry.path == (
        tmp_path / "grpo" / "test-run" / "iteration_001" / "trajectory_00000.pt"
    )
    assert entry.path.is_file()
    assert store.manifest_path.is_file()
    minibatches = list(store.iter_minibatches(1))
    assert len(minibatches) == 1
    loaded_decisions, loaded_advantages = minibatches[0]
    assert loaded_decisions[0].action_index == decision.action_index
    assert loaded_decisions[0].batch["scene_vectors"].device.type == "cpu"
    assert loaded_advantages.tolist() == [1.25]


def test_grpo_update_keeps_minibatch_weighting_and_defers_host_reads(monkeypatch):
    policy = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0)
    decisions = [_decision(history_length=i, scene_length=i, action_index=0) for i in range(3)]
    config = GrpoConfig(minibatch_size=2, inner_updates=2)

    def loss_fn(model, batch, **kwargs):
        assert "repetition_penalty_mask" in batch
        size = batch["candidate_skill_ids"].shape[0]
        loss = model.weight.sum() * size
        return loss, {key: loss.detach() * 0 + size for key in ("loss", "policy_loss", "kl", "entropy", "clip_fraction", "mean_ratio")}

    monkeypatch.setattr("grpo.trainer.compute_grpo_loss", loss_fn)
    monkeypatch.setattr(torch.Tensor, "item", lambda *_args, **_kwargs: pytest.fail("GRPO metrics must not call item"))
    metrics = _update_policy(
        policy, optimizer, None, decisions, torch.ones(3), grpo=config,
        repetition=repetition_module.RepetitionConfig("blacklist", ("fire_iii",), 1.0),
        device=torch.device("cpu"), precision="float32",
    )
    assert metrics.pop("optimizer_updates") == 4.0
    assert set(metrics.values()) == {1.5}


def test_grpo_update_streams_disk_rollouts_without_full_decision_buffer(monkeypatch, tmp_path):
    policy = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0)
    store = GrpoRolloutStore(tmp_path / "grpo", iteration=1, run_id="stream-test")
    for index in range(3):
        store.write_trajectory(
            scene_json_path=tmp_path / f"scene-{index}.json",
            decisions=(_decision(history_length=index, scene_length=index, action_index=0),),
            ppg=float(index),
            greedy_ppg=0.0,
            reward=float(index),
        )
    store.set_advantages((1.0, -1.0, 0.5))

    def loss_fn(model, batch, **kwargs):
        size = batch["candidate_skill_ids"].shape[0]
        loss = model.weight.sum() * size
        return loss, {
            key: loss.detach() * 0 + size
            for key in ("loss", "policy_loss", "kl", "entropy", "clip_fraction", "mean_ratio")
        }

    monkeypatch.setattr("grpo.trainer.compute_grpo_loss", loss_fn)
    metrics = _update_policy(
        policy,
        optimizer,
        None,
        store,
        None,
        grpo=GrpoConfig(minibatch_size=2, inner_updates=2),
        repetition=repetition_module.RepetitionConfig("blacklist", ("fire_iii",), 1.0),
        device=torch.device("cpu"),
        precision="float32",
    )

    assert metrics["optimizer_updates"] == 4.0

"""C# 状态机进程内适配器测试。"""

from __future__ import annotations

import subprocess

import pytest

from scripts.common.inprocess_backend import InProcessBackend
from tests.scripts.conftest import _require_inprocess_backend


def test_inprocess_backend_submit_rejects_illegal_cooldown_action(cs_backend):
    """进程内状态机在动作产生任何状态变更前拒绝尚未转好的 oGCD。"""
    first = cs_backend.submit_action(0.0, "lucid_dreaming")
    assert first.accepted
    cs_backend.advance_to(21.0)

    before = cs_backend.observe_at(21.0, format="seconds").context
    rejected = cs_backend.submit_action(21.0, "lucid_dreaming")
    after = cs_backend.observe_at(21.0, format="seconds").context

    assert not rejected.accepted
    assert rejected.reason == "cooldown_locked"
    assert after == before


def test_inprocess_backend_accepts_observed_cast_duration_override(cs_backend):
    """日志缺少 begincast 时，转换层提供的实际读条时长应覆盖默认技能表读条。"""
    result = cs_backend.submit_action(
        0.0,
        "fire_iii",
        actual_cast_seconds=0.0,
    )

    assert result.accepted
    assert result.effect_timestamp == 0.0
    state = cs_backend.observe_at(0.0, format="seconds").context
    assert state["gcd_remaining_seconds"] == 2.5


def test_inprocess_backend_never_launches_a_host_process(monkeypatch):
    """进程内状态机即使首次装载 .NET runtime 也不得启动子进程。"""
    _require_inprocess_backend()

    def reject_process(*_args, **_kwargs):
        pytest.fail("InProcessBackend must not launch a subprocess")

    monkeypatch.setattr(subprocess, "Popen", reject_process)
    with InProcessBackend("black_mage") as backend:
        assert backend.validate_at(0.0, "fire_iii").legal


def test_inprocess_backend_returns_native_python_context():
    """观测结果由 PythonBridge 直接转换为 Python 原生容器。"""
    _require_inprocess_backend()
    with InProcessBackend("black_mage", actual_base_gcd=2.46, max_history=32) as backend:
        observation = backend.observe_at(
            0.0,
            format="vector",
            next_observation_timestamp=0.0,
        )

    def assert_native_python_tree(value):
        if type(value) is dict:
            for key, item in value.items():
                assert type(key) is str
                assert_native_python_tree(item)
        elif type(value) is list:
            for item in value:
                assert_native_python_tree(item)
        else:
            assert value is None or type(value) in (str, int, float, bool)

    assert_native_python_tree(observation.context)
    context = observation.context
    assert isinstance(context, dict)
    assert isinstance(context["skill_history_context"], list)
    assert isinstance(context["state_history_context"]["tokens"], list)
    assert isinstance(context["candidate_skill_context"], list)


@pytest.mark.parametrize(
    ("job_tag", "action_key"),
    [("black_mage", "fire_iii"), ("machinist", "heated_split_shot")],
)
def test_model_vectors_exclude_scheduling_fields_but_runtime_keeps_them(job_tag, action_key):
    """两个职业的模型不读取调度计时；运行时仍可使用完整秒制观测。"""
    _require_inprocess_backend()
    with InProcessBackend(job_tag, fight_remaining=60.0, max_history=32) as backend:
        assert backend.submit_action(0.0, action_key).accepted
        backend.advance_to(4.0)
        scalar = backend.observe_at(4.0, format="seconds").context
        canonical = backend.observe_at(4.0, format="vector", next_observation_timestamp=4.5).context

    assert scalar["gcd_index"] == 1
    assert scalar["fight_remaining_seconds"] == pytest.approx(56.0)
    assert "gcd_remaining_seconds" in scalar
    assert "weave_window_seconds" in scalar
    removed_fields = {
        "gcd_index", "fight_remaining_seconds", "gcd_remaining_gcds",
        "weave_window_gcds", "ogcd_window_gcds", "downtime_remaining_gcds",
        "gcd_remaining_seconds", "weave_window_seconds", "ogcd_window_seconds",
        "next_untargetable_in_gcds", "remaining_gcds",
    }
    for context_key in ("state_history_context", "candidate_state_context"):
        keys = canonical[context_key]["player_state_feature_keys"]
        assert len(keys) == 22
        all_keys = [
            feature_key
            for group_key, feature_keys in canonical[context_key].items()
            if group_key.endswith("_feature_keys")
            for feature_key in feature_keys
        ]
        assert not any(
            key.rsplit(".", 1)[-1] in removed_fields or key.endswith("_gcds")
            for key in all_keys
        )
        assert "before.current_gcd_seconds" in keys
        assert "after.downtime_remaining_seconds" in keys
    for context_key in ("skill_history_context", "candidate_skill_context"):
        assert canonical[context_key]
        assert all("gcd_index" not in token for token in canonical[context_key])

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

"""C# 状态机后端客户端（SidecarBackend）进程生命周期测试。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from common.contracts import SIDECAR_CONTRACT_VERSION
from scripts.common import cs_backend as cs_backend_mod
from scripts.common.cs_backend import SidecarBackend
from scripts.common.inprocess_backend import InProcessBackend


@pytest.fixture
def fake_sidecar_env(monkeypatch, tmp_path):
    """隔离构造路径：假 exe 存在 + 假 Popen，避免依赖真实 SidecarHost 进程。"""
    fake_exe = tmp_path / "SidecarHost.exe"
    fake_exe.write_bytes(b"fake")
    monkeypatch.setattr(cs_backend_mod, "_SIDECAR_EXE", fake_exe)
    fake_proc = MagicMock()
    fake_proc.stdout.readline.return_value = ""
    monkeypatch.setattr(cs_backend_mod.subprocess, "Popen", lambda *args, **kwargs: fake_proc)
    return fake_proc


def test_sidecar_backend_init_failure_recycles_child_process(fake_sidecar_env):
    """init 失败（进程崩溃）时子进程必须被回收，不能泄漏孤儿 SidecarHost。"""
    fake_proc = fake_sidecar_env

    with pytest.raises(RuntimeError, match="进程已退出"):
        SidecarBackend("black_mage")

    # close() 回收路径：发送 close 失败（进程已死）后必须 wait 回收
    assert fake_proc.wait.call_count >= 1


def test_sidecar_backend_rejected_init_recycles_child_process(fake_sidecar_env, monkeypatch):
    """init 被后端拒绝（ok:false，如不支持的 job_tag）时同样回收子进程。"""
    fake_proc = fake_sidecar_env
    fake_proc.stdout.readline.return_value = (
        '{"seq": 1, "ok": false, "error": "unsupported job tag"}'
    )

    with pytest.raises(RuntimeError, match="unsupported job tag"):
        SidecarBackend("unknown_job")

    assert fake_proc.wait.call_count >= 1


def test_sidecar_backend_close_kills_stuck_child(fake_sidecar_env):
    """close 时子进程 5s 内不退出则 kill 兜底，并 wait 回收。"""
    fake_proc = fake_sidecar_env
    fake_proc.stdout.readline.side_effect = [
        f'{{"seq": 1, "ok": true, "timestamp": 0.0, "sidecar_contract_version": {SIDECAR_CONTRACT_VERSION}}}',  # init 成功
        "",                          # close 命令时进程已崩溃
    ]
    fake_proc.wait.side_effect = [subprocess.TimeoutExpired("SidecarHost", 5), 0]

    backend = SidecarBackend("black_mage")
    backend.close()

    fake_proc.kill.assert_called_once()
    assert fake_proc.wait.call_count == 2


def test_sidecar_backend_init_failure_not_masked_by_close_error(fake_sidecar_env):
    """close() 清理自身抛意外异常时，不得覆盖原始构造异常。"""
    fake_proc = fake_sidecar_env
    fake_proc.stdout.readline.return_value = ""  # init 进程崩溃
    fake_proc.wait.side_effect = ValueError("unexpected close failure")

    with pytest.raises(RuntimeError, match="进程已退出"):
        SidecarBackend("black_mage")


def test_sidecar_backend_init_failure_logs_suppressed_close_error(fake_sidecar_env, caplog):
    """close 清理异常被抑制时记录日志细节，不静默丢弃。"""
    fake_proc = fake_sidecar_env
    fake_proc.stdout.readline.return_value = ""  # init 进程崩溃
    fake_proc.wait.side_effect = ValueError("unexpected close failure")

    with caplog.at_level("WARNING", logger="scripts.common.cs_backend"):
        with pytest.raises(RuntimeError, match="进程已退出"):
            SidecarBackend("black_mage")

    assert "close 清理异常被抑制" in caplog.text
    assert "unexpected close failure" in caplog.text


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


def test_sidecar_backend_close_is_idempotent_after_dead_child(fake_sidecar_env):
    """子进程已退出后 close 不抛错（正常转换 finally 路径）。"""
    fake_proc = fake_sidecar_env
    fake_proc.stdout.readline.side_effect = [
        f'{{"seq": 1, "ok": true, "timestamp": 0.0, "sidecar_contract_version": {SIDECAR_CONTRACT_VERSION}}}',  # init 成功
        "", "",                      # 两次 close 时进程已崩溃
    ]

    backend = SidecarBackend("black_mage")
    backend.close()
    backend.close()


def test_sidecar_backend_rejects_stale_runtime_contract(fake_sidecar_env):
    """旧 DLL 即使能启动，也必须在 init 阶段因输出契约过期而拒绝。"""
    fake_proc = fake_sidecar_env
    fake_proc.stdout.readline.return_value = (
        f'{{"seq": 1, "ok": true, "sidecar_contract_version": {SIDECAR_CONTRACT_VERSION - 1}}}'
    )

    with pytest.raises(RuntimeError, match="契约不匹配"):
        SidecarBackend("black_mage")

    assert fake_proc.wait.call_count >= 1


def test_sidecar_backend_reinit_preserves_max_history(fake_sidecar_env):
    """转换提取与训练阶段重复 init 时必须继续发送历史上限。"""
    fake_proc = fake_sidecar_env
    fake_proc.stdout.readline.side_effect = [
        f'{{"seq": 1, "ok": true, "timestamp": 0.0, "sidecar_contract_version": {SIDECAR_CONTRACT_VERSION}}}',
        f'{{"seq": 2, "ok": true, "timestamp": 0.0, "sidecar_contract_version": {SIDECAR_CONTRACT_VERSION}}}',
        "",  # close：模拟子进程已退出
    ]

    backend = SidecarBackend("black_mage", max_history=768)
    backend.init(actual_base_gcd=2.17)

    requests = [
        json.loads(call.args[0])
        for call in fake_proc.stdin.write.call_args_list[:2]
    ]
    assert requests[0]["max_history"] == 768
    assert requests[1]["max_history"] == 768

    backend.close()


def test_inprocess_backend_never_launches_a_host_process(monkeypatch):
    """进程内状态机即使首次装载 .NET runtime 也不得启动子进程。"""

    def reject_process(*_args, **_kwargs):
        pytest.fail("InProcessBackend must not launch SidecarHost")

    monkeypatch.setattr(subprocess, "Popen", reject_process)
    with InProcessBackend("black_mage") as backend:
        assert backend.validate_at(0.0, "fire_iii").legal


def test_inprocess_backend_matches_sidecar_state_and_outputs():
    """进程内调用与 Sidecar 协议使用同一状态机语义及 Python 输出形状。"""
    with (
        SidecarBackend("black_mage", actual_base_gcd=2.46, max_history=32) as sidecar,
        InProcessBackend("black_mage", actual_base_gcd=2.46, max_history=32) as direct,
    ):
        direct_observation = direct.observe_at(
            0.0,
            format="vector",
            next_observation_timestamp=0.0,
        )
        sidecar_observation = sidecar.observe_at(
            0.0,
            format="vector",
            next_observation_timestamp=0.0,
        )
        assert direct_observation == sidecar_observation

        direct_validation = direct.validate_at(0.0, "fire_iii")
        sidecar_validation = sidecar.validate_at(0.0, "fire_iii")
        assert direct_validation == sidecar_validation

        direct_submission = direct.submit_action(0.0, "fire_iii", actual_cast_seconds=0.0)
        sidecar_submission = sidecar.submit_action(0.0, "fire_iii", actual_cast_seconds=0.0)
        assert direct_submission.accepted == sidecar_submission.accepted
        assert direct_submission.queued == sidecar_submission.queued
        assert direct_submission.reason == sidecar_submission.reason
        assert direct_submission.request_timestamp == sidecar_submission.request_timestamp
        assert direct_submission.accepted_timestamp == sidecar_submission.accepted_timestamp
        assert direct_submission.effect_timestamp == sidecar_submission.effect_timestamp
        assert direct_submission.next_scheduled_event_time == sidecar_submission.next_scheduled_event_time

        assert direct.advance_to(0.0) == sidecar.advance_to(0.0)
        assert direct.observe_at(0.0, format="seconds") == sidecar.observe_at(
            0.0,
            format="seconds",
        )

        direct_event = direct.apply_external_event(
            1.0,
            "target_count_changed",
            target_count=2,
        )
        sidecar_event = sidecar.apply_external_event(
            1.0,
            "target_count_changed",
            target_count=2,
        )
        assert direct_event == sidecar_event
        assert direct.observe_at(1.0, format="seconds") == sidecar.observe_at(
            1.0,
            format="seconds",
        )
        direct_policy = direct.record_policy_action(1.0, "ogcd_wait", 3.46)
        sidecar_policy = sidecar.record_policy_action(1.0, "ogcd_wait", 3.46)
        assert direct_policy == sidecar_policy

"""报告转换流水线 —— 两级转换编排。"""

from __future__ import annotations

from collections import Counter

from .config import load_convert_fflogs_job_config
from .constants import DEFAULT_DOWNTIME_GAP_SECONDS
from .extraction import detect_gcd_from_logs, extract_supported_actions
from .fight_payload import build_fight_payload, build_output_fight_id
from .training import build_training_samples, resolve_initial_timestamp


def convert_report_payload(
    report_payload: dict[str, object],
    *,
    job_tag: str,
    project_config,
    skill_book,
    source_id: int,
    encounter_name: str | None,
    report_code: str,
    player_name: str,
    generated_at: str,
    downtime_gap_seconds: float = DEFAULT_DOWNTIME_GAP_SECONDS,
) -> tuple[dict[str, object] | None, Counter[tuple[int, str]]]:
    """把原始 FFLogs payload 转成训练样本输入载荷。

    本阶段不需要状态机：请求时刻与读条时长全部由日志事件解析得出。
    """
    convert_job_config = load_convert_fflogs_job_config(job_tag)
    gcd_time = detect_gcd_from_logs(
        report_payload.get("events", []),
        source_id,
        gcd_detection=convert_job_config.gcd_detection,
        haste_multiplier=project_config.job.timing.get("ley_lines_haste_multiplier"),
        skill_table_base_gcd=float(project_config.system.skill_table_base_gcd),
    )
    actions, ignored_skill_counts = extract_supported_actions(
        report_payload,
        source_id,
        skill_book,
        actual_base_gcd=gcd_time,
        skill_table_base_gcd=float(project_config.system.skill_table_base_gcd),
        action_queue_window_seconds=float(project_config.engine_timing.action_queue_window_seconds),
    )
    if not actions:
        return None, ignored_skill_counts

    fight_payload = build_fight_payload(
        actions,
        fight_id=build_output_fight_id(report_payload, report_code),
        player_name=player_name,
        encounter_name=encounter_name,
        report_code=report_code,
        source_id=source_id,
        job_tag=job_tag,
        gcd_time=gcd_time,
        generated_at=generated_at,
        downtime_gap_seconds=downtime_gap_seconds,
        downtime_boundary_margin=float(project_config.engine_timing.action_queue_window_seconds),
        raid_buff_marker_keys=project_config.system.raid_buff_window_marker_skills,
        raid_buff_window_duration=project_config.system.raid_buff_window_duration,
        raw_events=report_payload.get("events", []),
        skill_book=skill_book,
    )
    return fight_payload, ignored_skill_counts


def convert_report_to_training_payload(
    report_payload: dict[str, object],
    *,
    backend,
    project_config,
    skill_book,
    source_id: int,
    encounter_name: str | None,
    report_code: str,
    player_name: str,
    generated_at: str,
    downtime_gap_seconds: float = DEFAULT_DOWNTIME_GAP_SECONDS,
) -> tuple[dict[str, object] | None, Counter[tuple[int, str]]]:
    """直接把一份原始 FFLogs payload 转成最终训练样本。"""
    fight_payload, ignored_skill_counts = convert_report_payload(
        report_payload,
        job_tag=backend.job_tag,
        project_config=project_config,
        skill_book=skill_book,
        source_id=source_id,
        encounter_name=encounter_name,
        report_code=report_code,
        player_name=player_name,
        generated_at=generated_at,
        downtime_gap_seconds=downtime_gap_seconds,
    )
    if fight_payload is None:
        return None, ignored_skill_counts

    # fight payload 已把整场最早请求平移到 0。
    initial_timestamp = resolve_initial_timestamp(fight_payload)
    backend.init(
        actual_base_gcd=float(fight_payload["gcd_time"]),
        fight_remaining=float(fight_payload.get("duration", 0.0)) - initial_timestamp,
        initial_timestamp=initial_timestamp,
    )
    training_payload = build_training_samples(
        backend,
        skill_book,
        fight_payload,
    )
    return training_payload, ignored_skill_counts

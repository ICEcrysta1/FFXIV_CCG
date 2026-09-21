"""命令行入口。

这个文件只提供最小的手动检查命令；战斗状态机统一由 C#
`SidecarHost` 进程承载，Python 入口只负责静态配置、技能索引和命令编排。
"""

from __future__ import annotations

import argparse
import json

from common.config import load_project_config
from common.skills import SkillBook
from scripts.common.cs_backend import SidecarBackend


def _resolve_job_tag(job_tag: str | None) -> str:
    if job_tag:
        return job_tag
    return load_project_config().job.key


def cmd_validate(job_tag: str | None) -> None:
    """打印当前配置概览，用于检查配置是否装配正确（纯配置，无状态机）。"""
    job = _resolve_job_tag(job_tag)
    config = load_project_config(job_tag=job)
    print(
        json.dumps(
            {
                "project": config.name,
                "version": config.version,
                "job": job,
                "system_skill_count": len(config.system.skills),
                "job_skill_count": len(config.job.skills),
                "skill_count": len(config.system.skills) + len(config.job.skills),
                "system_status_count": len(config.system.statuses),
                "job_status_count": len(config.job.statuses),
                "status_count": len(config.system.statuses) + len(config.job.statuses),
            },
            indent=2,
        )
    )


def cmd_list_skills(job_tag: str | None) -> None:
    """列出当前职业可加载的所有启用技能（静态技能索引，无状态机）。"""
    job = _resolve_job_tag(job_tag)
    skill_book = SkillBook.from_project_config(load_project_config(job_tag=job))
    for skill in skill_book.enabled_skills():
        print(f"{skill.key:16} {skill.kind.value:4} id={skill.game_id}")


def cmd_list_actions(job_tag: str | None) -> None:
    """列出初始状态下的合法动作。"""
    job = _resolve_job_tag(job_tag)
    with SidecarBackend(job_tag=job) as backend:
        observation = backend.observe_at(
            0.0,
            format="vector",
            next_observation_timestamp=0.0,
        )
        context = observation.context
        if not isinstance(context, dict):
            raise RuntimeError("Sidecar vector observation must return a mapping context")
        for token in context["candidate_skill_context"]:
            if bool(token.get("is_legal", False)):
                print(str(token["skill_key"]))


def cmd_smoke(job_tag: str | None) -> None:
    """跑一段最小黑魔循环，确认状态机主链路能执行。"""
    job = _resolve_job_tag(job_tag)
    sequence = [
        "fire_iii",
        "fire_iv",
        "fire_iv",
        "fire_iv",
        "fire_iv",
        "despair",
    ]

    timestamp = 0.0
    with SidecarBackend(job_tag=job) as backend:
        for action in sequence:
            result = backend.submit_action(timestamp, action)
            if not result.accepted or result.accepted_timestamp is None:
                raise RuntimeError(
                    f"smoke action {action!r} was rejected: {result.reason}"
                )

            timestamp = max(timestamp, float(result.accepted_timestamp))
            backend.advance_to(timestamp)
            state = backend.observe_at(timestamp, format="seconds").context
            if not isinstance(state, dict):
                raise RuntimeError("Sidecar seconds observation must return a mapping context")

            wait_seconds = max(
                float(state.get("cast_remaining_seconds", 0.0) or 0.0),
                float(state.get("gcd_remaining_seconds", 0.0) or 0.0),
            )
            if wait_seconds > 0.0:
                timestamp += wait_seconds
                backend.advance_to(timestamp)

        state = backend.observe_at(timestamp, format="seconds").context
        if not isinstance(state, dict):
            raise RuntimeError("Sidecar seconds observation must return a mapping context")
    print(
        json.dumps(
            {
                "time": round(float(state.get("time_seconds", timestamp)), 2),
                "mp": int(state.get("mp", 0)),
                "astral_fire": int(state.get("astral_fire", 0)),
                "umbral_ice": int(state.get("umbral_ice", 0)),
                "umbral_hearts": int(state.get("umbral_hearts", 0)),
                "astral_soul": int(state.get("astral_soul", 0)),
                "polyglot": int(state.get("polyglot", 0)),
            },
            indent=2,
        )
    )


def main() -> None:
    """解析命令行并执行对应检查命令。"""
    parser = argparse.ArgumentParser(description="FFXIV_CCG combat simulator")
    parser.add_argument("command", choices=["validate", "list-skills", "list-actions", "smoke"])
    parser.add_argument("--job-tag", help="按职业 tag 选择状态机路由，默认读取根目录 .env 的 FFXIV_JOB_TAG")
    args = parser.parse_args()

    if args.command == "validate":
        cmd_validate(args.job_tag)
    elif args.command == "list-skills":
        cmd_list_skills(args.job_tag)
    elif args.command == "list-actions":
        cmd_list_actions(args.job_tag)
    else:
        cmd_smoke(args.job_tag)


if __name__ == "__main__":
    main()

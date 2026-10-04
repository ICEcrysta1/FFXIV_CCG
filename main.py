"""命令行入口。

这个文件只提供最小的手动检查命令；战斗状态机由 Python.NET 在当前进程中
直接调用 C# `FightEngine`，Python 入口只负责配置、技能索引和命令编排。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json

from common.config import load_project_config
from common.skills import SkillBook
from scripts.common.inprocess_backend import InProcessEngine


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


def _check_actions(backend):
    context = backend.observe_at(0.0, format="vector", next_observation_timestamp=0.0).context
    if not isinstance(context, dict):
        raise RuntimeError("C# vector observation must return a mapping context")
    return [
        str(action_key)
        for action_key, is_legal in zip(
            context["action_keys"], context["action_legal_mask"], strict=True,
        )
        if is_legal
    ]


def _check_smoke(backend):
    timestamp = 0.0
    for action in ("fire_iii", "fire_iv", "fire_iv", "fire_iv", "fire_iv", "despair"):
        result = backend.submit_action(timestamp, action)
        if not result.accepted or result.accepted_timestamp is None:
            raise RuntimeError(f"smoke action {action!r} was rejected: {result.reason}")
        timestamp = max(timestamp, float(result.accepted_timestamp))
        backend.advance_to(timestamp)
        state = backend.observe_at(timestamp, format="seconds").context
        if not isinstance(state, dict):
            raise RuntimeError("C# seconds observation must return a mapping context")
        wait_seconds = max(float(state.get("cast_remaining_seconds", 0) or 0),
                           float(state.get("gcd_remaining_seconds", 0) or 0))
        if wait_seconds > 0:
            timestamp += wait_seconds
            backend.advance_to(timestamp)
    state = backend.observe_at(timestamp, format="seconds").context
    if not isinstance(state, dict):
        raise RuntimeError("C# seconds observation must return a mapping context")
    return {"time": round(float(state.get("time_seconds", timestamp)), 2),
            **{key: int(state.get(key, 0)) for key in
               ("mp", "astral_fire", "umbral_ice", "umbral_hearts", "astral_soul", "polyglot")}}


def run_checks(command: str, job_tag: str | None, *, queues: int = 1):
    """单次或并行自检共用一个原生引擎，结果按队列顺序返回。"""
    if isinstance(queues, bool) or not isinstance(queues, int) or queues < 1:
        raise ValueError("queues must be a positive integer")
    check = {"list-actions": _check_actions, "smoke": _check_smoke}[command]
    with InProcessEngine(_resolve_job_tag(job_tag), capacity=queues) as engine:
        def run(_index):
            with engine.create_backend(max_history=None) as backend:
                return check(backend)
        with ThreadPoolExecutor(max_workers=queues, thread_name_prefix="sim-check") as pool:
            return list(pool.map(run, range(queues)))


def main() -> None:
    """解析命令行并执行对应检查命令。"""
    parser = argparse.ArgumentParser(description="FFXIV_CCG combat simulator")
    parser.add_argument("command", choices=["validate", "list-skills", "list-actions", "smoke"])
    parser.add_argument("--job-tag", help="按职业 tag 选择状态机路由，默认读取根目录 .env 的 FFXIV_JOB_TAG")
    parser.add_argument("--queues", type=int, default=1, help="自检并发队列数，共用一个引擎")
    args = parser.parse_args()
    if args.queues < 1:
        parser.error("--queues 必须是正整数")
    if args.queues != 1 and args.command in {"validate", "list-skills"}:
        parser.error("--queues 只用于 list-actions 或 smoke")

    if args.command == "validate":
        cmd_validate(args.job_tag)
    elif args.command == "list-skills":
        cmd_list_skills(args.job_tag)
    else:
        results = run_checks(args.command, args.job_tag, queues=args.queues)
        if args.queues > 1:
            print(json.dumps(results, indent=2))
        elif args.command == "list-actions":
            for action in results[0]:
                print(action)
        else:
            print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()

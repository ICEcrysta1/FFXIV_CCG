"""ONNX 正式发布门禁的场景与动作预算策略。"""

from __future__ import annotations


RELEASE_EMPTY_MIN_GCDS = 128
RELEASE_SCENE_MIN_ACTIONS = 100
RELEASE_EMPTY_ACTIONS_PER_GCD_BUDGET = 4
RELEASE_EMPTY_ACTION_BUDGET_RESERVE = 32


def minimum_empty_action_budget(max_gcds: int) -> int:
    """为每个 GCD 的动作与强制首步、额外 weave 留出正式门禁预算。"""
    resolved_gcds = int(max_gcds)
    if resolved_gcds < 1:
        raise ValueError(f"max_gcds must be >= 1, got {max_gcds}")
    return (
        resolved_gcds * RELEASE_EMPTY_ACTIONS_PER_GCD_BUDGET
        + RELEASE_EMPTY_ACTION_BUDGET_RESERVE
    )


def validate_release_scenario_request(
    *,
    scene_mode: str,
    max_steps: int,
    max_gcds: int | None,
) -> str:
    """在昂贵回放前后共用同一套正式门禁场景阈值。"""
    resolved_steps = int(max_steps)
    resolved_gcds = int(max_gcds or 0)
    if scene_mode == "empty" and resolved_gcds >= RELEASE_EMPTY_MIN_GCDS:
        required_steps = minimum_empty_action_budget(resolved_gcds)
        if resolved_steps < required_steps:
            raise ValueError(
                "formal empty-scene parity action budget is too small: "
                f"max_steps={resolved_steps}, required >= {required_steps} for "
                f"max_gcds={resolved_gcds} "
                f"({RELEASE_EMPTY_ACTIONS_PER_GCD_BUDGET} actions/GCD + "
                f"{RELEASE_EMPTY_ACTION_BUDGET_RESERVE} reserve)"
            )
        return "empty_128_gcd"
    if scene_mode == "cache" and resolved_steps >= RELEASE_SCENE_MIN_ACTIONS:
        # 保留既有证据键，避免仅因展示术语修正而让已发布报告失效。
        return "scene_100_decisions"
    raise ValueError(
        "formal release parity requires empty scene with max_gcds >= "
        f"{RELEASE_EMPTY_MIN_GCDS}, or cache scene with max_steps >= "
        f"{RELEASE_SCENE_MIN_ACTIONS} executed actions"
    )

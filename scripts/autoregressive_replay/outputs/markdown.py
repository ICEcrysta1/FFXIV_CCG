"""自回归回放 Markdown 输出。"""

from __future__ import annotations

from pathlib import Path

from ..replay import ReplayResult


SKILL_NAMES = {
    "ogcd_wait": "-",
    "potion": "爆发药",
    "fire_iii": "爆炎",
    "high_fire_ii": "高火",
    "fire_iv": "炽炎",
    "despair": "绝望",
    "flare": "核爆",
    "flare_star": "耀星",
    "blizzard_iii": "冰封",
    "blizzard_iv": "冰澈",
    "freeze": "玄冰",
    "high_blizzard_ii": "高冰冻",
    "paradox": "悖论",
    "xenoglossy": "异言",
    "foul": "秽浊",
    "high_thunder": "高闪雷",
    "high_thunder_ii": "高震雷",
    "transpose": "星灵移位",
    "umbral_soul": "灵极魂",
    "ley_lines": "黑魔纹",
    "retrace": "魔纹重置",
    "manafont": "魔泉",
    "triplecast": "三连咏唱",
    "amplifier": "详述",
    "swiftcast": "即刻咏唱",
    "lucid_dreaming": "醒梦",
    "manaward": "魔罩",
    "surecast": "沉稳咏唱",
}


def write_markdown(result: ReplayResult, output_path: Path) -> Path:
    """写出动作序列和每一步合法 Top-K 候选。"""
    metadata = [
        f"- backend: `{result.backend_name}`",
        f"- execution provider: `{result.execution_provider}`",
        f"- model source: `{result.model_source_path or result.checkpoint_path}`",
    ]
    if result.checkpoint_path is not None:
        metadata.append(f"- checkpoint: `{result.checkpoint_path}`")
    metadata.extend(
        [
            f"- scene raw JSON: `{result.scene_json_path}`",
            f"- input tensor device: `{result.device}`",
            f"- decisions: `{len(result.rows)}`",
            f"- history limit: `{result.history_limit}`",
            f"- PPG: `{result.ppg}`",
            (
                f"- backend metrics: `{result.backend_metrics}`"
                if result.backend_metrics is not None
                else ""
            ),
            (
                "- history experiment: fixed full autoregressive trajectory; only the model input history was truncated"
                if result.is_history_ablation
                else ""
            ),
        ]
    )
    lines = [
        "# Autoregressive 模型回放",
        "",
        "本回放使用真实战斗状态机推进，并从 raw JSON 对应的 compiled cache 读取有效 scene token。",
        "",
        *metadata,
        "",
        (
            "|Step|截断历史后的技能|完整轨迹技能|Top-1 概率|Top-K 合法候选|"
            if result.is_history_ablation
            else "|Step|技能|Top-1 概率|Top-K 合法候选|"
        ),
        (
            "|---:|---|---|---:|---|"
            if result.is_history_ablation
            else "|---:|---|---:|---|"
        ),
    ]
    for row in result.rows:
        top_k = (
            "强制首步"
            if row.forced
            else " / ".join(
                f"{SKILL_NAMES.get(key, key)} {prob:.3f}"
                for key, _, prob, _ in row.top_candidates
            )
        )
        action_name = SKILL_NAMES.get(row.action_key, row.action_key)
        if row.forced:
            action_name = f"{action_name}（强制）"
        if result.is_history_ablation:
            reference_name = SKILL_NAMES.get(
                row.reference_action_key or "",
                row.reference_action_key or "",
            )
            lines.append(
                f"|{row.gcd_step}|{action_name}|{reference_name}|"
                f"{row.probability:.3f}|{top_k}|"
            )
        else:
            lines.append(
                f"|{row.gcd_step}|{action_name}|"
                f"{row.probability:.3f}|{top_k}|"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return output_path

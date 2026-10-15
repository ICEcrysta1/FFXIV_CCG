"""转换缓存的唯一配置依赖清单，只收集会改变转换结果的生效配置。

模型、训练、优化器、GRPO、损失权重、下载及评估桥接配置不参与检测。
精度和归一化已经进入 cache_signature；读取侧不重新加载本清单。
"""

from __future__ import annotations

from dataclasses import asdict

from common.game.config import PROJECT_ROOT, load_project_config
from common.game.models import SkillDefinition
from common.policy.data.policy_actions import load_policy_actions
from common.context.schema import SCHEMA_CONFIG_PATH
from common.configuration.yaml import load_yaml_mapping

from ..config.config import load_action_quality_skill_policy, load_convert_fflogs_job_config


def build_conversion_config_dependencies(
    job_tag: str, *, source: int | None = None, encounter: str | None = None,
    downtime_gap_seconds: float,
) -> dict[str, object]:
    """集中收集当前职业的转换依赖；不遍历配置目录或记录 YAML 文本。"""
    project = load_project_config(job_tag=job_tag)
    convert = load_convert_fflogs_job_config(job_tag)
    system = asdict(project.system)
    system["skills"] = _skill_dependencies(project.system.skills)
    job = asdict(project.job)
    job.pop("name")
    job["skills"] = _skill_dependencies(project.job.skills)
    return {
        "convert_fflogs.movement_detection": asdict(convert.movement_detection),
        "convert_fflogs.gcd_detection": asdict(convert.gcd_detection),
        "convert_fflogs.action_quality": sorted(load_action_quality_skill_policy(job_tag)),
        "engine": {
            "timing": asdict(project.engine_timing),
            "resources": asdict(project.engine_resources),
        },
        "system": system,
        f"job.{job_tag}": job,
        "schema": _schema_dependencies(),
        # 控制动作的声明顺序参与词表行号分配，必须保留为有序列表。
        "policy_actions": [
            {key: value for key, value in asdict(action).items() if key != "name"}
            for action in load_policy_actions(PROJECT_ROOT / "config" / "policy_actions.yaml")
        ],
        "conversion_options": {
            "source": source, "encounter": encounter,
            "downtime_gap_seconds": float(downtime_gap_seconds),
        },
    }


def _skill_dependencies(skills: tuple[SkillDefinition, ...]) -> dict[str, object]:
    """技能名称只用于展示；value 会写入动作监督，仍须参与检测。"""
    return {
        skill.key: {
            **{key: value for key, value in asdict(skill).items() if key not in {"key", "name", "kind"}},
            "kind": skill.kind.value,
        }
        for skill in skills
    }


def _schema_dependencies() -> dict[str, object]:
    """只记录正式 schema 分区；状态分组的声明顺序属于向量布局契约。"""
    schema = load_yaml_mapping(SCHEMA_CONFIG_PATH)
    output = dict(schema["canonical_output"])
    output["state_vector_groups"] = list(output["state_vector_groups"].items())
    return {
        "contracts": schema["contracts"],
        "scene_context": schema["scene_context"],
        "canonical_output": output,
        "state_vector_fields": schema["state_vector_fields"],
    }

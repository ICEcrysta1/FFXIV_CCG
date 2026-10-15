"""黑魔技能配置的字段完整性和显式默认语义回归。"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from common.game.config import load_project_config
from common.game.models import SkillDefinition
from common.configuration.yaml import load_yaml_mapping


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/skills/black_mage.yaml"


def test_black_mage_skills_explicitly_declare_the_same_supported_fields():
    raw_skills = load_yaml_mapping(CONFIG_PATH)["skills"]
    expected_fields = {field.name for field in fields(SkillDefinition)} - {"key", "enabled"}
    field_order = tuple(next(iter(raw_skills.values())))

    for skill_key, payload in raw_skills.items():
        assert set(payload) == expected_fields, skill_key
        assert tuple(payload) == field_order, skill_key


@pytest.mark.parametrize(
    ("skill_key", "mp_cost", "mp_cost_floor"),
    [
        ("fire_iii", 2000, 2000),
        ("despair", "full", 800),
        ("flare", "full", 800),
        ("flare_star", 0, 0),
        ("ley_lines", 0, 0),
    ],
)
def test_black_mage_explicit_mp_fields_preserve_numeric_and_full_cost_semantics(
    skill_key: str,
    mp_cost: int | str,
    mp_cost_floor: int,
):
    skills = {skill.key: skill for skill in load_project_config(job_tag="black_mage").job.skills}

    assert skills[skill_key].mp_cost == mp_cost
    assert skills[skill_key].mp_cost_floor == mp_cost_floor


def test_black_mage_explicit_optional_fields_preserve_dot_and_buff_semantics():
    skills = {skill.key: skill for skill in load_project_config(job_tag="black_mage").job.skills}
    dot_skills = {"high_thunder", "high_thunder_ii"}

    for skill_key, skill in skills.items():
        assert skill.enabled
        assert skill.requires_target == (skill.potency > 0 or skill.dot_potency > 0)
        if skill_key in dot_skills:
            assert skill.dot_key == "high_thunder"
            assert skill.dot_potency > 0
            assert skill.dot_duration > 0.0
        else:
            assert skill.dot_key is None
            assert skill.dot_potency == 0
            assert skill.dot_duration == 0.0

    assert skills["ley_lines"].charges == 2
    assert skills["ley_lines"].applies_statuses == ("ley_lines",)
    assert skills["triplecast"].charges == 2
    assert skills["triplecast"].applies_statuses == ("triplecast",)
    assert skills["fire_iii"].value == 1.10

"""转换缓存配置依赖声明回归；只改变临时目录的 YAML。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from common.game import config as project_config
from common.policy.data.compiled_cache import changed_config_dependencies
from common.configuration.yaml import load_yaml_mapping
from scripts.convert_fflogs.cache import cache_config
from scripts.convert_fflogs.config import config as convert_config


@pytest.fixture
def config_root(tmp_path, monkeypatch):
    root = tmp_path / "project"
    paths = (
        "default.yaml", "system.yaml", "skills/black_mage.yaml", "schema.yaml",
        "policy_actions.yaml", "convert_fflogs/default.yaml",
        "convert_fflogs/jobs/black_mage.yaml", "convert_fflogs/action_quality/black_mage.yaml",
        "convert_fflogs/jobs/machinist.yaml", "convert_fflogs/action_quality/machinist.yaml",
        "skills/machinist.yaml", "precision.yaml", "policy/normalizer.yaml",
        "models/black_mage/artzip/config.yaml", "models/black_mage/artzip/model.yaml",
        "models/black_mage/artzip/training.yaml", "models/black_mage/artzip/optimizer.yaml",
        "models/black_mage/artzip/grpo.yaml", "models/black_mage/artzip/action_quality.yaml",
        "fflogs_scraper/batch.yaml", "fflogs_scraper/encounters.yaml", "action_quality.yaml",
    )
    for name in paths:
        target = root / "config" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((project_config.PROJECT_ROOT / "config" / name).read_bytes())
    monkeypatch.setattr(project_config, "PROJECT_ROOT", root)
    monkeypatch.setattr(project_config, "DEFAULT_CONFIG_PATH", root / "config/default.yaml")
    monkeypatch.setattr(cache_config, "PROJECT_ROOT", root)
    monkeypatch.setattr(cache_config, "SCHEMA_CONFIG_PATH", root / "config/schema.yaml")
    monkeypatch.setattr(convert_config, "CONVERT_DEFAULT_CONFIG_PATH", root / "config/convert_fflogs/default.yaml")
    monkeypatch.setattr(convert_config, "CONVERT_JOB_CONFIG_DIR", root / "config/convert_fflogs/jobs")
    monkeypatch.setattr(convert_config, "CONVERT_ACTION_QUALITY_CONFIG_DIR", root / "config/convert_fflogs/action_quality")
    return root


def dependencies(job_tag="black_mage", **kwargs):
    return cache_config.build_conversion_config_dependencies(
        job_tag, downtime_gap_seconds=4.0, **kwargs,
    )


def rewrite(path: Path, keys: tuple[str, ...], value: object):
    payload = load_yaml_mapping(path)
    node = payload
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")


@pytest.mark.parametrize("filename,keys,value,dependency", [
    ("convert_fflogs/default.yaml", ("convert_fflogs", "movement_detection", "speed_threshold"), 0.5, "convert_fflogs.movement_detection"),
    ("convert_fflogs/default.yaml", ("convert_fflogs", "gcd_detection_defaults", "min_probe_casts"), 5, "convert_fflogs.gcd_detection"),
    ("convert_fflogs/jobs/black_mage.yaml", ("gcd_detection", "probe_skill_cast_time_seconds"), 2.2, "convert_fflogs.gcd_detection"),
    ("convert_fflogs/action_quality/black_mage.yaml", ("action_quality", "skill_label_enabled", "fight_suggestions", "aoe_usages", "core.aoeusages.suggestion.content"), False, "convert_fflogs.action_quality"),
    ("default.yaml", ("engine_timing", "action_queue_window_seconds"), 0.7, "engine"),
    ("system.yaml", ("potency", "burst_potion_multiplier"), 1.09, "system"),
    ("skills/black_mage.yaml", ("skills", "fire_iv", "potency"), 301, "job.black_mage"),
    ("schema.yaml", ("contracts", "slidecast_window_seconds"), 0.4, "schema"),
    ("policy_actions.yaml", ("policy_actions", "ogcd_wait", "value"), 1.1, "policy_actions"),
    ("skills/black_mage.yaml", ("skills", "fire_iv", "value"), 1.2, "job.black_mage"),
])
def test_semantic_yaml_changes_require_rebuild(config_root, filename, keys, value, dependency):
    saved = dependencies()
    rewrite(config_root / "config" / filename, keys, value)
    assert changed_config_dependencies(saved, dependencies()) == (dependency,)


def test_comments_and_mapping_key_order_do_not_change_dependencies(config_root):
    saved = dependencies()

    def reorder(value, keys=()):
        if isinstance(value, dict):
            # 这两类 mapping 的顺序会改变向量布局或词表，属于生效配置。
            items = list(value.items())
            if keys not in {("canonical_output", "state_vector_groups"), ("policy_actions",)}:
                items.reverse()
            return {key: reorder(child, (*keys, key)) for key, child in items}
        if isinstance(value, list):
            return [reorder(child, keys) for child in value]
        return value

    for path in (config_root / "config").rglob("*.yaml"):
        payload = load_yaml_mapping(path)
        path.write_text(
            "# 新注释\n" + yaml.safe_dump(reorder(payload), allow_unicode=True, sort_keys=False),
            encoding="utf-8", newline="\n",
        )
    assert dependencies() == saved


def test_read_side_training_settings_and_other_job_routing_are_excluded(config_root, monkeypatch):
    saved = dependencies()
    path = config_root / "config/default.yaml"
    rewrite(path, ("project", "version"), "9.9.9")
    rewrite(path, ("decision_timing", "post_gcd_delay_seconds"), 0.9)
    rewrite(path, ("runtime", "job_configs", "machinist"), "missing-other-job.yaml")
    model_path = config_root / "config/models/black_mage/artzip/model.yaml"
    rewrite(model_path, ("model", "history_capacity"), 1)
    rewrite(model_path, ("model", "history_reset_keep"), 1)
    rewrite(model_path, ("model", "time_delta_scale"), 99)
    training_path = config_root / "config/models/black_mage/artzip/training.yaml"
    rewrite(training_path, ("training", "max_files"), 1)
    monkeypatch.setenv("CONVERT_FFLOGS_WORKERS", "99")
    assert dependencies() == saved


@pytest.mark.parametrize("filename", [
    "models/black_mage/artzip/config.yaml", "models/black_mage/artzip/model.yaml",
    "models/black_mage/artzip/training.yaml", "models/black_mage/artzip/optimizer.yaml",
    "models/black_mage/artzip/grpo.yaml", "models/black_mage/artzip/action_quality.yaml",
    "fflogs_scraper/batch.yaml", "fflogs_scraper/encounters.yaml", "action_quality.yaml",
    "skills/machinist.yaml", "convert_fflogs/jobs/machinist.yaml",
    "convert_fflogs/action_quality/machinist.yaml",
])
def test_unrelated_yaml_is_not_loaded(config_root, filename):
    """真正破坏现有文件，确认白名单之外的 YAML 不影响转换依赖收集。"""
    saved = dependencies()
    (config_root / "config" / filename).write_text("[", encoding="utf-8", newline="\n")
    assert dependencies() == saved


@pytest.mark.parametrize("filename,keys", [
    ("default.yaml", ("project", "name")),
    ("system.yaml", ("skills", "potion", "name")),
    ("skills/black_mage.yaml", ("job", "name")),
    ("skills/black_mage.yaml", ("skills", "fire_iv", "name")),
    ("policy_actions.yaml", ("policy_actions", "ogcd_wait", "name")),
    ("schema.yaml", ("description",)),
    ("system.yaml", ("description",)),
    ("skills/black_mage.yaml", ("description",)),
])
def test_display_and_unused_metadata_are_excluded(config_root, filename, keys):
    saved = dependencies()
    rewrite(config_root / "config" / filename, keys, "只影响展示的内容")
    assert dependencies() == saved


@pytest.mark.parametrize("job_tag", ["black_mage", "machinist"])
def test_dependencies_follow_the_selected_job(config_root, job_tag):
    snapshot = dependencies(job_tag)
    assert {name for name in snapshot if name.startswith("job.")} == {f"job.{job_tag}"}


def test_equivalent_default_values_do_not_require_rebuild(config_root):
    """显式写出与解析器相同的默认值，不应因原始 YAML 增加字段而重建。"""
    saved = dependencies()
    rewrite(config_root / "config/skills/black_mage.yaml", ("skills", "fire_iv", "requires_target"), True)
    assert dependencies() == saved


def test_schema_state_group_order_requires_rebuild(config_root):
    saved = dependencies()
    path = config_root / "config/schema.yaml"
    schema = load_yaml_mapping(path)
    groups = list(schema["canonical_output"]["state_vector_groups"].items())
    groups[0], groups[1] = groups[1], groups[0]
    schema["canonical_output"]["state_vector_groups"] = dict(groups)
    path.write_text(yaml.safe_dump(schema, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    assert changed_config_dependencies(saved, dependencies()) == ("schema",)


def test_policy_action_order_requires_rebuild(config_root):
    path = config_root / "config/policy_actions.yaml"
    payload = load_yaml_mapping(path)
    actions = payload["policy_actions"]
    actions["other_wait"] = {**actions["ogcd_wait"], "raw_id": -1}
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    saved = dependencies()
    payload["policy_actions"] = dict(reversed(list(actions.items())))
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n")
    assert changed_config_dependencies(saved, dependencies()) == ("policy_actions",)


def test_normalizer_and_precision_are_checked_by_existing_cache_signature(config_root):
    """精度和归一化仍由正式签名校验，配置快照不重复保存第二份。"""
    from common.pytorch.precision import load_precision_config
    from common.policy.data.compiled_cache import build_cache_signature
    from common.policy.data.normalizer import Normalizer

    saved = dependencies()
    source = config_root / "source.json.br"
    source.write_bytes(b"source")
    precision_path = config_root / "config/precision.yaml"
    normalizer_path = config_root / "config/policy/normalizer.yaml"

    def signature():
        precision = load_precision_config(precision_path)
        return build_cache_signature(
            source, int_dtype=precision.resolve_int_dtype(), float_dtype=precision.resolve_float_dtype(),
            normalizer=Normalizer(config_path=normalizer_path),
        )

    initial = signature()
    rewrite(normalizer_path, ("normalizer", "target_count_max"), 99)
    normalized = signature()
    assert normalized != initial
    rewrite(precision_path, ("precision", "float_dtype"), "float16")
    assert signature() != normalized
    assert dependencies() == saved


def test_conversion_source_and_encounter_overrides_require_rebuild(config_root):
    saved = dependencies()
    assert changed_config_dependencies(saved, dependencies(source=99)) == ("conversion_options",)
    assert changed_config_dependencies(saved, dependencies(encounter="other")) == ("conversion_options",)


def test_yaml_comments_formatting_and_mapping_order_do_not_require_rebuild(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("movement: {speed: 0.5, gaps: [1, 2]}\n", encoding="utf-8", newline="\n")
    saved = {"convert": load_yaml_mapping(path)}
    path.write_text(
        "# 仅修改注释、缩进和键顺序\nmovement:\n  gaps: [1, 2]\n  speed: 0.5  # 阈值不变\n",
        encoding="utf-8", newline="\n",
    )
    assert changed_config_dependencies(saved, {"convert": load_yaml_mapping(path)}) == ()


@pytest.mark.parametrize("current", [
    {"movement": {"speed": 0.6, "gaps": [1, 2]}},
    {"movement": {"speed": 0.5, "gaps": [2, 1]}},
    {"movement": {"speed": 0.5, "gaps": [1, 2], "limit": None}},
])
def test_changed_values_list_order_and_new_null_fields_require_rebuild(current):
    saved = {"movement": {"speed": 0.5, "gaps": [1, 2]}}
    assert changed_config_dependencies(saved, current) == ("movement",)


@pytest.mark.parametrize("saved", [None, [], "invalid-format", {}, {1: "invalid-key"}])
def test_missing_or_invalid_saved_dependencies_require_rebuild(saved):
    assert changed_config_dependencies(saved, {"movement": {"speed": 0.5}}) == ("movement",)


def test_removed_dependency_requires_rebuild():
    assert changed_config_dependencies({"movement": None, "job": 1}, {"job": 1}) == ("movement",)

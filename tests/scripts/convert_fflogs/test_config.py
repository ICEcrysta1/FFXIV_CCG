"""FFLogs 转换配置加载测试。"""

import pytest
import yaml

from common.project_config import resolve_positive_worker_count
from scripts.convert_fflogs.config import (
    load_convert_fflogs_config,
    load_convert_fflogs_job_config,
    resolve_convert_fflogs_job_tag,
)
from tests.helpers import DEFAULT_BASE_GCD


def test_convert_fflogs_config_loads_default_values():
    config = load_convert_fflogs_config()

    assert config.gcd_detection_defaults.fallback_seconds == pytest.approx(DEFAULT_BASE_GCD)
    assert config.gcd_detection_defaults.histogram_bin_width_ms == 10
    assert config.movement_detection.coordinate_scale == 100.0
    assert config.movement_detection.speed_threshold == pytest.approx(1 / 3)
    assert config.movement_detection.sample_step_seconds == 0.01
    assert config.movement_detection.merge_gap_gcds == 1.0
    assert config.movement_detection.minimum_window_gcds == 1.0


def test_legacy_convert_worker_config_is_rejected(tmp_path):
    path = tmp_path / "default.yaml"
    path.write_text(
        "convert_fflogs:\n  default_worker_count: 6\n",
        encoding="utf-8", newline="\n",
    )
    with pytest.raises(ValueError, match="CONVERT_FFLOGS_WORKERS"):
        load_convert_fflogs_config(path)


@pytest.mark.parametrize("value", ["0", "-2", "1.5", "many", ""])
def test_convert_worker_env_rejects_invalid_values(tmp_path, monkeypatch, value):
    monkeypatch.setenv("CONVERT_FFLOGS_WORKERS", value)
    with pytest.raises(ValueError, match="CONVERT_FFLOGS_WORKERS"):
        resolve_positive_worker_count(
            project_root=tmp_path, env_name="CONVERT_FFLOGS_WORKERS",
        )


def test_convert_worker_env_falls_back_to_serial(tmp_path, monkeypatch):
    monkeypatch.delenv("CONVERT_FFLOGS_WORKERS", raising=False)
    assert resolve_positive_worker_count(
        project_root=tmp_path, env_name="CONVERT_FFLOGS_WORKERS",
    ) == 1


def test_convert_fflogs_job_config_loads_black_mage_probe_skill():
    job_config = load_convert_fflogs_job_config("black_mage")

    assert job_config.job_tag == "black_mage"
    assert job_config.gcd_detection.probe_skill_game_id == 3577
    assert job_config.gcd_detection.haste_excluded_buff_game_ids == (1000737,)
    assert job_config.gcd_detection.fallback_seconds == pytest.approx(DEFAULT_BASE_GCD)
    assert job_config.gcd_detection.probe_skill_cast_time_seconds == pytest.approx(2.0)
    assert job_config.movement_detection == load_convert_fflogs_config().movement_detection


def test_convert_fflogs_job_config_loads_machinist_probe_skill():
    job_config = load_convert_fflogs_job_config("machinist")

    assert job_config.job_tag == "machinist"
    assert job_config.gcd_detection.probe_skill_game_id == 7411
    assert job_config.gcd_detection.haste_excluded_buff_game_ids == ()
    assert job_config.gcd_detection.fallback_seconds == pytest.approx(DEFAULT_BASE_GCD)


def test_resolve_convert_fflogs_job_tag_prefers_explicit_then_unified_env(monkeypatch):
    monkeypatch.setenv("FFXIV_JOB_TAG", "from_env")
    assert resolve_convert_fflogs_job_tag("black_mage") == "black_mage"
    assert resolve_convert_fflogs_job_tag(None) == "from_env"


@pytest.mark.parametrize("invalid_value", [0, -1.0])
def test_convert_fflogs_config_rejects_non_positive_probe_cast_time(tmp_path, invalid_value):
    from scripts.convert_fflogs.config import CONVERT_DEFAULT_CONFIG_PATH

    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        CONVERT_DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"),
        encoding="utf-8", newline="\n",
    )
    job_dir = tmp_path / "jobs"
    job_dir.mkdir()
    (job_dir / "black_mage.yaml").write_text(
        f"""job:
  key: black_mage
gcd_detection:
  probe_skill_game_id: 3577
  probe_skill_cast_time_seconds: {invalid_value}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="probe_skill_cast_time_seconds must be > 0"):
        load_convert_fflogs_job_config(
            "black_mage",
            default_path=default_path,
            job_dir=job_dir,
        )


@pytest.mark.parametrize("field", [
    "coordinate_scale", "speed_threshold", "sample_step_seconds",
    "merge_gap_gcds", "minimum_window_gcds",
])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_movement_detection_rejects_invalid_parameters(tmp_path, field, value):
    from scripts.convert_fflogs.config import CONVERT_DEFAULT_CONFIG_PATH

    payload = yaml.safe_load(CONVERT_DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["convert_fflogs"]["movement_detection"][field] = value
    path = tmp_path / "default.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match=f"movement_detection.{field}"):
        load_convert_fflogs_config(path)


def test_movement_detection_rejects_removed_expansion_parameter(tmp_path):
    """无效的旧参数必须报错，禁止静默接受并污染缓存签名。"""
    from scripts.convert_fflogs.config import CONVERT_DEFAULT_CONFIG_PATH

    payload = yaml.safe_load(CONVERT_DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["convert_fflogs"]["movement_detection"]["maximum_expansion_per_side_seconds"] = 0.5
    path = tmp_path / "default.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="movement_detection unknown fields.*maximum_expansion"):
        load_convert_fflogs_config(path)

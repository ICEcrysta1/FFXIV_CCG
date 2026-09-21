"""FFLogs 转换配置加载测试。"""

import pytest

from scripts.convert_fflogs.config import (
    load_convert_fflogs_config,
    load_convert_fflogs_job_config,
    resolve_convert_fflogs_job_tag,
)
from tests.helpers import DEFAULT_BASE_GCD


def test_convert_fflogs_config_loads_default_values():
    config = load_convert_fflogs_config()

    assert config.default_worker_count == 6
    assert config.gcd_detection_defaults.fallback_seconds == pytest.approx(DEFAULT_BASE_GCD)
    assert config.gcd_detection_defaults.histogram_bin_width_ms == 10


def test_convert_fflogs_job_config_loads_black_mage_probe_skill():
    job_config = load_convert_fflogs_job_config("black_mage")

    assert job_config.job_tag == "black_mage"
    assert job_config.gcd_detection.probe_skill_game_id == 3577
    assert job_config.gcd_detection.haste_excluded_buff_game_ids == (1000737,)
    assert job_config.gcd_detection.fallback_seconds == pytest.approx(DEFAULT_BASE_GCD)
    assert job_config.gcd_detection.probe_skill_cast_time_seconds == pytest.approx(2.0)


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
    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        """convert_fflogs:
  gcd_detection_defaults:
    probe_skill_game_id: 3577
    probe_skill_cast_time_seconds: 2.0
""",
        encoding="utf-8",
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

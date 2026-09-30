"""批量下载比例配置测试。"""

from decimal import Decimal

import pytest

from scripts.fflogs_scraper.config.batch import load_validation_ratio


def test_default_validation_ratio():
    assert load_validation_ratio() == Decimal("0.10")


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1", "true", "'0.2'", "null", ".nan"])
def test_invalid_validation_ratio_is_rejected(tmp_path, value):
    path = tmp_path / "batch.yaml"
    path.write_text(f"validation_ratio: {value}\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="validation_ratio"):
        load_validation_ratio(path)


def test_configured_validation_ratio(tmp_path):
    path = tmp_path / "batch.yaml"
    path.write_text("validation_ratio: 0.07\n", encoding="utf-8", newline="\n")
    assert load_validation_ratio(path) == Decimal("0.07")

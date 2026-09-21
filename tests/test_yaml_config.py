"""公共 YAML mapping 加载测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from common.yaml_config import load_yaml_mapping


@pytest.mark.parametrize("payload", [0, False, "", [], "scalar", [1]])
def test_load_yaml_mapping_rejects_non_mapping_falsey_values(
    tmp_path: Path,
    payload: object,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="YAML config must be a mapping"):
        load_yaml_mapping(config_path)


def test_load_yaml_mapping_treats_empty_document_as_empty_mapping(tmp_path: Path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    assert load_yaml_mapping(config_path) == {}

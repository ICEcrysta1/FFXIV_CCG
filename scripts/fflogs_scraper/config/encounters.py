"""下载目标代号与 FFLogs zone/encounter ID 的 YAML 契约。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.yaml_config import load_yaml_mapping


CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "fflogs_scraper" / "encounters.yaml"


@dataclass(frozen=True)
class DownloadEncounter:
    alias: str
    zone_id: int
    encounter_id: int
    group: str


def load_download_encounters(path: Path = CONFIG_PATH) -> dict[str, DownloadEncounter]:
    """验证可输入代号、目录名称和 ID，避免静默写入错误副本。"""
    raw = load_yaml_mapping(path, description="FFLogs encounter config")
    entries = raw.get("encounters")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("FFLogs encounter config requires a nonempty encounters mapping")
    result: dict[str, DownloadEncounter] = {}
    encounter_ids: set[int] = set()
    for alias, value in entries.items():
        if (not isinstance(alias, str) or not alias.isascii() or not alias.isalnum()
                or not alias[0].isalpha()):
            raise ValueError(f"invalid FFLogs encounter alias: {alias!r}")
        if not isinstance(value, dict):
            raise ValueError(f"invalid FFLogs encounter entry: {alias}")
        zone_id = value.get("zone_id")
        encounter_id = value.get("encounter_id")
        group = value.get("group")
        if (type(zone_id) is not int or zone_id < 1
                or type(encounter_id) is not int or encounter_id < 1
                or not isinstance(group, str) or not group.strip()):
            raise ValueError(f"invalid FFLogs encounter IDs or group: {alias}")
        if alias.casefold() in result or encounter_id in encounter_ids:
            raise ValueError(f"duplicate FFLogs encounter alias or ID: {alias}")
        result[alias.casefold()] = DownloadEncounter(alias, zone_id, encounter_id, group)
        encounter_ids.add(encounter_id)
    return result


def resolve_download_encounter(target: str, *, path: Path = CONFIG_PATH) -> DownloadEncounter:
    """名称不区分大小写；数字按 zone ID 解释，仅单战斗 zone 可自动选择。"""
    entries = load_download_encounters(path)
    if target.isdecimal():
        zone_id = int(target)
        matches = [entry for entry in entries.values() if entry.zone_id == zone_id]
        if len(matches) == 1:
            return matches[0]
        if matches:
            options = ", ".join(entry.alias for entry in matches)
            raise ValueError(f"Zone {zone_id} 含多个战斗，请输入副本代号：{options}")
        raise ValueError(f"未配置 Zone {zone_id}；请在 config/fflogs_scraper/encounters.yaml 中添加")
    entry = entries.get(target.casefold())
    if entry is None:
        raise ValueError(f"未配置副本 {target!r}；请在 config/fflogs_scraper/encounters.yaml 中添加")
    return entry

"""报告格式适配、战斗选择与下载包装字段。"""

from datetime import datetime
from typing import Optional

from ..config.constants import DOWNLOAD_SCHEMA_VERSION
from ..config.validation import _validate_integer
from .models import FightInfo, ReportMeta


def _adapt_report_metadata(report_code: str, report: dict) -> ReportMeta:
    """把 V2 元数据转成离线分析器使用的 FFLogs V1 报告契约。"""
    master = report.get("masterData") or {}
    lang = master.get("lang")
    if lang not in {"en", "ja", "de", "fr", "cn", "kr"}:
        raise ValueError(f"报告缺少可识别的原始语言，无法选择分析规则版本: {lang!r}")
    if not isinstance(report.get("startTime"), (int, float)) or report["startTime"] <= 0:
        raise ValueError("报告缺少真实 startTime，不能用下载日期代替")
    if not isinstance(report.get("endTime"), (int, float)) or report["endTime"] < report["startTime"]:
        raise ValueError("报告缺少有效 endTime")

    zone = report.get("zone") or {}
    raw_fights = report.get("fights") or []
    fights = []
    for raw in raw_fights:
        game_zone = raw.get("gameZone") or zone
        fight = {
            "id": raw["id"], "boss": raw["encounterID"], "name": raw["name"],
            "start_time": raw["startTime"], "end_time": raw["endTime"],
            "zoneID": game_zone["id"], "zoneName": game_zone["name"],
            "zone_name": zone.get("name", ""),
            "difficulty": raw.get("difficulty"), "kill": raw.get("kill", False),
            "size": raw.get("size"), "standardComposition": raw.get("standardComposition"),
        }
        if raw.get("combatTime") is not None:
            fight["combatTime"] = raw["combatTime"]
        # V2 返回百分数，V1 的战斗进度使用万分数。
        for key in ("bossPercentage", "fightPercentage"):
            if raw.get(key) is not None:
                fight[key] = raw[key] * 100
        fights.append(fight)

    actors = {actor["id"]: actor for actor in master.get("actors") or []}
    groups: dict[str, dict[int, dict]] = {
        key: {} for key in ("friendlies", "enemies", "friendlyPets", "enemyPets")
    }
    memberships = (
        ("friendlyPlayers", "friendlies"), ("friendlyNPCs", "friendlies"),
        ("enemyPlayers", "enemies"), ("enemyNPCs", "enemies"),
        ("friendlyPets", "friendlyPets"), ("enemyPets", "enemyPets"),
    )
    for fight in raw_fights:
        for field_name, group_name in memberships:
            for member in fight.get(field_name) or []:
                actor_id = member["id"] if isinstance(member, dict) else member
                if actor_id not in actors:
                    raise ValueError(f"fight={fight['id']} 的 actor={actor_id} 缺少 masterData")
                raw_actor = actors[actor_id]
                if actor_id not in groups[group_name]:
                    actor_type = raw_actor.get("subType") or raw_actor["type"]
                    if group_name.endswith("Pets"):
                        actor_type = "Pet"
                    actor = {
                        "id": actor_id, "guid": raw_actor["gameID"],
                        "name": raw_actor["name"], "type": actor_type, "fights": [],
                    }
                    if group_name.endswith("Pets"):
                        if raw_actor.get("petOwner") not in actors:
                            raise ValueError(f"宠物 actor={actor_id} 缺少有效 petOwner")
                        actor["petOwner"] = raw_actor["petOwner"]
                    groups[group_name][actor_id] = actor
                membership = {"id": fight["id"]}
                if isinstance(member, dict):
                    membership.update({
                        "instances": member.get("instanceCount") or 1,
                        "groups": member.get("groupCount") or 0,
                    })
                groups[group_name][actor_id]["fights"].append(membership)

    legacy = {
        "code": report_code, "loading": False, "start": report["startTime"],
        "end": report["endTime"], "title": report["title"],
        "owner": (report.get("owner") or {}).get("name", ""),
        "zone": zone.get("id", 0), "lang": lang, "fights": fights,
        # V2 没有 V1 的阶段名称表，当前报告解析只依赖战斗和角色元数据。
        "phases": [], **{key: list(value.values()) for key, value in groups.items()},
    }
    return ReportMeta(
        code=report_code,
        fights=[FightInfo(
            id=f["id"], name=f["name"], start_time=f["start_time"], end_time=f["end_time"],
            difficulty=f["difficulty"], kill=f["kill"], size=f["size"], zone_name=f["zone_name"],
        ) for f in fights],
        report=legacy,
    )


def _build_download_payload(meta: ReportMeta, fight: FightInfo | None, source_id: int | None) -> dict:
    """同一 JSON 保存完整分析上下文及现有训练入口需要的索引。"""
    if not meta.report:
        raise ValueError("下载报告缺少离线分析所需的元数据")
    selected_fights = [f for f in meta.report["fights"] if fight is None or f["id"] == fight.id]
    if not selected_fights:
        raise ValueError("报告没有可下载的战斗")
    selected_ids = {f["id"] for f in selected_fights}
    selected_actors = {}
    for group in ("friendlies", "enemies", "friendlyPets", "enemyPets"):
        selected_actors[group] = []
        for actor in meta.report[group]:
            memberships = [f for f in actor["fights"] if f["id"] in selected_ids]
            if memberships:
                selected_actors[group].append({**actor, "fights": memberships})
    if source_id is not None:
        source_id = _validate_integer(source_id, "source_id", minimum=1)
        if not any(actor["id"] == source_id for group in selected_actors.values() for actor in group):
            raise ValueError(f"选定战斗中不存在 source={source_id}")
    return {
        **meta.report, **selected_actors, "fights": selected_fights,
        "download_schema_version": DOWNLOAD_SCHEMA_VERSION,
        "report_code": meta.code, "fight_id": fight.id if fight else None,
        "source_id": source_id, "fetched_at": datetime.now().isoformat(),
        "events_complete": False,
    }


def _find_fight(meta: ReportMeta, fight_id: int) -> Optional[FightInfo]:
    """按 ID 查找战斗，-1 表示最后一场。"""
    if fight_id == -1:
        return meta.fights[-1] if meta.fights else None
    for f in meta.fights:
        if f.id == fight_id:
            return f
    return None

#!/usr/bin/env python3
"""FFLogs 数据拉取工具。

从 FFLogs V2 GraphQL API (OAuth) 拉取战斗报告事件数据。
事件模式保存 FFLogs V1 报告/事件契约，供离线分析器读取，并保留训练索引。
--source 选择分析玩家与伤害表，不裁剪整场事件；damage-only 不提供分析事件。

使用方式:
  python fflogs_scraper.py "https://www.fflogs.com/reports/JFLCXcQjBd9zgWR1?fight=6&type=damage-done&source=10"
  python fflogs_scraper.py single --report JFLCXcQjBd9zgWR1 --fight 6 --source 10
  python fflogs_scraper.py batch -e 1079 --spec-name BlackMage --max-pages 3
  python fflogs_scraper.py encounters -z 39

认证:
  在 FFLogs -> 我的 -> API 获取 V2 Client ID 和 Client Secret，设置到 .env:
    FFLOGS_V2_CLIENT_ID=your_client_id
    FFLOGS_V2_CLIENT_SECRET=your_client_secret
"""

import argparse
import json
import logging
import os
import re
import signal
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from curl_cffi import requests as cf_requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEFAULT_REQUEST_TIMEOUT = 60
BROWSER_FINGERPRINT = "chrome124"
REPORT_CODE_PATTERN = re.compile(r"^[A-Za-z0-9]+$")
INTEGER_PATTERN = re.compile(r"^-?\d+$")
RANKING_METRICS = frozenset({"dps", "rdps", "ndps", "adps"})
MAX_EVENTS = 100_000
DOWNLOAD_SCHEMA_VERSION = 2


def _validate_alphanumeric(value: str, field_name: str) -> str:
    """校验只能安全嵌入 GraphQL 字符串字面量的字母数字参数。"""
    if not isinstance(value, str) or not REPORT_CODE_PATTERN.fullmatch(value):
        raise ValueError(f"invalid {field_name}: {value!r}")
    return value


def _validate_integer(value: int | str, field_name: str, *, minimum: int | None = None) -> int:
    """把公开 API 的整数参数规范化为安全的 GraphQL Int 字面量。"""
    if isinstance(value, bool):
        raise ValueError(f"invalid {field_name}: {value!r}")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, str) and INTEGER_PATTERN.fullmatch(value):
        normalized = int(value)
    else:
        raise ValueError(f"invalid {field_name}: {value!r}")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"invalid {field_name}: {value!r}")
    return normalized


def _validate_integer_list(
    values: list[int],
    field_name: str,
    *,
    minimum: int | None = None,
) -> list[int]:
    """逐项校验整数列表，避免将列表原样拼入 GraphQL 查询。"""
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"invalid {field_name}: {values!r}")
    return [
        _validate_integer(value, f"{field_name}[{index}]", minimum=minimum)
        for index, value in enumerate(values)
    ]


def _validate_ranking_metric(metric: str) -> str:
    """校验排行指标，避免把公开 API 参数拼入任意 GraphQL 片段。"""
    if not isinstance(metric, str) or metric not in RANKING_METRICS:
        raise ValueError(f"invalid metric: {metric!r}")
    return metric

# 7.x 大版本补丁发布时间（Unix 毫秒）
PATCH_RELEASE_MS: dict[str, int] = {}
_patch_dates = {
    "7.0": "2024-07-02", "7.01": "2024-07-16", "7.05": "2024-07-30",
    "7.08": "2024-08-06", "7.1": "2024-11-12", "7.11": "2024-11-26",
    "7.15": "2024-12-17", "7.16": "2025-01-21", "7.18": "2025-02-25",
    "7.2": "2025-03-25", "7.21": "2025-04-01", "7.25": "2025-04-22",
    "7.3": "2025-05-20", "7.4": "2025-07-08", "7.5": "2025-09-30",
}
for _p, _d in _patch_dates.items():
    PATCH_RELEASE_MS[_p] = int(datetime.strptime(_d, "%Y-%m-%d").timestamp() * 1000)


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------


@dataclass
class FightInfo:
    """一场战斗的元信息。"""
    id: int
    name: str
    start_time: int
    end_time: int
    difficulty: Optional[int] = None
    kill: bool = False
    size: Optional[int] = None
    zone_name: str = ""


@dataclass
class ReportMeta:
    """一份报告的元信息。"""
    code: str
    fights: list[FightInfo] = field(default_factory=list)
    # FFLogs V1 报告元数据，训练选定玩家另存于下载包装字段。
    report: dict = field(default_factory=dict)


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


def _attach_analysis_events(result: dict, events: list[dict]) -> None:
    """保留嵌套技能与实例信息，并增加训练兼容字段和友好关系标记。"""
    friendly_by_fight: dict[int, set[int]] = {}
    for group in ("friendlies", "friendlyPets"):
        for actor in result[group]:
            for membership in actor["fights"]:
                friendly_by_fight.setdefault(membership["id"], set()).add(actor["id"])
    adapted = []
    for event in events:
        event = dict(event)
        ability = event.get("ability")
        if isinstance(ability, dict):
            event["abilityGameID"] = ability["guid"]
        elif "abilityGameID" in event or "abilityID" in event:
            raise ValueError("API 未返回嵌套 ability，不能保存为离线分析输入")
        extra = event.get("extraAbility")
        if isinstance(extra, dict):
            event["extraAbilityGameID"] = extra["guid"]
        friends = friendly_by_fight.get(event.get("fight"), set())
        event["sourceIsFriendly"] = event.get("sourceID") in friends
        event["targetIsFriendly"] = event.get("targetID") in friends
        adapted.append(event)
    result.update(
        events=adapted, event_count=len(adapted), events_complete=True,
        event_scope="fight" if result["fight_id"] is not None else "report",
    )


def _write_download_json(output_path: str, result: dict) -> None:
    """仅在完整序列化成功后替换目标，避免留下被批量任务误跳过的半文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=path.parent, suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# FFLogs V2 GraphQL API 客户端 (OAuth)
# ---------------------------------------------------------------------------


class FFLogsV2Client:
    """FFLogs V2 GraphQL API 客户端 (OAuth Client Credentials)。"""

    TOKEN_URL = "https://www.fflogs.com/oauth/token"
    GQL_URL = "https://www.fflogs.com/api/v2/client"

    def __init__(self, client_id: str, client_secret: str):
        self._client_id = client_id
        self._client_secret = client_secret
        self._token: Optional[str] = None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _ensure_token(self) -> str:
        if self._token:
            return self._token
        resp = cf_requests.post(
            self.TOKEN_URL,
            json={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
            impersonate=BROWSER_FINGERPRINT,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        logger.info("V2 token 已获取，有效期 %ds", data.get("expires_in", 0))
        return self._token

    def query(self, gql: str, variables: dict = None) -> dict:
        """执行 GraphQL 查询。"""
        token = self._ensure_token()
        payload: dict = {"query": gql}
        if variables:
            payload["variables"] = variables
        resp = cf_requests.post(
            self.GQL_URL,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            impersonate=BROWSER_FINGERPRINT,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            msgs = [e.get("message", "") for e in data["errors"]]
            logger.error("GQL 查询失败:\n%s", gql[:800])
            logger.error("GQL 错误详情: %s", data["errors"])
            raise RuntimeError(f"GQL errors: {msgs}")
        return data.get("data", {})

    # ---- 报告查询 ----

    def get_report_fights(self, report_code: str) -> ReportMeta:
        """获取报告、战斗及角色元数据，生成离线分析报告上下文。"""
        report_code = _validate_report_code(report_code)
        gql = f"""
        query {{
          reportData {{
            report(code: "{report_code}") {{
              title startTime endTime owner {{ name }} zone {{ id name }}
              masterData(translate: true) {{
                lang actors {{ id gameID name type subType petOwner }}
              }}
              fights {{
                id name encounterID startTime endTime combatTime difficulty kill size
                bossPercentage fightPercentage standardComposition gameZone {{ id name }}
                friendlyPlayers enemyPlayers
                friendlyNPCs {{ id instanceCount groupCount }}
                enemyNPCs {{ id instanceCount groupCount }}
                friendlyPets {{ id instanceCount groupCount }}
                enemyPets {{ id instanceCount groupCount }}
              }}
            }}
          }}
        }}
        """
        data = self.query(gql)
        report = data.get("reportData", {}).get("report", {})
        if not report:
            raise ValueError(f"报告不存在或无权读取: {report_code}")
        return _adapt_report_metadata(report_code, report)

    def get_report_players(self, report_code: str, fight_id: int) -> list[dict]:
        """获取报告中某场战斗的玩家列表。"""
        gql = f"""
        query {{
          reportData {{
            report(code: "{report_code}") {{
              playerDetails(fightIDs: [{fight_id}])
            }}
          }}
        }}
        """
        data = self.query(gql)
        report = data.get("reportData", {}).get("report", {})
        details = report.get("playerDetails", {}) if report else {}

        result = []
        if isinstance(details, dict):
            inner = details.get("data", details)
            if isinstance(inner, dict):
                players_by_role = inner.get("playerDetails", inner)
                if isinstance(players_by_role, dict):
                    for role in ("dps", "tanks", "healers", "tank", "healer"):
                        for p in players_by_role.get(role, []):
                            result.append(p)
                elif isinstance(players_by_role, list):
                    result = players_by_role
        return result

    def resolve_source_id(self, report_code: str, player_name: str, fight_id: int) -> Optional[int]:
        """根据玩家名查找 source ID。"""
        players = self.get_report_players(report_code, fight_id)
        for p in players:
            if p.get("name") == player_name:
                return p.get("id")
        return None

    def get_report_events(
        self,
        report_code: str,
        fight_ids: list[int] = None,
        source_id: Optional[int] = None,
        start_time: int = 0,
        end_time: int = 999999999999,
        max_pages: int = 50,
        *,
        require_complete: bool = False,
    ) -> list[dict]:
        """拉取报告事件（V2 分页）。"""
        report_code = _validate_report_code(report_code)
        max_pages = _validate_integer(max_pages, "max_pages", minimum=1)
        if source_id is not None:
            source_id = _validate_integer(source_id, "source_id", minimum=1)
        all_events: list[dict] = []
        next_ts: Optional[int] = start_time

        for _ in range(max_pages):
            if self._cancelled:
                raise RuntimeError("下载已取消，未保存不完整事件")
            requested_ts = next_ts
            args = [
                f'startTime: {next_ts}',
                f'endTime: {end_time}',
            ]
            if source_id is not None:
                args.append(f'sourceID: {source_id}')
            if fight_ids:
                normalized_fight_ids = _validate_integer_list(
                    fight_ids, "fight_ids", minimum=1
                )
                fids = ",".join(str(fight_id) for fight_id in normalized_fight_ids)
                args.append(f'fightIDs: [{fids}]')

            arg_str = ", ".join(args)

            gql = f"""
            query {{
              reportData {{
                report(code: "{report_code}") {{
                  events({arg_str} limit: 10000 includeResources: true
                         useAbilityIDs: false useActorIDs: true translate: true) {{
                    data
                    nextPageTimestamp
                  }}
                }}
              }}
            }}
            """
            data = self.query(gql)
            report = data.get("reportData", {}).get("report", {})
            events_wrapper = report.get("events") if report else None
            if not isinstance(events_wrapper, dict) or not isinstance(events_wrapper.get("data"), list):
                raise RuntimeError("FFLogs 未返回有效 events.data，不能视为完整空日志")
            events = events_wrapper["data"]
            next_ts = events_wrapper.get("nextPageTimestamp")

            remaining_events = MAX_EVENTS - len(all_events)
            if len(events) >= remaining_events:
                if require_complete and (
                    len(events) > remaining_events or (next_ts is not None and next_ts < end_time)
                ):
                    raise RuntimeError(f"事件超过上限 {MAX_EVENTS}，未保存不完整分析输入")
                accepted_events = events[:remaining_events]
                all_events.extend(accepted_events)
                logger.info(
                    "获取到 %d 条事件 (累计 %d)",
                    len(accepted_events),
                    len(all_events),
                )
                logger.warning("事件累计达到上限 %d，提前停止", MAX_EVENTS)
                break

            all_events.extend(events)
            logger.info("获取到 %d 条事件 (累计 %d)", len(events), len(all_events))

            if next_ts is None or next_ts >= end_time:
                return all_events
            if next_ts <= requested_ts:
                raise RuntimeError("FFLogs 事件分页时间未前进")

        if require_complete and next_ts is not None and next_ts < end_time:
            raise RuntimeError(f"事件分页超过上限 {max_pages}，未保存不完整分析输入")
        return all_events

    def get_fight_events(
        self, report_code: str, fight: FightInfo, source_id: Optional[int] = None,
        *, require_complete: bool = False,
    ) -> list[dict]:
        """拉取一场战斗的全部事件。"""
        return self.get_report_events(
            report_code,
            fight_ids=[fight.id],
            source_id=source_id,
            start_time=fight.start_time,
            end_time=fight.end_time,
            require_complete=require_complete,
        )

    def get_damage_table(
        self, report_code: str, fight: FightInfo, source_id: int,
    ) -> dict:
        """拉取伤害表。"""
        gql = f"""
        query {{
          reportData {{
            report(code: "{report_code}") {{
              table(startTime: {fight.start_time}, endTime: {fight.end_time}, sourceID: {source_id}, fightIDs: [{fight.id}])
            }}
          }}
        }}
        """
        data = self.query(gql)
        report = data.get("reportData", {}).get("report", {})
        return report.get("table", {}) if report else {}

    # ---- 排名查询 ----

    def get_zone_encounters(self, zone_id: int) -> list[dict]:
        """获取 zone 下所有 encounter。"""
        gql = f"""
        query {{
          worldData {{ zone(id: {zone_id}) {{ encounters {{ id name }} }} }}
        }}
        """
        data = self.query(gql)
        zone = data.get("worldData", {}).get("zone", {})
        return (zone or {}).get("encounters", [])

    def get_zones(self) -> list[dict]:
        """获取所有可用副本区域列表。"""
        gql = """
        query {
          worldData { zones { id name } }
        }
        """
        data = self.query(gql)
        zones = data.get("worldData", {}).get("zones", [])
        return zones if isinstance(zones, list) else []

    def get_encounter_rankings(
        self,
        encounter_id: int,
        spec_name: str = None,
        class_name: str = None,
        bracket: int = 0,
        page: int = 1,
        metric: str = "dps",
    ) -> dict:
        """获取 encounter 排行。"""
        encounter_id = _validate_integer(encounter_id, "encounter_id", minimum=1)
        bracket = _validate_integer(bracket, "bracket", minimum=0)
        page = _validate_integer(page, "page", minimum=1)
        metric = _validate_ranking_metric(metric)
        if spec_name is not None:
            spec_name = _validate_alphanumeric(spec_name, "spec_name")
        if class_name is not None:
            class_name = _validate_alphanumeric(class_name, "class_name")

        filters = []
        if spec_name:
            filters.append(f'specName: "{spec_name}"')
        if class_name:
            filters.append(f'className: "{class_name}"')
        filter_str = ", ".join(filters)
        gql = f"""
        query {{
          worldData {{
            encounter(id: {encounter_id}) {{
              characterRankings(
                metric: {metric}
                bracket: {bracket}
                page: {page}
                {filter_str}
              )
            }}
          }}
        }}
        """
        data = self.query(gql)
        enc = data.get("worldData", {}).get("encounter", {})
        rankings = enc.get("characterRankings", {}) if enc else {}
        return rankings if isinstance(rankings, dict) else {}

    def get_high_score_reports(
        self,
        encounter_id: int,
        spec_name: str = None,
        bracket: int = 0,
        max_pages: int = 10,
        metric: str = "rdps",
    ) -> list[tuple]:
        """获取高分报告列表。

        bracket: 0=全部, 6=金100%
        metric: dps / rdps / ndps / adps

        Returns
        -------
        [(report_code, fight_id, player_name, amount), ...]
        """
        reports: list[tuple] = []
        seen: set = set()

        for page in range(1, max_pages + 1):
            logger.info("查询排行: encounter=%d bracket=%d page=%d metric=%s",
                         encounter_id, bracket, page, metric)
            try:
                rankings = self.get_encounter_rankings(
                    encounter_id, spec_name=spec_name,
                    bracket=bracket, page=page, metric=metric,
                )
            except Exception as e:
                logger.warning("查询失败: %s", e)
                break

            entries = rankings.get("rankings", [])
            if not entries:
                break

            for r in entries:
                rep = r.get("report", {})
                rid = rep.get("code", "")
                fid = rep.get("fightID", 0)
                name = r.get("name", "")
                dps = r.get("amount", 0)

                if not rid or not fid:
                    continue

                key = f"{rid}_{fid}_{name}"
                if key not in seen:
                    seen.add(key)
                    reports.append((rid, fid, name, dps))

            if len(entries) < 100:
                break

        logger.info("共找到 %d 份报告 (encounter=%d, bracket=%d)",
                     len(reports), encounter_id, bracket)
        return reports


# ---------------------------------------------------------------------------
# URL 解析
# ---------------------------------------------------------------------------


def _validate_report_code(report_code: str) -> str:
    """校验报告编号，确保它只能作为文件名的一部分使用。"""
    return _validate_alphanumeric(report_code, "report code")


def parse_fflogs_url(url: str) -> dict:
    """解析 FFLogs URL，提取 report code / fight / source 等。"""
    parsed = urlparse(url)
    path_match = re.match(r"/reports/([A-Za-z0-9]+)", parsed.path)
    if not path_match:
        raise ValueError(f"无法从 URL 中提取 report code: {url}")

    report_code = _validate_report_code(path_match.group(1))
    params = parse_qs(parsed.query)

    result: dict = {"report_code": report_code}

    if "fight" in params:
        fight_val = params["fight"][0]
        if fight_val.isdigit():
            result["fight_id"] = int(fight_val)
        elif fight_val.lower() == "last":
            result["fight_id"] = -1

    if "source" in params:
        source_val = params["source"][0]
        if source_val.isdigit():
            result["source_id"] = int(source_val)

    return result


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _find_fight(meta: ReportMeta, fight_id: int) -> Optional[FightInfo]:
    """按 ID 查找战斗，-1 表示最后一场。"""
    if fight_id == -1:
        return meta.fights[-1] if meta.fights else None
    for f in meta.fights:
        if f.id == fight_id:
            return f
    return None


def _load_dotenv() -> None:
    """只加载项目根目录 `.env`（环境变量优先）。"""
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
        logger.debug("已加载项目根目录 .env: %s", env_path)


def _build_output_filename(report_code: str, fight_id: int, source_id: Optional[int]) -> str:
    report_code = _validate_report_code(report_code)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_source{source_id}" if source_id else ""
    return f"fflogs_{report_code}_f{fight_id}{suffix}_{ts}.json"


def _build_batch_output_filename(
    report_code: str,
    fight_id: int,
    player_name: str,
) -> str:
    """构造批量下载文件名，并隔离报告编号和角色名中的路径字符。"""
    report_code = _validate_report_code(report_code)
    safe_name = (
        str(player_name)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
    )
    return f"fflogs_{report_code}_f{fight_id}_{safe_name}.json"


# ---------------------------------------------------------------------------
# 命令函数
# ---------------------------------------------------------------------------


def _cmd_single(client: FFLogsV2Client, args) -> None:
    """单报告下载。"""
    if args.url:
        parsed = parse_fflogs_url(args.url)
        report_code = parsed["report_code"]
        fight_id = parsed.get("fight_id", args.fight)
        source_id = parsed.get("source_id", args.source)
    elif args.report:
        report_code = _validate_report_code(args.report)
        fight_id = args.fight
        source_id = args.source
    else:
        print("错误: 需要 report URL 或 --report 参数", file=sys.stderr)
        sys.exit(1)

    logger.info("目标: report=%s fight=%s source=%s", report_code, fight_id, source_id)

    logger.info("拉取战斗列表...")
    meta = client.get_report_fights(report_code)
    fight = _find_fight(meta, fight_id) if fight_id else None
    if fight_id and fight is None:
        raise ValueError(f"报告中不存在 fight={fight_id}")
    result = _build_download_payload(meta, fight, source_id)

    if fight:
        logger.info("选定战斗: [%d] %s", fight.id, fight.name)
    else:
        logger.info("共 %d 场战斗, 全部保留", len(meta.fights))

    if args.damage_only:
        if not fight or not source_id:
            print("错误: 拉取伤害表需要 --fight 和 --source", file=sys.stderr)
            sys.exit(1)
        logger.info("拉取伤害表 (fight=%d, source=%d)...", fight.id, source_id)
        table = client.get_damage_table(report_code, fight, source_id)
        result["damage_table"] = table
    else:
        if not args.events_only and source_id and fight:
            logger.info("拉取伤害表...")
            try:
                table = client.get_damage_table(report_code, fight, source_id)
                result["damage_table"] = table
            except Exception as e:
                logger.warning("获取伤害表失败: %s", e)
        # 分析器需要整场事实；选定玩家只用于训练索引与伤害表。
        if fight:
            events = client.get_fight_events(report_code, fight, require_complete=True)
        else:
            events = client.get_report_events(
                report_code, end_time=meta.report["end"] - meta.report["start"],
                require_complete=True,
            )
        _attach_analysis_events(result, events)

    output_path = args.output or os.path.join(
        args.output_dir or "data",
        _build_output_filename(report_code, result["fight_id"], source_id),
    )
    _write_download_json(output_path, result)

    logger.info("已写入 -> %s", output_path)
    print(f"输出文件: {output_path}")


def _cmd_batch(client: FFLogsV2Client, args) -> None:
    """V2 API 批量下载高分报告。"""
    if args.zone and not args.encounter:
        encounters = client.get_zone_encounters(args.zone)
        print(f"Zone {args.zone} encounters:")
        for e in encounters:
            print(f"  id={e['id']:<5} {e['name']}")
        print(f"\n用法: batch -e <id>")
        return

    if not args.encounter:
        print("错误: 需要 --encounter 参数", file=sys.stderr)
        sys.exit(1)

    logger.info("批量: encounter=%d spec=%s bracket=%d pages=%d metric=%s",
                 args.encounter, args.spec_name, args.bracket, args.max_pages, args.metric)

    reports = client.get_high_score_reports(
        encounter_id=args.encounter,
        spec_name=args.spec_name,
        bracket=args.bracket,
        max_pages=args.max_pages,
        metric=args.metric,
    )

    if not reports:
        print("未找到符合条件的报告")
        return

    print(f"\n找到 {len(reports)} 份报告，开始批量下载...\n")
    _batch_download(client, reports, output_dir=args.output, mode=args.mode)


def _batch_download(
    client: FFLogsV2Client,
    reports: list[tuple],
    output_dir: str = "data",
    mode: str = "default",
):
    """批量下载报告。

    Parameters
    ----------
    client: FFLogsV2Client 实例
    reports: [(report_code, fight_id, player_name, dps), ...]
    output_dir: 输出目录
    mode: "default" | "events-only" | "damage-only"
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(reports)
    success = 0
    skipped = 0
    failed = 0

    for i, (code, fid, name, dps) in enumerate(reports, 1):
        if client._cancelled:
            logger.info("收到中断信号，停止批量下载")
            break

        logger.info("[%d/%d] %s f=%d name=%s dps=%.0f",
                     i, total, code, fid, name, dps)

        try:
            output_filename = _build_batch_output_filename(code, fid, name)
        except (TypeError, ValueError) as e:
            logger.warning("  报告编号无效，跳过: %s", e)
            failed += 1
            continue

        output_path = os.path.join(output_dir, output_filename)
        if os.path.exists(output_path):
            logger.info("  -> 已存在，跳过")
            skipped += 1
            continue

        try:
            sid = client.resolve_source_id(code, name, fid)
            if not sid:
                logger.warning("  未找到玩家 %s 的 source ID，跳过", name)
                failed += 1
                continue

            meta = client.get_report_fights(code)
            fight = _find_fight(meta, fid)
            if not fight:
                logger.warning("  未找到 fight=%d，跳过", fid)
                failed += 1
                continue

            result = _build_download_payload(meta, fight, sid)
            result.update(player_name=name, player_dps=dps)

            if mode == "damage-only":
                table = client.get_damage_table(code, fight, sid)
                result["damage_table"] = table
            else:
                if mode != "events-only":
                    try:
                        table = client.get_damage_table(code, fight, sid)
                        result["damage_table"] = table
                    except Exception as e:
                        logger.warning("  伤害表获取失败: %s", e)
                events = client.get_fight_events(code, fight, require_complete=True)
                _attach_analysis_events(result, events)

            _write_download_json(output_path, result)
            logger.info("  -> %s", output_path)
            success += 1

        except Exception as e:
            logger.error("  -> 失败: %s", e)
            failed += 1

    logger.info("批量下载完成: 成功=%d 跳过=%d 失败=%d / 总计=%d",
                 success, skipped, failed, total)
    print(f"\n批量下载完成: 成功={success} 跳过={skipped} 失败={failed} / 总计={total}")


def _cmd_encounters(client: FFLogsV2Client, args) -> None:
    """列出 zone 下的 encounters 或全部 zones。"""
    if args.zone:
        encounters = client.get_zone_encounters(args.zone)
        print(f"Zone {args.zone} encounters:")
        print(f"{'ID':<6} {'名称'}")
        print("-" * 40)
        for e in encounters:
            print(f"{e['id']:<6} {e['name']}")
    else:
        zones = client.get_zones()
        if not zones:
            print("无法获取区域列表")
            return
        print(f"{'ID':<6} {'名称'}")
        print("-" * 60)
        for z in zones:
            print(f"{str(z.get('id', '')):<6} {z.get('name', '')}")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="FFLogs 战斗数据拉取工具 (V2 GraphQL API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s "https://www.fflogs.com/reports/JFLCXcQjBd9zgWR1?fight=6&type=damage-done&source=10"
  %(prog)s single --report JFLCXcQjBd9zgWR1 --fight 6 --source 10
  %(prog)s batch -e 1079 --spec-name BlackMage --max-pages 3
  %(prog)s encounters -z 39

认证:
  在 FFLogs -> 我的 -> API 获取 V2 Client ID 和 Client Secret, 设置到 .env:
    FFLOGS_V2_CLIENT_ID=your_client_id
    FFLOGS_V2_CLIENT_SECRET=your_client_secret
        """,
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    sub = parser.add_subparsers(dest="command", help="子命令")

    # ---- 子命令: single (默认行为) ----
    single = sub.add_parser("single", help="单报告下载 (默认行为)", add_help=False)
    single.add_argument("url", nargs="?", help="FFLogs 报告 URL")
    single.add_argument("--report", "-r", help="报告码")
    single.add_argument("--fight", "-f", type=int, default=0,
                        help="Fight ID (0=全部, -1=最后一场)")
    single.add_argument("--source", "-s", type=int, help="玩家 source ID")
    single.add_argument("--output", "-o", help="输出文件路径")
    single.add_argument("--output-dir", default="data", help="输出目录 (默认 data/)")
    single.add_argument("--events-only", action="store_true", help="只拉事件")
    single.add_argument("--damage-only", action="store_true", help="只拉伤害表")

    # ---- 子命令: batch ----
    batch_cmd = sub.add_parser("batch", help="V2 批量下载 (排行查询)")
    batch_cmd.add_argument("--encounter", "-e", type=int, default=None,
                           help="Encounter ID")
    batch_cmd.add_argument("--zone", "-z", type=int, help="Zone ID (用于列出 encounters)")
    batch_cmd.add_argument("--spec-name", default="BlackMage",
                           help="职业名 (默认 BlackMage)")
    batch_cmd.add_argument("--bracket", "-b", type=int, default=0, choices=[0, 6],
                            help="分段: 6=金100%%, 0=全部(默认)")
    batch_cmd.add_argument("--metric", default="rdps",
                            choices=["dps", "rdps", "ndps", "adps"],
                            help="排行指标 (默认 rdps)")
    batch_cmd.add_argument("--max-pages", type=int, default=10, help="最多翻页数")
    batch_cmd.add_argument("--mode", "-m",
                           choices=["default", "events-only", "damage-only"],
                           default="default", help="下载模式")
    batch_cmd.add_argument("--output", "-o", default="data", help="输出目录")

    # ---- 子命令: encounters ----
    enc_cmd = sub.add_parser("encounters", help="列出 zones 或 zone 下的 encounters")
    enc_cmd.add_argument("--zone", "-z", type=int, help="Zone ID (列出其 encounters)")

    args = parser.parse_args()

    _load_dotenv()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 认证
    client_id = os.environ.get("FFLOGS_V2_CLIENT_ID")
    client_secret = os.environ.get("FFLOGS_V2_CLIENT_SECRET")
    if not client_id or not client_secret:
        print(
            "错误: 缺少 FFLogs V2 API 凭证.\n"
            "  设置环境变量 FFLOGS_V2_CLIENT_ID 和 FFLOGS_V2_CLIENT_SECRET.\n"
            "  在 FFLogs -> 我的 -> API 获取 V2 Client ID 和 Client Secret.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = FFLogsV2Client(client_id, client_secret)

    def _on_interrupt(signum, frame):
        logger.info("收到 Ctrl+C，正在停止...")
        client.cancel()

    signal.signal(signal.SIGINT, _on_interrupt)

    # 子命令分发
    if args.command == "batch":
        _cmd_batch(client, args)
    elif args.command == "encounters":
        _cmd_encounters(client, args)
    elif args.command == "single" or args.command is None:
        _cmd_single(client, args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

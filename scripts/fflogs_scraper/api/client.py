"""FFLogs V2 OAuth、GraphQL 查询、事件分页与排行 API。"""

import logging
from typing import Optional

from curl_cffi import requests as cf_requests

from ..config.constants import BROWSER_FINGERPRINT, MAX_EVENTS
from ..config.validation import (
    _validate_alphanumeric,
    _validate_integer,
    _validate_integer_list,
    _validate_ranking_metric,
    _validate_report_code,
)
from ..contracts.models import FightInfo, ReportMeta
from ..contracts.report import _adapt_report_metadata

logger = logging.getLogger(__name__)


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

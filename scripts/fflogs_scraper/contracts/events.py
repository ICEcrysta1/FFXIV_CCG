"""事件格式适配与完整性标记。"""


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

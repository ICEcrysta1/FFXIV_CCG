"""FFLogs 下载器测试共用的报告元数据。"""

import pytest

from scripts.fflogs_scraper.contracts import report


@pytest.fixture
def analysis_meta():
    """覆盖 V2 玩家、敌人实例和宠物归属。"""
    return report._adapt_report_metadata("ABC123", {
        "title": "中文报告", "startTime": 1778332651064, "endTime": 1778332661064,
        "owner": {"name": "Uploader"}, "zone": {"id": 66, "name": "Ranking zone"},
        "masterData": {"lang": "cn", "actors": [
            {"id": 1, "gameID": 1000001, "name": "黑魔", "type": "Player", "subType": "BlackMage"},
            {"id": 2, "gameID": 17839, "name": "Boss", "type": "NPC", "subType": "Boss"},
            {"id": 3, "gameID": 13961, "name": "宠物", "type": "Pet", "petOwner": 1},
        ]},
        "fights": [{
            "id": 33, "name": "Futures Rewritten", "encounterID": 1079,
            "startTime": 5000, "endTime": 10000, "combatTime": 5000,
            "difficulty": 100, "kill": True, "size": 8,
            "bossPercentage": 12.5, "fightPercentage": 25.0,
            "gameZone": {"id": 1238, "name": "A Future Rewritten"},
            "friendlyPlayers": [1], "enemyPlayers": [], "friendlyNPCs": [],
            "enemyNPCs": [{"id": 2, "instanceCount": 2, "groupCount": 1}],
            "friendlyPets": [{"id": 3, "instanceCount": 3}], "enemyPets": [],
        }],
    })

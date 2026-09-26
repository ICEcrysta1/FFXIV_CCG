"""副本区域和战斗目录查询展示。"""

from ..api.client import FFLogsV2Client


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

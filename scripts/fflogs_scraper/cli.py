"""FFLogs 下载命令行参数、认证与中断处理。"""

import argparse
import logging
import os
import signal
import sys

from .api.client import FFLogsV2Client
from .config.encounters import resolve_download_encounter
from .config.environment import _load_dotenv
from .config.validation import _validate_integer
from .download.batch import _cmd_batch
from .download.encounters import _cmd_encounters
from .download.single import _cmd_single

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="FFLogs 战斗数据拉取工具 (V2 GraphQL API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s single "https://www.fflogs.com/reports/JFLCXcQjBd9zgWR1?fight=6&type=damage-done&source=10"
  %(prog)s single --report JFLCXcQjBd9zgWR1 --fight 6 --source 10
  %(prog)s batch FRU --count 200
  %(prog)s batch 65 --count 200
  %(prog)s batch -e 1079 --spec-name BlackMage --count 200 --output data/human/job/black_mage/raw/FRU
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
    batch_cmd = sub.add_parser("batch", help="国服十档训练下载，并自动补充美服高分验证数据")
    batch_cmd.add_argument("target", nargs="?", help="配置中的副本代号；单战斗 Zone 也可输入 Zone ID")
    batch_cmd.add_argument("--encounter", "-e", type=int, default=None,
                           help="Encounter ID")
    batch_cmd.add_argument("--zone", "-z", type=int, help="Zone ID (用于列出 encounters)")
    batch_cmd.add_argument("--spec-name", default=None,
                           help="职业名；副本代号模式默认采用当前模型职业，旧 -e 模式默认 BlackMage")
    batch_cmd.add_argument("--count", type=int, default=200,
                            help="国服训练目标份数，自动均分十档；美服验证目标按 config/fflogs_scraper/batch.yaml 比例计算")
    batch_cmd.add_argument("--partition", type=int, default=None,
                           help="API 排名分区 ID；不指定时使用 API 默认分区")
    batch_cmd.add_argument("--bracket", "-b", type=int, default=0,
                           help="角色发现使用的补丁分组 ID；0=不限制，非百分位档位")
    batch_cmd.add_argument("--metric", default="rdps",
                            choices=["dps", "rdps", "ndps", "adps"],
                            help="排行指标 (默认 rdps)")
    batch_cmd.add_argument("--max-pages", type=int, default=10,
                           help="所选分区最多查询的角色榜单页数 (默认 10)")
    batch_cmd.add_argument("--mode", "-m",
                           choices=["default", "events-only", "damage-only"],
                           default="default", help="下载模式")
    batch_cmd.add_argument("--output", "-o",
                            help="训练副本目录，必须指向 raw/<副本>；验证写入 raw/VAL/<副本>")

    # ---- 子命令: encounters ----
    enc_cmd = sub.add_parser("encounters", help="列出 zones 或 zone 下的 encounters")
    enc_cmd.add_argument("--zone", "-z", type=int, help="Zone ID (列出其 encounters)")

    args = parser.parse_args()

    if args.command == "batch":
        try:
            for field, minimum in (("count", 1), ("max_pages", 1), ("bracket", 0)):
                _validate_integer(getattr(args, field), field, minimum=minimum)
            if args.partition is not None:
                _validate_integer(args.partition, "partition", minimum=1)
        except ValueError as error:
            parser.error(str(error))
        if args.target is not None:
            if args.encounter is not None or args.output is not None or args.zone is not None:
                parser.error("副本代号不能与 --encounter、--zone 或 --output 同时使用")
            try:
                resolve_download_encounter(args.target)
            except ValueError as error:
                parser.error(str(error))
        elif args.encounter is not None and args.output is None:
            parser.error("batch 下载需要 --output raw/<副本>")

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

    clients = (
        (FFLogsV2Client(client_id, client_secret, server_region="CN"),
         FFLogsV2Client(client_id, client_secret, server_region="NA"))
        if args.command == "batch" else (FFLogsV2Client(client_id, client_secret),)
    )

    def _on_interrupt(signum, frame):
        logger.info("收到 Ctrl+C，正在停止...")
        for client in clients:
            client.cancel()

    signal.signal(signal.SIGINT, _on_interrupt)

    # 子命令分发
    try:
        if args.command == "batch":
            _cmd_batch(clients[0], clients[1], args)
        elif args.command == "encounters":
            _cmd_encounters(clients[0], args)
        elif args.command == "single" or args.command is None:
            _cmd_single(clients[0], args)
        else:
            parser.print_help()
            sys.exit(1)
    finally:
        for client in clients:
            client.close()

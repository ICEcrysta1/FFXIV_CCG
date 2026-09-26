"""FFLogs 下载命令行参数、认证与中断处理。"""

import argparse
import logging
import os
import signal
import sys

from .api.client import FFLogsV2Client
from .config.environment import _load_dotenv
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

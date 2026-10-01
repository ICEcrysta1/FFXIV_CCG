"""离线标注 CLI，复用公共数据发现与环境变量加载。"""

import argparse
import logging
import signal
import subprocess
from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Event

from common.dataset_layout import find_dataset_json_files
from scripts.common.json_io import is_json_file

from .config import default_raw_root, load_bridge_config
from .runner import analyzer_commit, annotation_is_current, annotate_file, output_path_for_source

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="raw JSON -> annotated JSON 离线动作质量标注")
    parser.add_argument("inputs", nargs="*", type=Path, help="raw JSON 文件或目录；默认从模型输入阶段定位同级 raw")
    parser.add_argument("--output-root", type=Path, help="评估输出根目录；默认 raw 同级 annotated")
    parser.add_argument("--source-root", type=Path, help="自定义输入阶段根目录，保留其下相对层级")
    parser.add_argument("--limit", type=int, help="只处理前 N 份，便于验证")
    parser.add_argument("--force", action="store_true", help="忽略已有标注，重新分析所选文件")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = load_bridge_config()
    inputs = args.inputs or [default_raw_root()]
    files = set()
    for path in inputs:
        if path.is_dir():
            files.update(find_dataset_json_files(path))
        elif path.is_file() and is_json_file(path):
            files.add(path.resolve())
        else:
            parser.error(f"JSON input not found: {path}")
    selected = sorted(files, key=lambda path: str(path).casefold())
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        logger.error("没有找到 JSON 文件")
        return 1
    destinations = [output_path_for_source(path, args.output_root, args.source_root) for path in selected]
    if len(set(destinations)) != len(destinations):
        parser.error("multiple inputs map to the same output; use separate output roots")
    commit = None if args.force else analyzer_commit(config)
    if commit is None and not args.force:
        logger.warning("无法确认分析器 commit，本次不跳过已有标注")

    stop = Event()

    def process(source: Path, destination: Path):
        try:
            if stop.is_set():
                raise CancelledError("标注已取消")
            if not args.force and annotation_is_current(source, destination, commit=commit):
                return "skipped", destination, None
            output = annotate_file(
                source, config=config, output_root=args.output_root,
                source_root=args.source_root, stop=stop,
            )
        except (OSError, TypeError, ValueError, RuntimeError, UnicodeError, subprocess.TimeoutExpired) as error:
            return "failed", None, error
        return "saved", output, None

    counts = {"saved": 0, "skipped": 0, "failed": 0}
    previous_handlers = {}

    def cancel(signum, frame):
        # 重复 Ctrl+C 只设置取消状态，不打断子进程回收和线程退出。
        stop.set()

    executor = ThreadPoolExecutor(max_workers=min(config.max_workers, len(selected)))
    try:
        signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGBREAK"):
            signals.append(signal.SIGBREAK)
        for signum in signals:
            previous_handlers[signum] = signal.signal(signum, cancel)
        remaining = iter(zip(selected, destinations, strict=True))
        futures = {}
        index = 0
        while not stop.is_set():
            # 只提交当前并发所需任务，中断后不再从长队列启动 Node。
            while len(futures) < config.max_workers and not stop.is_set():
                item = next(remaining, None)
                if item is None:
                    break
                source, destination = item
                futures[executor.submit(process, source, destination)] = source
            if not futures:
                break
            completed, _ = wait(futures, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in completed:
                source = futures.pop(future)
                status, output, error = future.result()
                index += 1
                counts[status] += 1
                if error is not None:
                    logger.error("分析失败 %s: %s", source, error)
                else:
                    logger.info("%d/%d %s: %s", index, len(selected), "跳过" if status == "skipped" else "已保存", output)
    except (KeyboardInterrupt, CancelledError):
        stop.set()
    finally:
        cancelled = stop.is_set()
        stop.set()
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    if cancelled:
        logger.warning("标注已取消，已回收分析进程；已保存的结果保留，下次运行可继续。")
        return 130
    logger.info("完成: 新标注=%d 跳过=%d 失败=%d", counts["saved"], counts["skipped"], counts["failed"])
    return int(counts["failed"] > 0)

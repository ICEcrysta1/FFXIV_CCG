"""离线标注 CLI，复用公共数据发现与环境变量加载。"""

import argparse
import logging
import subprocess
from pathlib import Path

from common.dataset_layout import find_dataset_json_files

from .config import default_raw_root, load_bridge_config
from .runner import annotate_file, output_path_for_source

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="raw JSON -> annotated JSON 离线动作质量标注")
    parser.add_argument("inputs", nargs="*", type=Path, help="JSON 文件或目录；默认读取模型 YAML 的 raw_data_dir")
    parser.add_argument("--output-root", type=Path, help="评估输出根目录；默认 raw 同级 annotated")
    parser.add_argument("--source-root", type=Path, help="自定义输入阶段根目录，保留其下相对层级")
    parser.add_argument("--limit", type=int, help="只处理前 N 份，便于验证")
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
        elif path.is_file() and path.suffix.lower() == ".json":
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
    failed = 0
    for index, source in enumerate(selected, 1):
        logger.info("分析 %d/%d: %s", index, len(selected), source)
        try:
            output = annotate_file(
                source, config=config, output_root=args.output_root,
                source_root=args.source_root,
            )
        except (OSError, TypeError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            logger.error("分析失败 %s: %s", source, error)
            failed += 1
        else:
            logger.info("已保存: %s", output)
    logger.info("完成: 成功=%d 失败=%d", len(selected) - failed, failed)
    return int(failed > 0)

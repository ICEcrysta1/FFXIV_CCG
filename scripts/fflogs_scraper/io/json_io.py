"""下载 JSON 的 UTF-8/LF 原子写入。"""

from scripts.common.json_io import atomic_write_json


def _write_download_json(output_path: str, result: dict) -> None:
    """仅在完整序列化成功后替换目标，避免留下被批量任务误跳过的半文件。"""
    atomic_write_json(output_path, result)

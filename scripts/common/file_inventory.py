"""复用未变化文件的校验结果；业务兼容性由各流程自己的校验器决定。"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterable
from pathlib import Path

from .json_io import atomic_write_json

logger = logging.getLogger(__name__)


def _stamp(path: Path) -> list[int] | None:
    """只读文件元数据；替换、改写和删除都会使旧记录失效。"""
    try:
        stat = path.stat()
        return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino]
    except OSError:
        return None


class FileInventory:
    """一次加载、批量查验、原子保存的可丢弃索引，不替代正式文件契约。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: dict[str, dict] = {}
        self._dirty = False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("version") == 1 and isinstance(payload.get("entries"), dict):
                self._entries = payload["entries"]
        except FileNotFoundError:
            pass
        except (OSError, ValueError, UnicodeError):
            logger.warning("文件校验索引不可用，将重新校验：%s", self.path)

    def lookup(self, key: Path, *, context: object, paths: Iterable[Path]) -> object:
        """只查轻量索引，返回 None 表示需要正式校验。"""
        # 这里只规范化路径文本；命中判断只需 stat，避免 resolve 逐层访问磁盘。
        key = os.path.abspath(key)
        inputs = [os.path.abspath(path) for path in paths]
        context = json.dumps(context, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        entry = self._entries.get(key)
        if (isinstance(entry, dict) and entry.get("context") == context
                and entry.get("inputs") == inputs
                and isinstance(entry.get("files"), dict)
                and all(path in entry["files"] for path in inputs)
                and all(_stamp(Path(path)) == stamp for path, stamp in entry["files"].items())
                and "value" in entry):
            return entry["value"]
        return None

    @staticmethod
    def snapshot(paths: Iterable[Path]) -> dict[str, list[int] | None]:
        """在异步校验提交前冻结文件身份，完成后才能确认结果仍可复用。"""
        return {os.path.abspath(path): _stamp(Path(path)) for path in paths}

    def check(
        self, key: Path, *, context: object, paths: Iterable[Path],
        validate: Callable[[], tuple[object, Iterable[Path]]],
        before: dict[str, list[int] | None] | None = None,
    ) -> object:
        """命中时仅 stat；否则正式校验，并记录额外依赖（例如 PT 分片）。"""
        paths = tuple(paths)
        cached = self.lookup(key, context=context, paths=paths)
        if cached is not None:
            return cached
        key = os.path.abspath(key)
        inputs = [os.path.abspath(path) for path in paths]
        context = json.dumps(context, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        before = self.snapshot(paths) if before is None else before
        value, extra_paths = validate()
        watched = dict.fromkeys([*inputs, *(os.path.abspath(path) for path in extra_paths)])
        stamps = {path: _stamp(Path(path)) for path in watched}
        # 校验期间变化的文件不能以新的时间戳背书旧结果。
        if value and all(stamps[path] == stamp for path, stamp in before.items()):
            self._entries[key] = {"context": context, "inputs": inputs, "files": stamps, "value": value}
            self._dirty = True
        else:
            if key in self._entries:
                del self._entries[key]
                self._dirty = True
            if any(stamps[path] != stamp for path, stamp in before.items()):
                return None
        return value

    def save(self) -> None:
        """索引写入失败只损失加速，不能阻止正常业务处理。"""
        if not self._dirty:
            return
        try:
            atomic_write_json(self.path, {"version": 1, "entries": self._entries})
        except OSError as error:
            logger.warning("无法保存文件校验索引 %s: %s", self.path, error)
        else:
            self._dirty = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.save()

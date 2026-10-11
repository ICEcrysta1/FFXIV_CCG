"""标注清单预检：索引命中直接复用，冷文件在有界进程中校验。"""

from __future__ import annotations

import logging
import signal
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import islice
from pathlib import Path
from threading import Event

from scripts.common.file_inventory import FileInventory

from .runner import annotation_is_current

logger = logging.getLogger(__name__)


def _ignore_interrupts() -> None:
    """取消由主进程协调，子进程完成当前一份校验后统一回收。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)


class AnnotationInventory:
    """业务兼容性仍由正式 validator 决定，索引只由主进程更新。"""

    def __init__(self, *, commit: str | None, validator=annotation_is_current):
        self.commit = commit
        self.validator = validator
        self.context = {"validator": 1, "commit": commit, "schema": 2, "bridge": 4}
        self.inventories: dict[Path, FileInventory] = {}

    def _inventory(self, destination: Path) -> FileInventory:
        if destination.parent not in self.inventories:
            self.inventories[destination.parent] = FileInventory(destination.parent / ".annotation-inventory.json")
        return self.inventories[destination.parent]

    def is_current(self, source: Path, destination: Path, *, result=None, before=None) -> bool:
        if self.commit is None:
            return False
        return self._inventory(destination).check(
            destination, context=self.context, paths=[source, destination], before=before,
            validate=lambda: (self.validator(source, destination, commit=self.commit) if result is None else result, ()),
        ) is True

    def save(self) -> None:
        for inventory in self.inventories.values():
            inventory.save()

    def partition(
        self, pairs: list[tuple[Path, Path]], *, max_workers: int, stop: Event,
    ) -> tuple[list[tuple[Path, Path]], int]:
        """先只查元数据，再并行校验冷文件；保持原候选顺序并周期保存进度。"""
        if max_workers < 1:
            raise ValueError("annotation scan workers must be positive")
        if self.commit is None:
            return pairs, 0
        started = time.monotonic()
        valid: set[Path] = set()
        unchecked = []
        discovered_count = 0
        for source, destination in pairs:
            if stop.is_set():
                break
            discovered_count += 1
            if self._inventory(destination).lookup(
                destination, context=self.context, paths=[source, destination],
            ) is True:
                valid.add(source)
            elif destination.is_file():
                unchecked.append((source, destination))
        completed_count = discovered_count - len(unchecked)
        logger.info(
            "标注索引预检: 总数=%d 索引命中=%d 缺失标注=%d 需完整校验=%d",
            len(pairs), len(valid), completed_count - len(valid), len(unchecked),
        )
        last_saved = time.monotonic()

        def accept(source, destination, result, before=None):
            nonlocal completed_count, last_saved
            if self.is_current(source, destination, result=result, before=before):
                valid.add(source)
            completed_count += 1
            if completed_count % 25 == 0 or time.monotonic() - last_saved >= 2.0:
                self.save()
                logger.info("标注预检进度: %d/%d 已有有效=%d 用时=%.1fs",
                            completed_count, len(pairs), len(valid), time.monotonic() - started)
                last_saved = time.monotonic()

        try:
            worker_count = min(max_workers, len(unchecked))
            if worker_count <= 1:
                for source, destination in unchecked:
                    if stop.is_set():
                        break
                    before = FileInventory.snapshot([source, destination])
                    accept(source, destination, self.validator(source, destination, commit=self.commit), before)
            elif not stop.is_set():
                # 每个 worker 同时只持有一对日志；不传递大事件对象、不无限提交 Future。
                remaining = iter(unchecked)
                with ProcessPoolExecutor(max_workers=worker_count, initializer=_ignore_interrupts) as executor:
                    futures = {}

                    def submit(pair):
                        source, destination = pair
                        before = FileInventory.snapshot(pair)
                        future = executor.submit(self.validator, source, destination, commit=self.commit)
                        futures[future] = (source, destination, before)

                    for pair in islice(remaining, worker_count):
                        submit(pair)
                    while futures and not stop.is_set():
                        completed, _ = wait(futures, timeout=0.2, return_when=FIRST_COMPLETED)
                        for future in completed:
                            source, destination, before = futures.pop(future)
                            accept(source, destination, future.result(), before)
                            pair = next(remaining, None)
                            if pair is not None and not stop.is_set():
                                submit(pair)
        finally:
            self.save()
        return [pair for pair in pairs if pair[0] not in valid], len(valid)

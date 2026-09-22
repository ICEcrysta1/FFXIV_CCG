"""训练 raw JSON 路径选择与副本配额分组。"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawTrainingPathGroup:
    """一个副本目录的目标有效文件数和有序 raw JSON 候选。"""

    directory_name: str
    target_count: int
    candidates: tuple[Path, ...]

    @property
    def primary_paths(self) -> tuple[Path, ...]:
        """返回按比例首轮尝试的文件，剩余候选用于失败补位。"""
        return self.candidates[: self.target_count]


def select_training_raw_path_groups(
    data_dir: Path,
    max_files: int | None = None,
) -> tuple[RawTrainingPathGroup, ...]:
    """按副本比例选择目标配额，并为每个副本保留失败补位候选。"""
    data_dir = Path(data_dir)
    directory_groups = []
    if data_dir.is_dir():
        for directory in sorted(path for path in data_dir.iterdir() if path.is_dir()):
            files = sorted(directory.rglob("*.json"))
            if files:
                directory_groups.append((directory.name, files))

    if not directory_groups:
        files = sorted(data_dir.rglob("*.json"))
        if not files:
            return ()
        target = len(files) if max_files is None or max_files <= 0 else min(int(max_files), len(files))
        return (RawTrainingPathGroup(data_dir.name or str(data_dir), target, tuple(files)),)

    total_files = sum(len(files) for _, files in directory_groups)
    if max_files is None or max_files <= 0 or max_files >= total_files:
        target_files = total_files
        allocations = [len(files) for _, files in directory_groups]
    else:
        target_files = int(max_files)
        allocations = []
        remainders: list[tuple[int, str, int]] = []
        for index, (directory_name, files) in enumerate(directory_groups):
            quotient, remainder = divmod(target_files * len(files), total_files)
            allocations.append(quotient)
            remainders.append((remainder, directory_name, index))

        remaining = target_files - sum(allocations)
        for _, _, index in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
            allocations[index] += 1

    selected_by_group = [
        [
            files[min(len(files) - 1, math.floor((index + 0.5) * len(files) / allocation))]
            for index in range(allocation)
        ]
        for (_, files), allocation in zip(directory_groups, allocations)
    ]
    selected = [path for paths in selected_by_group for path in paths]
    if len(selected) != target_files:
        raise RuntimeError(
            f"proportional raw selection mismatch: selected={len(selected)} target={target_files}"
        )

    groups = []
    for (directory_name, files), selected_paths, allocation in zip(
        directory_groups,
        selected_by_group,
        allocations,
    ):
        selected_set = set(selected_paths)
        groups.append(
            RawTrainingPathGroup(
                directory_name,
                allocation,
                tuple(selected_paths) + tuple(path for path in files if path not in selected_set),
            )
        )
    logger.info(
        "按副本比例选择 raw JSON: %s",
        ", ".join(
            f"{directory_name}={allocation}"
            for (directory_name, _), allocation in zip(directory_groups, allocations)
        ),
    )
    return tuple(groups)


def select_training_raw_paths(data_dir: Path, max_files: int | None = None) -> list[Path]:
    """按副本目录中的 raw JSON 文件数量比例选择首轮训练文件。"""
    return [
        path
        for group in select_training_raw_path_groups(data_dir, max_files)
        for path in group.primary_paths
    ]

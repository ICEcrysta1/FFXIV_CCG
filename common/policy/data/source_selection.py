"""按副本比例选择训练数据来源，并保留失败文件的补位顺序。"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from common.dataset_layout import (
    DATASET_STAGES,
    PERCENTILE_BUCKETS,
    find_dataset_json_files,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawTrainingPathGroup:
    """一个副本/区间的目标有效文件数和有序 JSON 候选。"""

    directory_name: str
    target_count: int
    candidates: tuple[Path, ...]

    @property
    def primary_paths(self) -> tuple[Path, ...]:
        """返回首轮文件，剩余候选用于失败补位。"""
        return self.candidates[: self.target_count]


def _encounter_and_bucket(root: Path, source: Path) -> tuple[str, str | None]:
    relative = source.relative_to(root).parts
    if root.name in PERCENTILE_BUCKETS:
        return root.parent.name, root.name
    if root.name not in DATASET_STAGES and len(relative) >= 2 and relative[0] in PERCENTILE_BUCKETS:
        return root.name, relative[0]
    if len(relative) == 1:
        return root.name or str(root), None
    encounter = relative[0]
    bucket = relative[1] if len(relative) >= 3 and relative[1] in PERCENTILE_BUCKETS else None
    return encounter, bucket


def _proportional_allocations(sizes: list[int], names: list[str], target: int) -> list[int]:
    total = sum(sizes)
    if target >= total:
        return sizes[:]
    allocations = []
    remainders = []
    for index, (size, name) in enumerate(zip(sizes, names)):
        quotient, remainder = divmod(target * size, total)
        allocations.append(quotient)
        remainders.append((remainder, name, index))
    for _, _, index in sorted(remainders, key=lambda item: (-item[0], item[1]))[:target - sum(allocations)]:
        allocations[index] += 1
    return allocations


def _balanced_allocations(sizes: list[int], target: int) -> list[int]:
    """同副本各区间尽量均分，输入不足的区间不占用空配额。"""
    if target >= sum(sizes):
        return sizes[:]
    allocations = [0] * len(sizes)
    available = [index for index, size in enumerate(sizes) if size > 0]
    if 0 < target < len(available):
        positions = (
            [round((len(available) - 1) / 2)]
            if target == 1 else [
                round(index * (len(available) - 1) / (target - 1))
                for index in range(target)
            ]
        )
        for position in positions:
            allocations[available[position]] = 1
        return allocations
    remaining = target
    while remaining:
        for index, size in enumerate(sizes):
            if allocations[index] < size and remaining:
                allocations[index] += 1
                remaining -= 1
    return allocations


def select_training_raw_path_groups(
    data_dir: Path,
    max_files: int | None = None,
) -> tuple[RawTrainingPathGroup, ...]:
    """先按副本分配总额，再按其已有区间均分；失败只在同区间补位。"""
    data_dir = Path(data_dir)
    discovered = find_dataset_json_files(data_dir)
    if data_dir.name in DATASET_STAGES:
        discovered = [
            source for source in discovered
            if source.relative_to(data_dir).parts[0] != "VAL"
        ]
    if not discovered:
        return ()
    by_encounter: dict[str, dict[str | None, list[Path]]] = {}
    for source in discovered:
        encounter, bucket = _encounter_and_bucket(data_dir, source)
        by_encounter.setdefault(encounter, {}).setdefault(bucket, []).append(source)

    encounters = sorted(by_encounter)
    total_files = len(discovered)
    target_files = total_files if max_files is None or max_files <= 0 else min(int(max_files), total_files)
    encounter_sizes = [sum(len(files) for files in by_encounter[name].values()) for name in encounters]
    encounter_targets = _proportional_allocations(encounter_sizes, encounters, target_files)
    directory_groups: list[tuple[str, list[Path], int]] = []
    for encounter, encounter_target in zip(encounters, encounter_targets):
        buckets = by_encounter[encounter]
        ordered_buckets = sorted(
            buckets,
            key=lambda bucket: (bucket is None, PERCENTILE_BUCKETS.index(bucket) if bucket is not None else 0),
        )
        bucket_sizes = [len(buckets[bucket]) for bucket in ordered_buckets]
        bucket_targets = _balanced_allocations(bucket_sizes, encounter_target)
        for bucket, allocation in zip(ordered_buckets, bucket_targets):
            name = encounter if bucket is None else f"{encounter}/{bucket}"
            directory_groups.append((name, buckets[bucket], allocation))

    selected_by_group = [
        [
            files[min(len(files) - 1, math.floor((index + 0.5) * len(files) / allocation))]
            for index in range(allocation)
        ]
        for _, files, allocation in directory_groups
    ]
    selected = [path for paths in selected_by_group for path in paths]
    if len(selected) != target_files:
        raise RuntimeError(
            f"proportional raw selection mismatch: selected={len(selected)} target={target_files}"
        )

    groups = []
    for (directory_name, files, allocation), selected_paths in zip(directory_groups, selected_by_group):
        selected_set = set(selected_paths)
        groups.append(RawTrainingPathGroup(
            directory_name, allocation,
            tuple(selected_paths) + tuple(path for path in files if path not in selected_set),
        ))
    logger.info(
        "按副本/区间选择 JSON: %s",
        ", ".join(f"{name}={allocation}" for name, _, allocation in directory_groups),
    )
    return tuple(groups)


def select_training_raw_paths(data_dir: Path, max_files: int | None = None) -> list[Path]:
    """按副本和区间配额选择首轮训练文件。"""
    return [
        path
        for group in select_training_raw_path_groups(data_dir, max_files)
        for path in group.primary_paths
    ]


def select_validation_raw_path_groups(
    data_dir: Path, max_files: int,
) -> tuple[RawTrainingPathGroup, ...]:
    """按验证副本固定均分配额；只从各自 VAL 目录补位，不跨副本。"""
    if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files < 1:
        raise ValueError("validation max_files must be a positive integer")
    data_dir = validation_stage_root(data_dir)
    validation_root = data_dir / "VAL"
    encounter_names = {path.name for path in validation_root.iterdir() if path.is_dir()} if validation_root.is_dir() else set()
    if data_dir.name == "annotated":
        raw_validation_root = data_dir.parent / "raw" / "VAL"
        if raw_validation_root.is_dir():
            encounter_names.update(path.name for path in raw_validation_root.iterdir() if path.is_dir())
    if not encounter_names:
        raise FileNotFoundError(f"没有验证副本目录：{validation_root}")
    encounters = sorted(encounter_names)
    if max_files < len(encounters):
        raise ValueError(f"validation_files={max_files} 小于验证副本数 {len(encounters)}")
    base, remainder = divmod(max_files, len(encounters))
    return tuple(
        RawTrainingPathGroup(
            encounter,
            base + (index < remainder),
            tuple(sorted((validation_root / encounter).glob("*.json.br"))),
        )
        for index, encounter in enumerate(encounters)
    )


def validation_stage_root(data_dir: Path) -> Path:
    """从阶段根或其副本/区间子目录定位独立 VAL 的同级阶段根。"""
    source = Path(data_dir).resolve()
    stage = next(
        (parent for parent in (source, *source.parents)
         if parent.name in {"raw", "annotated"}),
        None,
    )
    if stage is None:
        raise ValueError(f"validation source must be under raw or annotated: {source}")
    return stage

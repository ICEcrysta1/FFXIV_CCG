"""FFLogs 原始数据到训练样本的转换包。"""

from __future__ import annotations

from .config import (
    ConvertFflogsConfig,
    ConvertFflogsJobConfig,
    GcdDetectionConfig,
    load_convert_fflogs_config,
    load_convert_fflogs_dotenv,
    load_convert_fflogs_job_config,
    resolve_convert_fflogs_job_tag,
)
from .config.constants import (
    DEFAULT_DOWNTIME_GAP_SECONDS,
    FORCED_MOVEMENT_CONTEXT_KEY,
    FORCED_MOVEMENT_FEATURE_KEYS,
    MOVEMENT_MERGE_GAP,
    MOVE_DIST_THRESHOLD,
    POTION_BUFF_ID,
    POTION_SKILL_ID,
    RAID_BUFF_WINDOW_CONTEXT_KEY,
    RAID_BUFF_WINDOW_FEATURE_KEYS,
    RAID_BUFF_WINDOW_MARKER_KEYS,
    SCENE_CONTEXT_KEYS,
    SCENE_EPSILON,
    TARGET_COUNT_WINDOW_CONTEXT_KEY,
    TARGET_COUNT_WINDOW_FEATURE_KEYS,
    TARGETABLE_SEGMENT_KINDS,
    TARGETABLE_WINDOW_CONTEXT_KEY,
    TARGETABLE_WINDOW_FEATURE_KEYS,
)
from .cache import (
    load_raw_compiled_cache,
    prepare_training_caches,
    precompile_raw_training_caches,
)
from .extraction.downtime import detect_downtime_windows
from .extraction.extraction import (
    annotate_action_movement,
    detect_forced_movement_windows,
    detect_gcd_from_logs,
    extract_supported_actions,
)
from .extraction.fight_payload import build_fight_payload, build_output_fight_id
from .pipeline import convert_report_payload, convert_report_to_training_payload
from .source.raw_source import convert_raw_file
from .scene.scene_context import (
    build_forced_movement_context,
    build_raid_buff_window_context,
    build_scene_context,
    build_scene_context_view,
    build_target_count_window_context,
    build_target_count_window_token,
    build_targetable_window_context,
    normalize_scene_context,
    resolve_anchor,
)
from .training.training import build_training_samples, resolve_initial_timestamp
from .utils import build_backend, build_skill_book, load_job_project_config, merge_timestamps_to_windows


def main() -> None:
    """延迟加载 CLI，避免 `python -m scripts.convert_fflogs.cli` 重复初始化模块。"""
    from .cli import main as cli_main

    cli_main()

__all__ = [
    "ConvertFflogsConfig",
    "ConvertFflogsJobConfig",
    "DEFAULT_DOWNTIME_GAP_SECONDS",
    "FORCED_MOVEMENT_CONTEXT_KEY",
    "FORCED_MOVEMENT_FEATURE_KEYS",
    "GcdDetectionConfig",
    "MOVEMENT_MERGE_GAP",
    "MOVE_DIST_THRESHOLD",
    "POTION_BUFF_ID",
    "POTION_SKILL_ID",
    "RAID_BUFF_WINDOW_CONTEXT_KEY",
    "RAID_BUFF_WINDOW_FEATURE_KEYS",
    "RAID_BUFF_WINDOW_MARKER_KEYS",
    "SCENE_CONTEXT_KEYS",
    "SCENE_EPSILON",
    "TARGET_COUNT_WINDOW_CONTEXT_KEY",
    "TARGET_COUNT_WINDOW_FEATURE_KEYS",
    "TARGETABLE_SEGMENT_KINDS",
    "TARGETABLE_WINDOW_CONTEXT_KEY",
    "TARGETABLE_WINDOW_FEATURE_KEYS",
    "annotate_action_movement",
    "build_fight_payload",
    "build_forced_movement_context",
    "build_backend",
    "build_skill_book",
    "load_job_project_config",
    "build_output_fight_id",
    "build_raid_buff_window_context",
    "build_scene_context",
    "build_scene_context_view",
    "build_target_count_window_context",
    "build_target_count_window_token",
    "build_targetable_window_context",
    "build_training_samples",
    "convert_report_payload",
    "convert_report_to_training_payload",
    "detect_downtime_windows",
    "detect_forced_movement_windows",
    "detect_gcd_from_logs",
    "extract_supported_actions",
    "load_convert_fflogs_config",
    "load_convert_fflogs_dotenv",
    "load_convert_fflogs_job_config",
    "main",
    "merge_timestamps_to_windows",
    "normalize_scene_context",
    "resolve_convert_fflogs_job_tag",
    "resolve_anchor",
    "resolve_initial_timestamp",
    "convert_raw_file",
    "load_raw_compiled_cache",
    "precompile_raw_training_caches",
    "prepare_training_caches",
]

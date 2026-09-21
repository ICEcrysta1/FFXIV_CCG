"""跨状态机、输出和训练层共享的稳定契约常量。

定义由 `config/schema.yaml` 提供（C# 与 Python 单一事实来源），
本模块只做装配。
"""

from __future__ import annotations

from .schema_config import load_schema_config

_contracts = load_schema_config()["contracts"]
_scene_context_keys = _contracts["scene_context_keys"]

TARGETABLE_WINDOW_CONTEXT_KEY = str(_scene_context_keys["targetable"])
FORCED_MOVEMENT_CONTEXT_KEY = str(_scene_context_keys["forced_movement"])
RAID_BUFF_WINDOW_CONTEXT_KEY = str(_scene_context_keys["raid_buff"])
TARGET_COUNT_WINDOW_CONTEXT_KEY = str(_scene_context_keys["target_count"])
SCENE_CONTEXT_ABSOLUTE_MODE = str(_contracts["scene_context_absolute_mode"])
SLIDECAST_WINDOW_SECONDS = float(_contracts["slidecast_window_seconds"])
SCENE_EPSILON = float(_contracts["scene_epsilon"])
SIDECAR_CONTRACT_VERSION = int(_contracts["sidecar_contract_version"])

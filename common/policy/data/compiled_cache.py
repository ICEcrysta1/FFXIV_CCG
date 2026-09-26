"""最终 compiled cache 的只读 manifest、shard reader 和签名工具。"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

from common.dataset_layout import map_dataset_output_path
from common.torch_dependencies import import_torch
from common.torch_serialization import safe_torch_load

from .schema import SceneWindowSchema, TrainingSchema

# v12：黑魔候选集合移除 retrace、manaward、surecast，compiled cache 的候选布局
# 不再与旧缓存兼容；同时保留 v11 的提前效果结算语义。
CACHE_FORMAT = "raw_json_compiled_samples_v12_black_mage_candidate_set"
# v11：C# 状态机把硬读条的服务器效果结算与完整读条锁结束拆开；转换请求时刻
# 仍按统一滑步窗口恢复，日志抖动只由容量一动作队列吸收。旧缓存的效果状态时序不可复用。
# v10：硬读条请求时刻改由 `cast − 实际读条时长 + 0.5 秒滑步窗口` 解析，
# 整场动作与场景事实再统一平移到最早请求为 0；网络抖动由状态机动作队列吸收。
# v9：请求时刻改由 `cast − 实际读条时长` 解析（消除服务器提前结算偏差），
# 场景事实只注入 Boss 可选中/目标数/团辅，移动与停手标量由输出层改写，
# policy 动作改走 record_policy_action。旧缓存的时序与场景语义不可复用。
DEFAULT_CONVERSION_VERSION = "raw_json_to_compiled_v12_black_mage_candidate_set"
# `weights_only=True` 的安全 unpickler 对 protocol 2 支持最稳定；compiled
# cache 的样本数据只需要普通 mapping 和 tensor，不需要更高协议。
CACHE_PICKLE_PROTOCOL = 2
DEFAULT_CACHE_SHARD_SIZE = 512
DEFAULT_CACHE_MAX_SHARDS = 8


class CompiledShardCache:
    """跨所有 raw source 共享的有限 shard LRU。"""

    def __init__(self, max_shards: int = DEFAULT_CACHE_MAX_SHARDS):
        if max_shards < 1:
            raise ValueError("max_shards must be >= 1")
        self.max_shards = int(max_shards)
        self._items: OrderedDict[tuple[str, int], list[dict[str, object]]] = OrderedDict()

    def get(
        self,
        key: tuple[str, int],
        loader: Callable[[], list[dict[str, object]]],
    ) -> list[dict[str, object]]:
        cached = self._items.get(key)
        if cached is not None:
            self._items.move_to_end(key)
            return cached

        loaded = loader()
        self._items[key] = loaded
        self._items.move_to_end(key)
        while len(self._items) > self.max_shards:
            self._items.popitem(last=False)
        return loaded

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


class CompiledCacheReader:
    """只读取 raw JSON 对应的 manifest，并按样本访问 compiled shard。"""

    def __init__(
        self,
        payload: dict[str, object],
        *,
        cache_path: Path,
        shard_cache: CompiledShardCache,
    ):
        self._payload = payload
        self._cache_path = Path(cache_path)
        self._shard_cache = shard_cache
        self._schema = payload["schema"]
        self._job_tag = str(payload["job_tag"])
        self._fight_id = str(payload.get("fight_id", ""))
        self._num_samples = int(payload["num_samples"])
        if self._num_samples < 0:
            raise ValueError("compiled cache num_samples must be >= 0")
        self._num_candidates = int(payload["num_candidates"])
        self._skill_feature_names = tuple(payload["skill_feature_names"])
        self._candidate_action_keys = tuple(payload["candidate_action_keys"])
        self._vocab_signature = tuple(payload.get("vocab_signature", ()))
        self._shard_size = int(payload["shard_size"])
        self._history_bank_id = str(
            payload.get("history_bank_id", self._cache_path.resolve())
        )
        history_bank = payload["history_bank"]
        if not isinstance(history_bank, dict):
            raise ValueError("compiled cache history_bank must be a mapping")
        required_bank_keys = (
            "skill_ids",
            "skill_features",
            "state_vectors",
            "state_null_mask",
            "action_keys",
            "skill_potencies",
            "cumulative_dot_potencies",
        )
        if any(key not in history_bank for key in required_bank_keys):
            raise ValueError("compiled cache history_bank is missing required fields")
        action_keys = history_bank["action_keys"]
        if not isinstance(action_keys, (list, tuple)):
            raise ValueError("compiled cache history_bank action_keys must be a list or tuple")
        if not all(isinstance(action_key, str) for action_key in action_keys):
            raise ValueError("compiled cache history_bank action_keys must contain strings")
        bank_size = len(action_keys)
        max_bank_size = max(1, self._num_samples)
        if not 1 <= bank_size <= max_bank_size:
            raise ValueError(
                "compiled cache history_bank size exceeds the sample history bound: "
                f"bank_size={bank_size} max={max_bank_size} "
                f"num_samples={self._num_samples}"
            )
        torch = import_torch()
        tensor_bank_keys = (
            "skill_ids",
            "skill_features",
            "state_vectors",
            "state_null_mask",
            "skill_potencies",
            "cumulative_dot_potencies",
        )
        for key in tensor_bank_keys:
            field = history_bank[key]
            if not isinstance(field, torch.Tensor) or field.ndim < 1:
                raise ValueError(
                    f"compiled cache history_bank field must be a tensor: {key}"
                )
            if field.shape[0] != bank_size:
                raise ValueError(f"compiled cache history_bank field length mismatch: {key}")
        if bank_size < 1 or action_keys[0] != "":
            raise ValueError("compiled cache history_bank must start with an empty sentinel row")
        self._history_bank = history_bank
        self._shard_paths = tuple(
            self._cache_path.parent / str(relative_path)
            for relative_path in payload["shard_files"]
        )

    @property
    def schema(self):
        return self._schema

    @property
    def job_tag(self) -> str:
        return self._job_tag

    @property
    def fight_id(self) -> str:
        return self._fight_id

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def num_candidates(self) -> int:
        return self._num_candidates

    @property
    def shard_size(self) -> int:
        return self._shard_size

    @property
    def skill_feature_names(self) -> tuple[str, ...]:
        return self._skill_feature_names

    @property
    def vocab_signature(self) -> tuple[tuple[int, int], ...]:
        return self._vocab_signature

    @property
    def history_bank(self) -> dict[str, object]:
        """返回 source 级历史 bank；调用方只读，不得原地修改其 tensor。"""
        return self._history_bank

    @property
    def history_bank_id(self) -> str:
        """返回 source bank 的稳定身份标识，供 batch 合并去重。"""
        return self._history_bank_id

    def candidate_action_keys(self, _sample_idx: int) -> list[str]:
        return list(self._candidate_action_keys)

    def step_metadata(self, sample_idx: int) -> dict[str, object]:
        sample = self.sample(sample_idx)
        metadata = sample.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            "fight_id": self.fight_id,
            "job_tag": self.job_tag,
            "step": int(metadata.get("step", 0)),
            "source_step": int(metadata.get("source_step", metadata.get("step", 0))),
            "time_offset": float(metadata.get("time_offset", 0.0)),
        }

    def scene_tokens(self, sample_idx: int, *, float_dtype, int_dtype):
        torch = import_torch()
        sample = self.sample(sample_idx)
        scene_vectors = sample.get("scene_vectors")
        scene_types = sample.get("scene_types")
        if scene_vectors is None or scene_types is None:
            return (
                torch.zeros((0, self.schema.scene_feature_dim()), dtype=float_dtype),
                torch.zeros((0,), dtype=int_dtype),
            )
        return (
            torch.as_tensor(scene_vectors, dtype=float_dtype),
            torch.as_tensor(scene_types, dtype=int_dtype),
        )

    def sample(self, sample_idx: int) -> dict[str, object]:
        if not 0 <= sample_idx < self._num_samples:
            raise IndexError(f"compiled sample index out of range: {sample_idx}")
        shard_index, offset = divmod(sample_idx, self._shard_size)
        return self._get_shard(shard_index)[offset]

    def samples(self, sample_indices: Sequence[int]) -> list[dict[str, object]]:
        if not sample_indices:
            return []

        offsets: list[tuple[int, int]] = []
        shard_indices: set[int] = set()
        for sample_idx in sample_indices:
            if not 0 <= sample_idx < self._num_samples:
                raise IndexError(f"compiled sample index out of range: {sample_idx}")
            shard_index, offset = divmod(sample_idx, self._shard_size)
            offsets.append((shard_index, offset))
            shard_indices.add(shard_index)

        loaded_shards = {
            shard_index: self._get_shard(shard_index)
            for shard_index in shard_indices
        }
        return [loaded_shards[shard_index][offset] for shard_index, offset in offsets]

    def _get_shard(self, shard_index: int) -> list[dict[str, object]]:
        return self._shard_cache.get(
            (str(self._cache_path), shard_index),
            lambda shard_index=shard_index: self._load_shard(shard_index),
        )

    def _load_shard(self, shard_index: int) -> list[dict[str, object]]:
        shard_path = self._shard_paths[shard_index]
        payload = safe_torch_load(
            shard_path,
            mmap=True,
            safe_globals=(SceneWindowSchema, TrainingSchema),
        )
        if not isinstance(payload, dict) or payload.get("cache_format") != CACHE_FORMAT:
            raise ValueError(f"invalid compiled cache shard: {shard_path}")
        samples = payload.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"compiled cache shard samples must be a list: {shard_path}")
        return samples


def cache_path_for_source(cache_dir: Path, source_path: Path) -> Path:
    """复用数据阶段布局，生成保留副本/区间且不冲突的 manifest 路径。"""
    output_path = map_dataset_output_path(source_path, output_root=cache_dir)
    return output_path.with_name(_cache_filename_for_source(source_path))


def _cache_filename_for_source(source_path: Path) -> str:
    """保留现有缓存文件身份，目录布局不参与编译签名。"""
    source_key = str(source_path.resolve()).lower().encode("utf-8")
    digest = hashlib.sha1(source_key).hexdigest()[:12]
    return f"{source_path.stem}.{digest}.compiled.pt"


def load_compiled_cache_for_source(
    cache_dir: Path,
    source_path: Path,
    *,
    signature: dict[str, object],
    shard_cache: CompiledShardCache,
) -> CompiledCacheReader | None:
    """统一定位新布局缓存，签名一致时也复用旧的平铺 manifest 和分片。"""
    cache_path = cache_path_for_source(cache_dir, source_path)
    legacy_path = Path(cache_dir).resolve() / _cache_filename_for_source(source_path)
    paths = (cache_path,) if cache_path == legacy_path else (cache_path, legacy_path)
    for path in paths:
        cached = load_compiled_cache(path, source_path, signature=signature, shard_cache=shard_cache)
        if cached is not None:
            return cached
    return None


def build_cache_signature(
    source_path: Path,
    *,
    int_dtype,
    float_dtype,
    normalizer,
    shard_size: int = DEFAULT_CACHE_SHARD_SIZE,
    conversion_version: str = DEFAULT_CONVERSION_VERSION,
) -> dict[str, object]:
    stat = Path(source_path).stat()
    normalizer_signature = getattr(normalizer, "cache_signature", None)
    if normalizer_signature is None:
        normalizer_config = getattr(normalizer, "_config", None)
        normalizer_signature = normalizer_config
    return {
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "int_dtype": str(int_dtype),
        "float_dtype": str(float_dtype),
        "normalizer": normalizer_signature,
        "shard_size": int(shard_size),
        "source_format": "raw_json",
        "conversion_version": str(conversion_version),
    }


def load_compiled_cache(
    cache_path: Path,
    source_path: Path,
    *,
    signature: dict[str, object],
    shard_cache: CompiledShardCache,
) -> CompiledCacheReader | None:
    """读取仍对应当前 raw JSON 和编译参数的 manifest。"""
    del source_path
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        return None

    try:
        payload = safe_torch_load(
            cache_path,
            mmap=True,
            safe_globals=(SceneWindowSchema, TrainingSchema),
        )
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("cache_format") != CACHE_FORMAT:
        return None
    if payload.get("cache_signature") != signature:
        return None
    shard_files = payload.get("shard_files")
    if not isinstance(shard_files, list):
        return None
    if any(not (cache_path.parent / str(path)).is_file() for path in shard_files):
        return None
    try:
        return CompiledCacheReader(
            payload,
            cache_path=cache_path,
            shard_cache=shard_cache,
        )
    # 仅把 manifest 结构解析错误视为缓存损坏；MemoryError 等运行时错误必须继续抛出。
    except (AttributeError, IndexError, KeyError, OverflowError, TypeError, ValueError):
        return None

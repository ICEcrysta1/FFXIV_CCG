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
from common.output_context_schema import CANONICAL_CONTEXT_SCHEMA_VERSION

from .schema import SceneWindowSchema, TrainingSchema, TRAINING_SAMPLE_SCHEMA_VERSION
from .action_space import ActionSpace

# v15：样本只保存配置无关的动作质量等级代码 1/2/3；旧权重缓存必须重编译。
# v14：样本新增排名区间、标注状态和配置映射后的数值等级权重；旧缓存必须重编译。
# v13：样本增加动作质量标签和原始事件索引；旧缓存没有该监督信息，必须重编译。
# v12：黑魔候选集合移除 retrace、manaward、surecast，compiled cache 的候选布局
# 不再与旧缓存兼容；同时保留 v11 的提前效果结算语义。
# v16：技能和状态 bank 移除累计 GCD 索引、精确战斗剩余时间及冗余 GCD 窗口。
# v17：状态 bank 移除三个调度窗口秒数字段以及全部 GCD 单位时间字段。
# v18：状态 bank 只保留资源 before/after，移除 weave 字段与黑魔残留辅助 Buff。
# v19：固定动作词表和当前请求状态；不保存任何候选输入，旧缓存必须重编译。
# v20：模型历史保存请求时冻结的跨步状态，真实执行统计与状态向量分离。
# v21：技能字段与完整 history bank 移除绝对时间列，状态时间仍保留。
CACHE_FORMAT = "raw_json_compiled_samples_v21_timeless_skills"
# v11：C# 状态机把硬读条的服务器效果结算与完整读条锁结束拆开；转换请求时刻
# 仍按统一滑步窗口恢复，日志抖动只由容量一动作队列吸收。旧缓存的效果状态时序不可复用。
# v10：硬读条请求时刻改由 `cast − 实际读条时长 + 0.5 秒滑步窗口` 解析，
# 整场动作与场景事实再统一平移到最早请求为 0；网络抖动由状态机动作队列吸收。
# v9：请求时刻改由 `cast − 实际读条时长` 解析（消除服务器提前结算偏差），
# 场景事实只注入 Boss 可选中/目标数/团辅，移动与停手标量由输出层改写，
# policy 动作改走 record_policy_action。旧缓存的时序与场景语义不可复用。
# 调整动作质量标签的准入语义时，手动提升此版本以重编译旧缓存；
# 模型等级权重只在损失计算时映射，修改权重或注释无需重编译。
# v16：移除编译时模型等级权重依赖，输出稳定等级代码；旧结果不能复用。
# v15：数值权重和来源档位进入 compiled 样本；旧 v14 转换结果不能复用。
# v14：真实技能样本补齐 step/source_step；旧 v13 缓存中的零步号不能复用。
# v17：按新的模型 token 契约重建完整 history bank；读取窗口仍不影响缓存身份。
# v18：按纯秒制时间输入重建完整 history bank，调度窗口不进入模型。
# v19：按精简后的状态 token 契约重建完整 history bank，技能侧继续保留资源消耗。
# v20：当前请求状态和只读合法性/value 直接生成，不进行未来动作预演。
# v21：状态使用上一动作后与当前请求快照；wait 不预演未来，场景按各段自身时间查询。
# v22：真实技能与等待按统一历史写入序号合并，并拒绝不稳定的历史前缀。
# v23：所有技能 token 移除绝对时间，按精简后的字段重建完整 history bank。
DEFAULT_CONVERSION_VERSION = "raw_json_to_compiled_v23_timeless_skills"
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
        from threading import RLock
        self._lock = RLock()

    def get(
        self,
        key: tuple[str, int],
        loader: Callable[[], list[dict[str, object]]],
    ) -> list[dict[str, object]]:
        with self._lock:
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
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __getstate__(self):
        # Windows DataLoader spawn 不传递线程锁，子进程重建自己的锁。
        with self._lock:
            return {"max_shards": self.max_shards, "_items": self._items.copy()}

    def __setstate__(self, state):
        from threading import RLock
        self.__dict__.update(state)
        self._lock = RLock()


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
        if payload.get("cache_format") != CACHE_FORMAT:
            raise ValueError("unsupported compiled cache format")
        self._cache_path = Path(cache_path)
        self._shard_cache = shard_cache
        self._schema = payload["schema"]
        if not isinstance(self._schema, TrainingSchema):
            raise ValueError("compiled cache schema must be a TrainingSchema")
        if (self._schema.sample_schema_version != TRAINING_SAMPLE_SCHEMA_VERSION
                or self._schema.context_schema_version != CANONICAL_CONTEXT_SCHEMA_VERSION):
            raise ValueError("unsupported compiled cache schema version; recompile raw source")
        if "time_seconds" in self._schema.skill_history_fields:
            raise ValueError("compiled cache contains removed skill time_seconds field")
        self._job_tag = str(payload["job_tag"])
        self._fight_id = str(payload.get("fight_id", ""))
        self._num_samples = int(payload["num_samples"])
        if self._num_samples < 0:
            raise ValueError("compiled cache num_samples must be >= 0")
        self._num_actions = int(payload["num_actions"])
        self._skill_feature_names = tuple(payload["skill_feature_names"])
        if "time_seconds" in self._skill_feature_names:
            raise ValueError("compiled cache contains removed skill time_seconds feature")
        self._action_keys = tuple(payload["action_keys"])
        self._action_to_vocab_id = tuple(int(value) for value in payload["action_to_vocab_id"])
        self._action_is_gcd = tuple(payload["action_is_gcd"])
        if self._num_actions < 1 or len(self._action_keys) != self._num_actions or len(set(self._action_keys)) != self._num_actions:
            raise ValueError("compiled cache action_keys must match the fixed output space")
        if len(self._action_to_vocab_id) != self._num_actions or any(value <= 0 for value in self._action_to_vocab_id) or len(set(self._action_to_vocab_id)) != self._num_actions:
            raise ValueError("compiled cache output mapping must contain distinct non-padding vocabulary rows")
        if len(self._action_is_gcd) != self._num_actions or any(type(value) is not bool for value in self._action_is_gcd):
            raise ValueError("compiled cache action_is_gcd must contain one boolean per output action")
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
        if history_bank["skill_features"].shape != (bank_size, len(self._skill_feature_names)):
            raise ValueError("compiled cache history_bank skill feature width mismatch")
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
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def action_keys(self) -> tuple[str, ...]:
        return self._action_keys

    @property
    def action_to_vocab_id(self) -> tuple[int, ...]:
        return self._action_to_vocab_id

    @property
    def action_is_gcd(self) -> tuple[bool, ...]:
        return self._action_is_gcd

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
            "percentile": metadata.get("percentile"),
            "percentile_bucket": metadata.get("percentile_bucket"),
            "quality_label_status": metadata.get("quality_label_status", "unannotated"),
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
        torch = import_torch()
        for sample in samples:
            if not isinstance(sample, dict) or tuple(sample.get("action_keys", ())) != self.action_keys:
                raise ValueError("compiled sample output action order mismatch")
            for key in ("current_state_vectors", "current_state_null_mask"):
                value = sample.get(key)
                if not isinstance(value, torch.Tensor) or value.shape != (self.schema.state_vector_dim(),):
                    raise ValueError(f"compiled sample {key} shape mismatch")
            for key in ("action_values", "action_legal_mask"):
                value = sample.get(key)
                if not isinstance(value, torch.Tensor) or value.shape != (self.num_actions,):
                    raise ValueError(f"compiled sample {key} shape mismatch")
            index = int(sample.get("label_index", -1))
            if not 0 <= index < self.num_actions or sample.get("label_action_key") != self.action_keys[index]:
                raise ValueError("compiled sample label must match its fixed output action index")
        return samples


def cache_path_for_source(cache_dir: Path, source_path: Path) -> Path:
    """复用数据阶段布局，生成保留副本/区间且不冲突的 manifest 路径。"""
    output_path = map_dataset_output_path(source_path, output_root=cache_dir)
    return output_path.with_name(_cache_filename_for_source(source_path))


def _cache_filename_for_source(source_path: Path) -> str:
    """保留现有缓存文件身份，目录布局不参与编译签名。"""
    source_key = str(source_path.resolve()).lower().encode("utf-8")
    digest = hashlib.sha1(source_key).hexdigest()[:12]
    source_name = source_path.name.removesuffix(".json.br")
    return f"{source_name}.{digest}.compiled.pt"


def load_compiled_cache_for_source(
    cache_dir: Path,
    source_path: Path,
    *,
    signature: dict[str, object],
    expected_action_space: ActionSpace,
    shard_cache: CompiledShardCache,
) -> CompiledCacheReader | None:
    """统一定位新布局缓存，签名一致时也复用旧的平铺 manifest 和分片。"""
    cache_path = cache_path_for_source(cache_dir, source_path)
    legacy_path = Path(cache_dir).resolve() / _cache_filename_for_source(source_path)
    paths = (cache_path,) if cache_path == legacy_path else (cache_path, legacy_path)
    for path in paths:
        cached = load_compiled_cache(
            path, source_path, signature=signature,
            expected_action_space=expected_action_space, shard_cache=shard_cache,
        )
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
        "source_format": "json_brotli",
        "conversion_version": str(conversion_version),
    }


def load_compiled_cache(
    cache_path: Path,
    source_path: Path,
    *,
    signature: dict[str, object],
    expected_action_space: ActionSpace,
    shard_cache: CompiledShardCache,
) -> CompiledCacheReader | None:
    """按 raw 签名和调用方指定的动作契约校验 manifest，不读取本机 YAML。"""
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
    job_tag = payload.get("job_tag")
    if not isinstance(job_tag, str) or not job_tag:
        return None
    # 新训练由调用方提供当前配置，离线恢复由调用方提供模型保存的 DataSpec。
    if (tuple(payload.get("action_keys", ())) != expected_action_space.action_keys
            or tuple(payload.get("action_to_vocab_id", ())) != expected_action_space.action_to_vocab_id
            or tuple(payload.get("action_is_gcd", ())) != expected_action_space.action_is_gcd):
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

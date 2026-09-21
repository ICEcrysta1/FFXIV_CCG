"""自回归回放共用的 PyTorch / ONNX Runtime policy backend。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import torch

from common.torch_runtime import autocast_context, model_dtype
from common.torch_serialization import safe_torch_load
from scripts.onnx_export import DeploymentManifest, TENSOR_INPUT_NAMES
from scripts.onnx_export.runtime.ort_runtime import (
    ORT_PROVIDER_CUDA,
    create_ort_session,
)
from scripts.onnx_export.runtime.precision import (
    PRECISION_BF16,
    onnx_torch_dtype,
)
from scripts.onnx_export.release.release import RELEASE_REPORT_FILENAME, verify_release
from scripts.onnx_export.runtime.tensor_runtime import run_ort_tensors
from common.policy.data import DataSpec, ModelInputContract, SkillVocab
from common.policy.model import (
    CandidateTransformerModel,
    RepetitionConfig,
    repetition_config_from_checkpoint,
    parse_repetition_config,
)


@dataclass(frozen=True)
class BackendMetrics:
    """单个 backend 在当前进程内累计的推理性能数据。"""

    calls: int
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    latency_ms_max: float
    process_peak_working_set_bytes: int | None
    cuda_peak_allocated_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PolicyBackend(Protocol):
    """状态机回放唯一依赖的模型推理边界。"""

    name: str
    source_path: Path
    input_device: torch.device
    data_spec: DataSpec
    input_contract: ModelInputContract
    repetition: RepetitionConfig
    vocab_entries: tuple[tuple[int, int], ...]
    execution_provider: str

    def raw_logits(
        self,
        batch: Mapping[str, object],
        candidate_action_keys: Sequence[str],
    ) -> torch.Tensor:
        """返回 `[batch, candidate]`、未应用宿主策略的 raw logits。"""

    def configure_cache(self, enabled: bool) -> None:
        """切换 backend 自身支持的推理 cache；不支持时必须安全关闭。"""

    def metrics(self) -> BackendMetrics:
        """返回当前累计性能指标。"""


class _MeasuredBackend:
    """集中维护延迟和进程峰值，避免两个 backend 各写一套统计。"""

    def __init__(self) -> None:
        self._latencies_ms: list[float] = []
        self._process_peak_bytes = _process_peak_working_set_bytes()
        self._cuda_peak_bytes: int | None = None

    def _start_measurement(self) -> float:
        self._synchronize()
        return perf_counter()

    def _finish_measurement(self, started_at: float) -> None:
        self._synchronize()
        self._latencies_ms.append((perf_counter() - started_at) * 1000.0)
        current_peak = _process_peak_working_set_bytes()
        if current_peak is not None:
            self._process_peak_bytes = max(self._process_peak_bytes or 0, current_peak)

    def _synchronize(self) -> None:
        return None

    def metrics(self) -> BackendMetrics:
        values = sorted(self._latencies_ms)
        return BackendMetrics(
            calls=len(values),
            latency_ms_p50=_percentile(values, 0.50),
            latency_ms_p95=_percentile(values, 0.95),
            latency_ms_p99=_percentile(values, 0.99),
            latency_ms_max=values[-1] if values else 0.0,
            process_peak_working_set_bytes=self._process_peak_bytes,
            cuda_peak_allocated_bytes=self._cuda_peak_bytes,
        )


class PyTorchPolicyBackend(_MeasuredBackend):
    """加载原始 checkpoint，并返回应用重复惩罚前的 PyTorch logits。"""

    name = "pytorch"

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str,
        use_kv_cache: bool,
        precision: str | None = None,
    ):
        super().__init__()
        self.source_path = Path(checkpoint_path).resolve()
        self.input_device = _resolve_torch_device(device)
        self.execution_provider = f"PyTorch:{self.input_device.type}"
        checkpoint = safe_torch_load(self.source_path)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("autoregressive checkpoint must be a mapping")
        self.checkpoint = checkpoint
        self.data_spec = DataSpec.from_dict(dict(checkpoint["data_spec"]))
        self.input_contract = ModelInputContract.from_checkpoint(checkpoint)
        self.input_contract.assert_matches_data_spec(self.data_spec)
        self.repetition = repetition_config_from_checkpoint(checkpoint)
        vocab = SkillVocab.build_from_job_tag(self.data_spec.job_tag)
        self.vocab_entries = tuple(vocab)
        embedding = checkpoint["model_state_dict"].get(
            "input_encoder.skill_embed.weight"
        )
        if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
            raise ValueError("checkpoint missing skill embedding weight")
        vocab_size = int(embedding.shape[0])
        checkpoint_precision = str(checkpoint.get("training_precision", "float32"))
        requested_precision = precision or checkpoint_precision
        if requested_precision in {"bf16", "float16"} and self.input_device.type != "cuda":
            raise ValueError(
                f"PyTorch replay precision {requested_precision} requires CUDA; "
                "the deployment reference is not silently promoted to float32"
            )
        if requested_precision == PRECISION_BF16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("current CUDA device does not support native BF16")
        self.precision = requested_precision
        # 重复惩罚统一移到宿主；这里故意使用默认关闭的 repetition。
        self.model = CandidateTransformerModel(
            self.data_spec,
            CandidateTransformerModel.checkpoint_model_config(dict(checkpoint)),
            vocab_size=vocab_size,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(
            device=self.input_device,
            dtype=model_dtype(self.precision),
        )
        self.model.eval()
        self.configure_cache(use_kv_cache)

    def raw_logits(
        self,
        batch: Mapping[str, object],
        candidate_action_keys: Sequence[str],
    ) -> torch.Tensor:
        _validate_candidate_order(self.data_spec, candidate_action_keys)
        started_at = self._start_measurement()
        try:
            with torch.no_grad(), autocast_context(self.input_device, self.precision):
                output = self.model(dict(batch))
                return output["logits"].float()
        finally:
            self._finish_measurement(started_at)

    def configure_cache(self, enabled: bool) -> None:
        self.model.enable_kv_cache(enabled)

    def _synchronize(self) -> None:
        if self.input_device.type == "cuda":
            torch.cuda.synchronize(self.input_device)
            self._cuda_peak_bytes = int(torch.cuda.max_memory_allocated(self.input_device))


class OrtPolicyBackend(_MeasuredBackend):
    """严格校验部署 manifest 后创建 ORT session 的固定容量 backend。"""

    name = "onnxruntime"

    def __init__(self, package_path: Path, *, provider: str):
        super().__init__()
        package_path = Path(package_path).resolve()
        self.package_dir = package_path if package_path.is_dir() else package_path.parent
        manifest_path = self.package_dir / "manifest.json"
        self.manifest = DeploymentManifest.load(manifest_path, verify_files=True)
        self.contract = self.manifest.contract
        release_path = self.package_dir / RELEASE_REPORT_FILENAME
        self.release = (
            verify_release(self.package_dir, require_validated=False)
            if release_path.is_file()
            else None
        )
        self.release_status = (
            "graph_validated" if self.release is None else str(self.release["status"])
        )
        if self.contract.precision == PRECISION_BF16 and provider != ORT_PROVIDER_CUDA:
            raise ValueError(
                "BF16 ONNX replay requires explicit CUDAExecutionProvider; "
                "auto/CPU fallback is disabled for the BF16 deployment contract"
            )
        self.data_spec = self.contract.data_spec
        self.input_contract = self.contract.input_contract
        self.repetition = parse_repetition_config(self.contract.repetition_config)
        self.vocab_entries = self.contract.vocab_entries
        self.input_device = torch.device("cpu")
        model_payload = self.manifest.payload["model"]
        model_path = self.package_dir / str(model_payload["filename"])
        if package_path.is_file() and package_path != model_path:
            raise ValueError(
                f"configured ONNX model differs from manifest: {package_path} != {model_path}"
            )
        self.source_path = model_path
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ONNX Runtime replay requires requirements-onnx.txt"
            ) from exc
        # manifest 和配套文件全部校验完成后才允许创建 session。
        self.session, _requested_providers, self.providers = create_ort_session(
            ort,
            model_path,
            provider,
        )
        self.provider = self.providers[0]
        self.execution_provider = self.provider
        self._validate_runtime_contract()

    def raw_logits(
        self,
        batch: Mapping[str, object],
        candidate_action_keys: Sequence[str],
    ) -> torch.Tensor:
        legal_mask = _require_tensor(batch, "candidate_legal_mask").detach().cpu().bool()
        self.contract.validate_host_candidate_order(candidate_action_keys, legal_mask)
        inputs = build_fixed_ort_inputs(batch, self.contract)
        started_at = self._start_measurement()
        try:
            output = run_ort_tensors(
                self.session,
                dict(zip(TENSOR_INPUT_NAMES, inputs, strict=True)),
                ("raw_logits",),
            )[0]
        finally:
            self._finish_measurement(started_at)
        if not torch.isfinite(output).all():
            raise RuntimeError("ORT raw_logits contains NaN/Inf")
        return output.float().cpu()

    def configure_cache(self, enabled: bool) -> None:
        if enabled:
            raise ValueError("ONNX Runtime replay v1 does not support KV cache")

    def _validate_runtime_contract(self) -> None:
        runtime_inputs = self.session.get_inputs()
        runtime_outputs = self.session.get_outputs()
        expected_inputs = self.contract.tensor_inputs()
        expected_outputs = self.contract.tensor_outputs()
        actual_inputs = [
            (item.name, item.type, tuple(item.shape)) for item in runtime_inputs
        ]
        actual_outputs = [
            (item.name, item.type, tuple(item.shape)) for item in runtime_outputs
        ]
        if actual_inputs != [
            (item.name, item.dtype, item.shape) for item in expected_inputs
        ]:
            raise ValueError("ORT session inputs differ from deployment manifest")
        if actual_outputs != [
            (item.name, item.dtype, item.shape) for item in expected_outputs
        ]:
            raise ValueError("ORT session outputs differ from deployment manifest")

    def _synchronize(self) -> None:
        if self.provider == ORT_PROVIDER_CUDA:
            torch.cuda.synchronize()


class ParityPolicyBackend:
    """同一决策点同时执行 PT/ORT；只有完全过门槛才返回参考 logits。"""

    name = "pytorch+onnxruntime-parity"

    def __init__(
        self,
        reference: PolicyBackend,
        candidate: PolicyBackend,
        *,
        tolerance: float = 1e-4,
    ):
        if reference.data_spec != candidate.data_spec:
            raise ValueError("parity backends have different DataSpec")
        if reference.input_contract.to_dict() != candidate.input_contract.to_dict():
            raise ValueError("parity backends have different ModelInputContract")
        if reference.repetition != candidate.repetition:
            raise ValueError("parity backends have different repetition policy")
        if reference.vocab_entries != candidate.vocab_entries:
            raise ValueError("parity backends have different SkillVocab")
        self.reference = reference
        self.candidate = candidate
        self.tolerance = float(tolerance)
        self.source_path = candidate.source_path
        self.input_device = reference.input_device
        self.data_spec = reference.data_spec
        self.input_contract = reference.input_contract
        self.repetition = reference.repetition
        self.vocab_entries = reference.vocab_entries
        self.execution_provider = (
            f"{reference.execution_provider}+{candidate.execution_provider}"
        )
        self.rows: list[dict[str, object]] = []

    def raw_logits(
        self,
        batch: Mapping[str, object],
        candidate_action_keys: Sequence[str],
    ) -> torch.Tensor:
        # 完全因果布局下 position id 按序列长度生成，动态 PT 与固定容量
        # ORT 不再逐 token 等价；parity 统一使用同一份固定容量输入双跑，
        # 并放在 reference 的 device 上（ORT 侧会自行搬回 CPU）。
        fixed_batch = batch
        contract = getattr(self.candidate, "contract", None)
        if getattr(contract, "capacity", None) is not None:
            fixed_batch = _to_fixed_capacity_batch(
                batch,
                contract,
                device=self.reference.input_device,
            )
        reference_logits = self.reference.raw_logits(fixed_batch, candidate_action_keys)
        candidate_logits = self.candidate.raw_logits(fixed_batch, candidate_action_keys)
        row = _compare_logits_values(
            reference_logits,
            candidate_logits,
            candidate_action_keys,
        )
        legal_mask = (
            _require_tensor(fixed_batch, "candidate_legal_mask")
            .detach()
            .cpu()
            .bool()
        )
        reference_final = _apply_host_policy(
            reference_logits.detach().cpu(),
            batch,
            self.repetition,
            legal_mask,
        )
        candidate_final = _apply_host_policy(
            candidate_logits.detach().cpu(),
            batch,
            self.repetition,
            legal_mask,
        )
        reference_selection = int(reference_final.argmax(dim=-1).item())
        candidate_selection = int(candidate_final.argmax(dim=-1).item())
        row.update(
            {
                "decision_index": len(self.rows),
                "reference_final_action": str(
                    candidate_action_keys[reference_selection]
                ),
                "candidate_final_action": str(
                    candidate_action_keys[candidate_selection]
                ),
                "final_selection_match": reference_selection == candidate_selection,
            }
        )
        row["passed"] = bool(
            row["max_abs_diff"] <= self.tolerance
            and row["top1_match"]
            and row["top3_set_match"]
            and row["final_selection_match"]
        )
        self.rows.append(row)
        return reference_logits

    def configure_cache(self, enabled: bool) -> None:
        if enabled:
            raise ValueError("parity replay requires KV cache to be disabled")
        self.reference.configure_cache(False)
        self.candidate.configure_cache(False)

    def metrics(self) -> BackendMetrics:
        return self.reference.metrics()

    def report(self) -> dict[str, object]:
        rows = self.rows
        first_divergence = next(
            (row for row in rows if not row["passed"]),
            None,
        )
        return {
            "passed": bool(rows) and first_divergence is None,
            "decision_count": len(rows),
            "max_abs_diff": max(
                (float(row["max_abs_diff"]) for row in rows),
                default=0.0,
            ),
            "top1_match_rate": _match_rate(rows, "top1_match"),
            "top3_set_match_rate": _match_rate(rows, "top3_set_match"),
            "final_selection_match_rate": _match_rate(
                rows,
                "final_selection_match",
            ),
            "first_divergence": first_divergence,
            "decisions": list(rows),
            "reference_backend": self.reference.name,
            "candidate_backend": self.candidate.name,
            "reference_execution_provider": self.reference.execution_provider,
            "candidate_execution_provider": self.candidate.execution_provider,
            "reference_metrics": self.reference.metrics().to_dict(),
            "candidate_metrics": self.candidate.metrics().to_dict(),
        }


def build_fixed_ort_inputs(
    batch: Mapping[str, object],
    contract,
    *,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, ...]:
    """把同一个 LiveBatchBuilder 的动态 batch 右补位到 manifest 固定容量。

    默认在 CPU 上构造（ORT session 输入）；PT reference 在 CUDA 时可由
    调用方指定 device，直接构造在目标设备上，避免先搬 CPU 再搬回 GPU。
    """
    scene_length = int(_require_tensor(batch, "scene_vectors").shape[1])
    history_length = int(_require_tensor(batch, "history_skill_ids").shape[1])
    if scene_length > contract.capacity.scene_capacity:
        raise ValueError(
            "live scene exceeds ONNX capacity: "
            f"{scene_length} > {contract.capacity.scene_capacity}"
        )
    if history_length > contract.capacity.history_capacity:
        raise ValueError(
            "live history exceeds ONNX capacity: "
            f"{history_length} > {contract.capacity.history_capacity}"
        )

    values: list[torch.Tensor] = []
    for spec in contract.tensor_inputs():
        source = _require_tensor(batch, spec.name).detach().to(device)
        target = torch.zeros(spec.shape, dtype=onnx_torch_dtype(spec.dtype), device=device)
        source = source.to(dtype=target.dtype)
        if spec.name.startswith("scene_"):
            target[:, :scene_length] = source
        elif spec.name.startswith("history_"):
            target[:, :history_length] = source
        else:
            if tuple(source.shape) != spec.shape:
                raise ValueError(
                    f"live {spec.name} shape mismatch: {tuple(source.shape)} != {spec.shape}"
                )
            target.copy_(source)
        values.append(target)
    result = tuple(values)
    contract.validate_tensor_inputs(result)
    return result


def _to_fixed_capacity_batch(
    batch: Mapping[str, object],
    contract,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """把动态 batch 补位成 manifest 固定容量 batch，供 PT/ORT 同容量对比。

    固定容量张量直接构造在 reference 的 device 上：PT reference 直接消费；
    ORT 侧 ``build_fixed_ort_inputs`` 会自行搬回 CPU，传 CUDA 张量安全。
    """
    fixed_inputs = build_fixed_ort_inputs(batch, contract, device=device)
    fixed_batch = dict(zip(TENSOR_INPUT_NAMES, fixed_inputs, strict=True))
    fixed_batch["candidate_legal_mask"] = (
        _require_tensor(batch, "candidate_legal_mask").detach().cpu().bool()
    )
    for key in ("history_action_keys", "candidate_action_keys"):
        if key in batch:
            fixed_batch[key] = batch[key]
    return fixed_batch


def compare_backend_logits(
    reference: PolicyBackend,
    candidate: PolicyBackend,
    batch: Mapping[str, object],
    candidate_action_keys: Sequence[str],
    *,
    tolerance: float = 1e-4,
) -> dict[str, object]:
    """比较同一决策点，并在失败时指出首个候选及两个 logit。

    完全因果布局下动态 PT 与固定容量 ORT 的 position id 不同，
    候选后端持有部署契约时统一在固定容量输入上双跑。
    """
    comparison_batch = batch
    contract = getattr(candidate, "contract", None)
    if getattr(contract, "capacity", None) is not None:
        comparison_batch = _to_fixed_capacity_batch(
            batch,
            contract,
            device=reference.input_device,
        )
    result = _compare_logits_values(
        reference.raw_logits(comparison_batch, candidate_action_keys),
        candidate.raw_logits(comparison_batch, candidate_action_keys),
        candidate_action_keys,
    )
    max_difference = float(result["max_abs_diff"])
    if max_difference > tolerance or not result["top1_match"] or not result["top3_set_match"]:
        raise AssertionError(
            "policy backend parity failed: "
            f"candidate={result['max_diff_candidate']!r}, "
            f"reference_logit={result['reference_logit']:.8f}, "
            f"candidate_logit={result['candidate_logit']:.8f}, "
            f"max_abs_diff={result['max_abs_diff']:.8f}, "
            f"reference_top1={result['reference_top1']!r}, "
            f"candidate_top1={result['candidate_top1']!r}, "
            f"top3_set_match={result['top3_set_match']}"
        )
    return result


def _compare_logits_values(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    candidate_action_keys: Sequence[str],
) -> dict[str, object]:
    reference_row = reference_logits[0].detach().cpu().float()
    candidate_row = candidate_logits[0].detach().cpu().float()
    if reference_row.shape != candidate_row.shape:
        raise AssertionError(
            "policy backend logit shape mismatch: "
            f"{tuple(reference_row.shape)} != {tuple(candidate_row.shape)}"
        )
    differences = torch.abs(reference_row - candidate_row)
    max_difference, max_index_tensor = differences.max(dim=0)
    max_index = int(max_index_tensor.item())
    reference_top1 = int(reference_row.argmax().item())
    candidate_top1 = int(candidate_row.argmax().item())
    top_k = min(3, len(candidate_action_keys))
    reference_top3_indices = reference_row.topk(top_k).indices.tolist()
    candidate_top3_indices = candidate_row.topk(top_k).indices.tolist()
    reference_top3 = frozenset(reference_top3_indices)
    candidate_top3 = frozenset(candidate_top3_indices)
    return {
        "max_abs_diff": float(max_difference.item()),
        "max_diff_index": max_index,
        "max_diff_candidate": str(candidate_action_keys[max_index]),
        "reference_logit": float(reference_row[max_index].item()),
        "candidate_logit": float(candidate_row[max_index].item()),
        "top1_match": reference_top1 == candidate_top1,
        "top3_set_match": reference_top3 == candidate_top3,
        "reference_top1": str(candidate_action_keys[reference_top1]),
        "candidate_top1": str(candidate_action_keys[candidate_top1]),
        "reference_top3": [
            str(candidate_action_keys[index]) for index in reference_top3_indices
        ],
        "candidate_top3": [
            str(candidate_action_keys[index]) for index in candidate_top3_indices
        ],
    }


def _apply_host_policy(logits, batch, repetition, legal_mask):
    from common.policy.model.repetition import apply_repetition_penalty

    penalized = apply_repetition_penalty(logits, dict(batch), repetition)
    return penalized.masked_fill(~legal_mask, float("-inf"))


def _match_rate(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 1.0
    return sum(bool(row[key]) for row in rows) / len(rows)


def validate_backend_vocab(backend: PolicyBackend, vocab: SkillVocab) -> None:
    actual = tuple(vocab)
    if actual != backend.vocab_entries:
        raise ValueError(
            "state-machine SkillVocab differs from policy deployment contract"
        )


def _validate_candidate_order(data_spec: DataSpec, values: Sequence[str]) -> None:
    if tuple(str(value) for value in values) != data_spec.candidate_action_keys:
        raise ValueError("live candidate order differs from policy DataSpec")


def _require_tensor(batch: Mapping[str, object], name: str) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"live batch missing tensor {name}")
    return value


def _resolve_torch_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "autoregressive replay requested CUDA, but CUDA is unavailable"
            )
        return torch.device("cuda")
    if name != "cpu":
        raise ValueError(f"unsupported PyTorch replay device: {name!r}")
    return torch.device("cpu")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _process_peak_working_set_bytes() -> int | None:
    """读取进程生命周期峰值；不引入 psutil 运行时依赖。"""
    try:
        import os

        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_current_process.restype = wintypes.HANDLE
            get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            )
            get_process_memory_info.restype = wintypes.BOOL
            process = get_current_process()
            if not get_process_memory_info(
                process,
                ctypes.byref(counters),
                counters.cb,
            ):
                return None
            return int(counters.PeakWorkingSetSize)

        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if os.uname().sysname == "Darwin" else peak * 1024
    except (AttributeError, ImportError, OSError, ValueError):
        return None

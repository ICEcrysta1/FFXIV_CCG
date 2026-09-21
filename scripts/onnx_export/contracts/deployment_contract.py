"""从 checkpoint 权威对象派生的完整 ONNX 部署契约。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import torch

from common.policy.data import DataSpec, ModelInputContract
from common.policy.model.repetition import parse_repetition_config

from ..io.artifact_io import file_sha256
from .contract import CapacityContract, OUTPUT_NAMES, TENSOR_INPUT_NAMES
from .deployment_profile import stable_sha256
from ..runtime.precision import (
    SUPPORTED_PRECISIONS,
    onnx_torch_dtype,
    precision_onnx_dtype,
)
from ..runtime.tensor_runtime import GOLDEN_FORMAT, golden_encoding


# 候选合法性与 Sidecar 执行语义变化后，旧导出包不得继续被部署侧接受。
DEPLOYMENT_CONTRACT_VERSION = 9
DEPLOYMENT_MANIFEST_VERSION = 7
MANIFEST_SCHEMA_FILENAME = "manifest.schema.json"


@dataclass(frozen=True)
class TensorSpec:
    """单个部署张量的名称、dtype、固定 shape 和语义。"""

    name: str
    dtype: str
    shape: tuple[int, ...]
    semantic: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "semantic": self.semantic,
        }


@dataclass(frozen=True)
class DeploymentContract:
    """状态机宿主与 ONNX 图之间不可变、可验签的完整语义契约。"""

    precision: str
    capacity: CapacityContract
    data_spec: DataSpec
    input_contract: ModelInputContract
    model_config: dict[str, object]
    repetition_config: dict[str, object]
    vocab_entries: tuple[tuple[int, int], ...]
    capacity_report_sha256: str
    capacity_evidence: dict[str, object]

    @classmethod
    def create(
        cls,
        *,
        precision: str,
        capacity: CapacityContract,
        data_spec: DataSpec,
        input_contract: ModelInputContract,
        model_config: Mapping[str, object],
        repetition_config: Mapping[str, object],
        vocab_entries: Sequence[tuple[int, int]],
        capacity_report: Mapping[str, object],
        embedding_vocab_size: int,
    ) -> "DeploymentContract":
        normalized_vocab = tuple(
            (int(raw_skill_id), int(vocab_id))
            for raw_skill_id, vocab_id in vocab_entries
        )
        capacity_report_sha256 = str(capacity_report.get("semantic_sha256", ""))
        capacity_evidence = {
            "scene_capacity": int(capacity_report["scene_capacity"]),
            "history_capacity": int(capacity_report["history_capacity"]),
            "evidence": dict(capacity_report["evidence"]),
        }
        contract = cls(
            precision=str(precision),
            capacity=capacity,
            data_spec=data_spec,
            input_contract=input_contract,
            model_config=dict(model_config),
            repetition_config=_json_value(
                asdict(parse_repetition_config(repetition_config))
            ),
            vocab_entries=normalized_vocab,
            capacity_report_sha256=capacity_report_sha256,
            capacity_evidence=capacity_evidence,
        )
        contract.validate(embedding_vocab_size=embedding_vocab_size)
        return contract

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DeploymentContract":
        _require_exact_keys(
            payload,
            {
                "contract_version",
                "job_tag",
                "precision",
                "capacity",
                "data_spec",
                "model_input_contract",
                "model_config",
                "vocab",
                "state_layout",
                "scene_layout",
                "tensor_inputs",
                "tensor_outputs",
                "host_postprocessing",
                "capacity_provenance",
                "signatures",
            },
            path="contract",
        )
        version = int(payload["contract_version"])
        if version != DEPLOYMENT_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported deployment contract version: "
                f"{version} != {DEPLOYMENT_CONTRACT_VERSION}"
            )
        data_spec_payload = _mapping(payload["data_spec"], "contract.data_spec")
        input_payload = _mapping(
            payload["model_input_contract"],
            "contract.model_input_contract",
        )
        capacity_payload = _mapping(payload["capacity"], "contract.capacity")
        model_config = dict(_mapping(payload["model_config"], "contract.model_config"))
        host_postprocessing = _mapping(
            payload["host_postprocessing"],
            "contract.host_postprocessing",
        )
        repetition_config = _json_value(
            asdict(
                parse_repetition_config(
                    _mapping(
                        host_postprocessing["repetition"],
                        "contract.host_postprocessing.repetition",
                    )
                )
            )
        )
        vocab_payload = _mapping(payload["vocab"], "contract.vocab")
        entries = vocab_payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError("contract.vocab.entries must be a list")
        vocab_entries = tuple(
            (
                int(_mapping(entry, "contract.vocab.entries[]")["raw_skill_id"]),
                int(_mapping(entry, "contract.vocab.entries[]")["vocab_id"]),
            )
            for entry in entries
        )
        provenance = _mapping(
            payload["capacity_provenance"],
            "contract.capacity_provenance",
        )
        evidence = _mapping(provenance["evidence"], "capacity evidence")
        data_spec = DataSpec.from_dict(dict(data_spec_payload))
        parsed_input_contract = ModelInputContract.from_dict(input_payload)
        state_layout = payload["state_layout"]
        if not isinstance(state_layout, list):
            raise ValueError("contract.state_layout must be a list")
        ordered_state_groups: dict[str, tuple[str, ...]] = {}
        for item in state_layout:
            layout = _mapping(item, "contract.state_layout[]")
            feature_keys = layout.get("feature_keys")
            if not isinstance(feature_keys, list):
                raise ValueError("contract.state_layout.feature_keys must be a list")
            ordered_state_groups[str(layout["group_key"])] = tuple(
                str(feature_key) for feature_key in feature_keys
            )
        parsed_schema = replace(
            parsed_input_contract.schema,
            state_group_feature_keys=ordered_state_groups,
        )
        # JSON 会把 DataSpec 中的 tuple 序列化为 list；部署加载后统一恢复为
        # checkpoint 内部的权威 DataSpec 表示，再执行严格相等校验。
        input_contract = ModelInputContract(
            job_tag=parsed_input_contract.job_tag,
            data_spec=asdict(data_spec),
            schema=parsed_schema,
            normalizer_contract=parsed_input_contract.normalizer_contract,
        )
        capacity = CapacityContract.from_dict(capacity_payload)
        contract = cls(
            precision=str(payload["precision"]),
            capacity=capacity,
            data_spec=data_spec,
            input_contract=input_contract,
            model_config=model_config,
            repetition_config=repetition_config,
            vocab_entries=vocab_entries,
            capacity_report_sha256=str(provenance["semantic_sha256"]),
            capacity_evidence=dict(evidence),
        )
        embedding_vocab_size = int(vocab_payload["size"])
        contract.validate(embedding_vocab_size=embedding_vocab_size)
        expected = contract.to_dict()
        if _json_value(payload) != expected:
            difference = _first_difference(_json_value(payload), expected)
            raise ValueError(
                "deployment contract fields or signatures differ from authoritative layouts: "
                f"{difference}"
            )
        return contract

    def validate(self, *, embedding_vocab_size: int) -> None:
        if self.precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                "deployment precision must be bf16, float32 or float16"
            )
        normalized_repetition = _json_value(
            asdict(parse_repetition_config(self.repetition_config))
        )
        if self.repetition_config != normalized_repetition:
            raise ValueError("deployment repetition config is not canonical")
        self.input_contract.assert_matches_data_spec(self.data_spec)
        if self.input_contract.job_tag != self.data_spec.job_tag:
            raise ValueError("deployment job_tag differs from model input contract")
        if self.capacity.candidate_count != self.data_spec.num_candidates:
            raise ValueError("deployment candidate capacity differs from checkpoint DataSpec")
        if "max_sequence_length" in self.model_config:
            raise ValueError(
                "deployment model_config contains removed max_sequence_length"
            )
        try:
            model_scene_capacity = int(self.model_config["scene_capacity"])
            model_history_capacity = int(self.model_config["history_capacity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "deployment model_config must contain scene_capacity and history_capacity"
            ) from exc
        if model_scene_capacity != self.capacity.scene_capacity:
            raise ValueError("deployment scene capacity differs from model_config")
        if model_history_capacity != self.capacity.history_capacity:
            raise ValueError("deployment history capacity differs from model_config")
        self.capacity.validate()
        if len(self.data_spec.candidate_action_keys) != self.data_spec.num_candidates:
            raise ValueError("candidate action order length differs from candidate count")
        if len(self.data_spec.skill_feature_names) != self.data_spec.skill_feature_dim:
            raise ValueError("skill feature order length differs from skill feature dimension")
        schema = self.input_contract.schema
        if schema.state_vector_dim() != self.data_spec.state_dim:
            raise ValueError("state schema dimension differs from checkpoint DataSpec")
        if schema.scene_feature_dim() != self.data_spec.scene_dim:
            raise ValueError("scene schema dimension differs from checkpoint DataSpec")
        scene_type_ids = tuple(window.scene_type_id for window in schema.scene_windows)
        if scene_type_ids != tuple(range(self.data_spec.num_scene_types)):
            raise ValueError("scene type ids must be ordered and contiguous from zero")
        if not self.capacity_report_sha256 or len(self.capacity_report_sha256) != 64:
            raise ValueError("capacity report signature is invalid")
        scene_capacity = int(self.capacity_evidence["scene_capacity"])
        history_capacity = int(self.capacity_evidence["history_capacity"])
        if self.capacity.scene_capacity != scene_capacity:
            raise ValueError(
                "scene capacity must equal the fixed deployment profile: "
                f"{self.capacity.scene_capacity} != {scene_capacity}"
            )
        if history_capacity != self.capacity.history_capacity:
            raise ValueError(
                "capacity report history differs from checkpoint full-history boundary: "
                f"{history_capacity} != {self.capacity.history_capacity}"
            )
        vocab_ids = tuple(vocab_id for _raw_skill_id, vocab_id in self.vocab_entries)
        raw_skill_ids = tuple(raw_skill_id for raw_skill_id, _vocab_id in self.vocab_entries)
        if len(set(self.vocab_entries)) != len(self.vocab_entries):
            raise ValueError("vocab entries must be unique")
        if len(set(raw_skill_ids)) != len(raw_skill_ids):
            raise ValueError("raw skill ids in deployment vocab must be unique")
        if sorted(vocab_ids) != list(range(1, embedding_vocab_size)):
            raise ValueError("vocab ids must exactly cover checkpoint embedding rows 1..N")
        if embedding_vocab_size != len(self.vocab_entries) + 1:
            raise ValueError("vocab size differs from checkpoint skill embedding")

    def to_dict(self) -> dict[str, object]:
        core = self._unsigned_dict()
        signatures = {
            "candidate_order_sha256": stable_sha256(
                list(self.data_spec.candidate_action_keys)
            ),
            "skill_feature_order_sha256": stable_sha256(
                list(self.data_spec.skill_feature_names)
            ),
            "state_schema_sha256": stable_sha256(core["state_layout"]),
            "scene_schema_sha256": stable_sha256(core["scene_layout"]),
            "vocab_sha256": stable_sha256(core["vocab"]),
            "normalizer_sha256": stable_sha256(
                core["model_input_contract"]["normalizer"]
            ),
            "model_input_contract_sha256": stable_sha256(
                core["model_input_contract"]
            ),
            "model_config_sha256": stable_sha256(core["model_config"]),
        }
        signed = {**core, "signatures": signatures}
        signatures["deployment_contract_sha256"] = stable_sha256(signed)
        return _json_value({**core, "signatures": signatures})

    def tensor_inputs(self) -> tuple[TensorSpec, ...]:
        b = self.capacity.batch_size
        s = self.capacity.scene_capacity
        h = self.capacity.history_capacity
        c = self.data_spec.num_candidates
        sd = self.data_spec.state_dim
        fd = self.data_spec.skill_feature_dim
        xd = self.data_spec.scene_dim
        float_dtype = precision_onnx_dtype(self.precision)
        specs = (
            TensorSpec("scene_vectors", float_dtype, (b, s, xd), "right-padded scene vectors"),
            TensorSpec("scene_types", "tensor(int64)", (b, s), "scene type ids"),
            TensorSpec("scene_mask", "tensor(bool)", (b, s), "true for valid scene tokens"),
            TensorSpec("history_skill_ids", "tensor(int64)", (b, h), "right-padded vocab ids"),
            TensorSpec("history_skill_features", float_dtype, (b, h, fd), "ordered skill features"),
            TensorSpec("history_state_vectors", float_dtype, (b, h, sd), "ordered state vectors"),
            TensorSpec(
                "history_state_null_mask",
                "tensor(bool)",
                (b, h, sd),
                "history state null flags; right-padded positions are true",
            ),
            TensorSpec("history_mask", "tensor(bool)", (b, h), "true for valid history tokens"),
            TensorSpec("candidate_skill_ids", "tensor(int64)", (b, c), "candidate vocab ids in canonical order"),
            TensorSpec("candidate_skill_features", float_dtype, (b, c, fd), "candidate features in canonical order"),
            TensorSpec("candidate_state_vectors", float_dtype, (b, c, sd), "candidate states in canonical order"),
            TensorSpec("candidate_state_null_mask", "tensor(bool)", (b, c, sd), "candidate state null flags"),
        )
        assert tuple(spec.name for spec in specs) == TENSOR_INPUT_NAMES
        return specs

    def tensor_outputs(self) -> tuple[TensorSpec, ...]:
        float_dtype = precision_onnx_dtype(self.precision)
        return (
            TensorSpec(
                OUTPUT_NAMES[0],
                float_dtype,
                (self.capacity.batch_size, self.data_spec.num_candidates),
                "raw logits in canonical candidate order before host masking/policy",
            ),
        )

    def validate_tensor_inputs(self, inputs: Sequence[torch.Tensor]) -> None:
        if len(inputs) != len(TENSOR_INPUT_NAMES):
            raise ValueError(f"expected {len(TENSOR_INPUT_NAMES)} tensor inputs")
        for tensor, spec in zip(inputs, self.tensor_inputs(), strict=True):
            if tuple(tensor.shape) != spec.shape:
                raise ValueError(
                    f"{spec.name} shape mismatch: {tuple(tensor.shape)} != {spec.shape}"
                )
            if tensor.dtype != onnx_torch_dtype(spec.dtype):
                raise ValueError(
                    f"{spec.name} dtype mismatch: {tensor.dtype} != {spec.dtype}"
                )
        scene_types = inputs[1]
        if bool(((scene_types < 0) | (scene_types >= self.data_spec.num_scene_types)).any()):
            raise ValueError("scene_types contains an id outside the deployment contract")
        history_ids = inputs[3]
        candidate_ids = inputs[8]
        vocab_size = len(self.vocab_entries) + 1
        if bool(((history_ids < 0) | (history_ids >= vocab_size)).any()):
            raise ValueError("history_skill_ids contains an id outside the deployment vocab")
        if bool(((candidate_ids <= 0) | (candidate_ids >= vocab_size)).any()):
            raise ValueError("candidate_skill_ids contains an invalid deployment vocab id")
        _validate_right_padding(inputs[2], "scene_mask")
        _validate_right_padding(inputs[7], "history_mask")

    def validate_host_candidate_order(
        self,
        candidate_action_keys: Sequence[str],
        candidate_legal_mask: torch.Tensor,
    ) -> None:
        if tuple(candidate_action_keys) != self.data_spec.candidate_action_keys:
            raise ValueError("host candidate order differs from raw_logits contract")
        expected_shape = (
            self.capacity.batch_size,
            self.data_spec.num_candidates,
        )
        if tuple(candidate_legal_mask.shape) != expected_shape:
            raise ValueError("candidate_legal_mask shape differs from raw_logits")
        if candidate_legal_mask.dtype != torch.bool:
            raise ValueError("candidate_legal_mask must use bool dtype")

    def _unsigned_dict(self) -> dict[str, object]:
        schema = self.input_contract.schema
        state_layout = [
            {"group_key": group_key, "feature_keys": list(feature_keys)}
            for group_key, feature_keys in schema.state_group_feature_keys.items()
        ]
        scene_layout = [
            {
                "scene_type_id": window.scene_type_id,
                "context_key": window.context_key,
                "feature_keys": list(window.feature_keys),
            }
            for window in schema.scene_windows
        ]
        vocab = {
            "size": len(self.vocab_entries) + 1,
            "padding_vocab_id": 0,
            "entries": [
                {"raw_skill_id": raw_skill_id, "vocab_id": vocab_id}
                for raw_skill_id, vocab_id in self.vocab_entries
            ],
        }
        return {
            "contract_version": DEPLOYMENT_CONTRACT_VERSION,
            "job_tag": self.data_spec.job_tag,
            "precision": self.precision,
            "capacity": self.capacity.to_dict(),
            "data_spec": asdict(self.data_spec),
            "model_input_contract": self.input_contract.to_dict(),
            "model_config": dict(self.model_config),
            "vocab": vocab,
            "state_layout": state_layout,
            "scene_layout": scene_layout,
            "tensor_inputs": [spec.to_dict() for spec in self.tensor_inputs()],
            "tensor_outputs": [spec.to_dict() for spec in self.tensor_outputs()],
            "host_postprocessing": {
                "candidate_legal_mask": {
                    "dtype": "tensor(bool)",
                    "shape": [
                        self.capacity.batch_size,
                        self.data_spec.num_candidates,
                    ],
                    "order": "data_spec.candidate_action_keys",
                },
                "raw_logits_order": "data_spec.candidate_action_keys",
                "repetition": dict(self.repetition_config),
                "repetition_policy": "host",
                "sampling_policy": "host",
            },
            "capacity_provenance": {
                "history_capacity_source": "checkpoint.model_config.history_capacity",
                "scene_capacity_source": "checkpoint.model_config.scene_capacity",
                "over_capacity": "reject",
                "report_filename": "capacity_report.json",
                "semantic_sha256": self.capacity_report_sha256,
                "evidence": dict(self.capacity_evidence),
            },
        }


@dataclass(frozen=True)
class DeploymentManifest:
    """部署包 manifest loader；负责 schema、版本、签名和文件哈希拒绝。"""

    payload: dict[str, object]
    contract: DeploymentContract

    @classmethod
    def load(cls, path: Path, *, verify_files: bool = True) -> "DeploymentManifest":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("deployment manifest must be an object")
        try:
            version = int(payload.get("manifest_version", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError("deployment manifest_version must be an integer") from exc
        if version != DEPLOYMENT_MANIFEST_VERSION:
            raise ValueError(
                "unsupported deployment manifest version; re-export the package: "
                f"{version} != {DEPLOYMENT_MANIFEST_VERSION}"
            )
        schema_path = Path(__file__).resolve().parents[1] / MANIFEST_SCHEMA_FILENAME
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        try:
            import jsonschema
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "manifest validation requires requirements-onnx.txt"
            ) from exc
        jsonschema.Draft202012Validator(schema).validate(payload)
        _require_exact_keys(
            payload,
            {
                "$schema",
                "manifest_version",
                "format",
                "opset",
                "contract",
                "checkpoint",
                "model",
                "capacity_report",
                "golden",
                "exporter",
            },
            path="manifest",
        )
        if payload["format"] != "onnx":
            raise ValueError("deployment manifest format must be onnx")
        contract = DeploymentContract.from_dict(
            _mapping(payload["contract"], "manifest.contract")
        )
        golden = _mapping(payload["golden"], "manifest.golden")
        if golden["format"] != GOLDEN_FORMAT:
            raise ValueError("deployment golden format is unsupported")
        if golden["float_encoding"] != golden_encoding(contract.precision):
            raise ValueError(
                "deployment golden encoding differs from contract precision"
            )
        manifest = cls(payload=dict(payload), contract=contract)
        if verify_files:
            manifest.verify_files(path.parent)
        return manifest

    def verify_files(self, package_dir: Path) -> None:
        package_dir = Path(package_dir)
        model = _mapping(self.payload["model"], "manifest.model")
        signatures = _mapping(
            _mapping(self.payload["contract"], "manifest.contract")["signatures"],
            "manifest.contract.signatures",
        )
        if model["contract_sha256"] != signatures["deployment_contract_sha256"]:
            raise ValueError("model deployment contract signature mismatch")
        checkpoint = _mapping(self.payload["checkpoint"], "manifest.checkpoint")
        if model["checkpoint_sha256"] != checkpoint["sha256"]:
            raise ValueError("model checkpoint signature mismatch")
        artifacts = (
            model,
            _mapping(self.payload["capacity_report"], "manifest.capacity_report"),
            _mapping(
                _mapping(self.payload["golden"], "manifest.golden")["inputs"],
                "manifest.golden.inputs",
            ),
            _mapping(
                _mapping(self.payload["golden"], "manifest.golden")["outputs"],
                "manifest.golden.outputs",
            ),
        )
        for artifact in artifacts:
            artifact_path = package_dir / str(artifact["filename"])
            if not artifact_path.is_file():
                raise FileNotFoundError(f"deployment artifact missing: {artifact_path}")
            if file_sha256(artifact_path) != artifact["sha256"]:
                raise ValueError(f"deployment artifact hash mismatch: {artifact_path}")
        for artifact in model["external_files"]:
            external = _mapping(artifact, "manifest.model.external_files[]")
            external_path = package_dir / str(external["filename"])
            if not external_path.is_file():
                raise FileNotFoundError(f"ONNX external data missing: {external_path}")
            if file_sha256(external_path) != external["sha256"]:
                raise ValueError(f"ONNX external data hash mismatch: {external_path}")
        exporter = _mapping(self.payload["exporter"], "manifest.exporter")
        schema_path = package_dir / MANIFEST_SCHEMA_FILENAME
        if file_sha256(schema_path) != exporter["manifest_schema_sha256"]:
            raise ValueError("packaged manifest schema hash mismatch")
        report_payload = json.loads(
            (package_dir / "capacity_report.json").read_text(encoding="utf-8")
        )
        semantic = dict(report_payload)
        claimed = semantic.pop("semantic_sha256", None)
        if claimed != stable_sha256(semantic):
            raise ValueError("capacity report semantic signature mismatch")
        if claimed != self.contract.capacity_report_sha256:
            raise ValueError("capacity report differs from deployment contract")


def _validate_right_padding(mask: torch.Tensor, name: str) -> None:
    if mask.shape[1] <= 1:
        return
    if bool((~mask[:, :-1] & mask[:, 1:]).any()):
        raise ValueError(f"{name} must use contiguous right padding")


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _require_exact_keys(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    path: str,
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{path} keys mismatch: missing={missing}, extra={extra}")


def _json_value(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _first_difference(actual: object, expected: object, path: str = "contract") -> str:
    if type(actual) is not type(expected):
        return f"{path} type {type(actual).__name__} != {type(expected).__name__}"
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            return f"{path} keys differ"
        for key in actual:
            if actual[key] != expected[key]:
                return _first_difference(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            return f"{path} length {len(actual)} != {len(expected)}"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            if actual_item != expected_item:
                return _first_difference(
                    actual_item,
                    expected_item,
                    f"{path}[{index}]",
                )
    elif actual != expected:
        return f"{path} value {actual!r} != {expected!r}"
    return f"{path} differs"

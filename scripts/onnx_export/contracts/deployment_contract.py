"""从 checkpoint 权威对象派生的完整 ONNX 部署契约。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import torch

from common.policy.config import ModelConfig
from common.policy.data import DataSpec, ModelInputContract, SkillVocab
from common.policy.data.input_contract import residual_composition_contract
from common.policy.data.context_fields import MODEL_INPUT_FIELDS
from common.policy.model.repetition import parse_repetition_config

from ..io.artifact_io import file_sha256
from .contract import CapacityContract, OUTPUT_NAMES, TENSOR_INPUT_NAMES
from .deployment_profile import stable_sha256
from ..runtime.precision import (
    SUPPORTED_PRECISIONS,
    onnx_torch_dtype,
    precision_onnx_dtype,
    validate_deployment_logit_softcap,
)
from ..runtime.tensor_runtime import GOLDEN_FORMAT, golden_encoding


# 输入与状态机执行语义变化后，旧导出包不得继续被部署侧接受。
# 10：技能和状态输入收缩为新的秒制窗口契约，禁止复用旧部署包。
# 11：移除模型调度窗口和 GCD 单位时间字段，状态维度与归一化契约变化。
# 12：状态输入移除资源 consumed、weave 字段及黑魔残留辅助 Buff，拒绝旧维度。
# 13：移除 CLS token 和对应评分器输入，部署图的内部 token 布局变化。
# 14：移除候选输入，单因果序列与共享技能词表输出；固定动作类型随契约保存。
# 15：技能与状态独立为 d_model token，删除融合/输出适配，容量按每动作两个 token 计算。
# 16：状态改为请求时冻结的跨步快照，同宽旧状态语义与旧部署包不兼容。
# 17：技能数值输入删除绝对时间列，旧技能宽度与对应部署包不兼容。
# 18：历史 token 改为状态、技能顺序，拒绝同宽但因果语义不同的旧部署包。
# 19：删除独立内容 LayerNorm，内容与角色相加后统一无参数 RMS 归一化。
# 20：主干两处子层归一化及最终归一化统一为无参数 RMSNorm，拒绝旧 LayerNorm 图。
# 21：普通残差路径改为每层可学习 r/a 与初始输入混合，旧普通残差图不兼容。
# 22：RoPE 后逐 head 归一化 Q/K，尺度由 checkpoint 保存的 model_config 提供。
# 23：独立无 bias 动作输出头与 FP32 softcap，尺度由 checkpoint 保存的配置提供。
# 24：宿主 FP32 重锚 ABS/DELTA 与场景裁剪差分，增加逐字段 reset mask。
DEPLOYMENT_CONTRACT_VERSION = 24
DEPLOYMENT_MANIFEST_VERSION = 15
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
        normalized_vocab = tuple(SkillVocab.from_entries(vocab_entries))
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
                "residual_composition",
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
        vocab_entries = tuple(SkillVocab.from_dict(vocab_payload))
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
            skill_vocab_entries=parsed_input_contract.skill_vocab_entries,
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
        residual_composition_contract(self.model_config)
        if "qk_norm_scale" not in self.model_config:
            raise ValueError("deployment model_config missing qk_norm_scale; re-export the package")
        if "logit_softcap" not in self.model_config:
            raise ValueError("deployment model_config missing logit_softcap; re-export the package")
        # 复用模型配置的数值校验，但禁止给旧部署元数据补默认尺度。
        parsed_model_config = ModelConfig.from_mapping(self.model_config)
        if self.precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                "deployment precision must be bf16, float32 or float16"
            )
        validate_deployment_logit_softcap(parsed_model_config.logit_softcap, precision=self.precision)
        normalized_repetition = _json_value(
            asdict(parse_repetition_config(self.repetition_config))
        )
        if self.repetition_config != normalized_repetition:
            raise ValueError("deployment repetition config is not canonical")
        self.input_contract.assert_matches_data_spec(self.data_spec)
        self.input_contract.assert_matches_embedding(embedding_vocab_size)
        self.input_contract.create_skill_vocab().assert_matches(
            self.vocab_entries, context="deployment",
        )
        if self.input_contract.job_tag != self.data_spec.job_tag:
            raise ValueError("deployment job_tag differs from model input contract")
        if "max_sequence_length" in self.model_config:
            raise ValueError(
                "deployment model_config contains removed max_sequence_length"
            )
        removed = {"pair_embedding_dim", "pair_fusion", "output_adapter"}.intersection(self.model_config)
        if removed:
            raise ValueError("deployment model_config contains removed fusion options: " + ", ".join(sorted(removed)))
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
        if len(self.data_spec.action_keys) != self.data_spec.num_actions:
            raise ValueError("action order length differs from num_actions")
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

    def to_dict(self) -> dict[str, object]:
        core = self._unsigned_dict()
        signatures = {
            "action_order_sha256": stable_sha256(
                list(self.data_spec.action_keys)
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
        sd = self.data_spec.state_dim
        fd = self.data_spec.skill_feature_dim
        xd = self.data_spec.scene_dim
        float_dtype = precision_onnx_dtype(self.precision)
        layouts = {
            "scene_vectors": ((b, s, xd), "host FP32 anchor-clipped then time-differenced scene vectors; cast after encoding"),
            "scene_types": ((b, s), "scene type ids"),
            "scene_mask": ((b, s), "true for valid scene tokens"),
            "history_skill_ids": ((b, h), "right-padded vocab ids"),
            "history_skill_features": ((b, h, fd), "ordered skill features"),
            "history_state_vectors": ((b, h, sd), "host FP32 first-visible ABS anchor then raw numeric DELTA; normalized before cast"),
            "history_state_null_mask": ((b, h, sd), "history state null flags; right-padded positions are true"),
            "history_mask": ((b, h), "true for valid actions; shared by independent skill and state tokens"),
            "current_state_vectors": ((b, sd), "host FP32 current DELTA from last visible history, or ABS without history"),
            "current_state_null_mask": ((b, sd), "current request state null flags"),
            "history_state_reset_mask": ((b, h, sd), "per-field ABS reset markers; first visible state is absolute"),
            "current_state_reset_mask": ((b, sd), "current per-field ABS reset markers"),
        }
        # 宿主 FP32 和技能浮点载荷在图入口统一转换到部署精度。
        dtypes = {"float32": float_dtype, "float": float_dtype, "index": "tensor(int64)", "bool": "tensor(bool)"}
        return tuple(TensorSpec(field.name, dtypes[field.dtype], *layouts[field.name]) for field in MODEL_INPUT_FIELDS)

    def tensor_outputs(self) -> tuple[TensorSpec, ...]:
        float_dtype = precision_onnx_dtype(self.precision)
        return (
            TensorSpec(
                OUTPUT_NAMES[0],
                float_dtype,
                (self.capacity.batch_size, self.data_spec.num_actions),
                "softcapped logits in fixed action order before host repetition/masking/policy",
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
        named_inputs = dict(zip(TENSOR_INPUT_NAMES, inputs, strict=True))
        scene_types = named_inputs["scene_types"]
        if bool(((scene_types < 0) | (scene_types >= self.data_spec.num_scene_types)).any()):
            raise ValueError("scene_types contains an id outside the deployment contract")
        history_ids = named_inputs["history_skill_ids"]
        vocab_size = len(self.vocab_entries) + 1
        if bool(((history_ids < 0) | (history_ids >= vocab_size)).any()):
            raise ValueError("history_skill_ids contains an id outside the deployment vocab")
        _validate_right_padding(named_inputs["scene_mask"], "scene_mask")
        _validate_right_padding(named_inputs["history_mask"], "history_mask")

    def validate_host_action_order(
        self,
        action_keys: Sequence[str],
        action_legal_mask: torch.Tensor,
    ) -> None:
        if tuple(action_keys) != self.data_spec.action_keys:
            raise ValueError("host action order differs from raw_logits contract")
        expected_shape = (
            self.capacity.batch_size,
            self.data_spec.num_actions,
        )
        if tuple(action_legal_mask.shape) != expected_shape:
            raise ValueError("action_legal_mask shape differs from raw_logits")
        if action_legal_mask.dtype != torch.bool:
            raise ValueError("action_legal_mask must use bool dtype")

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
        vocab = SkillVocab.from_entries(self.vocab_entries).to_dict()
        return {
            "contract_version": DEPLOYMENT_CONTRACT_VERSION,
            "job_tag": self.data_spec.job_tag,
            "precision": self.precision,
            "capacity": self.capacity.to_dict(),
            "data_spec": asdict(self.data_spec),
            "model_input_contract": self.input_contract.to_dict(),
            "model_config": dict(self.model_config),
            "residual_composition": residual_composition_contract(self.model_config),
            "vocab": vocab,
            "state_layout": state_layout,
            "scene_layout": scene_layout,
            "tensor_inputs": [spec.to_dict() for spec in self.tensor_inputs()],
            "tensor_outputs": [spec.to_dict() for spec in self.tensor_outputs()],
            "host_postprocessing": {
                "action_legal_mask": {
                    "dtype": "tensor(bool)",
                    "shape": [
                        self.capacity.batch_size,
                        self.data_spec.num_actions,
                    ],
                    "order": "data_spec.action_keys",
                },
                "raw_logits_order": "data_spec.action_keys",
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

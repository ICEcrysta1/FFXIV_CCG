"""从 compiled cache 契约推导模型输入规格。"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class DataSpec:
    """模型输入维度与固定动作输出词表。"""

    job_tag: str
    num_actions: int
    state_dim: int
    scene_dim: int
    skill_feature_dim: int
    num_scene_types: int
    action_keys: tuple[str, ...]
    skill_feature_names: tuple[str, ...]
    action_to_vocab_id: tuple[int, ...]
    action_is_gcd: tuple[bool, ...]

    def __post_init__(self) -> None:
        if "time_seconds" in self.skill_feature_names:
            raise ValueError("removed skill time_seconds feature; rebuild model input")
        if self.skill_feature_dim != len(self.skill_feature_names):
            raise ValueError("skill feature order length differs from skill feature dimension")
        if self.num_actions < 1 or len(self.action_keys) != self.num_actions:
            raise ValueError("action_keys must match the positive num_actions")
        if len(set(self.action_keys)) != self.num_actions:
            raise ValueError("action_keys must be unique")
        if len(self.action_to_vocab_id) != self.num_actions:
            raise ValueError("action_to_vocab_id must match num_actions")
        if any(value <= 0 for value in self.action_to_vocab_id):
            raise ValueError("output actions must map to non-padding vocabulary rows")
        if len(set(self.action_to_vocab_id)) != self.num_actions:
            raise ValueError("output actions must map to distinct vocabulary rows")
        if len(self.action_is_gcd) != self.num_actions or any(type(value) is not bool for value in self.action_is_gcd):
            raise ValueError("action_is_gcd must contain one boolean per output action")

    @classmethod
    def from_dataset(cls, dataset) -> DataSpec:
        return cls(
            job_tag=dataset.job_tag,
            num_actions=dataset.num_actions,
            state_dim=dataset.state_dim,
            scene_dim=dataset.scene_dim,
            skill_feature_dim=len(dataset.skill_feature_names),
            num_scene_types=dataset.num_scene_types,
            action_keys=tuple(dataset.action_keys),
            skill_feature_names=tuple(dataset.skill_feature_names),
            action_to_vocab_id=tuple(dataset.action_to_vocab_id),
            action_is_gcd=tuple(dataset.action_is_gcd),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DataSpec:
        return cls(
            job_tag=str(payload["job_tag"]),
            num_actions=int(payload["num_actions"]),
            state_dim=int(payload["state_dim"]),
            scene_dim=int(payload["scene_dim"]),
            skill_feature_dim=int(payload["skill_feature_dim"]),
            num_scene_types=int(payload["num_scene_types"]),
            action_keys=tuple(str(value) for value in payload["action_keys"]),
            skill_feature_names=tuple(str(value) for value in payload["skill_feature_names"]),
            action_to_vocab_id=tuple(int(value) for value in payload["action_to_vocab_id"]),
            action_is_gcd=tuple(payload["action_is_gcd"]),
        )

    def assert_compatible_with(self, other: DataSpec) -> None:
        if self != other:
            differences = [
                f"{field.name}: {getattr(self, field.name)!r} != {getattr(other, field.name)!r}"
                for field in fields(self)
                if getattr(self, field.name) != getattr(other, field.name)
            ]
            raise ValueError("training data spec mismatch: " + "; ".join(differences))

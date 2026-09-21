"""从 compiled cache 契约推导模型输入规格。"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class DataSpec:
    """模型输入所需的全部数据维度与候选顺序。"""

    job_tag: str
    num_candidates: int
    state_dim: int
    scene_dim: int
    skill_feature_dim: int
    num_scene_types: int
    candidate_action_keys: tuple[str, ...]
    skill_feature_names: tuple[str, ...]

    @classmethod
    def from_dataset(cls, dataset) -> DataSpec:
        return cls(
            job_tag=dataset.job_tag,
            num_candidates=dataset.num_candidates,
            state_dim=dataset.state_dim,
            scene_dim=dataset.scene_dim,
            skill_feature_dim=len(dataset.skill_feature_names),
            num_scene_types=dataset.num_scene_types,
            candidate_action_keys=tuple(dataset.candidate_action_keys),
            skill_feature_names=tuple(dataset.skill_feature_names),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DataSpec:
        return cls(
            job_tag=str(payload["job_tag"]),
            num_candidates=int(payload["num_candidates"]),
            state_dim=int(payload["state_dim"]),
            scene_dim=int(payload["scene_dim"]),
            skill_feature_dim=int(payload["skill_feature_dim"]),
            num_scene_types=int(payload["num_scene_types"]),
            candidate_action_keys=tuple(str(value) for value in payload["candidate_action_keys"]),
            skill_feature_names=tuple(str(value) for value in payload["skill_feature_names"]),
        )

    def assert_compatible_with(self, other: DataSpec) -> None:
        if self != other:
            differences = [
                f"{field.name}: {getattr(self, field.name)!r} != {getattr(other, field.name)!r}"
                for field in fields(self)
                if getattr(self, field.name) != getattr(other, field.name)
            ]
            raise ValueError("training data spec mismatch: " + "; ".join(differences))

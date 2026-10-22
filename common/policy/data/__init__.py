"""策略输入数据契约与预处理组件。"""

from .cache.compiled_cache import CompiledCacheReader
from .contracts.action_space import ActionSpace
from .contracts.input_contract import ModelInputContract
from .contracts.policy_actions import PolicyActionDefinition, load_policy_actions
from .contracts.skill_vocab import SkillVocab
from .contracts.spec import DataSpec
from .encoding.normalization import NormalizerConfig, load_normalizer_config
from .encoding.normalizer import Normalizer
from .schema import (
    SceneWindowSchema,
    StateFeatureGroup,
    StateFeatureLayout,
    TrainingSchema,
)

__all__ = [
    "ActionSpace",
    "CompiledCacheReader",
    "DataSpec",
    "ModelInputContract",
    "Normalizer",
    "NormalizerConfig",
    "PolicyActionDefinition",
    "SceneWindowSchema",
    "SkillVocab",
    "StateFeatureGroup",
    "StateFeatureLayout",
    "TrainingSchema",
    "load_normalizer_config",
    "load_policy_actions",
]

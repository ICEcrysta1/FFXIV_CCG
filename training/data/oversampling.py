"""职业级动作序列过采样规则。

规则匹配以一个 raw source 的完整 label 时间线为单位：忽略动作不参与动作模式比较，
但位于命中首尾动作之间的全部样本都会获得该规则权重。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Sequence

from common.policy.config import load_policy_config


_CAST_TIME_EPSILON = 1e-6
_ACTION_SUFFIX_PATTERN = re.compile(r"^(?P<action>.+?)\s*\((?P<mode>[01])\)$")
_DEFAULT_IGNORED_ACTIONS = frozenset({"ogcd_wait"})


@dataclass(frozen=True)
class ActionPattern:
    """一项规则动作及其可选的实际读条约束。"""

    action_key: str
    cast_mode: int | None = None


@dataclass(frozen=True)
class OversamplingRule:
    """一条完整动作序列的过采样规则。"""

    sequence: tuple[ActionPattern, ...]
    weight: int


@dataclass(frozen=True)
class _ActionEvent:
    """单个训练 label 对应的动作事件。"""

    sample_offset: int
    action_key: str
    cast_time_seconds: object


class SequenceOversampler:
    """把命中的完整动作区间转换为样本权重。"""

    def __init__(
        self,
        *,
        ignored_actions: frozenset[str],
        rules: Sequence[OversamplingRule],
    ):
        self._ignored_actions = ignored_actions
        self._rules = tuple(
            sorted(
                rules,
                key=lambda rule: (
                    len(rule.sequence),
                    sum(pattern.cast_mode is not None for pattern in rule.sequence),
                ),
                reverse=True,
            )
        )

    def build_source_weights(
        self,
        samples: Sequence[dict[str, object]],
        *,
        skill_feature_names: Sequence[str],
    ) -> tuple[int, ...]:
        """为单一 raw source 的连续样本生成权重。

        忽略动作会从模式匹配序列中移除，但命中区间内的忽略动作仍保留权重，
        以便被重复的 batch 仍能复现完整上下文。
        """
        events = tuple(
            _build_action_event(
                sample,
                sample_offset=sample_offset,
                skill_feature_names=skill_feature_names,
            )
            for sample_offset, sample in enumerate(samples)
        )
        return self.build_source_weights_from_events(events, sample_count=len(samples))

    def build_source_weights_from_events(
        self,
        events: Iterable[_ActionEvent],
        *,
        sample_count: int,
    ) -> tuple[int, ...]:
        """从事件流构建权重，只保留最长规则所需的匹配窗口。"""
        if sample_count < 0:
            raise ValueError("sample_count must be >= 0")
        weights = [1] * sample_count
        if not sample_count or not self._rules:
            return tuple(weights)

        max_sequence_length = max(len(rule.sequence) for rule in self._rules)
        matched_events: list[_ActionEvent] = []
        for event in events:
            if event.action_key in self._ignored_actions:
                continue
            matched_events.append(event)
            if len(matched_events) > max_sequence_length:
                del matched_events[0]

            rule = self._match_ending_at(matched_events, len(matched_events) - 1)
            if rule is None:
                continue
            start_index = len(matched_events) - len(rule.sequence)
            start_offset = matched_events[start_index].sample_offset
            end_offset = event.sample_offset
            for sample_offset in range(start_offset, end_offset + 1):
                weights[sample_offset] = max(weights[sample_offset], rule.weight)
        return tuple(weights)

    def _match_ending_at(
        self,
        events: Sequence[_ActionEvent],
        end_index: int,
    ) -> OversamplingRule | None:
        for rule in self._rules:
            if len(rule.sequence) > end_index + 1:
                continue
            start_index = end_index - len(rule.sequence) + 1
            if all(
                _event_matches_pattern(event, pattern)
                for event, pattern in zip(
                    events[start_index : end_index + 1],
                    rule.sequence,
                    strict=True,
                )
            ):
                return rule
        return None


def load_sequence_oversampler(config_path: Path | None) -> SequenceOversampler | None:
    """从职业模型 YAML 加载完整序列过采样规则。"""
    if config_path is None:
        return None
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        return None

    raw = load_policy_config(config_path)
    oversampling = raw.get("oversampling", {}) or {}
    if not isinstance(oversampling, dict) or not bool(oversampling.get("enabled", True)):
        return None

    ignored_actions_raw = oversampling.get("ignored_actions", ()) or ()
    if not isinstance(ignored_actions_raw, (list, tuple)):
        raise ValueError(f"oversampling ignored_actions must be a list: {config_path}")
    ignored_actions = _DEFAULT_IGNORED_ACTIONS.union(
        str(action).strip()
        for action in ignored_actions_raw
        if str(action).strip()
    )

    rules: list[OversamplingRule] = []
    seen_sequences: set[tuple[ActionPattern, ...]] = set()
    for raw_rule in oversampling.get("rules", []):
        if not isinstance(raw_rule, dict):
            raise ValueError(f"oversampling rules must be mappings: {config_path}")
        sequence = tuple(
            _parse_action_pattern(action, config_path=config_path)
            for action in raw_rule.get("sequence", ())
        )
        sequence = tuple(pattern for pattern in sequence if pattern.action_key)
        if not sequence:
            raise ValueError(f"oversampling rule sequence must not be empty: {config_path}")
        if sequence in seen_sequences:
            raise ValueError(f"duplicate oversampling rule sequence: {sequence!r}")
        seen_sequences.add(sequence)
        weight = int(raw_rule.get("weight", 1))
        if weight < 1:
            raise ValueError(f"oversampling rule weight must be >= 1: {config_path}")
        rules.append(OversamplingRule(sequence=sequence, weight=weight))

    if not rules:
        return None
    return SequenceOversampler(ignored_actions=frozenset(ignored_actions), rules=rules)


def build_sample_weights(dataset, oversampler: SequenceOversampler | None) -> tuple[int, ...] | None:
    """按 raw source 识别命中区间，并把完整序列映射成训练样本权重。"""
    if oversampler is None:
        return None

    weights = [1] * len(dataset)
    for source_indices in dataset.iter_source_index_ranges():
        events = (
            _build_action_event(
                dataset[index],
                sample_offset=sample_offset,
                skill_feature_names=dataset.skill_feature_names,
            )
            for sample_offset, index in enumerate(source_indices)
        )
        source_weights = oversampler.build_source_weights_from_events(
            events,
            sample_count=len(source_indices),
        )
        for dataset_index, weight in zip(source_indices, source_weights, strict=True):
            if weight < 1:
                raise ValueError(
                    f"oversampling sample weight must be >= 1: index={dataset_index}, weight={weight}"
                )
            weights[dataset_index] = weight
    return tuple(weights)


def _parse_action_pattern(value: object, *, config_path: Path) -> ActionPattern:
    """解析 `技能`、`技能(0)` 和 `技能(1)` 三种规则动作写法。"""
    text = str(value).strip()
    if not text:
        raise ValueError(f"oversampling action pattern must not be empty: {config_path}")

    match = _ACTION_SUFFIX_PATTERN.fullmatch(text)
    if match is not None:
        action = match.group("action").strip()
        if not action:
            raise ValueError(f"oversampling action pattern must include an action: {config_path}")
        return ActionPattern(action_key=action, cast_mode=int(match.group("mode")))

    if "(" in text or ")" in text:
        raise ValueError(
            f"oversampling action cast mode must be '(0)' or '(1)': {text!r} ({config_path})"
        )
    return ActionPattern(action_key=text)


def _build_action_event(
    sample: dict[str, object],
    *,
    sample_offset: int,
    skill_feature_names: Sequence[str],
) -> _ActionEvent:
    cast_time_index = _feature_index(skill_feature_names, "cast_time.seconds")
    label_index = sample.get("label_index")
    label_row = _row_value(
        sample.get("candidate_skill_features"),
        None if label_index is None else int(label_index),
    )
    return _ActionEvent(
        sample_offset=sample_offset,
        action_key=str(sample.get("label_action_key", "")),
        cast_time_seconds=_row_value(label_row, cast_time_index),
    )


def _event_matches_pattern(event: _ActionEvent, pattern: ActionPattern) -> bool:
    if event.action_key != pattern.action_key:
        return False
    if pattern.cast_mode is None:
        return True
    return _cast_time_is_mode(event.cast_time_seconds, pattern.cast_mode)


def _cast_time_is_mode(cast_time: object, mode: int) -> bool:
    """按实际读条秒数判断瞬发 `(0)` 或读条 `(1)`。"""
    if cast_time is None:
        return False
    is_instant = float(cast_time) <= _CAST_TIME_EPSILON
    return is_instant if mode == 0 else not is_instant


def _feature_index(feature_names: Sequence[str], feature_name: str) -> int | None:
    try:
        return list(feature_names).index(feature_name)
    except ValueError:
        return None


def _row_value(row: object, index: int | None) -> object:
    if index is None or row is None:
        return None
    try:
        return row[index]
    except (IndexError, KeyError, TypeError):
        return None

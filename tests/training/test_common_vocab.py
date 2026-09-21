"""训练公共层 batch sampler 与序列过采样测试。"""

from __future__ import annotations

import yaml

from training import ShardBatchSampler, WeightedShardBatchSampler, build_sample_weights, load_sequence_oversampler
from tests.training._common_fixtures import enabled_black_mage_config


def test_shard_batch_sampler_keeps_batches_local_and_shuffles_each_epoch():
    sampler = ShardBatchSampler(
        [(0, 1, 2, 3), (4, 5, 6, 7)],
        batch_size=2,
        seed=42,
        shuffle=True,
    )

    first_epoch = list(sampler)
    second_epoch = list(sampler)

    assert len(sampler) == 4
    assert sorted(index for batch in first_epoch for index in batch) == list(range(8))
    assert all(len({index // 4 for index in batch}) == 1 for batch in first_epoch)
    assert first_epoch != second_epoch


def test_weighted_shard_batch_sampler_repeats_rare_samples_without_crossing_shards():
    sampler = WeightedShardBatchSampler(
        [(0, 1), (2, 3)],
        (1, 3, 1, 1),
        batch_size=2,
        seed=42,
        shuffle=False,
    )

    batches = list(sampler)

    assert sum(len(batch) for batch in batches) == 6
    assert sum(index == 1 for batch in batches for index in batch) == 3
    assert all(len({0 if index < 2 else 1 for index in batch}) == 1 for batch in batches)


def test_black_mage_oversampling_strengthens_entire_matched_sequence(tmp_path):
    config_path = enabled_black_mage_config(tmp_path)
    oversampler = load_sequence_oversampler(config_path)
    assert oversampler is not None
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured_weights = {
        tuple(rule["sequence"]): int(rule["weight"])
        for rule in config["oversampling"]["rules"]
    }

    samples = _oversampling_samples(
        [
            ("paradox", 0.0),
            ("transpose", 0.0),
            ("high_thunder", 0.0),
            ("paradox", 0.0),
            ("fire_iii", 0.0),
            ("fire_iv", 2.5),
        ]
    )

    weights = oversampler.build_source_weights(samples, skill_feature_names=("cast_time.seconds",))

    assert weights == (
        configured_weights[("paradox", "transpose", "paradox", "fire_iii(0)")],
    ) * 5 + (1,)


def test_oversampling_matches_cast_time_mode_for_entire_sequence(tmp_path):
    config_path = tmp_path / "oversampling.yaml"
    config_path.write_text(
        "oversampling:\n"
        "  rules:\n"
        "    - sequence: [fire_iii(1)]\n"
        "      weight: 3\n",
        encoding="utf-8",
    )
    oversampler = load_sequence_oversampler(config_path)
    assert oversampler is not None

    assert oversampler.build_source_weights(
        _oversampling_samples([("fire_iii", 2.5)]),
        skill_feature_names=("cast_time.seconds",),
    ) == (3,)
    assert oversampler.build_source_weights(
        _oversampling_samples([("fire_iii", 0.0)]),
        skill_feature_names=("cast_time.seconds",),
    ) == (1,)


def test_oversampling_ignores_actions_for_match_but_includes_them_in_weighted_span(tmp_path):
    config_path = tmp_path / "oversampling.yaml"
    config_path.write_text(
        "oversampling:\n"
        "  ignored_actions: [high_thunder, high_thunder_ii]\n"
        "  rules:\n"
        "    - sequence: [transpose, paradox, fire_iii(0)]\n"
        "      weight: 4\n",
        encoding="utf-8",
    )
    oversampler = load_sequence_oversampler(config_path)
    assert oversampler is not None

    samples = _oversampling_samples(
        [
            ("transpose", 0.0),
            ("high_thunder", 0.0),
            ("paradox", 0.0),
            ("ogcd_wait", 0.0),
            ("fire_iii", 0.0),
            ("fire_iv", 2.5),
        ]
    )
    assert oversampler.build_source_weights(samples, skill_feature_names=("cast_time.seconds",)) == (4, 4, 4, 4, 4, 1)


def test_oversampling_uses_highest_weight_for_overlapping_sequences(tmp_path):
    config_path = tmp_path / "oversampling.yaml"
    config_path.write_text(
        "oversampling:\n"
        "  rules:\n"
        "    - sequence: [a, b]\n"
        "      weight: 3\n"
        "    - sequence: [b, c]\n"
        "      weight: 5\n",
        encoding="utf-8",
    )
    oversampler = load_sequence_oversampler(config_path)
    assert oversampler is not None

    assert oversampler.build_source_weights(
        _oversampling_samples([("a", 0.0), ("b", 0.0), ("c", 0.0)]),
        skill_feature_names=("cast_time.seconds",),
    ) == (3, 5, 5)


def test_oversampling_does_not_match_across_pt_sources(tmp_path):
    config_path = tmp_path / "oversampling.yaml"
    config_path.write_text(
        "oversampling:\n"
        "  rules:\n"
        "    - sequence: [a, b]\n"
        "      weight: 4\n",
        encoding="utf-8",
    )
    oversampler = load_sequence_oversampler(config_path)
    assert oversampler is not None
    dataset = _OversamplingDataset(
        _oversampling_samples([("a", 0.0), ("b", 0.0)]),
        source_ranges=(range(0, 1), range(1, 2)),
    )

    assert build_sample_weights(dataset, oversampler) == (1, 1)


def _oversampling_samples(actions_and_cast_times):
    return [
        {
            "label_action_key": action_key,
            "label_index": 0,
            "candidate_skill_features": [[cast_time]],
        }
        for action_key, cast_time in actions_and_cast_times
    ]


class _OversamplingDataset:
    """只实现序列权重构建所需的数据集接口。"""

    skill_feature_names = ("cast_time.seconds",)

    def __init__(self, samples, *, source_ranges):
        self._samples = samples
        self._source_ranges = source_ranges

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        return self._samples[index]

    def iter_source_index_ranges(self):
        yield from self._source_ranges

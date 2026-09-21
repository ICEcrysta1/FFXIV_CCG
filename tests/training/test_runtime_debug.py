from __future__ import annotations

import json

import torch

from training.runtime.runtime_debug import (
    RuntimeDebugConfig,
    RuntimeDebugRecorder,
    aggregate_events,
)
from common.policy.model.trace import TraceableTransformerEncoderLayer
from common.policy.model.position_encoding import RotaryPositionEncoding


def _read_report(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_runtime_debug_writes_batch_context_and_aggregates(tmp_path):
    output_path = tmp_path / "runtime_memory.jsonl"
    recorder = RuntimeDebugRecorder(
        device=torch.device("cpu"),
        config=RuntimeDebugConfig(enabled=True, max_steps=1, synchronize=False),
        output_path=output_path,
    )
    batch = {
        "label_index": torch.zeros(2, dtype=torch.int64),
        "history_skill_ids": torch.zeros((2, 5), dtype=torch.int64),
        "scene_vectors": torch.zeros((2, 3, 4)),
        "candidate_skill_ids": torch.zeros((2, 7), dtype=torch.int64),
    }

    recorder.begin_step(epoch=2, step=4, batch=batch)
    with recorder.stage("input_encoder"):
        pass
    with recorder.stage("input_encoder"):
        pass
    recorder.end_step()
    recorder.begin_step(epoch=2, step=5, batch=batch)
    recorder.end_step()

    reports = _read_report(output_path)
    assert len(reports) == 1
    report = reports[0]
    assert report["schema_version"] == 1
    assert report["epoch"] == 2
    assert report["step"] == 4
    assert report["batch"]["batch_size"] == 2
    assert report["batch"]["history_length"] == 5
    assert report["batch"]["scene_length"] == 3
    assert report["batch"]["candidate_count"] == 7
    assert report["batch"]["sequence_length"] == 16
    assert report["aggregates"]["input_encoder"]["calls"] == 2
    assert report["step_peak_allocated_mib"] == 0.0


def test_runtime_debug_records_transformer_attention_and_ffn(tmp_path):
    from common.policy.model.split_encoder import run_split_encoder

    output_path = tmp_path / "runtime_memory.jsonl"
    recorder = RuntimeDebugRecorder(
        device=torch.device("cpu"),
        config=RuntimeDebugConfig(enabled=True, max_steps=1, synchronize=False),
        output_path=output_path,
    )
    layer = TraceableTransformerEncoderLayer(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    encoder = torch.nn.TransformerEncoder(
        layer,
        1,
        norm=torch.nn.LayerNorm(8),
    ).train()
    encoder.rotary_position_encoding = RotaryPositionEncoding(
        encoder.layers[0].self_attn.head_dim
    )
    encoder.layers[0].set_runtime_debug(recorder, layer_index=3)
    encoder.layers[0].set_activation_checkpoint_ffn(True)
    source = torch.randn(2, 5, 8, requires_grad=True)

    recorder.begin_step(
        epoch=1,
        step=1,
        batch={"label_index": torch.zeros(2, dtype=torch.int64)},
    )
    encoded = {
        "tokens": source,
        "prefix_length": 2,
        "candidate_count": 2,
        "prefix_valid": torch.ones((2, 2), dtype=torch.bool),
        "candidate_valid": torch.ones((2, 2), dtype=torch.bool),
        "cls_valid": torch.ones((2, 1), dtype=torch.bool),
    }
    outputs = run_split_encoder(encoder, encoded)
    sum(value.square().mean() for value in outputs[:3]).backward()
    recorder.end_step()

    report = _read_report(output_path)[0]
    aggregate_names = set(report["aggregates"])
    assert "encoder.layer.3.attention" in aggregate_names
    assert "encoder.layer.3.ffn" in aggregate_names
    assert report["aggregates"]["encoder.layer.3.ffn"]["calls"] >= 2


def test_aggregate_events_keeps_peak_maxima_and_time_sum():
    events = [
        {
            "name": "ffn",
            "elapsed_ms": 3.0,
            "peak_allocated_mib": 100.0,
            "new_peak_mib": 40.0,
            "peak_reserved_mib": 120.0,
        },
        {
            "name": "ffn",
            "elapsed_ms": 2.0,
            "peak_allocated_mib": 110.0,
            "new_peak_mib": 10.0,
            "peak_reserved_mib": 125.0,
        },
    ]

    aggregate = aggregate_events(events)["ffn"]

    assert aggregate["calls"] == 2
    assert aggregate["elapsed_ms"] == 5.0
    assert aggregate["max_peak_allocated_mib"] == 110.0
    assert aggregate["max_new_peak_mib"] == 40.0
    assert aggregate["max_peak_reserved_mib"] == 125.0

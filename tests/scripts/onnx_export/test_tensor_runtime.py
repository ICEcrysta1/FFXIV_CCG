"""ONNX Runtime Tensor I/O 与 BF16 golden 编码的职责级测试。"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.onnx_export.io.artifact_io import write_deterministic_npz
from scripts.onnx_export.runtime.tensor_runtime import (
    run_ort_tensors,
    tensor_to_golden_array,
)


def test_cpu_tensor_runtime_uses_numpy_only_for_bool(monkeypatch):
    class FakeOrtValue:
        dlpack_inputs: list[torch.Tensor] = []
        numpy_inputs: list[tuple[np.ndarray, str, int]] = []

        @classmethod
        def from_dlpack(cls, value):
            cls.dlpack_inputs.append(value)
            return value

        @classmethod
        def ortvalue_from_numpy(cls, value, *, device_type, device_id):
            cls.numpy_inputs.append((value, device_type, device_id))
            return torch.from_numpy(value.copy())

    class FakeBinding:
        def __init__(self):
            self.inputs = {}
            self.outputs = []

        def bind_ortvalue_input(self, name, value):
            self.inputs[name] = value

        def bind_output(self, _name, _device_type, _device_id):
            return None

        def get_outputs(self):
            return self.outputs

    class FakeSession:
        def __init__(self):
            self.binding = FakeBinding()

        @staticmethod
        def get_providers():
            return ("CPUExecutionProvider",)

        def io_binding(self):
            return self.binding

        def run_with_iobinding(self, binding):
            binding.outputs = [binding.inputs["values"]]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(OrtValue=FakeOrtValue),
    )
    values = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    mask = torch.tensor([[True, False]])

    outputs = run_ort_tensors(
        FakeSession(),
        {"values": values, "mask": mask},
        ("output",),
    )

    assert torch.equal(outputs[0], values)
    assert len(FakeOrtValue.dlpack_inputs) == 1
    dlpack_value = FakeOrtValue.dlpack_inputs[0]
    assert torch.equal(dlpack_value, values)
    assert dlpack_value.dtype == torch.float32
    assert dlpack_value.device.type == "cpu"
    assert len(FakeOrtValue.numpy_inputs) == 1
    bool_array, device_type, device_id = FakeOrtValue.numpy_inputs[0]
    assert bool_array.dtype == np.bool_
    assert device_type == "cpu"
    assert device_id == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bf16_golden_npz_is_byte_reproducible(tmp_path):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")
    source = torch.tensor(
        [1.0, -2.5, 0.0, float("inf")],
        dtype=torch.bfloat16,
        device="cuda",
    )
    arrays = {"raw_logits": tensor_to_golden_array(source)}
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    write_deterministic_npz(first, arrays)
    write_deterministic_npz(second, arrays)

    assert first.read_bytes() == second.read_bytes()
    with np.load(first) as payload:
        encoded = payload["raw_logits"]
    restored = torch.from_numpy(encoded.copy()).view(torch.bfloat16)
    assert encoded.dtype == np.uint16
    assert torch.equal(restored, source.cpu())

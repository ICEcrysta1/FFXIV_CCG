"""冻结技能表和基础状态共享布局、读取与装配的回归。"""

from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest
import torch

from common.policy.data.context_fields import MODEL_INPUT_FIELDS, tensor_dimensions
from common.policy.data.input_contract import ModelInputContract
from common.policy.data.schema import StateFeatureLayout, TrainingSchema
from common.policy.data.state_features import assemble_state_inputs, read_state_tokens
from tests.training._causal_fixtures import make_data_spec, make_input_contract, make_state_groups


def _layout(actions=("ice", "fire", "wait")):
    return StateFeatureLayout(
        make_state_groups({"player_state": ("previous_action_after.mp", "request_state.time_seconds")}, actions),
        ("previous_action_after", "request_state"), actions,
    )


def _context(layout, *, tokens=True):
    return {
        **{group.feature_keys_field: list(group.feature_keys) for group in layout.groups},
        "tokens": [{"player_state": [None, 600.004], "skill_availability": [0, 1, 0, 1, 0, 1]}] if tokens else [],
    }


def test_layout_dimensions_follow_saved_columns_and_action_order():
    layout = _layout()
    assert (layout.base_state_dim, layout.availability_dim, layout.state_dim) == (2, 6, 8)
    assert layout.request_time_index == 1
    assert layout.group_slices == {"player_state": slice(0, 2), "skill_availability": slice(2, 8)}
    with pytest.raises(ValueError, match="feature order"):
        replace(layout, action_keys=tuple(reversed(layout.action_keys)))
    with pytest.raises(ValueError, match="unique"):
        replace(layout, action_keys=("ice", "ice", "wait"))


def test_restored_group_records_are_revalidated_at_parent_layout_boundary():
    layout = deepcopy(_layout())
    object.__setattr__(layout.groups[0], "encoding", "unrecognized")
    with pytest.raises(ValueError, match="unknown state encoding"):
        replace(layout)


def test_reader_preserves_fp32_base_null_and_absolute_boolean_table():
    layout = _layout()
    context = _context(layout)
    original = deepcopy(context)
    raw = read_state_tokens(context, layout)
    assert raw.base_values.dtype == torch.float32
    assert raw.base_values.tolist() == [[0.0, 600.0040283203125]]
    assert raw.base_null_mask.tolist() == [[True, False]]
    assert raw.availability.dtype == torch.bool
    assert raw.availability.tolist() == [[False, True, False, True, False, True]]
    assert context == original
    empty = read_state_tokens(_context(layout, tokens=False), layout)
    assert empty.base_values.shape == (0, 2)
    assert empty.availability.shape == (0, 6)


@pytest.mark.parametrize("bad", [None, True, "1", 0.5, -1, 2, float("nan"), float("inf")])
def test_reader_rejects_binary_values_before_bool_conversion(bad):
    layout = _layout()
    context = _context(layout)
    context["tokens"][0]["skill_availability"][0] = bad
    with pytest.raises(ValueError, match="numeric 0/1"):
        read_state_tokens(context, layout)


@pytest.mark.parametrize("bad", [None, "600", float("nan"), float("inf"), 1e40])
def test_reader_rejects_missing_or_nonfinite_request_time(bad):
    layout = _layout()
    context = _context(layout)
    context["tokens"][0]["player_state"][1] = bad
    with pytest.raises(ValueError):
        read_state_tokens(context, layout)


def test_reader_rejects_same_width_reordered_keys_and_missing_table():
    layout = _layout()
    context = _context(layout)
    context["skill_availability_feature_keys"].reverse()
    with pytest.raises(ValueError, match="feature keys"):
        read_state_tokens(context, layout)
    context = _context(layout)
    del context["tokens"][0]["skill_availability"]
    with pytest.raises(ValueError, match="width"):
        read_state_tokens(context, layout)


def test_assembly_keeps_base_values_and_masks_availability_padding():
    layout = _layout()
    values = torch.tensor([[[0.2, -0.5], [9.0, -1.0]]])
    availability = torch.tensor([[[1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1]]], dtype=torch.bool)
    result = assemble_state_inputs(values, availability, torch.tensor([[True, False]]), layout)
    torch.testing.assert_close(result[0, 0, :2], values[0, 0], rtol=0, atol=0)
    assert result[0, 0, 2:].tolist() == [1, 0, 1, 0, 1, 0]
    assert not result[0, 1].any()


def test_saved_schema_survives_json_key_sorting_and_rejects_spec_drift():
    contract = make_input_contract()
    restored = ModelInputContract.from_dict(json.loads(json.dumps(contract.to_dict(), sort_keys=True)))
    assert restored.schema == contract.schema
    restored.assert_matches_data_spec(make_data_spec())
    bad = contract.to_dict()
    bad["data_spec"]["base_state_dim"] += 1
    with pytest.raises(ValueError, match="state dimensions"):
        ModelInputContract.from_dict(bad)
    schema_payload = asdict(contract.schema)
    schema_payload["state_groups"] = dict(enumerate(schema_payload["state_groups"]))
    with pytest.raises(ValueError, match="ordered list"):
        TrainingSchema.from_dict(schema_payload)


def test_model_fields_share_combined_values_and_base_mask_dimensions():
    spec = make_data_spec(base_state_dim=86, num_actions=25,
                          action_keys=tuple(f"action_{i}" for i in range(25)),
                          action_to_vocab_id=tuple(range(1, 26)), action_is_gcd=(True,) * 25)
    dimensions = tensor_dimensions(spec, batch=2, history=3, scene=4)
    fields = {field.name: field for field in MODEL_INPUT_FIELDS}
    assert len(fields) == 12
    assert fields["history_state_vectors"].resolve_shape(dimensions) == (2, 3, 136)
    assert fields["history_state_null_mask"].resolve_shape(dimensions) == (2, 3, 86)
    assert fields["current_state_reset_mask"].resolve_shape(dimensions) == (2, 86)

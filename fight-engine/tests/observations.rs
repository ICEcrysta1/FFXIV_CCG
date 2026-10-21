// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

//! 实际基线重放：输出不推进队列，冻结历史、缓存恢复和模型列式载荷共用同一状态。

mod common;
#[path = "common/probe.rs"]
mod probe;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use fight_engine::config::{PrecisionConfig, SchemaConfig, TensorDtype};
use fight_engine::engine::ActionRequest;
use fight_engine::engine::simulator::{EnginePayload, Simulator};
use fight_engine::outputs::{
    OutputFormatter, build_death_token, output_schema_metadata, pack_tensor_payload, scene_context,
};
use fight_engine::state::ExternalCombatEvent;
use fight_engine::system::timeline::SimulationSnapshot;
use serde_json::{Value, json};

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/observations.json")).unwrap()
}

fn round_legacy(value: &mut Value) {
    if let Some(number) = value.as_f64() {
        *value = json!((number * 10_000.0).round_ties_even() / 10_000.0);
    }
}

fn round_metrics(metrics: &mut Value) {
    for key in ["cumulative_potency", "cumulative_dot_potency"] {
        round_legacy(&mut metrics[key]);
    }
}

// 只对旧实现明确舍入过的字段投影四位小数；生产输出与内部计算仍保持 F64。
fn legacy_vector(mut output: Value, float_resources: &BTreeSet<String>) -> Value {
    output = common::legacy_skill_state(output);
    for key in ["state_history_context", "current_state_context"] {
        let context = &mut output[key];
        let mut rounded_indices = BTreeMap::new();
        for group in ["buff_state", "target_buff_state", "resource_state"] {
            let indices = context[format!("{group}_feature_keys")]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .filter_map(|(index, feature)| {
                    let feature = feature.as_str().unwrap();
                    let resource = feature.split_once('.').unwrap().1;
                    let rounded = feature.ends_with(".remaining_seconds")
                        || feature.ends_with(".current_potency")
                        || feature.ends_with(".current_gcd_dot_potency")
                        || (group == "resource_state" && float_resources.contains(resource))
                        || (group == "buff_state"
                            && resource
                                .strip_prefix("resource.")
                                .is_some_and(|key| float_resources.contains(key)));
                    rounded.then_some(index)
                })
                .collect::<Vec<_>>();
            rounded_indices.insert(group, indices);
        }
        for token in context["tokens"].as_array_mut().unwrap() {
            for (group, indices) in &rounded_indices {
                for &index in indices {
                    round_legacy(&mut token[*group][index]);
                }
            }
        }
        if let Some(metrics) = context
            .get_mut("execution_metrics")
            .and_then(Value::as_array_mut)
        {
            for metrics in metrics {
                round_metrics(metrics);
            }
        }
    }
    for token in output["skill_history_context"].as_array_mut().unwrap() {
        round_legacy(&mut token["cast_time"]["seconds"]);
        round_legacy(&mut token["gcd_window"]["seconds"]);
        round_legacy(&mut token["next_cooldown_seconds"]);
        for key in float_resources {
            round_legacy(&mut token["job_resources_consumed"][key]);
        }
    }
    output
}

#[test]
fn snapshots_vectors_tensors_and_cache_restoration_match_baseline() {
    let baseline = fixture();
    let rules = probe::rules(&baseline, None);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let schema = Arc::new(SchemaConfig::load(root).unwrap());
    let precision = PrecisionConfig::load(root).unwrap();
    let formatter = |limit| {
        OutputFormatter::new(
            rules.project().clone(),
            rules.system().clone(),
            rules.job().clone(),
            schema.clone(),
            limit,
        )
        .unwrap()
    };
    let output = formatter(None);
    let floats = rules
        .job()
        .runtime()
        .resources
        .definitions()
        .iter()
        .filter(|(_, definition)| definition.resource_type == "float")
        .map(|(key, _)| key.clone())
        .collect::<BTreeSet<_>>();
    let mut simulator = Simulator::from_state(
        rules.clone(),
        probe::initial_state(&rules, &baseline["initial"]),
    )
    .unwrap();
    let mut snapshots: BTreeMap<String, SimulationSnapshot<EnginePayload>> = BTreeMap::new();
    for (index, step) in baseline["steps"].as_array().unwrap().iter().enumerate() {
        let input = &step["input"];
        let actual = match input["op"].as_str().unwrap() {
            "submit" => {
                let result = simulator
                    .submit_action(
                        ActionRequest {
                            timestamp: input["timestamp"].as_f64().unwrap(),
                            skill_key: input["skill"].as_str().unwrap().into(),
                        },
                        None,
                    )
                    .unwrap();
                json!({"accepted":result.accepted,"queued":result.queued,"reason":result.reason})
            }
            "advance" => {
                simulator
                    .advance_to(input["timestamp"].as_f64().unwrap())
                    .unwrap();
                json!(true)
            }
            "external" => {
                let events: Vec<ExternalCombatEvent> =
                    serde_json::from_value(input["events"].clone()).unwrap();
                assert!(simulator.apply_external_events(&events).unwrap().accepted);
                json!(true)
            }
            "snapshot" => {
                snapshots.insert(
                    input["name"].as_str().unwrap().into(),
                    simulator.create_snapshot(),
                );
                json!(true)
            }
            "restore" => {
                simulator
                    .restore_snapshot(&snapshots[input["name"].as_str().unwrap()])
                    .unwrap();
                json!(true)
            }
            "capture" => {
                let before = simulator.create_snapshot();
                let state = simulator.get_state();
                let current = simulator.capture_model_state(&state).unwrap();
                let keys = rules.action_keys();
                let values = rules.action_values(&state).unwrap();
                let vector = output
                    .format_vectors(&state, keys, &values, &current, None)
                    .unwrap();
                let tensor = output
                    .format_tensors(&state, keys, &values, &current, None, &precision)
                    .unwrap();
                // 格式器实际使用完整 F64 canonical，在最后一步才按配置转换。
                assert_eq!(tensor, pack_tensor_payload(&vector, &precision).unwrap());
                let metadata =
                    common::legacy_skill_state(output_schema_metadata(&vector, &schema).unwrap());
                let vector = legacy_vector(vector, &floats);
                let mut seconds = output.format(&state, "seconds").unwrap();
                let mut gcd = output.format(&state, "gcd").unwrap();
                round_metrics(&mut seconds["execution_metrics"]);
                for key in &floats {
                    round_legacy(&mut seconds[key]);
                    let definition = &rules.job().runtime().resources.definitions()[key];
                    if definition.display_unit == "seconds" {
                        let field = format!("{key}_gcd");
                        let unit = gcd["current_gcd_seconds"].as_f64().unwrap();
                        if unit > 0.0 {
                            let seconds = gcd[&field].as_f64().unwrap() * unit;
                            gcd[&field] =
                                json!((seconds * 10_000.0).round_ties_even() / 10_000.0 / unit);
                        }
                    } else {
                        round_legacy(&mut gcd[key]);
                    }
                }
                let retained = step["output"]["retained"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| {
                        let limit = row["limit"].as_u64().unwrap() as usize;
                        let vector = formatter(Some(limit))
                            .format_vectors(&state, keys, &values, &current, None)
                            .unwrap();
                        json!({"limit":limit,"vector":legacy_vector(vector,&floats)})
                    })
                    .collect::<Vec<_>>();
                assert_eq!(before, simulator.create_snapshot(), "输出改变了时间线");
                json!({"seconds":seconds,"gcd":gcd,"tensor":pack_tensor_payload(&vector, &precision).unwrap(),
                    "metadata":metadata,"vector":vector,"retained":retained})
            }
            op => panic!("输出基线含未知操作: {op}"),
        };
        common::assert_equivalent(&step["output"], &actual, &format!("steps[{index}]"));
    }
}

#[test]
fn registered_skill_states_expand_automatically_and_keep_charges_when_unavailable() {
    let mut baseline = fixture();
    let mut added = baseline["project"]["job"]["skills"][0].clone();
    added["key"] = json!("added_skill");
    added["game_id"] = json!(7654321);
    baseline["project"]["job"]["skills"]
        .as_array_mut()
        .unwrap()
        .push(added);
    let rules = probe::rules(&baseline, None);
    let mut state = rules.initial_state(None, None, 0.0).unwrap();
    for (moving, mp) in [(true, 10000), (false, 0)] {
        state.is_moving = moving;
        state.mp = mp;
        let frame = rules.capture_model_frame(&state, false).unwrap();
        let cast = &frame.skill_state["cast"];
        assert!(!cast.available);
        assert_eq!(cast.current_charges, 1);
        assert_eq!(cast.next_cooldown_seconds, 0.0);
    }
    let current = rules.capture_model_state(&state, false).unwrap();
    assert!(
        current
            .request_state
            .skill_state
            .contains_key("added_skill")
    );
    assert!(!current.request_state.skill_state.contains_key("disabled"));

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let schema = Arc::new(SchemaConfig::load(root).unwrap());
    let output = OutputFormatter::new(
        rules.project().clone(),
        rules.system().clone(),
        rules.job().clone(),
        schema,
        None,
    )
    .unwrap();
    let vector = output
        .format_vectors(
            &state,
            rules.action_keys(),
            &rules.action_values(&state).unwrap(),
            &current,
            None,
        )
        .unwrap();
    let fields = ["available", "current_charges", "next_cooldown_seconds"];
    let expected: Vec<_> = ["previous_action_after", "request_state"]
        .into_iter()
        .flat_map(|snapshot| {
            rules.action_keys().iter().flat_map(move |key| {
                fields
                    .into_iter()
                    .map(move |field| format!("{snapshot}.{key}.{field}"))
            })
        })
        .collect();
    assert_eq!(
        vector["current_state_context"]["skill_state_feature_keys"],
        json!(expected)
    );
    assert_eq!(
        vector["current_state_context"]["tokens"][0]["skill_state"]
            .as_array()
            .unwrap()
            .len(),
        2 * rules.action_keys().len() * fields.len()
    );
}

#[test]
fn skill_states_freeze_each_snapshot_and_survive_charge_recovery_and_restore() {
    let baseline = fixture();
    let rules = probe::rules(&baseline, None);
    let mut simulator = Simulator::new(rules, 0.0, None).unwrap();
    let initial = simulator
        .capture_model_state(&simulator.get_state())
        .unwrap();
    assert_eq!(
        initial.request_state.skill_state["charge"].current_charges,
        2
    );
    assert!(
        simulator
            .submit_action(
                ActionRequest {
                    timestamp: 0.0,
                    skill_key: "charge".into(),
                },
                None
            )
            .unwrap()
            .accepted
    );
    let saved = simulator.create_snapshot();
    let mut depleted = simulator.fork();
    assert!(
        depleted
            .submit_action(
                ActionRequest {
                    timestamp: 0.0,
                    skill_key: "charge".into(),
                },
                None
            )
            .unwrap()
            .accepted
    );
    let empty = depleted.capture_model_state(&depleted.get_state()).unwrap();
    assert_eq!(empty.request_state.skill_state["charge"].current_charges, 0);
    assert!(!empty.request_state.skill_state["charge"].available);
    assert_eq!(
        empty.request_state.skill_state["charge"].next_cooldown_seconds,
        3.0
    );
    simulator.advance_to(1.234567891).unwrap();
    let current = simulator
        .capture_model_state(&simulator.get_state())
        .unwrap();
    let previous = &current.previous_action_after.skill_state["charge"];
    let request = &current.request_state.skill_state["charge"];
    assert_eq!(previous.current_charges, 1);
    assert_eq!(previous.next_cooldown_seconds, 3.0);
    assert_eq!(request.current_charges, 1);
    assert_eq!(request.next_cooldown_seconds, 3.0 - 1.234567891);
    simulator.advance_to(3.0).unwrap();
    let recovered = simulator
        .capture_model_state(&simulator.get_state())
        .unwrap();
    assert_eq!(
        recovered.request_state.skill_state["charge"].current_charges,
        2
    );
    assert_eq!(
        recovered.request_state.skill_state["charge"].next_cooldown_seconds,
        0.0
    );
    assert_eq!(
        current.request_state.skill_state["charge"].current_charges,
        1
    );
    assert_eq!(
        current.request_state.skill_state["charge"].next_cooldown_seconds,
        3.0 - 1.234567891
    );
    assert_eq!(
        initial.request_state.skill_state["charge"].current_charges,
        2
    );
    assert_eq!(
        recovered.previous_action_after,
        current.previous_action_after
    );
    simulator.restore_snapshot(&saved).unwrap();
    let restored = simulator
        .capture_model_state(&simulator.get_state())
        .unwrap();
    assert_eq!(restored.request_state.skill_state["charge"], *previous);
    assert_eq!(
        restored.previous_action_after,
        current.previous_action_after
    );
}

#[test]
fn numeric_layouts_scene_schema_and_death_marker_match_baseline() {
    let baseline = fixture();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let schema = SchemaConfig::load(root).unwrap();
    let precision = PrecisionConfig::load(root).unwrap();
    common::assert_equivalent(
        &baseline["packing"]["output"],
        &pack_tensor_payload(&baseline["packing"]["input"], &precision).unwrap(),
        "packing",
    );
    common::assert_equivalent(
        &baseline["empty_scene"],
        &scene_context(&schema).unwrap(),
        "empty_scene",
    );
    let death = &baseline["death_token"];
    common::assert_equivalent(
        death,
        &build_death_token(
            death["timestamp"].as_f64().unwrap(),
            death["after_history_length"].as_u64().unwrap(),
        ),
        "death_token",
    );
}

#[test]
fn tensor_precision_is_applied_without_changing_canonical_or_layout() {
    let baseline = fixture();
    let rules = probe::rules(&baseline, None);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let output = OutputFormatter::new(
        rules.project().clone(),
        rules.system().clone(),
        rules.job().clone(),
        Arc::new(SchemaConfig::load(root).unwrap()),
        None,
    )
    .unwrap();
    let simulator = Simulator::from_state(
        rules.clone(),
        probe::initial_state(&rules, &baseline["initial"]),
    )
    .unwrap();
    let state = simulator.get_state();
    let current = simulator.capture_model_state(&state).unwrap();
    let keys = rules.action_keys();
    let values = vec![0.4; keys.len()];
    let canonical = output
        .format_vectors(&state, keys, &values, &current, None)
        .unwrap();
    let mut precision = PrecisionConfig::load(root).unwrap();
    for (dtype, expected) in [
        (TensorDtype::Float32, 0.4000000059604645),
        (TensorDtype::Float16, 0.39990234375),
        (TensorDtype::Bfloat16, 0.400390625),
    ] {
        precision.float_dtype = dtype;
        let tensor = output
            .format_tensors(&state, keys, &values, &current, None, &precision)
            .unwrap();
        assert_eq!(tensor["action_values"]["values"][0], json!(expected));
        assert_eq!(tensor["action_values"]["is_null"][0], json!(false));
        assert_eq!(
            pack_tensor_payload(
                &json!([
                    {"value": 0.4, "missing": null, "flag": true, "id": 1},
                    {"value": -0.4, "missing": 1.0, "flag": false, "id": 2},
                ]),
                &precision,
            )
            .unwrap(),
            json!({
                "value": {"values": [expected, -expected], "is_null": [false, false]},
                "missing": {"values": [0.0, 1.0], "is_null": [true, false]},
                "flag": [true, false], "id": [1, 2],
            }),
        );
        assert_eq!(
            canonical,
            output
                .format_vectors(&state, keys, &values, &current, None)
                .unwrap(),
        );
        assert_eq!(canonical["action_values"][0], json!(0.4));
    }
}

#[test]
fn tensor_precision_rounds_once_at_midpoints_subnormals_and_overflow() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut precision = PrecisionConfig::load(root).unwrap();
    for (dtype, ulp, smallest, largest, overflow_midpoint) in [
        (
            TensorDtype::Float32,
            2.0_f64.powi(-23),
            2.0_f64.powi(-149),
            f64::from(f32::MAX),
            2.0_f64.powi(128) - 2.0_f64.powi(103),
        ),
        (
            TensorDtype::Float16,
            2.0_f64.powi(-10),
            2.0_f64.powi(-24),
            65504.0,
            65520.0,
        ),
        (
            TensorDtype::Bfloat16,
            2.0_f64.powi(-7),
            2.0_f64.powi(-133),
            f64::from(f32::from_bits(0x7f7f0000)),
            2.0_f64.powi(128) - 2.0_f64.powi(119),
        ),
    ] {
        precision.float_dtype = dtype;
        let midpoint = 1.0 + ulp / 2.0;
        for (raw, expected) in [
            (0.0, 0.0),
            (-0.0, -0.0),
            (midpoint.next_down(), 1.0),
            (midpoint, 1.0),
            (midpoint.next_up(), 1.0 + ulp),
            (1.0 + 1.5 * ulp, 1.0 + 2.0 * ulp),
            (-midpoint.next_up(), -1.0 - ulp),
            (smallest / 2.0, 0.0),
            ((smallest / 2.0).next_up(), smallest),
            (-smallest / 2.0, -0.0),
            (smallest, smallest),
            (overflow_midpoint.next_down(), largest),
            (largest, largest),
        ] {
            let packed = pack_tensor_payload(&json!([raw]), &precision).unwrap();
            assert_eq!(
                packed["values"][0].as_f64().unwrap().to_bits(),
                expected.to_bits(),
                "{dtype:?}: {raw:e}",
            );
        }
        for raw in [overflow_midpoint, -overflow_midpoint, f64::MAX] {
            assert!(pack_tensor_payload(&json!([raw]), &precision).is_err());
        }
    }
}

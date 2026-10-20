//! 实际基线重放：输出不推进队列，冻结历史、缓存恢复和模型列式载荷共用同一状态。

mod common;
#[path = "common/probe.rs"]
mod probe;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use fight_engine::config::SchemaConfig;
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
                    .format_tensors(&state, keys, &values, &current, None)
                    .unwrap();
                // 格式器实际使用完整 F64 canonical，在最后一步才转换 FP32。
                assert_eq!(tensor, pack_tensor_payload(&vector).unwrap());
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
                json!({"seconds":seconds,"gcd":gcd,"tensor":pack_tensor_payload(&vector).unwrap(),
                    "metadata":output_schema_metadata(&vector,&schema).unwrap(),"vector":vector,"retained":retained})
            }
            op => panic!("输出基线含未知操作: {op}"),
        };
        common::assert_equivalent(&step["output"], &actual, &format!("steps[{index}]"));
    }
}

#[test]
fn numeric_layouts_scene_schema_and_death_marker_match_baseline() {
    let baseline = fixture();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let schema = SchemaConfig::load(root).unwrap();
    common::assert_equivalent(
        &baseline["packing"]["output"],
        &pack_tensor_payload(&baseline["packing"]["input"]).unwrap(),
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

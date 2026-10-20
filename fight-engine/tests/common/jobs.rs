//! 职业 crate 共用的纯 Rust 基线重放器，不依赖任何具体职业。

use std::collections::BTreeMap;
use std::sync::Arc;

use fight_engine::config::{ProjectConfig, ResourceValue, SchemaConfig};
use fight_engine::contracts::JobRegistry;
use fight_engine::engine::simulator::Simulator;
use fight_engine::engine::{ActionRequest, EngineRules};
use fight_engine::state::{CombatState, CooldownState, ExternalCombatEvent, ModelStateFrame};
use serde_json::{Value, json};

use super::common;

fn project(baseline: &Value) -> ProjectConfig {
    let mut project: ProjectConfig = serde_json::from_value(baseline["project"].clone()).unwrap();
    for (definitions, order) in [
        (
            &mut project.system.statuses,
            &baseline["status_order"]["system"],
        ),
        (&mut project.job.statuses, &baseline["status_order"]["job"]),
    ] {
        let original = std::mem::take(definitions);
        for key in order.as_array().unwrap() {
            let key = key.as_str().unwrap();
            definitions.insert(key.into(), original[key].clone());
        }
    }
    project
}

fn initial_state(rules: &EngineRules, input: &Value) -> CombatState {
    let now = input["timestamp"].as_f64().unwrap();
    let mut state = rules.initial_state(None, None, now).unwrap();
    state.mp = input["mp"].as_i64().unwrap() as i32;
    state.base_gcd = input["base_gcd"].as_f64();
    state.gcd_ready_at = input["gcd_ready_at"].as_f64().unwrap_or(state.gcd_ready_at);
    state.cast_ends_at = input["cast_ends_at"].as_f64().unwrap_or(state.cast_ends_at);
    state.weave_ends_at = input["weave_ends_at"]
        .as_f64()
        .unwrap_or(state.weave_ends_at);
    state.is_moving = input["is_moving"].as_bool().unwrap();
    state.set_boss_targetable(input["boss_targetable"].as_bool().unwrap());
    state.target_count = input["target_count"].as_i64().unwrap() as i32;
    for (key, value) in input["resources"].as_object().unwrap() {
        rules
            .job()
            .runtime()
            .resources
            .set(
                &mut state,
                key,
                serde_json::from_value(value.clone()).unwrap(),
            )
            .unwrap();
    }
    for status in input["job_statuses"].as_array().unwrap() {
        rules
            .job()
            .runtime()
            .grant_status(
                &mut state,
                status["key"].as_str().unwrap(),
                status["remaining"].as_f64(),
                status["stacks"].as_i64().map(|value| value as i32),
            )
            .unwrap();
    }
    for status in input["system_statuses"].as_array().unwrap() {
        rules
            .system()
            .grant_status(
                &mut state,
                status["key"].as_str().unwrap(),
                status["remaining"].as_f64(),
                status["stacks"].as_i64().map(|value| value as i32),
            )
            .unwrap();
    }
    for (key, value) in input["job_deadlines"].as_object().unwrap() {
        state
            .job
            .timeline_deadlines
            .insert(key.clone(), value.as_f64().unwrap());
    }
    for (key, value) in input["cooldowns"].as_object().unwrap() {
        state.cooldowns.insert(
            key.clone(),
            CooldownState {
                available_charges: value["available_charges"].as_i64().unwrap() as i32,
                recharge_ready_at: value["recharge_ready_at"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_f64().unwrap())
                    .collect(),
            },
        );
    }
    state
}

// 仅在对比旧输出投影时回到其四位小数，运行状态与职业计算始终保留 F64。
fn legacy_resources(resources: &BTreeMap<String, ResourceValue>) -> Value {
    json!(
        resources
            .iter()
            .map(|(key, value)| {
                let value = match value {
                    ResourceValue::Float(value) => {
                        ResourceValue::Float((value * 10_000.0).round_ties_even() / 10_000.0)
                    }
                    value => *value,
                };
                (key, value)
            })
            .collect::<BTreeMap<_, _>>()
    )
}

fn frame(frame: &ModelStateFrame) -> Value {
    json!({"time":frame.state.player.time_seconds,"mp":frame.state.player.mp,
        "resources":legacy_resources(&frame.state.resources)})
}

fn state_projection(simulator: &Simulator, previous_cursor: u64) -> Value {
    let state = simulator.get_state();
    let history_delta = state.history.iter().filter(|entry| entry.history_sequence > previous_cursor)
        .map(|entry| json!({
            "skill_key":entry.skill_key,"history_sequence":entry.history_sequence,"time_seconds":entry.time_seconds,
            "request_timestamp":entry.request_timestamp,"cast_completed_timestamp":entry.cast_completed_timestamp,
            "effect_timestamp":entry.effect_timestamp,"gcd_index":entry.gcd_index,
            "mp_before":entry.mp_before,"mp_after":entry.mp_after,"potency":entry.potency,"value":entry.value,
            "cast_time_seconds":entry.cast_time_seconds,
            "next_cooldown_seconds":(entry.next_cooldown_seconds*10_000.0).round_ties_even()/10_000.0,
            "available_charges":entry.available_charges,"max_charges":entry.max_charges,
            "resources_before":legacy_resources(&entry.job_resources_before),
            "resources_after":legacy_resources(&entry.job_resources_after),
            "consumed":legacy_resources(&entry.job_resources_consumed),
            "model_request":entry.model_state.as_ref().map(|model| frame(&model.request_state)),
            "model_previous":entry.model_state.as_ref().map(|model| frame(&model.previous_action_after)),
        })).collect::<Vec<_>>();
    json!({
        "time":state.time(),"mp":state.mp,"gcd_index":state.gcd_index,
        "cast_ends_at":state.cast_ends_at,"gcd_ready_at":state.gcd_ready_at,
        "weave_ends_at":state.weave_ends_at,"ogcds_weaved":state.ogcds_weaved,
        "is_dead":state.is_dead,"is_moving":state.is_moving,"boss_targetable":state.boss_targetable(),
        "target_count":state.target_count,"cumulative_potency":state.cumulative_potency,
        "cumulative_dot_potency":state.cumulative_dot_potency,"current_potency":state.current_potency,
        "current_gcd_dot_potency":state.current_gcd_dot_potency,"resources":state.job.resources,
        "resource_snapshot":legacy_resources(&simulator.rules().job().runtime().resource_snapshot(&state).unwrap()),
        "job_deadlines":state.job.timeline_deadlines,"statuses":state.statuses,"job_statuses":state.job.statuses,
        "cooldowns":state.cooldowns,"dots":state.dots.iter().map(|(key,dot)| (key,json!({
            "expires_at":dot.expires_at,"next_tick_at":dot.next_tick_at,"tick_interval":dot.tick_interval,
            "potency_per_tick":dot.potency_per_tick,
        }))).collect::<BTreeMap<_,_>>(),
        "history_cursor":simulator.history_cursor(),"has_queued_action":simulator.has_queued_action(),
        "next_scheduled_event_time":simulator.next_scheduled_event_time(),"history_delta":history_delta,
    })
}

fn execute(simulator: &mut Simulator, input: &Value) -> Value {
    let time = input["timestamp"].as_f64().unwrap();
    match input["op"].as_str().unwrap() {
        "submit" => {
            if let Some(legal) = input["legal_keys"].as_array() {
                common::assert_equivalent(
                    &json!(legal),
                    &json!(simulator.available_action_keys_at(time).unwrap()),
                    "legal_keys",
                );
            }
            let result = simulator
                .submit_action(
                    ActionRequest {
                        timestamp: time,
                        skill_key: input["skill"].as_str().unwrap().into(),
                    },
                    input["cast_override"].as_f64(),
                )
                .unwrap();
            let mut result = serde_json::to_value(result).unwrap();
            let id = result
                .as_object_mut()
                .unwrap()
                .remove("action_instance_id")
                .unwrap();
            result["has_action_id"] = json!(!id.is_null());
            result
        }
        "advance" => {
            simulator.advance_to(time).unwrap();
            json!(true)
        }
        "validate" => json!(
            simulator
                .validate_action_at(time, input["skill"].as_str().unwrap())
                .unwrap()
        ),
        "external" => {
            let events: Vec<ExternalCombatEvent> =
                serde_json::from_value(input["events"].clone()).unwrap();
            json!(simulator.apply_external_events(&events).unwrap())
        }
        op => panic!("职业基线含未知操作: {op}"),
    }
}

pub fn run_fixture(baseline: &Value, registry: &JobRegistry, schema: Arc<SchemaConfig>) {
    for scene in baseline["scenarios"].as_array().unwrap() {
        let name = scene["name"].as_str().unwrap();
        let rules = Arc::new(
            EngineRules::from_project(
                Arc::new(project(baseline)),
                schema.clone(),
                registry,
                scene["max_history"].as_u64().map(|value| value as usize),
            )
            .unwrap(),
        );
        let mut registration = baseline["registration"].clone();
        for (key, maximum) in &rules.project().job.resource_limits {
            registration["resources"][key]["max_value"] = json!(maximum);
        }
        common::assert_equivalent(
            &registration["resources"],
            &json!(rules.job().runtime().resources.definitions()),
            "registration.resources",
        );
        common::assert_equivalent(
            &registration["statuses"],
            &json!(rules.job().runtime().statuses.definitions()),
            "registration.statuses",
        );
        let state = initial_state(&rules, &scene["initial"]);
        let mut simulator = Simulator::from_state(rules, state).unwrap();
        for (index, step) in scene["steps"].as_array().unwrap().iter().enumerate() {
            let previous = simulator.history_cursor();
            let result = execute(&mut simulator, &step["input"]);
            let path = format!("{name}.steps[{index}]");
            common::assert_equivalent(&step["result"], &result, &format!("{path}.result"));
            let actual_state = state_projection(&simulator, previous);
            common::assert_equivalent(&step["state"], &actual_state, &format!("{path}.state"));
        }
    }
}

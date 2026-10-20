//! 通过合成职业重放真实 C# 探针；输入与基线同时保存在 JSON 中。

mod common;
#[path = "common/probe.rs"]
mod probe;
use probe::{initial_state, rules};

use fight_engine::engine::ActionRequest;
use fight_engine::engine::simulator::{EnginePayload, Simulator};
use fight_engine::outputs::StateContextBuilder;
use fight_engine::state::{
    DotState, ExternalCombatEvent, ModelStateFrame, StateContext, StatusState,
};
use fight_engine::system::timeline::SimulationSnapshot;
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/engine.json")).unwrap()
}

fn context_projection(context: &StateContext) -> Value {
    json!({
        "time": context.player.time_seconds, "mp": context.player.mp,
        "gcd_index": context.player.gcd_index, "gcd_remaining": context.player.gcd_remaining_seconds,
        "cast_remaining": context.player.cast_remaining_seconds,
        "cumulative_potency": context.target.cumulative_potency, "resources": context.resources,
    })
}

fn frame(frame: Option<&ModelStateFrame>) -> Value {
    frame.map_or(Value::Null, |frame| {
        json!({"state": context_projection(&frame.state), "availability": frame.skill_availability})
    })
}

fn state_projection(simulator: &Simulator, previous_cursor: u64) -> Value {
    let state = simulator.get_state();
    let history_delta = state.history.iter().filter(|entry| entry.history_sequence>previous_cursor).map(|entry| json!({
        "skill_key":entry.skill_key,"history_sequence":entry.history_sequence,"time_seconds":entry.time_seconds,
        "request_timestamp":entry.request_timestamp,"cast_completed_timestamp":entry.cast_completed_timestamp,
        "effect_timestamp":entry.effect_timestamp,"gcd_index":entry.gcd_index,
        "mp_before":entry.mp_before,"mp_after":entry.mp_after,"potency":entry.potency,"value":entry.value,
        "cast_time_seconds":entry.cast_time_seconds,"cast_time_gcds":entry.cast_time_gcds,
        "gcd_window_seconds":entry.gcd_window_seconds,"gcd_window_gcds":entry.gcd_window_gcds,
        "next_cooldown_seconds":entry.next_cooldown_seconds,"available_charges":entry.available_charges,
        "max_charges":entry.max_charges,"resources_before":entry.job_resources_before,
        "resources_after":entry.job_resources_after,"consumed":entry.job_resources_consumed,
        "before":context_projection(&entry.state_before),"after":context_projection(&entry.state_after),
        "model_previous":frame(entry.model_state.as_ref().map(|model| model.previous_action_after.as_ref())),
        "model_request":frame(entry.model_state.as_ref().map(|model| model.request_state.as_ref())),
    })).collect::<Vec<_>>();
    json!({
        "time": state.time(), "mp": state.mp, "gcd_index": state.gcd_index,
        "cast_ends_at": state.cast_ends_at, "gcd_ready_at": state.gcd_ready_at,
        "weave_ends_at": state.weave_ends_at, "ogcds_weaved": state.ogcds_weaved,
        "is_dead": state.is_dead, "boss_targetable": state.boss_targetable(),
        "is_moving": state.is_moving, "target_count": state.target_count,
        "cumulative_potency": state.cumulative_potency, "current_potency": state.current_potency,
        "cumulative_dot_potency": state.cumulative_dot_potency,
        "current_gcd_dot_potency": state.current_gcd_dot_potency,
        "resources": state.job.resources, "statuses": state.statuses, "job_statuses": state.job.statuses,
        "cooldowns": state.cooldowns,
        "dots": state.dots.iter().map(|(key,dot)| (key,json!({
            "expires_at": dot.expires_at,"next_tick_at": dot.next_tick_at,"potency_per_tick": dot.potency_per_tick,
        }))).collect::<BTreeMap<_,_>>(),
        "history_cursor": simulator.history_cursor(), "has_queued_action": simulator.has_queued_action(),
        "next_scheduled_event_time": simulator.next_scheduled_event_time(),
        "pending": simulator.create_snapshot().pending_events.iter().map(|event| json!({
            "time":event.timestamp,"kind":format!("{:?}",event.kind),"owner":event.owner_key,
        })).collect::<Vec<_>>(),
        "has_last_decision_id": state.last_decision_id.is_some(),
        "last_decision_after": frame(state.last_decision_after.as_deref()),
        "last_completed_after": frame(state.last_completed_decision_after.as_deref()),
        "history": state.history.iter().map(|entry| json!({
            "skill_key":entry.skill_key,"history_sequence":entry.history_sequence,"time_seconds":entry.time_seconds,
        })).collect::<Vec<_>>(),
        "history_delta": history_delta,
    })
}

fn execute(
    simulator: &mut Simulator,
    snapshots: &mut BTreeMap<String, SimulationSnapshot<EnginePayload>>,
    input: &Value,
) -> Result<Value, String> {
    let timestamp = || input["timestamp"].as_f64().unwrap();
    let skill = || input["skill"].as_str().unwrap();
    match input["op"].as_str().unwrap() {
        "submit" => {
            let result = simulator.submit_action(
                ActionRequest {
                    timestamp: timestamp(),
                    skill_key: skill().into(),
                },
                input["cast_override"].as_f64(),
            )?;
            let mut projection = serde_json::to_value(&result).unwrap();
            projection
                .as_object_mut()
                .unwrap()
                .remove("action_instance_id");
            projection["has_action_id"] = json!(result.action_instance_id.is_some());
            Ok(projection)
        }
        "advance" => {
            simulator.advance_to(timestamp())?;
            Ok(json!(true))
        }
        "validate" => Ok(json!(simulator.validate_action_at(timestamp(), skill())?)),
        "available" => Ok(json!(simulator.available_action_keys_at(timestamp())?)),
        "external" => {
            let events: Vec<ExternalCombatEvent> =
                serde_json::from_value(input["events"].clone()).unwrap();
            Ok(json!(simulator.apply_external_events(&events)?))
        }
        "snapshot" => {
            snapshots.insert(
                input["snapshot"].as_str().unwrap().into(),
                simulator.create_snapshot(),
            );
            Ok(json!(true))
        }
        "restore" => {
            simulator.restore_snapshot(&snapshots[input["snapshot"].as_str().unwrap()])?;
            Ok(json!(true))
        }
        _ => panic!("fixture 包含未知操作"),
    }
}

#[test]
fn action_lifecycles_queues_history_and_snapshots_match_csharp() {
    let baseline = fixture();
    for scene in baseline["scenarios"].as_array().unwrap() {
        let name = scene["name"].as_str().unwrap();
        let rules = rules(
            &baseline,
            scene["max_history"].as_u64().map(|value| value as usize),
        );
        assert!(
            rules
                .system()
                .registered_target_dot_keys()
                .contains("probe_dot")
        );
        let initial = initial_state(&rules, &scene["initial"]);
        let mut simulators = BTreeMap::from([(
            "main".to_owned(),
            Simulator::from_state(rules, initial).unwrap(),
        )]);
        let mut snapshots = BTreeMap::new();
        for (index, step) in scene["steps"].as_array().unwrap().iter().enumerate() {
            let input = &step["input"];
            let branch = input["branch"].as_str().unwrap_or("main");
            let fork = input["op"] == "fork";
            let previous_cursor = simulators[if fork { "main" } else { branch }].history_cursor();
            let result = if fork {
                let fork = simulators["main"].fork();
                simulators.insert(branch.into(), fork);
                json!(true)
            } else {
                execute(simulators.get_mut(branch).unwrap(), &mut snapshots, input)
                    .unwrap_or_else(|_| json!({"error":true}))
            };
            let path = format!("{name}.steps[{index}]");
            common::assert_equivalent(&step["result"], &result, &format!("{path}.result"));
            common::assert_equivalent(
                &step["state"],
                &state_projection(&simulators[branch], previous_cursor),
                &format!("{path}.state"),
            );
        }
    }
}

#[test]
fn context_boundaries_preserve_order_and_at_least_csharp_precision() {
    let baseline = fixture();
    let rules = rules(&baseline, None);
    let builder = StateContextBuilder::new(
        rules.project().clone(),
        rules.system().clone(),
        rules.job().clone(),
    );
    for case in baseline["contexts"].as_array().unwrap() {
        let input = &case["input"];
        let now = input["timestamp"].as_f64().unwrap();
        let mut state = rules
            .initial_state(None, Some(input["max_mp"].as_i64().unwrap() as i32), now)
            .unwrap();
        state.mp = input["mp"].as_i64().unwrap() as i32;
        state.base_gcd = input["base_gcd"].as_f64();
        state.gcd_ready_at = input["gcd_ready_at"].as_f64().unwrap();
        state.cast_ends_at = input["cast_ends_at"].as_f64().unwrap();
        state.weave_ends_at = input["weave_ends_at"].as_f64().unwrap();
        state.next_downtime_starts_at = input["next_downtime"].as_f64();
        state.set_downtime_remaining(input["downtime_remaining"].as_f64().unwrap());
        state.cumulative_potency = input["target"]["cumulative_potency"].as_f64().unwrap();
        state.current_potency = input["target"]["current_potency"].as_f64().unwrap();
        state.cumulative_dot_potency = input["target"]["cumulative_dot_potency"].as_f64().unwrap();
        state.current_gcd_dot_potency =
            input["target"]["current_gcd_dot_potency"].as_f64().unwrap();
        for (key, value) in input["job_deadlines"].as_object().unwrap() {
            state
                .job
                .timeline_deadlines
                .insert(key.clone(), value.as_f64().unwrap());
        }
        for status in input["statuses"].as_array().unwrap() {
            let statuses = if status["scope"] == "system" {
                &mut state.statuses
            } else {
                &mut state.job.statuses
            };
            statuses.insert(
                status["key"].as_str().unwrap().into(),
                StatusState::new(
                    now,
                    status["remaining"].as_f64().unwrap(),
                    status["stacks"].as_i64().unwrap() as i32,
                ),
            );
        }
        for dot in input["dots"].as_array().unwrap() {
            state.dots.insert(
                dot["key"].as_str().unwrap().into(),
                DotState::new(
                    now,
                    dot["remaining"].as_f64().unwrap(),
                    dot["potency_per_tick"].as_f64().unwrap(),
                    dot["tick_interval"].as_f64().unwrap(),
                    3.0,
                ),
            );
        }
        let mut actual = serde_json::to_value(builder.build(&state).unwrap()).unwrap();
        // C# 的这些输出原先 Round4；Rust 保留 F64，只在比较投影时回到旧精度。
        for value in actual["target"].as_object_mut().unwrap().values_mut() {
            *value = json!((value.as_f64().unwrap() * 10_000.0).round_ties_even() / 10_000.0);
        }
        for (group, fields) in [
            ("buffs", &["remaining_seconds", "remaining_gcds"][..]),
            (
                "dots",
                &[
                    "remaining_seconds",
                    "remaining_gcds",
                    "tick_interval_seconds",
                ][..],
            ),
        ] {
            for item in actual[group].as_array_mut().unwrap() {
                for field in fields {
                    let value = item[field].as_f64().unwrap();
                    item[field] = json!((value * 10_000.0).round_ties_even() / 10_000.0);
                }
            }
        }
        common::assert_equivalent(&case["result"], &actual, input["name"].as_str().unwrap());
    }
}

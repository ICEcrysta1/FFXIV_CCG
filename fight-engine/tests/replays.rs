// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

//! 真实 C# 回放门面对照：绝对请求、固定随机流、完整快照和独立队列。

mod common;
#[path = "common/probe.rs"]
mod probe;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use fight_engine::config::{PolicyActionRegistry, ResourceValue, SkillBook};
use fight_engine::engine::ActionRequest;
use fight_engine::engine::replay::{ReplayStateCache, run_action_sequence, run_random_sequence};
use fight_engine::engine::sessions::{SimulationEngine, SimulationSession};
use fight_engine::engine::simulator::{ActionSubmissionResult, EnginePayload, Simulator};
use fight_engine::state::CombatState;
use fight_engine::system::timeline::SimulationSnapshot;
use serde_json::{Value, json};

type Snapshot = SimulationSnapshot<EnginePayload>;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/replays.json")).unwrap()
}

fn state(state: &CombatState) -> Value {
    let statuses = |items: &indexmap::IndexMap<String, fight_engine::state::StatusState>| {
        items
            .iter()
            .map(|(key, value)| {
                (
                    key.clone(),
                    json!({"expires_at":value.expires_at,"stacks":value.stacks}),
                )
            })
            .collect::<BTreeMap<_, _>>()
    };
    json!({
        "time":state.time(),"mp":state.mp,"gcd_index":state.gcd_index,
        "cast_ends_at":state.cast_ends_at,"gcd_ready_at":state.gcd_ready_at,
        "weave_ends_at":state.weave_ends_at,"ogcds_weaved":state.ogcds_weaved,
        "cumulative_potency":state.cumulative_potency,"current_potency":state.current_potency,
        "cumulative_dot_potency":state.cumulative_dot_potency,"current_gcd_dot_potency":state.current_gcd_dot_potency,
        "resources":state.job.resources,"history_cursor":state.last_history_sequence,
        "statuses":statuses(&state.statuses),"job_statuses":statuses(&state.job.statuses),
        "cooldowns":state.cooldowns.iter().map(|(key,value)|(key.clone(),json!({"available_charges":value.available_charges,"recharge_ready_at":value.recharge_ready_at}))).collect::<BTreeMap<_,_>>(),
        "dots":state.dots.iter().map(|(key,value)|(key.clone(),json!({"expires_at":value.expires_at,"next_tick_at":value.next_tick_at,"potency_per_tick":value.potency_per_tick}))).collect::<BTreeMap<_,_>>(),
        "has_last_decision_id":state.last_decision_id.is_some(),
        "last_decision_in_history":state.history.iter().any(|entry|entry.action_instance_id==state.last_decision_id),
        "has_last_decision_after":state.last_decision_after.is_some(),
        "has_last_completed_after":state.last_completed_decision_after.is_some(),
        "history":state.history.iter().map(|entry|json!({"skill_key":entry.skill_key,"history_sequence":entry.history_sequence,
            "request_timestamp":entry.request_timestamp,"cast_completed_timestamp":entry.cast_completed_timestamp,
            "effect_timestamp":entry.effect_timestamp,"mp_before":entry.mp_before,"mp_after":entry.mp_after,"potency":entry.potency,
            "has_action_id":entry.action_instance_id.is_some(),"resources_before":entry.job_resources_before,
            "resources_after":entry.job_resources_after,"consumed":entry.job_resources_consumed})).collect::<Vec<_>>()
    })
}

fn submission(result: &ActionSubmissionResult, state: &CombatState) -> Value {
    json!({"accepted":result.accepted,"queued":result.queued,"reason":result.reason,
        "request_timestamp":result.request_timestamp,"accepted_timestamp":result.accepted_timestamp,
        "effect_timestamp":result.effect_timestamp,"next_scheduled_event_time":result.next_scheduled_event_time,
        "has_action_id":result.action_instance_id.is_some(),
        "id_is_last_decision":result.action_instance_id.is_some()&&result.action_instance_id==state.last_decision_id,
        "id_in_history":result.action_instance_id.is_some()&&state.history.iter().any(|entry|entry.action_instance_id==result.action_instance_id)})
}

fn snapshot(snapshot: &Snapshot) -> Value {
    json!({"state":state(&snapshot.state),"next_sequence":snapshot.next_sequence,
        "pending":snapshot.pending_events.iter().map(|event| {
            let request_state=match &event.payload {
                Some(EnginePayload::Action(payload))=>json!({"mp":payload.request_state.mp,"request_timestamp":payload.request.timestamp,"gauge":payload.request_state.job.resources["gauge"]}),
                _=>Value::Null,
            };
            json!({"timestamp":event.timestamp,"kind":format!("{:?}",event.kind),"owner_key":event.owner_key,
                "has_action_id":event.action_instance_id.is_some(),
                "id_is_last_decision":event.action_instance_id.is_some()&&event.action_instance_id==snapshot.state.last_decision_id,
                "request_state":request_state})
        }).collect::<Vec<_>>(),"pending_settlement_count":snapshot.pending_settlements.len()})
}

fn session_state(session: &SimulationSession) -> Value {
    let stats = session.statistics().unwrap();
    let observed = session
        .observe_at(stats.timestamp, "seconds", None)
        .unwrap();
    json!({"stats":{"timestamp":stats.timestamp,"action_history_count":stats.action_history_count,
        "policy_history_count":stats.policy_history_count,"pending_event_count":stats.pending_event_count,
        "pending_settlement_count":stats.pending_settlement_count},"seconds":observed.value,
        "next_scheduled_event_time":observed.next_scheduled_event_time})
}

fn create_engine(baseline: &Value, capacity: usize) -> Arc<SimulationEngine> {
    let rules = probe::rules(baseline, None);
    let book = SkillBook::from_project_config(rules.project()).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let registry = Arc::new(PolicyActionRegistry::load(root, &book).unwrap());
    SimulationEngine::new(rules, registry, capacity).unwrap()
}

// 绝对时钟、截止时刻和请求时刻逐值相等；不以时间容差放宽派发结果。
fn assert_parity(expected: &Value, actual: &Value, path: &str) {
    match (expected, actual) {
        (Value::Object(expected), Value::Object(actual)) => {
            assert_eq!(expected.len(), actual.len(), "{path}: 字段数量不同");
            for (key, value) in expected {
                assert_parity(value, &actual[key], &format!("{path}.{key}"));
            }
        }
        (Value::Array(expected), Value::Array(actual)) => {
            assert_eq!(expected.len(), actual.len(), "{path}: 数组长度不同");
            for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                assert_parity(expected, actual, &format!("{path}[{index}]"));
            }
        }
        (Value::Number(expected), Value::Number(actual))
            if path.contains("time") || path.contains("_at") =>
        {
            assert_eq!(expected.as_f64(), actual.as_f64(), "{path}: 时刻不同");
        }
        _ => common::assert_equivalent(expected, actual, path),
    }
}

fn normalize_sequence(mut output: Value) -> Value {
    let mut ids = BTreeSet::new();
    for row in output["applied_actions"].as_array_mut().unwrap() {
        let id = row
            .as_object_mut()
            .unwrap()
            .remove("action_instance_id")
            .unwrap();
        row["has_action_id"] = json!(id.as_u64().is_some_and(|id| id > 0));
        assert!(ids.insert(id.as_u64().unwrap()), "动作实例 ID 重复");
        for key in [
            "request_time_seconds",
            "accepted_time_seconds",
            "effect_time_seconds",
        ] {
            let value = row[key].as_f64().unwrap();
            row[key] = json!((value * 10_000.0).round_ties_even() / 10_000.0);
        }
    }
    output["action_ids_unique"] = json!(true);
    common::legacy_skill_state(output)
}

#[test]
fn absolute_sequences_and_seeded_random_traces_match_csharp() {
    let baseline = fixture();
    assert_eq!(baseline["source_commit"], "9ad36ad");
    for case in baseline["sequences"].as_array().unwrap() {
        let engine = create_engine(&baseline, 1);
        let session = engine
            .create_session(serde_json::from_value(case["session_options"].clone()).unwrap())
            .unwrap();
        let requests: Vec<ActionRequest> =
            serde_json::from_value(case["requests"].clone()).unwrap();
        let output = run_action_sequence(&session, &requests)
            .map(|output| {
                if case["name"] == "precise_request_metadata" {
                    for (row, request) in output["applied_actions"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .zip(&requests)
                    {
                        for key in [
                            "request_time_seconds",
                            "accepted_time_seconds",
                            "effect_time_seconds",
                        ] {
                            assert_eq!(
                                row[key].as_f64(),
                                Some(request.timestamp),
                                "{key}: 原始 F64 精度丢失"
                            );
                        }
                    }
                }
                normalize_sequence(output)
            })
            .unwrap_or_else(|_| json!({"error":true}));
        let path = case["name"].as_str().unwrap();
        assert_parity(&case["output"], &output, path);
        assert_parity(
            &case["state"],
            &session_state(&session),
            &format!("{path}.state"),
        );
    }
    for case in baseline["random"].as_array().unwrap() {
        let engine = create_engine(&baseline, 1);
        let session = engine
            .create_session(serde_json::from_value(case["session_options"].clone()).unwrap())
            .unwrap();
        let steps = run_random_sequence(
            &session,
            case["seed"].as_i64().unwrap(),
            case["max_steps"].as_i64().unwrap() as i32,
        )
        .unwrap();
        let output = steps.iter().map(|step|json!({"kind":step.kind,"step_index":step.step_index,"legal_keys":step.legal_keys,
            "chosen_index":step.chosen_index,"action":step.action,"timestamp":step.timestamp,
            "result":step.result.as_ref().map(|result|submission(result,&step.state_after)),"state_after":state(&step.state_after)})).collect::<Vec<_>>();
        let path = case["name"].as_str().unwrap();
        assert_parity(&case["steps"], &json!(output), path);
        assert_parity(
            &case["state"],
            &session_state(&session),
            &format!("{path}.state"),
        );
    }
}

fn mutate(snapshot: &mut Snapshot) {
    snapshot.state.mp = -321;
    snapshot
        .state
        .job
        .resources
        .insert("gauge".into(), ResourceValue::Integer(-5));
    if let Some(EnginePayload::Action(payload)) = snapshot
        .pending_events
        .iter_mut()
        .find_map(|event| {
            matches!(event.payload, Some(EnginePayload::Action(_))).then_some(&mut event.payload)
        })
        .and_then(Option::as_mut)
    {
        payload.request_state.mp = -654;
        payload
            .request_state
            .job
            .resources
            .insert("gauge".into(), ResourceValue::Integer(-6));
    }
}

fn cache_operation(
    cache: &mut ReplayStateCache,
    source: &mut Simulator,
    saved: &mut BTreeMap<String, Snapshot>,
    returned: &mut BTreeMap<String, Snapshot>,
    input: &Value,
) -> Result<Value, String> {
    Ok(match input["op"].as_str().unwrap() {
        "snapshot" => {
            let item = cache.push_snapshot();
            let name = input["name"].as_str().unwrap().to_owned();
            saved.insert(name.clone(), item.clone());
            let output = snapshot(&item);
            returned.insert(name, item);
            output
        }
        "mutate_returned_snapshot" => {
            mutate(returned.get_mut(input["name"].as_str().unwrap()).unwrap());
            json!(true)
        }
        "mutate_snapshot_at" => {
            mutate(&mut cache.snapshot_at(input["index"].as_u64().unwrap() as usize)?);
            json!(true)
        }
        "snapshot_at" => snapshot(&cache.snapshot_at(input["index"].as_u64().unwrap() as usize)?),
        "mutate_current_state" => {
            let mut state = cache.current_state();
            state.mp = -123;
            state
                .job
                .resources
                .insert("gauge".into(), ResourceValue::Integer(-1));
            json!(true)
        }
        "source_advance" => {
            source.advance_to(input["timestamp"].as_f64().unwrap())?;
            snapshot(&source.create_snapshot())
        }
        "advance" => state(&cache.advance_to(
            input["timestamp"].as_f64().unwrap(),
            input["snapshot"].as_bool().unwrap(),
        )?),
        "submit" => {
            let result = cache.submit_action(
                ActionRequest {
                    timestamp: input["timestamp"].as_f64().unwrap(),
                    skill_key: input["skill_key"].as_str().unwrap().into(),
                },
                input["snapshot"].as_bool().unwrap(),
            )?;
            submission(&result, &cache.current_state())
        }
        "restore" => {
            cache.restore_snapshot(&saved[input["name"].as_str().unwrap()])?;
            json!(true)
        }
        "mutate_restored_snapshot" => {
            mutate(saved.get_mut(input["name"].as_str().unwrap()).unwrap());
            json!(true)
        }
        other => return Err(format!("unknown fixture operation {other}")),
    })
}

#[test]
fn replay_cache_snapshots_are_isolated_and_restore_queued_lifecycle() {
    let baseline = fixture();
    let case = &baseline["cache"];
    let rules = probe::rules(&baseline, None);
    let mut source = Simulator::from_state(
        rules.clone(),
        probe::initial_state(&rules, &case["initial"]),
    )
    .unwrap();
    for request in serde_json::from_value::<Vec<ActionRequest>>(case["prepare"].clone()).unwrap() {
        assert!(source.submit_action(request, None).unwrap().accepted);
    }
    let mut cache = ReplayStateCache::new(&source);
    let mut saved = BTreeMap::new();
    let mut returned = BTreeMap::new();
    for (index, step) in case["steps"].as_array().unwrap().iter().enumerate() {
        let output = cache_operation(
            &mut cache,
            &mut source,
            &mut saved,
            &mut returned,
            &step["input"],
        )
        .unwrap_or_else(|_| json!({"error":true}));
        assert_parity(&step["output"], &output, &format!("cache[{index}].output"));
        assert_parity(
            &step["state"],
            &state(&cache.current_state()),
            &format!("cache[{index}].state"),
        );
    }
    assert_parity(
        &case["source_final"],
        &snapshot(&source.create_snapshot()),
        "cache.source_final",
    );
}

#[test]
fn parallel_replays_keep_sessions_independent() {
    let baseline = fixture();
    let expected = &baseline["sessions"];
    let engine = create_engine(&baseline, 2);
    let sessions = expected["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|options| {
            engine
                .create_session(serde_json::from_value(options.clone()).unwrap())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let workers = expected["runs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|input| {
            let session = sessions[input["queue"].as_u64().unwrap() as usize].clone();
            let seed = input["seed"].as_i64().unwrap();
            let count = input["max_steps"].as_i64().unwrap() as i32;
            std::thread::spawn(move || run_random_sequence(&session, seed, count).unwrap())
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(
        expected["ids_distinct"],
        json!(sessions[0].id() != sessions[1].id())
    );
    let actual = sessions
        .iter()
        .map(|session| session_state(session))
        .collect::<Vec<_>>();
    assert_parity(&expected["states"], &json!(actual), "sessions.parallel");
}

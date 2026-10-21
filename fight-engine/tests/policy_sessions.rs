// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

//! 真实 C# 基线覆盖策略游标、增量回放和独立队列生命周期。
mod common;
#[path = "common/probe.rs"]
mod probe;

use fight_engine::config::{PolicyActionDefinition, PolicyActionRegistry, SkillBook};
use fight_engine::engine::ActionRequest;
use fight_engine::engine::sessions::{SessionOptions, SimulationEngine, SimulationSession};
use fight_engine::engine::simulator::Simulator;
use fight_engine::policy::{PolicyDecision, PolicySession, replay};
use serde_json::{Value, json};
use std::sync::Arc;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/policy_sessions.json")).unwrap()
}
fn registry(baseline: &Value, extended: bool) -> Arc<PolicyActionRegistry> {
    let project = serde_json::from_value(baseline["project"].clone()).unwrap();
    let book = SkillBook::from_project_config(&project).unwrap();
    let definitions: Vec<PolicyActionDefinition> = serde_json::from_value(
        baseline[if extended {
            "actions"
        } else {
            "default_actions"
        }]
        .clone(),
    )
    .unwrap();
    Arc::new(PolicyActionRegistry::new(definitions, &book).unwrap())
}
fn summary(sim: &Simulator, policy: &PolicySession) -> Value {
    let state = sim.get_state();
    json!({"time":sim.time(),"mp":state.mp,"history_cursor":sim.history_cursor(),
        "action_history_count":state.history.len(),"policy_history_count":policy.history_count(),
        "cumulative_potency":state.cumulative_potency,"cumulative_dot_potency":state.cumulative_dot_potency})
}
fn operation(
    sim: &mut Simulator,
    policy: &mut PolicySession,
    saved: &mut Vec<PolicyDecision>,
    input: &Value,
) -> Result<Value, String> {
    Ok(match input["op"].as_str().unwrap() {
        "submit" => {
            let r = sim.submit_action(
                ActionRequest {
                    timestamp: input["timestamp"].as_f64().unwrap(),
                    skill_key: input["skill"].as_str().unwrap().into(),
                },
                input["cast_override"].as_f64(),
            )?;
            json!({"accepted":r.accepted,"queued":r.queued,"reason":r.reason,"accepted_timestamp":r.accepted_timestamp,"effect_timestamp":r.effect_timestamp})
        }
        "advance" => {
            sim.advance_to(input["timestamp"].as_f64().unwrap())?;
            json!(true)
        }
        "record" => {
            let r = policy.record(
                sim,
                input["timestamp"].as_f64().unwrap(),
                input["key"].as_str().unwrap(),
                input["next"].as_f64().unwrap(),
            )?;
            json!({"history_sequence":r.history_sequence,"timestamp":r.timestamp,"gcd_index":r.gcd_index,"before_mp":r.state_before.mp,"after_mp":r.state_after.mp,
                "before_time":r.state_before.time(),"after_time":r.state_after.time(),"before_history":r.state_before.history.len(),"after_history":r.state_after.history.len()})
        }
        "capture" => {
            if let Some(after) = input["after"].as_u64() {
                policy.build_vector_delta_context(sim, input["next"].as_f64().unwrap(), after)?
            } else {
                policy.build_vector_context(sim, input["next"].as_f64().unwrap())?
            }
        }
        "snapshot" => {
            *saved = policy.create_snapshot();
            json!(true)
        }
        "restore" => {
            policy.restore_snapshot(saved);
            json!(true)
        }
        "mutate_snapshot" => {
            let mut changed = saved.clone();
            let mut frame = changed[0].model_state.request_state.as_ref().clone();
            frame.state.player.mp = 123;
            changed[0].model_state.request_state = Arc::new(frame);
            policy.restore_snapshot(&changed);
            json!(true)
        }
        "fork_pair" => {
            let mut fork_sim = sim.fork();
            let mut fork_policy = policy.fork();
            let time = fork_sim.time();
            fork_policy.record(&mut fork_sim, time, "ogcd_wait", 6.0)?;
            json!({"original":policy.build_vector_context(sim,6.0)?,"fork":fork_policy.build_vector_context(&fork_sim,6.0)?,"fork_state":summary(&fork_sim,&fork_policy)})
        }
        "pair_restore_record" => {
            let paired_sim = sim.create_snapshot();
            let paired_policy = policy.create_snapshot();
            let time = sim.time();
            policy.record(sim, time, "ogcd_wait", 6.0)?;
            let first = policy.build_vector_context(sim, 6.0)?;
            sim.restore_snapshot(&paired_sim)?;
            policy.restore_snapshot(&paired_policy);
            policy.record(sim, time, "ogcd_wait", 6.0)?;
            json!({"first":first,"restored":policy.build_vector_context(sim,6.0)?})
        }
        "corrupt" => {
            let original = policy.create_snapshot();
            let bad = match input["mode"].as_str().unwrap() {
                "missing" => original[1..].to_vec(),
                "duplicate" => {
                    let mut bad = original.clone();
                    bad.push(original[0].clone());
                    bad
                }
                _ => {
                    let mut bad = original.clone();
                    bad[0].history_sequence = sim.history_cursor() + 1;
                    bad
                }
            };
            policy.restore_snapshot(&bad);
            let rejected = policy.build_vector_delta_context(sim, 100.0, 0).is_err();
            policy.restore_snapshot(&original);
            json!({"error":rejected})
        }
        other => return Err(format!("unknown fixture operation {other}")),
    })
}

#[test]
fn policy_wait_history_delta_and_snapshot_match_baseline() {
    let baseline = fixture();
    for scene in baseline["scenarios"].as_array().unwrap() {
        let limit = scene["max_history"].as_u64().map(|n| n as usize);
        let rules = probe::rules(&baseline, limit);
        let mut sim = Simulator::from_state(
            rules.clone(),
            probe::initial_state(&rules, &scene["initial"]),
        )
        .unwrap();
        let mut policy = PolicySession::new(
            registry(&baseline, scene["extended"].as_bool().unwrap()),
            limit,
        );
        let mut saved = Vec::new();
        for (index, step) in scene["steps"].as_array().unwrap().iter().enumerate() {
            let path = format!("{}.steps[{index}]", scene["name"].as_str().unwrap());
            let output = operation(&mut sim, &mut policy, &mut saved, &step["input"])
                .unwrap_or_else(|_| json!({"error":true}));
            common::assert_equivalent(&step["output"], &output, &format!("{path}.output"));
            common::assert_equivalent(
                &step["state"],
                &summary(&sim, &policy),
                &format!("{path}.state"),
            );
        }
    }
}

#[test]
fn policy_record_rejects_invalid_times_without_mutation() {
    let baseline = fixture();
    for limit in [None, Some(0), Some(1)] {
        let rules = probe::rules(&baseline, limit);
        let mut sim = Simulator::new(rules, -2.0, None).unwrap();
        let mut policy = PolicySession::new(registry(&baseline, true), limit);
        let first = policy.record(&mut sim, -2.0, "ogcd_wait", -1.0).unwrap();
        let simulator_before = sim.create_snapshot();
        let policy_before = policy.create_snapshot();
        for (timestamp, next) in [
            (f64::NAN, -1.0),
            (f64::INFINITY, -1.0),
            (f64::NEG_INFINITY, -1.0),
            (-2.0, f64::NAN),
            (-2.0, f64::INFINITY),
            (-2.0, f64::NEG_INFINITY),
            (-3.0, -1.0),
            (-2.0, -3.0),
        ] {
            assert!(
                policy
                    .record(&mut sim, timestamp, "ogcd_wait", next)
                    .is_err(),
                "timestamp={timestamp}, next={next}, limit={limit:?}"
            );
            assert_eq!(simulator_before, sim.create_snapshot());
            assert_eq!(policy_before, policy.create_snapshot());
        }
        let second = policy.record(&mut sim, -2.0, "ogcd_wait", -2.0).unwrap();
        assert_eq!(first.history_sequence + 1, second.history_sequence);
    }
}

#[test]
fn policy_replay_preserves_external_batch_and_wait_order() {
    let baseline = fixture();
    for scene in baseline["replays"].as_array().unwrap() {
        let rules = probe::rules(&baseline, None);
        let mut sim = Simulator::from_state(
            rules.clone(),
            probe::initial_state(&rules, &scene["initial"]),
        )
        .unwrap();
        let mut policy = PolicySession::new(registry(&baseline, true), None);
        let output = replay::run(
            &mut sim,
            &mut policy,
            &scene["requests"].to_string(),
            &scene["facts"].to_string(),
            scene["fight_duration"].as_f64().unwrap(),
            scene["policy_key"].as_str().unwrap(),
        )
        .unwrap();
        let path = scene["name"].as_str().unwrap();
        common::assert_equivalent(&scene["output"], &output, path);
        common::assert_equivalent(
            &scene["state"],
            &summary(&sim, &policy),
            &format!("{path}.state"),
        );
    }
    for (kind, expected) in baseline["replay_errors"].as_object().unwrap() {
        let limit = (kind == "bounded").then_some(1);
        let rules = probe::rules(&baseline, limit);
        let mut sim = Simulator::new(rules, 0.0, None).unwrap();
        let mut policy = PolicySession::new(registry(&baseline, true), limit);
        let requests = match kind.as_str() {
            "empty" => json!([]),
            "negative_cast" => {
                json!([{"timestamp":0,"skill_key":"instant","actual_cast_seconds":-1}])
            }
            _ => json!([{"timestamp":0,"skill_key":"instant"}]),
        };
        let facts = if kind == "unsorted_facts" {
            json!([{"timestamp":2,"event_kind":"movement_changed","value":true},{"timestamp":1,"event_kind":"movement_changed","value":false}])
        } else {
            json!([])
        };
        assert_eq!(
            expected.as_bool().unwrap(),
            replay::run(
                &mut sim,
                &mut policy,
                &requests.to_string(),
                &facts.to_string(),
                20.0,
                "ogcd_wait"
            )
            .is_err(),
            "{kind}"
        );
    }
}

fn projection(session: &SimulationSession) -> Value {
    let mut stats = serde_json::to_value(session.statistics().unwrap()).unwrap();
    stats.as_object_mut().unwrap().remove("queue_entry_count");
    let seconds = session
        .observe_at(stats["timestamp"].as_f64().unwrap(), "seconds", None)
        .unwrap()
        .value;
    json!({"seconds":seconds,"stats":stats})
}
fn options(value: &Value) -> SessionOptions {
    serde_json::from_value(value.clone()).unwrap()
}

#[test]
fn independent_sessions_reset_close_and_parallel_execution_match_baseline() {
    let baseline = fixture();
    let expected = &baseline["sessions"];
    let rules = probe::rules(&baseline, None);
    let registry = registry(&baseline, true);
    let engine = SimulationEngine::new(rules.clone(), registry.clone(), 2).unwrap();
    let a = engine
        .create_session(options(&expected["options"][0]))
        .unwrap();
    let b = engine
        .create_session(options(&expected["options"][1]))
        .unwrap();
    let initial = json!({"capacity":engine.capacity(),"job_tag":engine.job_tag(),"active_count":engine.active_count().unwrap(),"ids":[a.id(),b.id()],"states":[projection(&a),projection(&b)]});
    common::assert_equivalent(&expected["initial"], &initial, "sessions.initial");
    assert_eq!(
        expected["capacity_error"],
        json!(engine.create_session(SessionOptions::default()).is_err())
    );
    a.submit_action(100.0, "instant", None).unwrap();
    a.record_policy_action(100.0, "ogcd_wait", 102.0).unwrap();
    let invalid_reset = a
        .reset(SessionOptions {
            actual_base_gcd: Some(0.0),
            ..Default::default()
        })
        .is_err();
    assert_eq!(expected["invalid_reset"], json!(invalid_reset));
    common::assert_equivalent(
        &expected["after_invalid_reset"],
        &projection(&a),
        "sessions.after_invalid_reset",
    );
    let validation = json!({"bad_observe":a.observe_at(105.0,"bad",None).is_err(),"bad_vector_next":a.observe_at(105.0,"vector",Some(104.0)).is_err(),
        "bounded_delta":a.observe_vector_delta_at(105.0,106.0,0).is_err(),"bad_cast":a.submit_action(105.0,"cast",Some(-1.0)).is_err(),"time_after":a.statistics().unwrap().timestamp});
    common::assert_equivalent(
        &expected["validation_errors"],
        &validation,
        "sessions.validation_errors",
    );
    let reset = a.reset(options(&expected["reset_options"])).unwrap();
    common::assert_equivalent(
        &expected["reset"],
        &serde_json::to_value(reset).unwrap(),
        "sessions.reset",
    );
    common::assert_equivalent(
        &expected["after_reset"],
        &projection(&a),
        "sessions.after_reset",
    );
    let delta = a.observe_vector_delta_at(10.0, 12.25, 0).unwrap();
    common::assert_equivalent(
        &expected["initial_delta"],
        &serde_json::to_value(delta).unwrap(),
        "sessions.initial_delta",
    );
    let workers = expected["parallel_inputs"]
        .as_array()
        .unwrap()
        .iter()
        .zip([a.clone(), b.clone()])
        .map(|(input, session)| {
            let start = input["start"].as_f64().unwrap();
            let gcd = input["gcd"].as_f64().unwrap();
            let count = input["count"].as_u64().unwrap();
            std::thread::spawn(move || {
                for index in 0..count {
                    assert!(
                        session
                            .submit_action(start + index as f64 * gcd, "instant", None)
                            .unwrap()
                            .value
                            .accepted
                    );
                }
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().unwrap();
    }
    common::assert_equivalent(
        &expected["parallel"],
        &json!([projection(&a), projection(&b)]),
        "sessions.parallel",
    );
    a.close().unwrap();
    let closed_error = a.advance_to(100.0).is_err();
    let closed_reset = a.reset(SessionOptions::default()).is_err();
    let closed = a.is_closed().unwrap();
    let active_after_close = engine.active_count().unwrap();
    let replacement = engine
        .create_session(SessionOptions {
            actual_base_gcd: Some(2.5),
            initial_timestamp: 1.0,
            fight_remaining: Some(10.0),
            ..Default::default()
        })
        .unwrap();
    let replacement_state = json!({"id":replacement.id(),"state":projection(&replacement),"active_count":engine.active_count().unwrap()});
    engine.close().unwrap();
    let lifecycle = json!({"closed":closed,"closed_error":closed_error,"closed_reset":closed_reset,"active_after_close":active_after_close,"replacement":replacement_state,
        "active_after_engine_close":engine.active_count().unwrap(),"all_closed":[b.is_closed().unwrap(),replacement.is_closed().unwrap()],
        "create_closed_error":engine.create_session(SessionOptions::default()).is_err(),"old_observe_error":b.observe_at(300.0,"seconds",None).is_err()});
    common::assert_equivalent(&expected["lifecycle"], &lifecycle, "sessions.lifecycle");
    let errors = json!({"zero":SimulationEngine::new(rules.clone(),registry.clone(),0).is_err(),"too_large":SimulationEngine::new(rules.clone(),registry.clone(),101).is_err()});
    assert_eq!(expected["capacity_errors"], errors);
    let owner = SimulationEngine::new(rules, registry, 1).unwrap();
    let orphan = owner.create_session(SessionOptions::default()).unwrap();
    drop(owner);
    assert!(orphan.is_closed().unwrap());
    assert!(orphan.advance_to(0.0).is_err());
}

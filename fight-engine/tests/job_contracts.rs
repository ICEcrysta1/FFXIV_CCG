mod common;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fight_engine::config::{JobStateRegistration, ProjectConfig, ResourceValue, SkillDefinition};
use fight_engine::contracts::{
    JobFactory, JobRegistry, JobRuntime, JobStateMachine, JobTimelineRegistry,
};
use fight_engine::state::{CombatState, ValidationResult};
use fight_engine::system::SystemRules;
use fight_engine::system::timeline::{
    Timeline, TimelineEvent, TimelineEventKind as Kind, TimelineEventPriority as Priority,
    TimelineMutation,
};
use serde_json::{Value, json};

use common::assert_equivalent;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/job_contracts.json")).unwrap()
}

fn empty_runtime() -> JobRuntime {
    JobRuntime::new(
        &JobStateRegistration {
            job_tag: "rust_probe_typed".into(),
            resources: BTreeMap::new(),
            statuses: Default::default(),
        },
        &BTreeMap::new(),
    )
    .unwrap()
}

static CONSTRUCTIONS: AtomicUsize = AtomicUsize::new(0);

struct Probe<const ID: u8> {
    runtime: JobRuntime,
}

impl<const ID: u8> JobStateMachine for Probe<ID> {
    fn runtime(&self) -> &JobRuntime {
        &self.runtime
    }
    fn default_fight_remaining(&self) -> Result<f64, String> {
        Ok(600.0)
    }
    fn current_gcd_duration(&self, _state: &CombatState) -> Result<f64, String> {
        Ok(2.5)
    }
    fn is_instant(&self, _state: &CombatState, _skill: &SkillDefinition) -> Result<bool, String> {
        Ok(true)
    }
    fn supports_behavior(&self, _behavior: &str) -> bool {
        false
    }
    fn validate_action(
        &self,
        _state: &CombatState,
        _skill: &SkillDefinition,
    ) -> Result<ValidationResult, String> {
        Ok(ValidationResult::valid())
    }
    fn apply_action(
        &self,
        _previous: &CombatState,
        _next: &mut CombatState,
        _skill: &SkillDefinition,
    ) -> Result<(), String> {
        Ok(())
    }
    fn consume_timing_resources(
        &self,
        _state: &mut CombatState,
        _skill: &SkillDefinition,
    ) -> Result<(), String> {
        Ok(())
    }
}

impl<const ID: u8> JobFactory for Probe<ID> {
    const TAG: &'static str = "rust_probe_typed";

    fn create(_project: Arc<ProjectConfig>, _system: Arc<SystemRules>) -> Result<Self, String> {
        CONSTRUCTIONS.fetch_add(1, Ordering::SeqCst);
        Ok(Self {
            runtime: empty_runtime(),
        })
    }
}

#[test]
fn registration_is_lazy_idempotent_and_returns_independent_instances() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let project = Arc::new(ProjectConfig::load(root, "black_mage").unwrap());
    let system =
        Arc::new(SystemRules::new(project.engine_timing.clone(), project.system.clone()).unwrap());
    let mut registry = JobRegistry::default();
    registry.register::<Probe<0>>().unwrap();
    registry.register::<Probe<0>>().unwrap();
    let after_registration_count = CONSTRUCTIONS.load(Ordering::SeqCst);
    let first = registry
        .get("rust_probe_typed", project.clone(), system.clone())
        .unwrap()
        .unwrap();
    let second = registry
        .get("rust_probe_typed", project.clone(), system.clone())
        .unwrap()
        .unwrap();
    let after_get_count = CONSTRUCTIONS.load(Ordering::SeqCst);
    let unknown_none = registry
        .get("missing", project.clone(), system.clone())
        .unwrap()
        .is_none();
    let typed_collision = registry.register::<Probe<1>>().is_err();
    let factory_collision = registry
        .register_factory("rust_probe_typed", |_, _| {
            Ok(Arc::new(Probe::<0> {
                runtime: empty_runtime(),
            }))
        })
        .is_err();
    let factory_count = Arc::new(AtomicUsize::new(0));
    let counter = factory_count.clone();
    registry
        .register_factory("rust_probe_factory", move |_, _| {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(Probe::<0> {
                runtime: empty_runtime(),
            }))
        })
        .unwrap();
    let factory_duplicate = registry
        .register_factory("rust_probe_factory", |_, _| {
            Ok(Arc::new(Probe::<0> {
                runtime: empty_runtime(),
            }))
        })
        .is_err();
    let factory_after_registration_count = factory_count.load(Ordering::SeqCst);
    registry
        .get("rust_probe_factory", project.clone(), system.clone())
        .unwrap();
    registry.get("rust_probe_factory", project, system).unwrap();
    assert_equivalent(
        &fixture()["registry"],
        &json!({
            "after_registration_count": after_registration_count,
            "after_get_count": after_get_count,
            "fresh_instances": !Arc::ptr_eq(&first, &second),
            "unknown_none": unknown_none,
            "typed_collision": typed_collision,
            "factory_collision": factory_collision,
            "factory_duplicate": factory_duplicate,
            "factory_after_registration_count": factory_after_registration_count,
            "factory_after_get_count": factory_count.load(Ordering::SeqCst),
            "tags": registry.registered_tags(),
        }),
        "registry",
    );
}

fn integer(state: &CombatState, key: &str) -> i32 {
    match state.job.resources[key] {
        ResourceValue::Integer(value) => value,
        _ => panic!("{key} 应为整数资源"),
    }
}

#[test]
fn job_timers_projections_and_expiry_hooks_match_baseline() {
    let expected = fixture();
    let registration: JobStateRegistration =
        serde_json::from_value(expected["timeline"]["registration"].clone()).unwrap();
    let mut runtime = JobRuntime::new(&registration, &BTreeMap::new()).unwrap();
    runtime
        .timeline
        .register_periodic(
            "a_disabled",
            10.0,
            "disabled",
            |_| None,
            |_, _| Ok(TimelineMutation::default()),
        )
        .unwrap();
    runtime
        .timeline
        .register_periodic(
            "z_periodic",
            3.0,
            "elapsed",
            |state| state.job.timeline_deadlines.get("p").copied(),
            |_, state| {
                let mut next = state.clone_without_history();
                next.job.resources.insert(
                    "ticks".into(),
                    ResourceValue::Integer(integer(state, "ticks") + 1),
                );
                next.job
                    .timeline_deadlines
                    .insert("p".into(), state.time() + 3.0);
                Ok(TimelineMutation {
                    state: Some(next),
                    ..TimelineMutation::default()
                })
            },
        )
        .unwrap();
    runtime
        .timeline
        .register_countdown(
            "a_countdown",
            "remaining",
            |state| state.job.timeline_deadlines.get("c").copied(),
            |event, state, targetable_at| {
                let mut next = state.clone_without_history();
                next.job.timeline_deadlines.remove("c");
                next.job.resources.insert(
                    "targetable".into(),
                    ResourceValue::Bool(targetable_at.unwrap()(event.timestamp)),
                );
                Ok(TimelineMutation {
                    state: Some(next),
                    ..TimelineMutation::default()
                })
            },
        )
        .unwrap();
    runtime
        .timeline
        .register_mp_modifier(|state, natural| natural + integer(state, "ticks") * 10)
        .unwrap();
    runtime
        .register_expiry_hook("charge", |state| {
            state.job.resources.insert(
                "residue".into(),
                ResourceValue::Integer(state.job.statuses["charge"].stacks),
            );
            Ok(())
        })
        .unwrap();
    let mut state = CombatState::default();
    runtime.initialize(&mut state).unwrap();
    state
        .job
        .timeline_deadlines
        .extend([("p".into(), 3.0), ("c".into(), 5.0)]);
    runtime
        .grant_status(&mut state, "charge", None, None)
        .unwrap();
    let mut timeline = Timeline::new(state).unwrap();
    let mut points = Vec::new();
    for time in [0.0, 2.0, 3.0, 5.0, 6.0] {
        timeline
            .advance_to(
                time,
                |state| {
                    let mut events = runtime.describe_status_events(state);
                    events.extend(runtime.timeline.describe_events(state));
                    events
                },
                |event, timeline| match event.kind {
                    Kind::JobPeriodicTick => {
                        runtime.timeline.handle_periodic(event, timeline.state())
                    }
                    Kind::JobTimerExpired => runtime.timeline.handle_countdown(
                        event,
                        timeline.state(),
                        Some(&|timestamp| timestamp < 5.0),
                    ),
                    Kind::StatusExpired => {
                        let mut state = timeline.get_state();
                        runtime.handle_status_expired(event, &mut state)?;
                        Ok(TimelineMutation {
                            state: Some(state),
                            ..TimelineMutation::default()
                        })
                    }
                    _ => Ok(TimelineMutation::default()),
                },
            )
            .unwrap();
        let state = timeline.state();
        points.push(json!({
            "time": state.time(),
            "events": runtime.timeline.describe_events::<()>(state).iter().map(|event| json!({
                "timestamp": event.timestamp,
                "priority": event.priority.rank(),
                "kind": format!("{:?}", event.kind),
                "owner": event.owner_key,
                "scope": format!("{:?}", event.scope),
            })).collect::<Vec<_>>(),
            "timers": runtime.timeline.timer_views(state).into_iter().map(|(key, value)| json!({
                "output_key": key, "value": value,
            })).collect::<Vec<_>>(),
            "resources": runtime.resource_snapshot(state).unwrap(),
            "raw_resources": state.job.resources,
            "deadlines": state.job.timeline_deadlines,
            "statuses": state.job.statuses,
            "mp_tick_amount": runtime.timeline.resolve_mp_tick(state, 200),
        }));
    }
    assert_equivalent(
        &expected["timeline"]["points"],
        &json!(points),
        "timeline.points",
    );
    assert_equivalent(
        &expected["timeline"]["default_mp_amount"],
        &json!(JobTimelineRegistry::default().resolve_mp_tick(timeline.state(), 200)),
        "default_mp_amount",
    );
    let mut negative = JobTimelineRegistry::default();
    negative.register_mp_modifier(|_, _| -1).unwrap();
    assert_equivalent(
        &expected["timeline"]["negative_mp_amount"],
        &json!(negative.resolve_mp_tick(timeline.state(), 200)),
        "negative_mp_amount",
    );
    let duplicate_periodic = runtime
        .timeline
        .register_periodic(
            "z_periodic",
            3.0,
            "elapsed",
            |_| None,
            |_, _| Ok(TimelineMutation::default()),
        )
        .is_err();
    let cross_kind_duplicate = runtime
        .timeline
        .register_countdown(
            "z_periodic",
            "remaining",
            |_| None,
            |_, _, _| Ok(TimelineMutation::default()),
        )
        .is_err();
    let duplicate_mp_modifier = runtime
        .timeline
        .register_mp_modifier(|_, natural| natural)
        .is_err();
    let invalid_interval = runtime
        .timeline
        .register_periodic(
            "invalid",
            0.0,
            "elapsed",
            |_| None,
            |_, _| Ok(TimelineMutation::default()),
        )
        .is_err();
    let blank_key = runtime
        .timeline
        .register_countdown(
            " ",
            "remaining",
            |_| None,
            |_, _, _| Ok(TimelineMutation::default()),
        )
        .is_err();
    let blank_output = runtime
        .timeline
        .register_countdown(
            "blank_output",
            " ",
            |_| None,
            |_, _, _| Ok(TimelineMutation::default()),
        )
        .is_err();
    let duplicate_expiry_hook = runtime.register_expiry_hook("charge", |_| Ok(())).is_err();
    assert_equivalent(
        &expected["timeline"]["errors"],
        &json!({
            "duplicate_periodic": duplicate_periodic,
            "cross_kind_duplicate": cross_kind_duplicate,
            "duplicate_mp_modifier": duplicate_mp_modifier,
            "invalid_interval": invalid_interval,
            "blank_key": blank_key,
            "blank_output": blank_output,
            "duplicate_expiry_hook": duplicate_expiry_hook,
        }),
        "timeline.errors",
    );
}

#[test]
fn job_callbacks_can_change_state_cancel_and_schedule_events() {
    let mut state = CombatState::default();
    state.mp = 0;
    state
        .job
        .timeline_deadlines
        .extend([("p".into(), 1.0), ("c".into(), 1.5)]);
    let mut timeline = Timeline::new(state).unwrap();
    let cancelled = ["cancel_periodic", "cancel_countdown"].map(|owner| {
        timeline
            .schedule(TimelineEvent {
                owner_key: Some(owner.into()),
                ..TimelineEvent::new(3.0, Priority::ExternalScene, Kind::SceneChanged)
            })
            .unwrap()
            .sequence
    });
    let mut registry = JobTimelineRegistry::default();
    registry
        .register_periodic(
            "periodic",
            1.0,
            "elapsed",
            |state| state.job.timeline_deadlines.get("p").copied(),
            move |_, state| {
                let mut next = state.clone();
                next.mp += 10;
                next.job.timeline_deadlines.remove("p");
                Ok(TimelineMutation {
                    state: Some(next),
                    cancel: vec![cancelled[0]],
                    schedule: vec![TimelineEvent {
                        owner_key: Some("periodic_follow".into()),
                        ..TimelineEvent::new(
                            2.0,
                            Priority::DecisionBoundary,
                            Kind::DecisionBoundary,
                        )
                    }],
                })
            },
        )
        .unwrap();
    registry
        .register_countdown(
            "countdown",
            "remaining",
            |state| state.job.timeline_deadlines.get("c").copied(),
            move |_, state, _| {
                let mut next = state.clone();
                next.mp += 100;
                next.job.timeline_deadlines.remove("c");
                Ok(TimelineMutation {
                    state: Some(next),
                    cancel: vec![cancelled[1]],
                    schedule: vec![TimelineEvent {
                        owner_key: Some("countdown_follow".into()),
                        ..TimelineEvent::new(
                            2.0,
                            Priority::ConfirmedActionEffect,
                            Kind::ActionEffect,
                        )
                    }],
                })
            },
        )
        .unwrap();
    let mut dispatch = Vec::new();
    timeline
        .advance_to(
            3.0,
            |state| registry.describe_events(state),
            |event, timeline| {
                dispatch.push(json!({
                    "time": event.timestamp, "kind": format!("{:?}", event.kind),
                    "owner": event.owner_key, "sequence": event.sequence,
                }));
                match event.kind {
                    Kind::JobPeriodicTick => registry.handle_periodic(event, timeline.state()),
                    Kind::JobTimerExpired => {
                        registry.handle_countdown(event, timeline.state(), None)
                    }
                    _ => {
                        let mut next = timeline.get_state();
                        next.mp += 1;
                        Ok(TimelineMutation {
                            state: Some(next),
                            ..TimelineMutation::default()
                        })
                    }
                }
            },
        )
        .unwrap();
    let actual = json!({
        "cancelled_owners_absent": dispatch.iter().all(|event| {
            !matches!(event["owner"].as_str(), Some("cancel_periodic" | "cancel_countdown"))
        }),
        "dispatch": dispatch,
        "mp": timeline.state().mp,
        "next_sequence": timeline.create_snapshot().next_sequence,
        "pending_events": timeline.pending_event_count(),
    });
    assert_equivalent(
        &fixture()["callback_mutations"],
        &actual,
        "callback_mutations",
    );
}

#[test]
fn system_and_job_expiry_hooks_preserve_grant_order_and_residual_stacks() {
    use std::collections::BTreeMap;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use fight_engine::config::{
        EngineTimingConfig, JobStateRegistration, StatusDefinition, SystemConfig,
    };
    use fight_engine::contracts::JobRuntime;
    use fight_engine::state::CombatState;
    use fight_engine::system::SystemRules;
    use fight_engine::system::timeline::{
        Timeline, TimelineEventKind, TimelineEventScope, TimelineMutation,
    };
    use serde_json::json;

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let expected = fixture();
    for (name, scope) in [
        ("system", TimelineEventScope::System),
        ("job", TimelineEventScope::Job),
    ] {
        let definitions: indexmap::IndexMap<_, _> = [("z", 1), ("a", 2)]
            .into_iter()
            .map(|(key, id)| {
                (
                    key.to_owned(),
                    StatusDefinition {
                        key: key.to_owned(),
                        game_id: id,
                        duration: 1.0,
                        max_stacks: 3,
                    },
                )
            })
            .collect();
        let mut config = SystemConfig::load(root.join("config/system.yaml")).unwrap();
        config.statuses = definitions.clone();
        let mut system = SystemRules::new(
            EngineTimingConfig {
                action_queue_window_seconds: 0.6,
            },
            config,
        )
        .unwrap();
        let mut job = JobRuntime::new(
            &JobStateRegistration {
                job_tag: "probe".to_owned(),
                resources: BTreeMap::new(),
                statuses: definitions,
            },
            &BTreeMap::new(),
        )
        .unwrap();
        let hooks = Arc::new(Mutex::new(Vec::new()));
        for (key, other) in [("z", "a"), ("a", "z")] {
            let hooks = hooks.clone();
            let hook = move |state: &mut CombatState| {
                let statuses = match scope {
                    TimelineEventScope::System => &state.statuses,
                    TimelineEventScope::Job => &state.job.statuses,
                };
                hooks.lock().unwrap().push(json!({
                    "key": key,
                    "other_present": statuses.contains_key(other),
                    "own_stacks": statuses[key].stacks,
                }));
                Ok(())
            };
            match scope {
                TimelineEventScope::System => system.register_expiry_hook(key, hook).unwrap(),
                TimelineEventScope::Job => job.register_expiry_hook(key, hook).unwrap(),
            }
        }
        let mut state = CombatState::default();
        for (key, stacks) in [("z", 3), ("a", 2)] {
            match scope {
                TimelineEventScope::System => {
                    system.grant_status(&mut state, key, None, Some(stacks))
                }
                TimelineEventScope::Job => job.grant_status(&mut state, key, None, Some(stacks)),
            }
            .unwrap();
        }
        let mut timeline = Timeline::<()>::new(state).unwrap();
        let mut dispatch = Vec::new();
        timeline
            .advance_to(
                1.0,
                |state| match scope {
                    TimelineEventScope::System => system
                        .describe_events(state)
                        .into_iter()
                        .filter(|event| event.kind == TimelineEventKind::StatusExpired)
                        .collect(),
                    TimelineEventScope::Job => job.describe_status_events(state),
                },
                |event, timeline| {
                    dispatch.push(event.owner_key.clone().unwrap());
                    let mut next = timeline.get_state();
                    match scope {
                        TimelineEventScope::System => {
                            system.settle_status_expired(&mut next, event)
                        }
                        TimelineEventScope::Job => job.handle_status_expired(event, &mut next),
                    }?;
                    Ok(TimelineMutation {
                        state: Some(next),
                        ..TimelineMutation::default()
                    })
                },
            )
            .unwrap();
        let remaining = match scope {
            TimelineEventScope::System => timeline
                .state()
                .statuses
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            TimelineEventScope::Job => timeline
                .state()
                .job
                .statuses
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
        };
        assert_equivalent(
            &expected["expiry_order"][name],
            &json!({ "hooks": *hooks.lock().unwrap(), "dispatch": dispatch, "remaining": remaining }),
            name,
        );
    }
}

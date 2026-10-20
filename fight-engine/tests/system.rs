// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

mod common;

use fight_engine::state::{CombatState, StatusState};
use fight_engine::system::timeline::{
    TIME_EPSILON, Timeline, TimelineEvent, TimelineEventKind as Kind,
    TimelineEventPriority as Priority, TimelineEventScope as Scope, TimelineMutation,
};
use serde_json::{Value, json};

use common::assert_equivalent;

fn assert_equivalent_at_root(expected: &Value, actual: &Value) {
    assert_equivalent(expected, actual, "root");
}

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/system.json")).unwrap()
}

fn event(time: f64, priority: Priority, kind: Kind, owner: &str) -> TimelineEvent {
    TimelineEvent {
        owner_key: Some(owner.into()),
        ..TimelineEvent::new(time, priority, kind)
    }
}

fn dispatch(event: &TimelineEvent) -> Value {
    json!({
        "time": event.timestamp,
        "kind": format!("{:?}", event.kind),
        "owner": event.owner_key,
        "sequence": event.sequence,
    })
}

fn scoped_dispatch(event: &TimelineEvent) -> Value {
    let mut result = dispatch(event);
    result["scope"] = json!(format!("{:?}", event.scope));
    result
}

fn no_resources(_: &CombatState) -> Vec<TimelineEvent> {
    Vec::new()
}

fn no_effect(_: &TimelineEvent, _: &Timeline) -> Result<TimelineMutation, String> {
    Ok(TimelineMutation::default())
}

#[test]
fn same_time_order_cancellation_and_followup_match_baseline() {
    let mut state = CombatState::default();
    state.mp = 0;
    let mut timeline = Timeline::new(state).unwrap();
    let cancelled = timeline
        .schedule(event(
            99.0,
            Priority::DecisionBoundary,
            Kind::DecisionBoundary,
            "cancel",
        ))
        .unwrap();
    for (priority, kind, owner) in [
        (Priority::ActionAccepted, Kind::ActionAccepted, "accept"),
        (Priority::DecisionBoundary, Kind::CastCompleted, "boundary"),
        (Priority::PlayerReset, Kind::PlayerDeath, "reset"),
        (
            Priority::ExpirationAndCooldown,
            Kind::StatusExpired,
            "expiry",
        ),
        (Priority::PeriodicSettlement, Kind::MpTick, "tick"),
        (
            Priority::ConfirmedActionEffect,
            Kind::ActionEffect,
            "effect",
        ),
        (Priority::ExternalScene, Kind::SceneChanged, "scene"),
    ] {
        timeline
            .schedule(event(2.0, priority, kind, owner))
            .unwrap();
    }
    timeline
        .schedule(event(
            1.0,
            Priority::DecisionBoundary,
            Kind::DecisionBoundary,
            "early",
        ))
        .unwrap();
    let cancel_results = [
        timeline.cancel(cancelled.sequence),
        timeline.cancel(cancelled.sequence),
    ];
    let mut log = Vec::new();
    timeline
        .advance_to(2.0, no_resources, |item, timeline| {
            log.push(dispatch(item));
            let mut state = timeline.get_state();
            state.mp += 1;
            Ok(TimelineMutation {
                state: Some(state),
                schedule: if item.owner_key.as_deref() == Some("early") {
                    vec![event(
                        2.0,
                        Priority::ConfirmedActionEffect,
                        Kind::ActionEffect,
                        "follow",
                    )]
                } else {
                    Vec::new()
                },
                ..TimelineMutation::default()
            })
        })
        .unwrap();
    assert_equivalent_at_root(
        &fixture()["ordering"],
        &json!({
            "dispatch": log,
            "cancel_results": cancel_results,
            "mp": timeline.state().mp,
            "time": timeline.current_time(),
            "next_sequence": timeline.create_snapshot().next_sequence,
            "pending_events": timeline.pending_event_count(),
        }),
    );
}

fn status_resources(state: &CombatState) -> Vec<TimelineEvent> {
    state
        .statuses
        .iter()
        .map(|(key, status)| {
            event(
                state.time().max(status.expires_at),
                Priority::ExpirationAndCooldown,
                Kind::StatusExpired,
                key,
            )
        })
        .collect()
}

#[test]
fn refreshing_a_due_resource_preserves_its_old_fact() {
    let mut state = CombatState::default();
    state
        .statuses
        .insert("probe".into(), StatusState::new(0.0, 3.0, 1));
    let mut timeline = Timeline::new(state).unwrap();
    timeline
        .synchronize_resources(&mut status_resources)
        .unwrap();
    timeline
        .schedule(event(
            3.0,
            Priority::ConfirmedActionEffect,
            Kind::ActionEffect,
            "refresh",
        ))
        .unwrap();
    let mut log = Vec::new();
    let mut actual = serde_json::Map::new();
    for time in [3.0, 8.0] {
        timeline
            .advance_to(time, status_resources, |item, timeline| {
                log.push(dispatch(item));
                let mut state = timeline.get_state();
                if item.kind == Kind::ActionEffect {
                    state
                        .statuses
                        .insert("probe".into(), StatusState::new(state.time(), 5.0, 1));
                } else if state.statuses["probe"].expires_at <= state.time() {
                    state.statuses.shift_remove("probe");
                }
                Ok(TimelineMutation {
                    state: Some(state),
                    ..TimelineMutation::default()
                })
            })
            .unwrap();
        let mut point = json!({
            "dispatch": log,
            "next_sequence": timeline.create_snapshot().next_sequence,
            "pending_events": timeline.pending_events().iter().map(dispatch).collect::<Vec<_>>(),
        });
        if time == 3.0 {
            point["expires_at"] = json!(timeline.state().statuses["probe"].expires_at);
        } else {
            point["has_status"] = json!(timeline.state().has_status("probe"));
        }
        actual.insert(format!("at_{}", time as i32), point);
    }
    assert_equivalent_at_root(&fixture()["refresh"], &Value::Object(actual));
}

fn scoped_status_resources(state: &CombatState) -> Vec<TimelineEvent> {
    let mut events = status_resources(state);
    events.extend(
        state
            .job
            .statuses
            .iter()
            .map(|(key, status)| TimelineEvent {
                scope: Scope::Job,
                ..event(
                    state.time().max(status.expires_at),
                    Priority::ExpirationAndCooldown,
                    Kind::StatusExpired,
                    key,
                )
            }),
    );
    events
}

fn settle_scoped_statuses(timeline: &mut Timeline) -> Value {
    let mut log = Vec::new();
    timeline
        .advance_to(6.0, scoped_status_resources, |item, timeline| {
            log.push(scoped_dispatch(item));
            let mut state = timeline.get_state();
            let statuses = if item.scope == Scope::System {
                &mut state.statuses
            } else {
                &mut state.job.statuses
            };
            statuses.shift_remove(item.owner_key.as_deref().unwrap());
            state.mp += 1;
            Ok(TimelineMutation {
                state: Some(state),
                ..TimelineMutation::default()
            })
        })
        .unwrap();
    json!({
        "time": timeline.current_time(),
        "mp": timeline.state().mp,
        "system_status_count": timeline.state().statuses.len(),
        "job_status_count": timeline.state().job.statuses.len(),
        "dispatch": log,
        "pending_events": timeline.pending_event_count(),
        "pending_settlements": timeline.pending_settlement_count(),
        "next_sequence": timeline.create_snapshot().next_sequence,
    })
}

#[test]
fn pending_facts_keep_scope_and_survive_snapshot_and_fork() {
    let mut state = CombatState::default();
    state.mp = 0;
    let status = StatusState::new(0.0, 5.0 + TIME_EPSILON / 2.0, 1);
    state.statuses.insert("probe".into(), status.clone());
    state.job.statuses.insert("probe".into(), status);
    let mut timeline = Timeline::new(state).unwrap();
    timeline
        .synchronize_resources(&mut scoped_status_resources)
        .unwrap();
    timeline
        .advance_to(5.0, scoped_status_resources, no_effect)
        .unwrap();
    let snapshot = timeline.create_snapshot();
    let mut fork = timeline.fork();
    let mut restored = Timeline::new(CombatState::default()).unwrap();
    restored.restore_snapshot(&snapshot).unwrap();
    let at_5 = json!({
        "time": timeline.current_time(),
        "pending_events": timeline.pending_event_count(),
        "pending_settlements": snapshot.pending_settlements.iter().map(scoped_dispatch).collect::<Vec<_>>(),
        "next_sequence": snapshot.next_sequence,
    });
    let branches = [&mut timeline, &mut fork, &mut restored].map(settle_scoped_statuses);
    let original_isolated = snapshot.state.time() == 5.0
        && snapshot.state.statuses["probe"].stacks == 1
        && snapshot.state.job.statuses["probe"].stacks == 1;
    assert_equivalent_at_root(
        &fixture()["snapshot"],
        &json!({
            "at_5": at_5,
            "branches": branches,
            "original_isolated": original_isolated,
        }),
    );
}

#[test]
fn invalid_requests_and_non_consuming_handlers_match_baseline() {
    let mut timeline = Timeline::new(CombatState::default()).unwrap();
    timeline.advance_to(5.0, no_resources, no_effect).unwrap();
    let schedule_past = timeline
        .schedule(event(
            4.0,
            Priority::DecisionBoundary,
            Kind::DecisionBoundary,
            "past",
        ))
        .is_err();
    let schedule_non_finite = timeline
        .schedule(event(
            f64::NAN,
            Priority::DecisionBoundary,
            Kind::DecisionBoundary,
            "nan",
        ))
        .is_err();
    let mut numbered = event(
        5.0,
        Priority::DecisionBoundary,
        Kind::DecisionBoundary,
        "numbered",
    );
    numbered.sequence = 1;
    let schedule_numbered = timeline.schedule(numbered).is_err();
    let backwards = timeline.advance_to(4.0, no_resources, no_effect).is_err();
    let mut data = serde_json::to_value(timeline.state()).unwrap();
    data["time"] = json!(6.0);
    let clock_mutation = timeline
        .apply_mutation(
            TimelineMutation {
                state: Some(serde_json::from_value(data).unwrap()),
                ..TimelineMutation::default()
            },
            &mut no_resources,
        )
        .is_err();
    let mut snapshot = timeline.create_snapshot();
    snapshot.next_sequence = 0;
    let invalid_snapshot = timeline.restore_snapshot(&snapshot).is_err();
    let mut broken = Timeline::new(CombatState::default()).unwrap();
    let mut dispatch_count = 0;
    let repeated_guard = broken
        .advance_to(
            0.0,
            |_| {
                vec![event(
                    0.0,
                    Priority::ExpirationAndCooldown,
                    Kind::JobTimerExpired,
                    "broken",
                )]
            },
            |_, _| {
                dispatch_count += 1;
                Ok(TimelineMutation::default())
            },
        )
        .is_err();
    assert_equivalent_at_root(
        &fixture()["validation"],
        &json!({
            "schedule_past": schedule_past,
            "schedule_non_finite": schedule_non_finite,
            "schedule_numbered": schedule_numbered,
            "backwards": backwards,
            "clock_mutation": clock_mutation,
            "invalid_snapshot": invalid_snapshot,
            "time_after_rejections": timeline.current_time(),
            "repeated_guard": repeated_guard,
            "repeat_dispatch_count": dispatch_count,
        }),
    );
}

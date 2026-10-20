mod common;

use std::collections::BTreeMap;

use fight_engine::config::{
    ActionKind, EngineTimingConfig, ResourceValue, SkillDefinition, StatusDefinition, SystemConfig,
    SystemMpRecoveryConfig, SystemPotencyConfig,
};
use fight_engine::state::{CombatState, ExternalCombatEvent, ExternalEventKind, StatusState};
use fight_engine::system::SystemRules;
use fight_engine::system::timeline::{
    Timeline, TimelineEvent, TimelineEventKind as Kind, TimelineEventPriority as Priority,
    TimelineEventScope, TimelineMutation,
};
use serde_json::{Value, json};

use common::assert_equivalent;

fn assert_equivalent_at_root(expected: &Value, actual: &Value) {
    assert_equivalent(expected, actual, "root");
}

fn fixture() -> Value {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/shared_system.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn rules() -> SystemRules {
    SystemRules::new(
        EngineTimingConfig {
            action_queue_window_seconds: 0.6,
        },
        SystemConfig {
            base_gcd: 2.5,
            skill_table_base_gcd: 2.5,
            mp_recovery: SystemMpRecoveryConfig {
                tick_interval_seconds: 3.0,
                in_combat_amount: 200,
                revival_ratio: 0.2,
            },
            potency: SystemPotencyConfig {
                burst_potion_multiplier: 1.08,
                raid_buff_window_multiplier: 1.1,
            },
            raid_buff_window_marker_skills: Vec::new(),
            raid_buff_window_duration: 15.0,
            statuses: [
                ("probe", 10.0, 3),
                ("burst_potion", 30.0, 1),
                ("raid_buff_window", 15.0, 1),
            ]
            .into_iter()
            .map(|(key, duration, max_stacks)| {
                (
                    key.into(),
                    StatusDefinition {
                        key: key.into(),
                        game_id: 0,
                        duration,
                        max_stacks,
                    },
                )
            })
            .collect(),
            skills: Vec::new(),
        },
    )
    .unwrap()
}

fn skill(key: &str, kind: ActionKind) -> SkillDefinition {
    SkillDefinition {
        key: key.into(),
        kind,
        ..SkillDefinition::default()
    }
}

fn clock_to(state: CombatState, time: f64) -> CombatState {
    Timeline::<()>::new(state)
        .unwrap()
        .advance_to(time, |_| Vec::new(), |_, _| Ok(TimelineMutation::default()))
        .unwrap()
}

fn event(state: &CombatState, kind: Kind, key: Option<&str>) -> TimelineEvent {
    TimelineEvent {
        owner_key: key.map(str::to_owned),
        ..TimelineEvent::new(state.time(), Priority::ExpirationAndCooldown, kind)
    }
}

#[test]
fn common_validation_preserves_lock_order_and_queue_exceptions() {
    let rules = rules();
    let mut cases = Vec::new();
    for name in [
        "ok",
        "dead",
        "boss",
        "cast",
        "gcd",
        "ogcd_limit",
        "movement",
        "cooldown",
        "non_temporal",
        "non_temporal_ogcd",
        "instant_moving",
        "threshold",
        "targetless",
    ] {
        let mut state = CombatState::default();
        let mut action = skill("probe", ActionKind::Gcd);
        action.requires_target = true;
        let mut instant = false;
        let mut charge = true;
        let mut temporal = true;
        match name {
            "dead" => {
                state.is_dead = true;
                state.set_boss_targetable(false);
            }
            "boss" => {
                state.set_boss_targetable(false);
                state.cast_ends_at = 1.0;
            }
            "cast" => {
                state.cast_ends_at = 1.0;
                state.gcd_ready_at = 1.0;
            }
            "gcd" => state.gcd_ready_at = 1.0,
            "ogcd_limit" | "non_temporal_ogcd" => {
                action.kind = ActionKind::Ogcd;
                state.gcd_ready_at = 1.0;
                state.ogcds_weaved = 3;
                temporal = name != "non_temporal_ogcd";
            }
            "movement" => state.is_moving = true,
            "cooldown" => charge = false,
            "non_temporal" => {
                state.cast_ends_at = 1.0;
                state.gcd_ready_at = 1.0;
                charge = false;
                temporal = false;
            }
            "instant_moving" => {
                state.is_moving = true;
                instant = true;
            }
            "threshold" => state.cast_ends_at = 0.0001,
            "targetless" => {
                state.set_boss_targetable(false);
                action.requires_target = false;
            }
            _ => {}
        }
        cases.push(json!({"name": name, "result": rules.validate_common_action(&state, &action, instant, charge, temporal)}));
    }
    assert_equivalent_at_root(&fixture()["validation"], &json!(cases));
}

fn timing_projection(state: &CombatState) -> Value {
    json!({
        "time": state.time(), "gcd_index": state.gcd_index,
        "cast_remaining": state.cast_remaining(), "cast_ends_at": state.cast_ends_at,
        "gcd_remaining": state.gcd_remaining(), "gcd_ready_at": state.gcd_ready_at,
        "weave_window_remaining": state.weave_window_remaining(), "weave_ends_at": state.weave_ends_at,
        "ogcds_weaved": state.ogcds_weaved, "fight_remaining": state.fight_remaining(),
        "fight_ends_at": state.fight_ends_at, "natural_mp_last_tick_at": state.natural_mp_last_tick_at,
        "downtime_ends_at": state.downtime_ends_at, "mp": state.mp, "max_mp": state.max_mp,
    })
}

#[test]
fn timing_preserves_request_based_recast_and_gcd_boundaries() {
    let rules = rules();
    let mut state = rules.initial_state(600.0, 15_000, 100.0).unwrap();
    let gcd = skill("gcd", ActionKind::Gcd);
    let ogcd = skill("ogcd", ActionKind::Ogcd);
    let mut actual = vec![json!({"name":"initial", "state":timing_projection(&state)})];
    rules.apply_action_timing(&mut state, &gcd, 2.5, 3.0, Some(99.8));
    actual.push(json!({"name":"cast", "state":timing_projection(&state)}));
    rules.apply_action_timing(&mut state, &ogcd, 0.0, 0.0, None);
    actual.push(json!({"name":"ogcd", "state":timing_projection(&state)}));
    state = clock_to(state, 103.0);
    rules.apply_action_timing(&mut state, &gcd, 2.5, 0.0, None);
    actual.push(json!({"name":"instant", "state":timing_projection(&state)}));
    rules.apply_action_timing(&mut state, &ogcd, 0.0, 0.0, None);
    state = clock_to(state, 105.5);
    let ready = event(&state, Kind::GcdReady, None);
    rules.settle_gcd_ready(&mut state, &ready);
    actual.push(json!({"name":"gcd_ready", "state":timing_projection(&state)}));
    assert_equivalent_at_root(&fixture()["timing"], &json!(actual));
}

fn cooldown_projection(
    rules: &SystemRules,
    state: &CombatState,
    action: &SkillDefinition,
    name: &str,
) -> Value {
    let (next, charges, max) = rules.build_cooldown_snapshot(state, action);
    json!({
        "name":name, "snapshot":{"next_cooldown_seconds":next,"available_charges":charges,"max_charges":max},
        "next_ready_at":rules.next_charge_ready_at(state, action),
        "ready_at":state.cooldowns.get(&action.key).map_or_else(Vec::new, |bucket| bucket.recharge_ready_at.clone()),
        "has_charge":rules.has_available_charge(state, action),
        "bucket_present":state.cooldowns.contains_key(&action.key),
    })
}

#[test]
fn charge_consumption_reductions_and_settlement_match_baseline() {
    let rules = rules();
    let mut state = rules.initial_state(600.0, 10_000, 100.0).unwrap();
    let action = SkillDefinition {
        cooldown: 10.0,
        charges: 2,
        ..skill("probe", ActionKind::Gcd)
    };
    let mut actual = vec![cooldown_projection(&rules, &state, &action, "ready")];
    rules.consume_cooldown(&mut state, &action);
    actual.push(cooldown_projection(&rules, &state, &action, "first"));
    state = clock_to(state, 103.0);
    rules.consume_cooldown(&mut state, &action);
    actual.push(cooldown_projection(&rules, &state, &action, "second"));
    rules.apply_cooldown_reduction(&mut state, &action, -1.0);
    actual.push(cooldown_projection(
        &rules,
        &state,
        &action,
        "negative_reduction",
    ));
    rules.apply_cooldown_reduction(&mut state, &action, 4.0);
    actual.push(cooldown_projection(&rules, &state, &action, "reduced"));
    state = clock_to(state, 106.0);
    let ready = event(&state, Kind::CooldownChargeReady, Some("probe"));
    rules.settle_cooldown_ready(&mut state, &ready, 2);
    actual.push(cooldown_projection(&rules, &state, &action, "one_ready"));
    rules.consume_cooldown(&mut state, &action);
    actual.push(cooldown_projection(
        &rules,
        &state,
        &action,
        "consumed_again",
    ));
    rules.apply_cooldown_reduction(&mut state, &action, 20.0);
    actual.push(cooldown_projection(&rules, &state, &action, "all_ready"));
    let free = skill("free", ActionKind::Gcd);
    rules.consume_cooldown(&mut state, &free);
    actual.push(cooldown_projection(&rules, &state, &free, "no_cooldown"));
    assert_equivalent_at_root(&fixture()["cooldown"], &json!(actual));
}

fn statuses(state: &CombatState) -> Value {
    json!(state.statuses.iter().map(|(key,status)| (key,json!({
        "remaining":status.remaining(state.time()),"expires_at":status.expires_at,"stacks":status.stacks,
    }))).collect::<BTreeMap<_,_>>())
}

#[test]
fn shared_buffs_preserve_stacks_refresh_validation_and_removal() {
    let rules = rules();
    let mut state = rules.initial_state(600.0, 10_000, 100.0).unwrap();
    let mut actual = Vec::new();
    for name in [
        "empty",
        "full",
        "consumed",
        "refreshed",
        "negative",
        "zero",
        "one",
        "clear",
        "missing_consume",
    ] {
        match name {
            "full" => rules.grant_status(&mut state, "probe", None, None).unwrap(),
            "consumed" | "missing_consume" => rules.consume_status_stack(&mut state, "probe"),
            "refreshed" => rules
                .grant_status(&mut state, "probe", Some(20.0), Some(99))
                .unwrap(),
            "negative" => rules
                .grant_status(&mut state, "probe", Some(-1.0), None)
                .unwrap(),
            "zero" => rules
                .grant_status(&mut state, "probe", Some(5.0), Some(0))
                .unwrap(),
            "one" => rules
                .grant_status(&mut state, "probe", None, Some(1))
                .unwrap(),
            "clear" => rules.clear_status(&mut state, "probe"),
            _ => {}
        }
        actual.push(json!({"name":name,"statuses":statuses(&state),"validation":rules.validate_status(&state,"probe").unwrap()}));
    }
    assert_equivalent_at_root(&fixture()["buffs"], &json!(actual));
}

#[test]
fn potency_preserves_target_caps_buff_expiration_and_request_snapshots() {
    let rules = rules();
    let mut source = rules.initial_state(600.0, 10_000, 100.0).unwrap();
    source.target_count = 5;
    let mut action = SkillDefinition {
        max_targets: 3,
        aoe_secondary_reduction: 0.5,
        ..skill("probe", ActionKind::Gcd)
    };
    let plain = rules.resolve_applied_potency(&source, 100.0, Some(&action));
    rules
        .grant_status(&mut source, "burst_potion", Some(10.0), None)
        .unwrap();
    let potion = rules.resolve_applied_potency(&source, 100.0, Some(&action));
    rules
        .grant_status(&mut source, "raid_buff_window", Some(5.0), None)
        .unwrap();
    let both = rules.resolve_applied_potency(&source, 100.0, Some(&action));
    let offsets = [0.0, 5.0, 10.0, -1.0]
        .map(|offset| rules.resolve_applied_potency_at_offset(&source, 100.0, offset));
    source.target_count = 0;
    let zero_targets = rules.resolve_applied_potency(&source, 100.0, Some(&action));
    source.target_count = 5;
    action.aoe_secondary_reduction = 1.5;
    let clamped_secondary = rules.resolve_applied_potency(&source, 100.0, Some(&action));
    let negative = rules.resolve_applied_potency(&source, -100.0, Some(&action));
    action.aoe_secondary_reduction = 0.5;
    let mut target = CombatState::default();
    target.cumulative_potency = 100.0;
    target.current_gcd_dot_potency = 10.0;
    rules.record_target_potency(&mut target, &source, &action, 100.0);
    let recorded = json!({"current_potency":target.current_potency,"cumulative_potency":target.cumulative_potency,"current_gcd_dot_potency":target.current_gcd_dot_potency});
    let resolved = [
        rules.record_delayed_potency(&mut target, -5.0),
        rules.record_delayed_potency(&mut target, 50.0),
    ];
    assert_equivalent_at_root(
        &fixture()["potency"],
        &json!({
            "plain":plain,"potion":potion,"both":both,"offsets":offsets,"zero_targets":zero_targets,
            "clamped_secondary":clamped_secondary,"negative":negative,"recorded":recorded,
            "delayed":{"resolved":resolved,"cumulative_potency":target.cumulative_potency},
        }),
    );
}

fn advance(rules: &SystemRules, timeline: &mut Timeline, time: f64) {
    timeline
        .advance_to(
            time,
            |state| rules.describe_events(state),
            |event, timeline| {
                let mut state = timeline.get_state();
                match event.kind {
                    Kind::MpTick => rules.settle_mp_tick(&mut state, event, 200),
                    Kind::DotTick => rules.settle_dot_tick(&mut state, event),
                    Kind::CooldownChargeReady => rules.settle_cooldown_ready(&mut state, event, 2),
                    Kind::StatusExpired => rules.settle_status_expired(&mut state, event)?,
                    Kind::GcdReady => rules.settle_gcd_ready(&mut state, event),
                    Kind::SceneChanged => rules.apply_scene_deadline(&mut state, event),
                    _ => return Err("共享规则测试收到了非系统事件".into()),
                }
                Ok(TimelineMutation {
                    state: Some(state),
                    ..TimelineMutation::default()
                })
            },
        )
        .unwrap();
}

fn external(state: &CombatState, kind: ExternalEventKind) -> ExternalCombatEvent {
    ExternalCombatEvent {
        timestamp: state.time(),
        kind,
        value: None,
        target_count: None,
        remaining_seconds: None,
        mp: None,
    }
}

fn apply_external(rules: &SystemRules, timeline: &mut Timeline, external: ExternalCombatEvent) {
    let mut state = timeline.get_state();
    rules.apply_external_event(&mut state, &external).unwrap();
    timeline
        .apply_mutation(
            TimelineMutation {
                state: Some(state),
                ..TimelineMutation::default()
            },
            &mut |state| rules.describe_events(state),
        )
        .unwrap();
}

fn lifecycle_projection(timeline: &Timeline, name: &str) -> Value {
    let state = timeline.state();
    json!({
        "name":name,"time":state.time(),"mp":state.mp,"is_dead":state.is_dead,
        "natural_mp_last_tick_at":state.natural_mp_last_tick_at,"boss_targetable":state.boss_targetable(),
        "is_moving":state.is_moving,"target_count":state.target_count,"statuses":statuses(state),
        "dots":state.dots.iter().map(|(key,dot)| (key,json!({"expires_at":dot.expires_at,"next_tick_at":dot.next_tick_at,"potency_per_tick":dot.potency_per_tick}))).collect::<BTreeMap<_,_>>(),
        "cooldowns":state.cooldowns.iter().map(|(key,bucket)| (key,json!({"available_charges":bucket.available_charges,"recharge_ready_at":bucket.recharge_ready_at}))).collect::<BTreeMap<_,_>>(),
        "cumulative_dot_potency":state.cumulative_dot_potency,"current_gcd_dot_potency":state.current_gcd_dot_potency,
    })
}

#[test]
fn death_and_revival_preserve_periodic_phase_cooldowns_and_target_dot() {
    let mut rules = rules();
    let dot = SkillDefinition {
        dot_duration: 9.0,
        dot_key: Some("shared_dot".into()),
        ..skill("dot", ActionKind::Gcd)
    };
    rules.register_target_dots([&dot]);
    let mut state = rules.initial_state(600.0, 10_000, 100.0).unwrap();
    state.mp = 9500;
    rules
        .grant_target_dot(&mut state, "shared_dot", 9.0, 40.0, 3.0)
        .unwrap();
    rules.grant_status(&mut state, "probe", None, None).unwrap();
    rules
        .grant_status(&mut state, "burst_potion", None, None)
        .unwrap();
    let cooldown = SkillDefinition {
        cooldown: 10.0,
        charges: 2,
        ..skill("probe", ActionKind::Gcd)
    };
    rules.consume_cooldown(&mut state, &cooldown);
    let mut timeline = Timeline::new(state).unwrap();
    timeline
        .synchronize_resources(&mut |state| rules.describe_events(state))
        .unwrap();
    let mut actual = vec![lifecycle_projection(&timeline, "initial")];
    advance(&rules, &mut timeline, 103.0);
    actual.push(lifecycle_projection(&timeline, "tick"));
    let mut fact = external(timeline.state(), ExternalEventKind::BossTargetableChanged);
    fact.value = Some(false);
    apply_external(&rules, &mut timeline, fact);
    actual.push(lifecycle_projection(&timeline, "untargetable"));
    advance(&rules, &mut timeline, 106.0);
    actual.push(lifecycle_projection(&timeline, "hidden_tick"));
    let fact = external(timeline.state(), ExternalEventKind::PlayerDeath);
    apply_external(&rules, &mut timeline, fact);
    actual.push(lifecycle_projection(&timeline, "death"));
    advance(&rules, &mut timeline, 109.0);
    actual.push(lifecycle_projection(&timeline, "dead_tick"));
    let fact = external(timeline.state(), ExternalEventKind::PlayerRevived);
    apply_external(&rules, &mut timeline, fact);
    let mut fact = external(timeline.state(), ExternalEventKind::BossTargetableChanged);
    fact.value = Some(true);
    apply_external(&rules, &mut timeline, fact);
    actual.push(lifecycle_projection(&timeline, "revived"));
    advance(&rules, &mut timeline, 112.0);
    actual.push(lifecycle_projection(&timeline, "resumed"));
    let fact = external(timeline.state(), ExternalEventKind::PlayerDeath);
    apply_external(&rules, &mut timeline, fact);
    let fact = external(timeline.state(), ExternalEventKind::PlayerRevivedLb3);
    apply_external(&rules, &mut timeline, fact);
    actual.push(lifecycle_projection(&timeline, "lb3"));
    let mut fact = external(timeline.state(), ExternalEventKind::MovementChanged);
    fact.value = Some(true);
    apply_external(&rules, &mut timeline, fact);
    let mut fact = external(timeline.state(), ExternalEventKind::TargetCountChanged);
    fact.target_count = Some(4);
    apply_external(&rules, &mut timeline, fact);
    let mut fact = external(timeline.state(), ExternalEventKind::RaidBuffWindowChanged);
    fact.value = Some(true);
    fact.remaining_seconds = Some(7.0);
    apply_external(&rules, &mut timeline, fact);
    actual.push(lifecycle_projection(&timeline, "scene"));
    let mut fact = external(timeline.state(), ExternalEventKind::RaidBuffWindowChanged);
    fact.value = Some(false);
    apply_external(&rules, &mut timeline, fact);
    actual.push(lifecycle_projection(&timeline, "raid_end"));
    assert_equivalent_at_root(&fixture()["lifecycle"], &json!(actual));
}

#[test]
fn scheduled_downtime_and_mp_helpers_match_baseline() {
    let rules = rules();
    let mut state = rules.initial_state(600.0, 10_000, 100.0).unwrap();
    state.next_downtime_starts_at = Some(102.0);
    state.set_downtime_remaining(3.0);
    let mut timeline = Timeline::new(state).unwrap();
    let mut actual = Vec::new();
    for time in [101.0, 102.0, 105.0] {
        advance(&rules, &mut timeline, time);
        let state = timeline.state();
        actual.push(json!({"time":state.time(),"boss_targetable":state.boss_targetable(),"next_downtime_eta":state.next_downtime_eta(),"downtime_remaining":state.downtime_remaining(),"downtime_ends_at":state.downtime_ends_at}));
    }
    assert_equivalent_at_root(&fixture()["downtime"], &json!(actual));
    let mut state = CombatState::default();
    state.mp = 100;
    rules.spend_mp(&mut state, 200);
    let spent = state.mp;
    rules.recover_mp(&mut state, 15_000);
    let recovered = state.mp;
    rules.spend_mp(&mut state, 500);
    let after_spend = state.mp;
    rules.restore_mp(&mut state);
    assert_equivalent_at_root(
        &fixture()["mp_helpers"],
        &json!({"spent":spent,"recovered":recovered,"after_spend":after_spend,"restored":state.mp}),
    );
}

#[test]
fn unregistered_resources_and_invalid_external_fields_are_rejected() {
    let mut rules = rules();
    let mut state = CombatState::default();
    assert!(
        rules
            .grant_status(&mut state, "missing", None, None)
            .is_err()
    );
    assert!(
        rules
            .grant_target_dot(&mut state, "missing", 9.0, 40.0, 3.0)
            .is_err()
    );
    let dot = SkillDefinition {
        dot_duration: 9.0,
        ..skill("dot", ActionKind::Gcd)
    };
    rules.register_target_dots([&dot]);
    assert!(
        rules
            .grant_target_dot(&mut state, "dot", 9.0, 40.0, 0.0)
            .is_err()
    );
    assert!(rules.initial_state(-1.0, 10_000, 0.0).is_err());
    assert!(rules.initial_state(600.0, 10_000, f64::INFINITY).is_err());
    for mut fact in [
        ExternalCombatEvent {
            value: Some(true),
            ..external(&state, ExternalEventKind::PlayerDeath)
        },
        ExternalCombatEvent {
            mp: Some(1),
            ..external(&state, ExternalEventKind::PlayerRevivedLb3)
        },
        ExternalCombatEvent {
            mp: Some(-1),
            ..external(&state, ExternalEventKind::PlayerRevived)
        },
        ExternalCombatEvent {
            target_count: Some(-1),
            ..external(&state, ExternalEventKind::TargetCountChanged)
        },
        ExternalCombatEvent {
            value: Some(false),
            remaining_seconds: Some(1.0),
            ..external(&state, ExternalEventKind::RaidBuffWindowChanged)
        },
    ] {
        assert!(rules.validate_external_event(&fact).is_err());
        fact.remaining_seconds = Some(f64::INFINITY);
        assert!(rules.validate_external_event(&fact).is_err());
    }
}

#[test]
fn system_rules_never_modify_job_owned_state() {
    let rules = rules();
    let mut state = rules.initial_state(600.0, 10_000, 0.0).unwrap();
    state
        .job
        .statuses
        .insert("probe".into(), StatusState::new(0.0, 20.0, 2));
    state
        .job
        .resources
        .insert("resource".into(), ResourceValue::Integer(9));
    state.job.timeline_deadlines.insert("timer".into(), 40.0);
    let job = state.job.clone();
    rules
        .grant_status(&mut state, "probe", Some(1.0), None)
        .unwrap();
    state = clock_to(state, 2.0);
    let expired = event(&state, Kind::StatusExpired, Some("probe"));
    rules.settle_status_expired(&mut state, &expired).unwrap();
    assert!(!state.statuses.contains_key("probe"));
    assert_eq!(state.job, job);
    let foreign = TimelineEvent {
        scope: TimelineEventScope::Job,
        ..expired
    };
    assert!(rules.settle_status_expired(&mut state, &foreign).is_err());
    rules.grant_status(&mut state, "probe", None, None).unwrap();
    rules.apply_death(&mut state);
    assert!(state.statuses.is_empty());
    assert_eq!(state.job, job);
    assert!(
        rules
            .describe_events::<()>(&state)
            .iter()
            .all(|event| event.scope == TimelineEventScope::System)
    );
}

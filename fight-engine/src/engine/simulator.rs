// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

//! 单队列模拟游标；动作、外部事实与分支均经过同一个逻辑时钟。

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::config::SkillDefinition;
use crate::state::{
    CombatState, ExternalCombatEvent, ExternalEventKind, ModelStateFrame, ModelStateSnapshot,
    ValidationResult,
};
use crate::system::timeline::{
    SimulationSnapshot, TIME_EPSILON, Timeline, TimelineEvent, TimelineEventKind as Kind,
    TimelineEventPriority as Priority, TimelineEventScope as Scope, TimelineMutation,
};

use super::{ActionHistoryMetadata, ActionRequest, ActionTimingPlan, EngineRules};

static NEXT_DECISION_ID: AtomicU64 = AtomicU64::new(1);

/// 请求关联在进程内唯一，不承担技能索引或模型数值的职责。
pub(crate) fn reserve_decision_id() -> Result<u64, String> {
    NEXT_DECISION_ID
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| "decision identifier exhausted".into())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionLifecycle {
    pub action_instance_id: u64,
    pub request: ActionRequest,
    pub skill: SkillDefinition,
    pub request_state: CombatState,
    pub timing: ActionTimingPlan,
    pub accepted_timestamp: f64,
    pub model_state: ModelStateSnapshot,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EnginePayload {
    Action(Box<ActionLifecycle>),
    External(ExternalCombatEvent),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionSubmissionResult {
    pub accepted: bool,
    pub queued: bool,
    pub reason: String,
    pub action_instance_id: Option<u64>,
    pub request_timestamp: f64,
    pub accepted_timestamp: Option<f64>,
    pub effect_timestamp: Option<f64>,
    pub next_scheduled_event_time: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExternalEventResult {
    pub accepted: bool,
    pub reason: String,
    pub timestamp: f64,
    pub token: Option<String>,
    pub history_cursor: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SimulationStatistics {
    pub history: usize,
    pub pending_events: usize,
    pub queue_entries: usize,
    pub pending_settlements: usize,
}

pub struct Simulator {
    rules: Arc<EngineRules>,
    timeline: Timeline<EnginePayload>,
    history_limit: Option<usize>,
}

impl Simulator {
    pub fn new(
        rules: Arc<EngineRules>,
        initial_timestamp: f64,
        fight_remaining: Option<f64>,
    ) -> Result<Self, String> {
        let state = rules.initial_state(fight_remaining, None, initial_timestamp)?;
        Self::from_state(rules, state)
    }

    pub fn from_state(rules: Arc<EngineRules>, state: CombatState) -> Result<Self, String> {
        let history_limit = rules.max_history;
        Self::with_history_limit(rules, state, history_limit)
    }

    pub fn with_history_limit(
        rules: Arc<EngineRules>,
        state: CombatState,
        history_limit: Option<usize>,
    ) -> Result<Self, String> {
        let mut timeline = Timeline::new(state)?;
        timeline.synchronize_resources(&mut |state| rules.describe_events(state))?;
        Ok(Self {
            rules,
            timeline,
            history_limit,
        })
    }

    pub fn rules(&self) -> &Arc<EngineRules> {
        &self.rules
    }

    pub fn time(&self) -> f64 {
        self.timeline.current_time()
    }

    pub fn history_cursor(&self) -> u64 {
        self.timeline.state().last_history_sequence
    }

    pub fn history_limit(&self) -> Option<usize> {
        self.history_limit
    }

    pub fn get_state(&self) -> CombatState {
        self.timeline.get_state()
    }

    pub fn get_state_without_history(&self) -> CombatState {
        self.timeline.state().clone_without_history()
    }

    pub fn has_queued_action(&self) -> bool {
        has_queued_action(&self.timeline)
    }

    pub fn next_scheduled_event_time(&self) -> Option<f64> {
        self.timeline.next_scheduled_event_time()
    }

    pub fn statistics(&self) -> SimulationStatistics {
        SimulationStatistics {
            history: self.timeline.state().history.len(),
            pending_events: self.timeline.pending_event_count(),
            queue_entries: self.timeline.pending_event_count(),
            pending_settlements: self.timeline.pending_settlement_count(),
        }
    }

    /// 未知技能在推进前报错；已知但非法的请求仍先结算请求时刻的事实。
    pub fn submit_action(
        &mut self,
        request: ActionRequest,
        actual_cast_seconds: Option<f64>,
    ) -> Result<ActionSubmissionResult, String> {
        let skill = self.rules.resolve_skill(&request.skill_key)?.clone();
        self.advance_clock_to(request.timestamp)?;
        let request_state = self.get_state_without_history();
        let submission = self.rules.evaluate_action_submission(
            &request_state,
            &skill,
            self.has_queued_action(),
            actual_cast_seconds,
        )?;
        if !submission.accepted {
            return Ok(ActionSubmissionResult {
                accepted: false,
                queued: false,
                reason: submission.reason,
                action_instance_id: None,
                request_timestamp: request.timestamp,
                accepted_timestamp: None,
                effect_timestamp: None,
                next_scheduled_event_time: self.next_scheduled_event_time(),
            });
        }

        let action_instance_id = reserve_decision_id()?;
        let timing =
            self.rules
                .build_action_timing_plan(&request_state, &skill, actual_cast_seconds)?;
        let effect_timestamp = submission.accepted_timestamp + timing.effect_delay_seconds;
        let (model_state, _) =
            self.record_model_decision(action_instance_id, &request_state, false)?;
        let request_timestamp = request.timestamp;
        let payload = ActionLifecycle {
            action_instance_id,
            request,
            skill,
            request_state,
            timing,
            accepted_timestamp: submission.accepted_timestamp,
            model_state,
        };
        self.timeline.schedule(TimelineEvent {
            owner_key: Some(payload.skill.key.clone()),
            action_instance_id: Some(action_instance_id),
            payload: Some(EnginePayload::Action(Box::new(payload))),
            ..TimelineEvent::new(
                submission.accepted_timestamp,
                Priority::ActionAccepted,
                Kind::ActionAccepted,
            )
        })?;
        if !submission.queued {
            self.advance_clock_to(request_timestamp)?;
        }
        Ok(ActionSubmissionResult {
            accepted: true,
            queued: submission.queued,
            reason: String::new(),
            action_instance_id: Some(action_instance_id),
            request_timestamp,
            accepted_timestamp: Some(submission.accepted_timestamp),
            effect_timestamp: Some(effect_timestamp),
            next_scheduled_event_time: self.next_scheduled_event_time(),
        })
    }

    pub fn advance_clock_to(&mut self, timestamp: f64) -> Result<(), String> {
        let rules = &self.rules;
        let history_limit = self.history_limit;
        self.timeline.advance_clock_to(
            timestamp,
            &mut |state| rules.describe_events(state),
            &mut |event, timeline| handle_event(rules, event, timeline, history_limit),
        )
    }

    pub fn advance_to(&mut self, timestamp: f64) -> Result<CombatState, String> {
        self.advance_clock_to(timestamp)?;
        Ok(self.get_state())
    }

    pub fn observe_at(&mut self, timestamp: f64) -> Result<CombatState, String> {
        self.advance_to(timestamp)
    }

    pub fn validate_action_at(
        &mut self,
        timestamp: f64,
        skill_key: &str,
    ) -> Result<ValidationResult, String> {
        let skill = self.rules.resolve_skill(skill_key)?.clone();
        self.advance_clock_to(timestamp)?;
        let submission = self.rules.evaluate_action_submission(
            self.timeline.state(),
            &skill,
            self.has_queued_action(),
            None,
        )?;
        Ok(ValidationResult {
            ok: submission.accepted,
            reason: submission.reason,
        })
    }

    pub fn available_action_keys_at(&mut self, timestamp: f64) -> Result<Vec<String>, String> {
        self.advance_clock_to(timestamp)?;
        let available = self
            .rules
            .skill_availability(self.timeline.state(), self.has_queued_action())?;
        Ok(self
            .rules
            .action_keys()
            .iter()
            .filter(|key| available[key.as_str()])
            .cloned()
            .collect())
    }

    /// 整批校验后再排程，失败的批次不改变时钟、状态或队列。
    pub fn apply_external_events(
        &mut self,
        events: &[ExternalCombatEvent],
    ) -> Result<ExternalEventResult, String> {
        let timestamp = events
            .first()
            .ok_or("external event batch must not be empty")?
            .timestamp;
        let mut kinds = Vec::new();
        let mut is_dead = self.timeline.state().is_dead;
        for event in events {
            if !event.timestamp.is_finite()
                || event.timestamp < self.time()
                || event.timestamp != timestamp
            {
                return Err("external event batch requires one finite, non-past timestamp".into());
            }
            if kinds.contains(&event.kind) {
                return Err("conflicting external facts".into());
            }
            kinds.push(event.kind);
            self.rules.system.validate_external_event(event)?;
            match event.kind {
                ExternalEventKind::PlayerDeath => is_dead = true,
                ExternalEventKind::PlayerRevived | ExternalEventKind::PlayerRevivedLb3 => {
                    if !is_dead {
                        return Err("revival requires a dead player".into());
                    }
                    is_dead = false;
                }
                _ => {}
            }
        }
        if kinds.contains(&ExternalEventKind::PlayerRevived)
            && kinds.contains(&ExternalEventKind::PlayerRevivedLb3)
        {
            return Err("conflicting revival facts".into());
        }
        for event in events {
            let kind = match event.kind {
                ExternalEventKind::PlayerDeath => Kind::PlayerDeath,
                ExternalEventKind::PlayerRevived | ExternalEventKind::PlayerRevivedLb3 => {
                    Kind::PlayerRevived
                }
                _ => Kind::SceneChanged,
            };
            self.timeline.schedule(TimelineEvent {
                owner_key: Some(external_kind_key(event.kind).into()),
                payload: Some(EnginePayload::External(event.clone())),
                ..TimelineEvent::new(
                    timestamp,
                    if kind == Kind::SceneChanged {
                        Priority::ExternalScene
                    } else {
                        Priority::PlayerReset
                    },
                    kind,
                )
            })?;
        }
        self.advance_clock_to(timestamp)?;
        let has_death = kinds.contains(&ExternalEventKind::PlayerDeath);
        Ok(ExternalEventResult {
            accepted: true,
            reason: String::new(),
            timestamp,
            token: has_death.then(|| "<dead/>".into()),
            history_cursor: has_death.then(|| self.history_cursor()),
        })
    }

    pub fn capture_model_frame(&self, state: &CombatState) -> Result<Arc<ModelStateFrame>, String> {
        self.rules
            .capture_model_frame(state, self.has_queued_action())
    }

    pub fn capture_model_state(&self, state: &CombatState) -> Result<ModelStateSnapshot, String> {
        self.rules
            .capture_model_state(state, self.has_queued_action())
    }

    pub(crate) fn record_model_decision(
        &mut self,
        decision_id: u64,
        request_state: &CombatState,
        completed: bool,
    ) -> Result<(ModelStateSnapshot, u64), String> {
        let model_state = self.capture_model_state(request_state)?;
        let mut next = self.get_state();
        next.last_decision_id = Some(decision_id);
        next.last_decision_after = completed.then(|| model_state.request_state.clone());
        let history_sequence = if completed {
            next.last_completed_decision_after = Some(model_state.request_state.clone());
            next.reserve_history_sequence()?
        } else {
            0
        };
        self.timeline.apply_mutation(
            TimelineMutation {
                state: Some(next),
                ..TimelineMutation::default()
            },
            &mut |state| self.rules.describe_events(state),
        )?;
        Ok((model_state, history_sequence))
    }

    pub fn create_snapshot(&self) -> SimulationSnapshot<EnginePayload> {
        self.timeline.create_snapshot()
    }

    pub fn restore_snapshot(
        &mut self,
        snapshot: &SimulationSnapshot<EnginePayload>,
    ) -> Result<(), String> {
        self.timeline.restore_snapshot(snapshot)
    }

    pub fn fork(&self) -> Self {
        Self {
            rules: self.rules.clone(),
            timeline: self.timeline.fork(),
            history_limit: self.history_limit,
        }
    }
}

fn has_queued_action(timeline: &Timeline<EnginePayload>) -> bool {
    timeline.iter_pending_events().any(|event| {
        event.kind == Kind::ActionAccepted
            && matches!(&event.payload, Some(EnginePayload::Action(payload))
                if payload.accepted_timestamp > payload.request.timestamp + TIME_EPSILON)
    })
}

fn handle_event(
    rules: &EngineRules,
    event: &TimelineEvent<EnginePayload>,
    timeline: &Timeline<EnginePayload>,
    history_limit: Option<usize>,
) -> Result<TimelineMutation<EnginePayload>, String> {
    if matches!(event.kind, Kind::JobPeriodicTick | Kind::JobTimerExpired) {
        let unit = unit_event(event);
        let mutation = if event.kind == Kind::JobPeriodicTick {
            rules
                .job
                .runtime()
                .timeline
                .handle_periodic(&unit, timeline.state())?
        } else {
            rules
                .job
                .runtime()
                .timeline
                .handle_countdown(&unit, timeline.state(), None)?
        };
        return Ok(lift_mutation(mutation));
    }
    let mut next = timeline.get_state();
    let mut cancel = Vec::new();
    let mut schedule = Vec::new();
    match event.kind {
        Kind::ActionAccepted => {
            let payload = action_payload(event)?;
            rules.apply_action_timing(
                &mut next,
                &payload.skill,
                payload.timing.effective_gcd_seconds,
                payload.timing.actual_cast_seconds,
                Some(payload.request.timestamp),
            )?;
            schedule.push(action_event(
                payload,
                event.timestamp + payload.timing.effect_delay_seconds,
                Kind::ActionEffect,
                Priority::ConfirmedActionEffect,
            ));
            if payload.timing.actual_cast_seconds
                > payload.timing.effect_delay_seconds + TIME_EPSILON
            {
                schedule.push(action_event(
                    payload,
                    event.timestamp + payload.timing.actual_cast_seconds,
                    Kind::CastCompleted,
                    Priority::DecisionBoundary,
                ));
            }
        }
        Kind::ActionEffect => {
            let payload = action_payload(event)?;
            rules.apply_action_effect(&payload.request_state, &mut next, &payload.skill)?;
            let completed = rules.capture_model_frame(&next, has_queued_action(timeline))?;
            next.last_completed_decision_after = Some(completed.clone());
            if next.last_decision_id == Some(payload.action_instance_id) {
                next.last_decision_after = Some(completed);
            }
            rules.record_timeline_action_history(
                &payload.request_state,
                &mut next,
                &payload.skill,
                &payload.timing,
                ActionHistoryMetadata {
                    action_instance_id: payload.action_instance_id,
                    request_timestamp: payload.request.timestamp,
                    cast_completed_timestamp: payload.accepted_timestamp
                        + payload.timing.actual_cast_seconds,
                    effect_timestamp: event.timestamp,
                    model_state: payload.model_state.clone(),
                },
            )?;
            if let Some(limit) = history_limit {
                let history = Arc::make_mut(&mut next.history);
                if history.len() > limit {
                    history.drain(..history.len() - limit);
                }
            }
        }
        Kind::PlayerDeath => {
            cancel.extend(
                timeline
                    .iter_pending_events()
                    .filter(|pending| {
                        matches!(
                            pending.kind,
                            Kind::ActionAccepted | Kind::ActionEffect | Kind::CastCompleted
                        )
                    })
                    .map(|pending| pending.sequence),
            );
            rules.apply_death(&mut next)?;
            next.cast_ends_at = next.time();
            next.weave_ends_at = next.time();
            next.ogcds_weaved = 0;
            next.last_decision_id = None;
            next.last_decision_after = next.last_completed_decision_after.clone();
        }
        Kind::SceneChanged | Kind::PlayerRevived => {
            if let Some(EnginePayload::External(external)) = &event.payload {
                rules.system.apply_external_event(&mut next, external)?;
            } else if event.kind == Kind::SceneChanged {
                rules.system.apply_scene_deadline(&mut next, event);
            }
        }
        Kind::GcdReady => rules.system.settle_gcd_ready(&mut next, event),
        Kind::CooldownChargeReady => {
            let skill = rules.resolve_skill(
                event
                    .owner_key
                    .as_deref()
                    .ok_or("cooldown event missing key")?,
            )?;
            rules
                .system
                .settle_cooldown_ready(&mut next, event, skill.charges);
        }
        Kind::StatusExpired => match event.scope {
            Scope::System => rules.system.settle_status_expired(&mut next, event)?,
            Scope::Job => rules
                .job
                .runtime()
                .handle_status_expired(event, &mut next)?,
        },
        Kind::DotTick => rules.system.settle_dot_tick(&mut next, event),
        Kind::MpTick => {
            let amount = rules.job.runtime().timeline.resolve_mp_tick(
                timeline.state(),
                rules.system.config().mp_recovery.in_combat_amount,
            );
            rules.system.settle_mp_tick(&mut next, event, amount);
        }
        Kind::CastCompleted | Kind::DecisionBoundary => return Ok(TimelineMutation::default()),
        Kind::JobPeriodicTick | Kind::JobTimerExpired => unreachable!("职业计时事件已单独派发"),
    }
    Ok(TimelineMutation {
        state: Some(next),
        cancel,
        schedule,
    })
}

fn action_payload(event: &TimelineEvent<EnginePayload>) -> Result<&ActionLifecycle, String> {
    match &event.payload {
        Some(EnginePayload::Action(payload)) => Ok(payload),
        _ => Err("action event missing lifecycle payload".into()),
    }
}

fn action_event(
    payload: &ActionLifecycle,
    timestamp: f64,
    kind: Kind,
    priority: Priority,
) -> TimelineEvent<EnginePayload> {
    TimelineEvent {
        owner_key: Some(payload.skill.key.clone()),
        action_instance_id: Some(payload.action_instance_id),
        payload: Some(EnginePayload::Action(Box::new(payload.clone()))),
        ..TimelineEvent::new(timestamp, priority, kind)
    }
}

fn unit_event(event: &TimelineEvent<EnginePayload>) -> TimelineEvent {
    TimelineEvent {
        timestamp: event.timestamp,
        priority: event.priority,
        kind: event.kind,
        owner_key: event.owner_key.clone(),
        action_instance_id: event.action_instance_id,
        payload: None,
        sequence: event.sequence,
        scope: event.scope,
    }
}

/// 职业契约不依赖动作载荷，转换时完整保留状态、取消和后续排程。
fn lift_mutation(mutation: TimelineMutation) -> TimelineMutation<EnginePayload> {
    TimelineMutation {
        state: mutation.state,
        cancel: mutation.cancel,
        schedule: mutation
            .schedule
            .into_iter()
            .map(|event| TimelineEvent {
                timestamp: event.timestamp,
                priority: event.priority,
                kind: event.kind,
                owner_key: event.owner_key,
                action_instance_id: event.action_instance_id,
                payload: None,
                sequence: event.sequence,
                scope: event.scope,
            })
            .collect(),
    }
}

fn external_kind_key(kind: ExternalEventKind) -> &'static str {
    match kind {
        ExternalEventKind::BossTargetableChanged => "boss_targetable_changed",
        ExternalEventKind::MovementChanged => "movement_changed",
        ExternalEventKind::RaidBuffWindowChanged => "raid_buff_window_changed",
        ExternalEventKind::TargetCountChanged => "target_count_changed",
        ExternalEventKind::PlayerDeath => "player_death",
        ExternalEventKind::PlayerRevived => "player_revived",
        ExternalEventKind::PlayerRevivedLb3 => "player_revived_lb3",
    }
}

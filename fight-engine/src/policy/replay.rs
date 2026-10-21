// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// 附加许可：FightEngine 链接例外 1.1 版，见 LICENSE-FightEngine-Linking-Exception。

//! 绝对时间策略回放：同步同刻场景事实，补充停止织入的等待并输出增量观测。

use serde::Deserialize;
use serde_json::{Value, json};

use crate::config::ActionKind;
use crate::engine::ActionRequest;
use crate::engine::simulator::{ActionSubmissionResult, Simulator};
use crate::outputs::build_death_token;
use crate::policy::PolicySession;
use crate::state::{ExternalCombatEvent, ExternalEventKind};

#[derive(Deserialize)]
struct ReplayRequest {
    timestamp: f64,
    skill_key: String,
    actual_cast_seconds: Option<f64>,
}

#[derive(Deserialize)]
struct ReplayFact {
    timestamp: f64,
    event_kind: ExternalEventKind,
    value: Option<bool>,
    target_count: Option<i32>,
    remaining_seconds: Option<f64>,
    mp: Option<i32>,
}

pub fn run(
    simulator: &mut Simulator,
    policy: &mut PolicySession,
    requests_json: &str,
    facts_json: &str,
    fight_duration: f64,
    policy_action_key: &str,
) -> Result<Value, String> {
    if simulator.history_limit().is_some() {
        return Err("policy sequence replay requires full history retention".into());
    }
    if !fight_duration.is_finite() {
        return Err("fight duration must be finite".into());
    }
    if policy_action_key.trim().is_empty() {
        return Err("policy action key must not be empty".into());
    }
    let requests: Vec<ReplayRequest> =
        serde_json::from_str(requests_json).map_err(|error| error.to_string())?;
    if requests.is_empty() {
        return Err("replay requests must not be empty".into());
    }
    for request in &requests {
        if !request.timestamp.is_finite() || request.skill_key.trim().is_empty() {
            return Err("replay requests require finite timestamps and skill keys".into());
        }
        if request
            .actual_cast_seconds
            .is_some_and(|duration| !duration.is_finite() || duration < 0.0)
        {
            return Err("actual cast duration must be finite and non-negative".into());
        }
    }
    let supplied: Vec<ReplayFact> =
        serde_json::from_str(facts_json).map_err(|error| error.to_string())?;
    let mut facts: Vec<ExternalCombatEvent> = Vec::with_capacity(supplied.len());
    for fact in supplied {
        if !fact.timestamp.is_finite()
            || fact.timestamp < simulator.time()
            || facts
                .last()
                .is_some_and(|previous| fact.timestamp < previous.timestamp)
        {
            return Err("replay facts must be ordered at or after the initial time".into());
        }
        facts.push(ExternalCombatEvent {
            timestamp: fact.timestamp,
            kind: fact.event_kind,
            value: fact.value,
            target_count: fact.target_count,
            remaining_seconds: fact.remaining_seconds,
            mp: fact.mp,
        });
    }
    let epsilon = simulator.rules().schema.scene_epsilon;
    let initial_time = simulator.time();
    let mut replay = Replay {
        simulator,
        policy,
        facts,
        fact_index: 0,
        synced_time: initial_time,
        history_cursor: 0,
        observations: Vec::new(),
        pending_markers: Vec::new(),
        pending_death: None,
    };
    let mut cursor = initial_time;
    for (index, request) in requests.iter().enumerate() {
        if cursor < request.timestamp - epsilon {
            let decision_time = cursor;
            replay.sync_and_advance(decision_time, false)?;
            let gcd_remaining = replay.simulator.get_state_without_history().gcd_remaining();
            if gcd_remaining > epsilon {
                let window_end = decision_time + gcd_remaining;
                let next_skill = replay.simulator.rules().resolve_skill(&request.skill_key)?;
                if next_skill.kind != ActionKind::Ogcd || request.timestamp >= window_end - epsilon
                {
                    let next_observation = window_end.max(request.timestamp);
                    let context = replay.observe(decision_time, next_observation)?;
                    replay.policy.record(
                        replay.simulator,
                        decision_time,
                        policy_action_key,
                        next_observation,
                    )?;
                    replay.add_frame(index, true, decision_time, context, Value::Null);
                }
            }
        }
        replay.sync_and_advance(request.timestamp, true)?;
        let next_request = requests
            .get(index + 1)
            .map_or(fight_duration, |next| next.timestamp);
        let context = replay.observe(request.timestamp, request.timestamp.max(next_request))?;
        let submission = replay.simulator.submit_action(
            ActionRequest {
                timestamp: request.timestamp,
                skill_key: request.skill_key.clone(),
            },
            request.actual_cast_seconds,
        )?;
        replay.add_frame(
            index,
            false,
            request.timestamp,
            context,
            submission_value(&submission),
        );
        if !submission.accepted {
            break;
        }
        if replay.pending_death.is_some()
            && requests
                .get(index + 1)
                .is_none_or(|next| next.timestamp != request.timestamp)
        {
            let death = replay
                .pending_death
                .take()
                .expect("pending death was checked");
            replay.apply_facts(&death)?;
        }
        // 生效时刻只参与下一轮等待判断，不能抢先越过后续请求。
        cursor = submission.effect_timestamp.unwrap_or(request.timestamp);
    }
    Ok(json!({"observations": replay.observations}))
}

struct Replay<'a> {
    simulator: &'a mut Simulator,
    policy: &'a mut PolicySession,
    facts: Vec<ExternalCombatEvent>,
    fact_index: usize,
    synced_time: f64,
    history_cursor: u64,
    observations: Vec<Value>,
    pending_markers: Vec<Value>,
    pending_death: Option<Vec<ExternalCombatEvent>>,
}

impl Replay<'_> {
    fn sync_and_advance(&mut self, timestamp: f64, before_action: bool) -> Result<(), String> {
        if timestamp < self.synced_time {
            return Err(
                "scene fact cursor cannot move backwards; reset the trajectory first".into(),
            );
        }
        while self.fact_index < self.facts.len()
            && self.facts[self.fact_index].timestamp <= timestamp
        {
            let mut end = self.fact_index + 1;
            while end < self.facts.len()
                && self.facts[end].timestamp == self.facts[self.fact_index].timestamp
            {
                end += 1;
            }
            let mut batch = self.facts[self.fact_index..end].to_vec();
            if before_action
                && batch[0].timestamp == timestamp
                && batch
                    .iter()
                    .any(|fact| fact.kind == ExternalEventKind::PlayerDeath)
                && !batch.iter().any(|fact| {
                    matches!(
                        fact.kind,
                        ExternalEventKind::PlayerRevived | ExternalEventKind::PlayerRevivedLb3
                    )
                })
            {
                self.pending_death = Some(
                    batch
                        .iter()
                        .filter(|fact| fact.kind == ExternalEventKind::PlayerDeath)
                        .cloned()
                        .collect(),
                );
                batch.retain(|fact| fact.kind != ExternalEventKind::PlayerDeath);
            }
            if !batch.is_empty() {
                self.apply_facts(&batch)?;
            }
            self.fact_index = end;
        }
        self.synced_time = timestamp;
        self.simulator.advance_clock_to(timestamp)
    }

    fn apply_facts(&mut self, batch: &[ExternalCombatEvent]) -> Result<(), String> {
        let applied = self.simulator.apply_external_events(batch)?;
        if !applied.accepted {
            return Err(format!(
                "scene facts rejected at {}: {}",
                batch[0].timestamp, applied.reason
            ));
        }
        if applied.token.is_some() {
            let cursor = applied
                .history_cursor
                .ok_or("death marker is missing its history cursor")?;
            let mut marker = build_death_token(applied.timestamp, cursor);
            marker["step"] = json!(self.observations.len() + 1);
            self.pending_markers.push(marker);
        }
        Ok(())
    }

    fn observe(&mut self, timestamp: f64, next_observation: f64) -> Result<Value, String> {
        self.simulator.advance_clock_to(timestamp)?;
        let context = self.policy.build_vector_delta_context(
            self.simulator,
            next_observation,
            self.history_cursor,
        )?;
        self.history_cursor = context["history_cursor"]
            .as_u64()
            .ok_or("vector observation is missing history_cursor")?;
        Ok(context)
    }

    fn add_frame(
        &mut self,
        source_index: usize,
        is_wait: bool,
        timestamp: f64,
        context: Value,
        submission: Value,
    ) {
        let mut frame = json!({
            "source_index":source_index,"is_wait":is_wait,"timestamp":timestamp,
            "context":context,"submission":submission,
        });
        if !self.pending_markers.is_empty() {
            frame["history_markers"] = json!(std::mem::take(&mut self.pending_markers));
        }
        self.observations.push(frame);
    }
}

fn submission_value(submission: &ActionSubmissionResult) -> Value {
    json!({
        "accepted":submission.accepted,"queued":submission.queued,"reason":submission.reason,
        "accepted_timestamp":submission.accepted_timestamp,"effect_timestamp":submission.effect_timestamp,
    })
}

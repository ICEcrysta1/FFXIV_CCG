// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

//! 多队列生命周期只使用注册锁；每条队列的执行与输出使用自己的锁。

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::PolicyActionRegistry;
use crate::outputs::OutputFormatter;
use crate::policy::{PolicyDecision, PolicySession};
use crate::state::{ExternalCombatEvent, ValidationResult};

use super::simulator::{ActionSubmissionResult, ExternalEventResult, Simulator};
use super::{ActionRequest, EngineRules};

pub const MAX_CAPACITY: usize = 100;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionOptions {
    pub history_limit: Option<usize>,
    pub actual_base_gcd: Option<f64>,
    pub initial_timestamp: f64,
    pub fight_remaining: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionResult<T> {
    pub timestamp: f64,
    pub next_scheduled_event_time: Option<f64>,
    pub value: T,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionStatistics {
    pub timestamp: f64,
    pub action_history_count: usize,
    pub policy_history_count: usize,
    pub pending_event_count: usize,
    pub queue_entry_count: usize,
    pub pending_settlement_count: usize,
}

struct EngineState {
    next_id: u64,
    closed: bool,
    sessions: BTreeMap<u64, Arc<SimulationSession>>,
}

pub struct SimulationEngine {
    rules: Arc<EngineRules>,
    policy_registry: Arc<PolicyActionRegistry>,
    capacity: usize,
    state: Mutex<EngineState>,
}

impl SimulationEngine {
    pub fn new(
        rules: Arc<EngineRules>,
        policy_registry: Arc<PolicyActionRegistry>,
        capacity: usize,
    ) -> Result<Arc<Self>, String> {
        if !(1..=MAX_CAPACITY).contains(&capacity) {
            return Err(format!(
                "engine capacity must be between 1 and {MAX_CAPACITY}"
            ));
        }
        Ok(Arc::new(Self {
            rules,
            policy_registry,
            capacity,
            state: Mutex::new(EngineState {
                next_id: 0,
                closed: false,
                sessions: BTreeMap::new(),
            }),
        }))
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn job_tag(&self) -> &str {
        self.rules.job_tag()
    }

    pub fn active_count(&self) -> Result<usize, String> {
        Ok(lock(&self.state)?.sessions.len())
    }

    pub fn create_session(
        self: &Arc<Self>,
        options: SessionOptions,
    ) -> Result<Arc<SimulationSession>, String> {
        let mut state = lock(&self.state)?;
        if state.closed {
            return Err("simulation engine is closed".into());
        }
        if state.sessions.len() >= self.capacity {
            return Err(format!(
                "simulation engine capacity reached: {}",
                self.capacity
            ));
        }
        let runtime = self.build_runtime(options)?;
        let id = state
            .next_id
            .checked_add(1)
            .ok_or("session identifier exhausted")?;
        let session = Arc::new(SimulationSession {
            id,
            owner: Arc::downgrade(self),
            runtime: Mutex::new(Some(runtime)),
        });
        state.next_id = id;
        state.sessions.insert(id, session.clone());
        Ok(session)
    }

    fn build_runtime(&self, options: SessionOptions) -> Result<SessionRuntime, String> {
        if !options.initial_timestamp.is_finite() {
            return Err("initial timestamp must be finite".into());
        }
        if options
            .actual_base_gcd
            .is_some_and(|gcd| !gcd.is_finite() || gcd <= 0.0)
        {
            return Err("actual base gcd must be finite and positive".into());
        }
        if options.fight_remaining.is_some_and(|duration| {
            !duration.is_finite()
                || duration < 0.0
                || !(options.initial_timestamp + duration).is_finite()
        }) {
            return Err("fight duration must be finite and non-negative".into());
        }
        let mut state =
            self.rules
                .initial_state(options.fight_remaining, None, options.initial_timestamp)?;
        state.base_gcd = options.actual_base_gcd;
        Ok(SessionRuntime {
            simulator: Simulator::with_history_limit(
                self.rules.clone(),
                state,
                options.history_limit,
            )?,
            policy: PolicySession::new(self.policy_registry.clone(), options.history_limit),
            formatter: OutputFormatter::new(
                self.rules.project.clone(),
                self.rules.system.clone(),
                self.rules.job.clone(),
                self.rules.schema.clone(),
                options.history_limit,
            )?,
        })
    }

    fn release(&self, id: u64) -> Result<(), String> {
        let mut state = lock(&self.state)?;
        if let Some(session) = state.sessions.remove(&id) {
            *lock(&session.runtime)? = None;
        }
        Ok(())
    }

    pub fn close(&self) -> Result<(), String> {
        let mut state = lock(&self.state)?;
        if !state.closed {
            state.closed = true;
            for session in state.sessions.values() {
                *lock(&session.runtime)? = None;
            }
            state.sessions.clear();
        }
        Ok(())
    }
}

impl Drop for SimulationEngine {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub(super) struct SessionRuntime {
    pub(super) simulator: Simulator,
    policy: PolicySession,
    pub(super) formatter: OutputFormatter,
}

pub struct SimulationSession {
    id: u64,
    owner: Weak<SimulationEngine>,
    runtime: Mutex<Option<SessionRuntime>>,
}

impl SimulationSession {
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn is_closed(&self) -> Result<bool, String> {
        Ok(lock(&self.runtime)?.is_none())
    }

    /// 新运行上下文完整构造成功后才替换，参数错误不会销毁旧队列。
    pub fn reset(&self, options: SessionOptions) -> Result<SessionResult<bool>, String> {
        // 宿主引用先于队列锁取得，确保最后一个宿主引用在解锁后才析构。
        let owner = self.owner.upgrade().ok_or("simulation engine is closed")?;
        let mut runtime = lock(&self.runtime)?;
        if runtime.is_none() {
            return Err("simulation session is closed".into());
        }
        let next = owner.build_runtime(options)?;
        let result = session_result(&next.simulator, true);
        *runtime = Some(next);
        Ok(result)
    }

    pub fn advance_to(&self, timestamp: f64) -> Result<SessionResult<bool>, String> {
        self.execute(|runtime| {
            runtime.simulator.advance_clock_to(timestamp)?;
            Ok(true)
        })
    }

    pub fn submit_action(
        &self,
        timestamp: f64,
        skill_key: &str,
        actual_cast_seconds: Option<f64>,
    ) -> Result<SessionResult<ActionSubmissionResult>, String> {
        self.execute(|runtime| {
            if actual_cast_seconds.is_some_and(|duration| !duration.is_finite() || duration < 0.0) {
                return Err("actual cast duration must be finite and non-negative".into());
            }
            runtime.simulator.submit_action(
                ActionRequest {
                    timestamp,
                    skill_key: skill_key.into(),
                },
                actual_cast_seconds,
            )
        })
    }

    pub fn validate_action_at(
        &self,
        timestamp: f64,
        skill_key: &str,
    ) -> Result<SessionResult<ValidationResult>, String> {
        self.execute(|runtime| runtime.simulator.validate_action_at(timestamp, skill_key))
    }

    pub fn apply_external_events(
        &self,
        events: &[ExternalCombatEvent],
    ) -> Result<SessionResult<ExternalEventResult>, String> {
        self.execute(|runtime| runtime.simulator.apply_external_events(events))
    }

    pub fn record_policy_action(
        &self,
        timestamp: f64,
        action_key: &str,
        next_observation_timestamp: f64,
    ) -> Result<SessionResult<PolicyDecision>, String> {
        self.execute(|runtime| {
            runtime.policy.record(
                &mut runtime.simulator,
                timestamp,
                action_key,
                next_observation_timestamp,
            )
        })
    }

    pub fn observe_at(
        &self,
        timestamp: f64,
        format: &str,
        next_observation_timestamp: Option<f64>,
    ) -> Result<SessionResult<Value>, String> {
        self.execute(|runtime| {
            if !matches!(format, "seconds" | "gcd" | "vector") {
                return Err(format!("unsupported observe format: {format}"));
            }
            if format == "vector"
                && next_observation_timestamp
                    .is_none_or(|next| !next.is_finite() || next < timestamp)
            {
                return Err("vector observation requires a finite, non-past next timestamp".into());
            }
            runtime.simulator.advance_clock_to(timestamp)?;
            if format == "vector" {
                runtime
                    .policy
                    .build_vector_context(&runtime.simulator, next_observation_timestamp.unwrap())
            } else {
                runtime
                    .formatter
                    .format(&runtime.simulator.get_state(), format)
            }
        })
    }

    pub fn observe_vector_delta_at(
        &self,
        timestamp: f64,
        next_observation_timestamp: f64,
        after_sequence: u64,
    ) -> Result<SessionResult<Value>, String> {
        self.execute(|runtime| {
            if runtime.simulator.history_limit().is_some() {
                return Err(
                    "incremental vector observation requires full history retention".into(),
                );
            }
            if after_sequence > runtime.simulator.history_cursor() {
                return Err("history cursor is outside the current history".into());
            }
            if !next_observation_timestamp.is_finite() || next_observation_timestamp < timestamp {
                return Err("next observation timestamp must be finite and non-past".into());
            }
            runtime.simulator.advance_clock_to(timestamp)?;
            runtime.policy.build_vector_delta_context(
                &runtime.simulator,
                next_observation_timestamp,
                after_sequence,
            )
        })
    }

    pub fn replay_policy_sequence(
        &self,
        requests_json: &str,
        facts_json: &str,
        fight_duration: f64,
        policy_action_key: &str,
    ) -> Result<SessionResult<Value>, String> {
        self.execute(|runtime| {
            crate::policy::replay::run(
                &mut runtime.simulator,
                &mut runtime.policy,
                requests_json,
                facts_json,
                fight_duration,
                policy_action_key,
            )
        })
    }

    pub fn statistics(&self) -> Result<SessionStatistics, String> {
        let runtime = lock(&self.runtime)?;
        let runtime = runtime.as_ref().ok_or("simulation session is closed")?;
        let counts = runtime.simulator.statistics();
        Ok(SessionStatistics {
            timestamp: runtime.simulator.time(),
            action_history_count: counts.history,
            policy_history_count: runtime.policy.history_count(),
            pending_event_count: counts.pending_events,
            queue_entry_count: counts.queue_entries,
            pending_settlement_count: counts.pending_settlements,
        })
    }

    /// 关闭不先取队列锁，保持注册锁到队列锁的统一顺序。
    pub fn close(&self) -> Result<(), String> {
        if let Some(owner) = self.owner.upgrade() {
            owner.release(self.id)
        } else {
            *lock(&self.runtime)? = None;
            Ok(())
        }
    }

    pub(super) fn execute<T>(
        &self,
        action: impl FnOnce(&mut SessionRuntime) -> Result<T, String>,
    ) -> Result<SessionResult<T>, String> {
        let mut runtime = lock(&self.runtime)?;
        let runtime = runtime.as_mut().ok_or("simulation session is closed")?;
        let value = action(runtime)?;
        Ok(session_result(&runtime.simulator, value))
    }
}

fn session_result<T>(simulator: &Simulator, value: T) -> SessionResult<T> {
    SessionResult {
        timestamp: simulator.time(),
        next_scheduled_event_time: simulator.next_scheduled_event_time(),
        value,
    }
}

fn lock<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>, String> {
    mutex
        .lock()
        .map_err(|_| "simulation lock was poisoned".into())
}

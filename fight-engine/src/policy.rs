// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// 附加许可：FightEngine 链接例外 1.1 版，见 LICENSE-FightEngine-Linking-Exception。

//! 纯策略决策与冻结模型历史；等待只记录当前事实，不推进游戏时间。

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::{PolicyActionDefinition, PolicyActionRegistry};
use crate::engine::EngineRules;
use crate::engine::simulator::{Simulator, reserve_decision_id};
use crate::outputs::{ModelHistoryRow, OutputContextBuilder, SkillTokenInput, build_skill_token};
use crate::state::skills::SkillState;
use crate::state::{CombatState, ModelStateFrame, ModelStateSnapshot};
use crate::system::timeline::TIME_EPSILON;

pub mod replay;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub action: PolicyActionDefinition,
    pub timestamp: f64,
    pub gcd_index: i32,
    pub state_before: CombatState,
    pub state_after: CombatState,
    pub model_state: ModelStateSnapshot,
    pub history_sequence: u64,
}

impl PolicyDecision {
    fn snapshot(&self) -> Self {
        Self {
            action: self.action.clone(),
            timestamp: self.timestamp,
            gcd_index: self.gcd_index,
            state_before: self.state_before.clone_without_history(),
            state_after: self.state_after.clone_without_history(),
            model_state: self.model_state.clone(),
            history_sequence: self.history_sequence,
        }
    }
}

struct ContextCache {
    rules: Arc<EngineRules>,
    output: OutputContextBuilder,
    action_keys: Vec<String>,
    real_rows: BTreeMap<usize, (Arc<ModelHistoryRow>, Arc<ModelHistoryRow>)>,
    wait_rows: BTreeMap<usize, (Arc<PolicyDecision>, Arc<ModelHistoryRow>)>,
}

pub struct PolicySession {
    registry: Arc<PolicyActionRegistry>,
    entries: Vec<Arc<PolicyDecision>>,
    history_limit: Option<usize>,
    context: Option<ContextCache>,
}

impl PolicySession {
    pub fn new(registry: Arc<PolicyActionRegistry>, history_limit: Option<usize>) -> Self {
        Self {
            registry,
            entries: Vec::new(),
            history_limit,
            context: None,
        }
    }

    pub fn history_count(&self) -> usize {
        self.entries.len()
    }

    pub fn history_limit(&self) -> Option<usize> {
        self.history_limit
    }

    pub fn record(
        &mut self,
        simulator: &mut Simulator,
        timestamp: f64,
        action_key: &str,
        next_observation_timestamp: f64,
    ) -> Result<PolicyDecision, String> {
        let action = self.registry.get(action_key)?.clone();
        if !timestamp.is_finite() || (simulator.time() - timestamp).abs() > TIME_EPSILON {
            return Err(format!(
                "policy decision timestamp must equal current observation time: {timestamp} != {}",
                simulator.time()
            ));
        }
        if !next_observation_timestamp.is_finite() || next_observation_timestamp < timestamp {
            return Err(
                "next observation timestamp must be finite and at or after the decision".into(),
            );
        }
        let before = simulator.get_state_without_history();
        let (model_state, history_sequence) =
            simulator.record_model_decision(reserve_decision_id()?, &before, true)?;
        let decision = PolicyDecision {
            action,
            timestamp,
            gcd_index: before.gcd_index,
            state_after: before.clone_without_history(),
            state_before: before,
            model_state,
            history_sequence,
        };
        self.entries.push(Arc::new(decision.snapshot()));
        trim(&mut self.entries, self.history_limit);
        Ok(decision)
    }

    pub fn create_snapshot(&self) -> Vec<PolicyDecision> {
        self.entries
            .iter()
            .map(|decision| decision.snapshot())
            .collect()
    }

    pub fn restore_snapshot(&mut self, snapshot: &[PolicyDecision]) {
        self.entries = snapshot
            .iter()
            .map(|decision| Arc::new(decision.snapshot()))
            .collect();
        trim(&mut self.entries, self.history_limit);
    }

    pub fn fork(&self) -> Self {
        let mut fork = Self::new(self.registry.clone(), self.history_limit);
        fork.restore_snapshot(&self.create_snapshot());
        fork
    }

    pub fn build_vector_context(
        &mut self,
        simulator: &Simulator,
        next_observation_timestamp: f64,
    ) -> Result<Value, String> {
        self.build_context(simulator, next_observation_timestamp, None)
    }

    pub fn build_vector_delta_context(
        &mut self,
        simulator: &Simulator,
        next_observation_timestamp: f64,
        after_sequence: u64,
    ) -> Result<Value, String> {
        self.build_context(simulator, next_observation_timestamp, Some(after_sequence))
    }

    fn build_context(
        &mut self,
        simulator: &Simulator,
        next_observation_timestamp: f64,
        after_sequence: Option<u64>,
    ) -> Result<Value, String> {
        let state = simulator.get_state();
        if !next_observation_timestamp.is_finite() || next_observation_timestamp < state.time() {
            return Err(
                "next observation timestamp must be finite and at or after current time".into(),
            );
        }
        if let Some(after) = after_sequence {
            if simulator.history_limit().is_some() || self.history_limit.is_some() {
                return Err(
                    "incremental vector observation requires full history retention".into(),
                );
            }
            if after > state.last_history_sequence {
                return Err("history start must be at or before history_cursor".into());
            }
        }
        if self
            .context
            .as_ref()
            .is_none_or(|cache| !Arc::ptr_eq(&cache.rules, simulator.rules()))
        {
            let rules = simulator.rules();
            let mut action_keys = rules.action_keys().to_vec();
            action_keys.extend(
                self.registry
                    .actions()
                    .iter()
                    .map(|action| action.key.clone()),
            );
            action_keys.sort();
            if action_keys.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err("policy and real action keys must be unique".into());
            }
            self.context = Some(ContextCache {
                rules: rules.clone(),
                output: OutputContextBuilder::new(
                    rules.project().clone(),
                    rules.system().clone(),
                    rules.job().clone(),
                    rules.schema.clone(),
                    // 实际历史已由所属模拟器裁剪，合并后再应用策略历史上限。
                    None,
                )?,
                action_keys,
                real_rows: BTreeMap::new(),
                wait_rows: BTreeMap::new(),
            });
        }
        let cache = self.context.as_mut().expect("context was just constructed");
        let mut values: BTreeMap<_, _> = simulator
            .rules()
            .action_keys()
            .iter()
            .cloned()
            .zip(simulator.rules().action_values(&state)?)
            .collect();
        values.extend(
            self.registry
                .actions()
                .iter()
                .map(|action| (action.key.clone(), action.value)),
        );
        let current = compose(&simulator.capture_model_state(&state)?, &self.registry)?;
        let mut merged = Vec::new();
        let mut real_rows = BTreeMap::new();
        for original in cache.output.collect_history(&state)? {
            let identity = Arc::as_ptr(&original) as usize;
            let row = if let Some((source, row)) = cache.real_rows.get(&identity)
                && Arc::ptr_eq(source, &original)
            {
                row.clone()
            } else {
                Arc::new(ModelHistoryRow {
                    model_state: compose(&original.model_state, &self.registry)?,
                    ..(*original).clone()
                })
            };
            real_rows.insert(identity, (original, row.clone()));
            merged.push(row);
        }
        cache.real_rows = real_rows;
        let mut wait_rows = BTreeMap::new();
        for decision in &self.entries {
            let identity = Arc::as_ptr(decision) as usize;
            let row = if let Some((source, row)) = cache.wait_rows.get(&identity)
                && Arc::ptr_eq(source, decision)
            {
                row.clone()
            } else {
                let consumed = cache
                    .output
                    .build_noop_resource_transition(&decision.state_before)?;
                let action = &decision.action;
                Arc::new(ModelHistoryRow {
                    sequence: decision.history_sequence,
                    model_state: compose(&decision.model_state, &self.registry)?,
                    skill: build_skill_token(SkillTokenInput {
                        skill_id: action.raw_id,
                        skill_key: &action.key,
                        skill_name: &action.name,
                        potency: 0.0,
                        value: action.value,
                        kind: &action.kind,
                        actual_mp_cost: 0,
                        cast_time_seconds: 0.0,
                        gcd_window_seconds: 0.0,
                        is_legal: true,
                        invalid_reason: "",
                        next_cooldown_seconds: 0.0,
                        available_charges: 1,
                        max_charges: 1,
                        job_resources_consumed: &consumed,
                    })?,
                    execution_metrics: BTreeMap::from([
                        (
                            "cumulative_potency".into(),
                            decision.state_after.cumulative_potency,
                        ),
                        (
                            "cumulative_dot_potency".into(),
                            decision.state_after.cumulative_dot_potency,
                        ),
                    ]),
                })
            };
            wait_rows.insert(identity, (decision.clone(), row.clone()));
            merged.push(row);
        }
        cache.wait_rows = wait_rows;
        merged.sort_by_key(|row| row.sequence);
        if merged.iter().any(|row| row.sequence == 0)
            || merged
                .windows(2)
                .any(|pair| pair[0].sequence == pair[1].sequence)
        {
            return Err("history sequence must be positive and unique".into());
        }
        if let Some(after) = after_sequence {
            if merged.len() as u64 != state.last_history_sequence
                || merged
                    .iter()
                    .enumerate()
                    .any(|(index, row)| row.sequence != index as u64 + 1)
            {
                return Err(
                    "incremental history sequence must be contiguous from 1 to history_cursor"
                        .into(),
                );
            }
            merged.retain(|row| row.sequence > after);
        } else {
            trim(&mut merged, self.history_limit);
        }
        let action_values = cache
            .action_keys
            .iter()
            .map(|key| values[key])
            .collect::<Vec<_>>();
        let mut output = cache.output.build_context(
            &state,
            &cache.action_keys,
            &action_values,
            &current,
            Some(&merged),
        )?;
        if let Some(after) = after_sequence {
            output["history_start"] = json!(after);
        }
        Ok(output)
    }
}

fn compose(
    snapshot: &ModelStateSnapshot,
    registry: &PolicyActionRegistry,
) -> Result<ModelStateSnapshot, String> {
    let frame = |frame: &ModelStateFrame| -> Result<Arc<ModelStateFrame>, String> {
        let mut skill_state = frame.skill_state.clone();
        for action in registry.actions() {
            if skill_state
                .insert(
                    action.key.clone(),
                    SkillState {
                        available: true,
                        current_charges: 1,
                        next_cooldown_seconds: 0.0,
                    },
                )
                .is_some()
            {
                return Err("policy and real action keys must be unique".into());
            }
        }
        Ok(Arc::new(ModelStateFrame {
            state: frame.state.clone(),
            skill_state,
        }))
    };
    Ok(ModelStateSnapshot {
        previous_action_after: frame(&snapshot.previous_action_after)?,
        request_state: frame(&snapshot.request_state)?,
    })
}

fn trim<T>(items: &mut Vec<T>, limit: Option<usize>) {
    if let Some(limit) = limit {
        let remove = items.len().saturating_sub(limit);
        items.drain(..remove);
    }
}

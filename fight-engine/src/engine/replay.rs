// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// 附加许可：FightEngine 链接例外 1.1 版，见 LICENSE-FightEngine-Linking-Exception。

//! 绝对请求序列、固定随机策略与完整模拟快照缓存。

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::state::CombatState;
use crate::system::timeline::SimulationSnapshot;

use super::ActionRequest;
use super::sessions::SimulationSession;
use super::simulator::{ActionSubmissionResult, EnginePayload, Simulator};

pub const DEFAULT_RANDOM_SEED: i64 = 20260814;
pub const DEFAULT_RANDOM_MAX_STEPS: i32 = 100;

/// 整段序列在队列锁内执行，最后只推进到已接受动作的最晚效果时刻。
pub fn run_action_sequence(
    session: &SimulationSession,
    sequence: &[ActionRequest],
) -> Result<Value, String> {
    session
        .execute(|runtime| {
            let simulator = &mut runtime.simulator;
            let mut applied_actions = Vec::with_capacity(sequence.len());
            let mut final_effect_timestamp: Option<f64> = None;
            for (index, request) in sequence.iter().enumerate() {
                let result = simulator.submit_action(request.clone(), None)?;
                if !result.accepted {
                    return Err(format!(
                        "{} at {} was rejected: {}",
                        request.skill_key, request.timestamp, result.reason
                    ));
                }
                let effect = result
                    .effect_timestamp
                    .or(result.accepted_timestamp)
                    .unwrap_or(request.timestamp);
                final_effect_timestamp = Some(
                    final_effect_timestamp
                        .unwrap_or(f64::NEG_INFINITY)
                        .max(effect),
                );
                applied_actions.push(json!({
                    "index": index + 1,
                    "action": request.skill_key,
                    "request_time_seconds": request.timestamp,
                    "accepted_time_seconds": result.accepted_timestamp.expect("已接受动作必须有接受时刻"),
                    "effect_time_seconds": result.effect_timestamp.expect("已接受动作必须有效果时刻"),
                    "queued": result.queued,
                    "action_instance_id": result.action_instance_id.expect("已接受动作必须有实例标识"),
                }));
            }
            if let Some(timestamp) = final_effect_timestamp
                && timestamp > simulator.time()
            {
                simulator.advance_clock_to(timestamp)?;
            }
            let final_output = if sequence.is_empty() {
                Value::Null
            } else {
                let state = simulator.get_state();
                let current = simulator.capture_model_state(&state)?;
                let action_values = simulator.rules().action_values(&state)?;
                runtime.formatter.format_vectors(
                    &state,
                    simulator.rules().action_keys(),
                    &action_values,
                    &current,
                    None,
                )?
            };
            Ok(json!({
                "requested_sequence": sequence.iter().map(|request| json!({
                    "timestamp": request.timestamp,
                    "action": request.skill_key,
                })).collect::<Vec<_>>(),
                "applied_actions": applied_actions,
                "final_output": final_output,
            }))
        })
        .map(|receipt| receipt.value)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomSequenceStepKind {
    SubmitAction,
    AdvanceTo,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RandomSequenceStep {
    pub kind: RandomSequenceStepKind,
    pub step_index: i32,
    pub legal_keys: Vec<String>,
    pub chosen_index: Option<usize>,
    pub action: Option<String>,
    pub timestamp: f64,
    pub result: Option<ActionSubmissionResult>,
    pub state_after: CombatState,
}

/// 无合法动作时显式推进到下一内部事件；只在实际选择动作时推进随机数状态。
pub fn run_random_sequence(
    session: &SimulationSession,
    seed: i64,
    max_steps: i32,
) -> Result<Vec<RandomSequenceStep>, String> {
    session
        .execute(|runtime| {
            let simulator = &mut runtime.simulator;
            let mut rng_state = seed as u64;
            let mut steps = Vec::new();
            for step_index in 1..=max_steps {
                let legal_keys = simulator.available_action_keys_at(simulator.time())?;
                if legal_keys.is_empty() {
                    let timestamp = simulator
                        .next_scheduled_event_time()
                        .filter(|next| *next > simulator.time())
                        .ok_or_else(|| {
                            format!("决策卡死：第 {step_index} 步无合法动作且没有未来事件")
                        })?;
                    let state_after = simulator.advance_to(timestamp)?;
                    steps.push(RandomSequenceStep {
                        kind: RandomSequenceStepKind::AdvanceTo,
                        step_index,
                        legal_keys,
                        chosen_index: None,
                        action: None,
                        timestamp,
                        result: None,
                        state_after,
                    });
                    continue;
                }
                let chosen_index = (next_u64(&mut rng_state) % legal_keys.len() as u64) as usize;
                let action = legal_keys[chosen_index].clone();
                let result = simulator.submit_action(
                    ActionRequest {
                        timestamp: simulator.time(),
                        skill_key: action.clone(),
                    },
                    None,
                )?;
                if !result.accepted {
                    return Err(format!(
                        "合法动作 {action} 在提交时被拒绝: {}",
                        result.reason
                    ));
                }
                steps.push(RandomSequenceStep {
                    kind: RandomSequenceStepKind::SubmitAction,
                    step_index,
                    legal_keys,
                    chosen_index: Some(chosen_index),
                    action: Some(action),
                    timestamp: simulator.time(),
                    result: Some(result),
                    state_after: simulator.get_state(),
                });
            }
            Ok(steps)
        })
        .map(|receipt| receipt.value)
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// 缓存从独立 fork 起步，保存与返回的快照分别克隆以隔离调用方修改。
pub struct ReplayStateCache {
    simulator: Simulator,
    snapshots: Vec<SimulationSnapshot<EnginePayload>>,
}

impl ReplayStateCache {
    pub fn new(simulator: &Simulator) -> Self {
        Self {
            simulator: simulator.fork(),
            snapshots: Vec::new(),
        }
    }

    pub fn current_state(&self) -> CombatState {
        self.simulator.get_state()
    }

    pub fn push_snapshot(&mut self) -> SimulationSnapshot<EnginePayload> {
        let snapshot = self.simulator.create_snapshot();
        self.snapshots.push(snapshot.clone());
        snapshot
    }

    pub fn snapshot_at(&self, index: usize) -> Result<SimulationSnapshot<EnginePayload>, String> {
        self.snapshots
            .get(index)
            .cloned()
            .ok_or_else(|| format!("snapshot index out of range: {index}"))
    }

    pub fn advance_to(&mut self, timestamp: f64, snapshot: bool) -> Result<CombatState, String> {
        let state = self.simulator.advance_to(timestamp)?;
        if snapshot {
            self.push_snapshot();
        }
        Ok(state)
    }

    pub fn submit_action(
        &mut self,
        request: ActionRequest,
        snapshot: bool,
    ) -> Result<ActionSubmissionResult, String> {
        let result = self.simulator.submit_action(request, None)?;
        if snapshot {
            self.push_snapshot();
        }
        Ok(result)
    }

    pub fn restore_snapshot(
        &mut self,
        snapshot: &SimulationSnapshot<EnginePayload>,
    ) -> Result<(), String> {
        self.simulator.restore_snapshot(snapshot)
    }
}

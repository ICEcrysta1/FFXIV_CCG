// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

//! 战斗状态与不可变历史；剩余时间始终由绝对截止时刻派生。

use std::collections::BTreeMap;
use std::sync::Arc;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::config::ResourceValue;

pub mod resources;
pub mod status;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationResult {
    pub ok: bool,
    pub reason: String,
}

impl ValidationResult {
    pub fn valid() -> Self {
        Self {
            ok: true,
            reason: String::new(),
        }
    }

    pub fn invalid(reason: impl Into<String>) -> Self {
        Self {
            ok: false,
            reason: reason.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalEventKind {
    BossTargetableChanged,
    MovementChanged,
    RaidBuffWindowChanged,
    TargetCountChanged,
    PlayerDeath,
    PlayerRevived,
    PlayerRevivedLb3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExternalCombatEvent {
    pub timestamp: f64,
    pub kind: ExternalEventKind,
    pub value: Option<bool>,
    pub target_count: Option<i32>,
    pub remaining_seconds: Option<f64>,
    pub mp: Option<i32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StatusState {
    pub expires_at: f64,
    pub stacks: i32,
}

impl StatusState {
    pub fn new(now: f64, duration: f64, stacks: i32) -> Self {
        Self {
            expires_at: now + duration,
            stacks,
        }
    }

    pub fn remaining(&self, now: f64) -> f64 {
        (self.expires_at - now).max(0.0)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DotState {
    pub expires_at: f64,
    pub potency_per_tick: f64,
    pub tick_interval: f64,
    pub next_tick_at: f64,
}

impl DotState {
    pub fn new(
        now: f64,
        duration: f64,
        potency_per_tick: f64,
        tick_interval: f64,
        next_tick_in_seconds: f64,
    ) -> Self {
        Self {
            expires_at: now + duration,
            potency_per_tick,
            tick_interval,
            next_tick_at: now + next_tick_in_seconds,
        }
    }

    pub fn remaining(&self, now: f64) -> f64 {
        (self.expires_at - now).max(0.0)
    }

    pub fn next_tick_in_seconds(&self, now: f64) -> f64 {
        (self.next_tick_at - now).max(0.0)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CooldownState {
    pub available_charges: i32,
    pub recharge_ready_at: Vec<f64>,
}

impl CooldownState {
    pub fn new(now: f64, available_charges: i32, recharge_timers: Vec<f64>) -> Self {
        Self {
            available_charges,
            recharge_ready_at: recharge_timers.into_iter().map(|time| now + time).collect(),
        }
    }

    pub fn recharge_timers(&self, now: f64) -> Vec<f64> {
        self.recharge_ready_at
            .iter()
            .map(|deadline| (deadline - now).max(0.0))
            .collect()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JobState {
    pub resources: BTreeMap<String, ResourceValue>,
    pub statuses: IndexMap<String, StatusState>,
    pub timeline_deadlines: BTreeMap<String, f64>,
}

impl JobState {
    pub fn has_status(&self, now: f64, key: &str) -> bool {
        self.statuses
            .get(key)
            .is_some_and(|status| status.remaining(now) > 0.0 && status.stacks > 0)
    }

    pub fn status_stacks(&self, now: f64, key: &str) -> i32 {
        self.statuses
            .get(key)
            .filter(|status| status.remaining(now) > 0.0)
            .map_or(0, |status| status.stacks)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CombatState {
    /// 只由中心时间线推进，职业与系统规则只读取。
    pub(crate) time: f64,
    pub gcd_index: i32,
    pub base_gcd: Option<f64>,
    pub fight_ends_at: f64,
    pub next_downtime_starts_at: Option<f64>,
    pub downtime_ends_at: f64,
    downtime_duration: f64,
    pub mp: i32,
    pub max_mp: i32,
    pub is_dead: bool,
    pub natural_mp_last_tick_at: f64,
    pub cast_ends_at: f64,
    pub gcd_ready_at: f64,
    pub weave_ends_at: f64,
    pub ogcds_weaved: i32,
    pub max_ogcd_per_window: i32,
    pub is_moving: bool,
    boss_targetable: bool,
    pub target_count: i32,
    pub cumulative_potency: f64,
    pub cumulative_dot_potency: f64,
    pub current_potency: f64,
    pub current_gcd_dot_potency: f64,
    pub job: JobState,
    pub cooldowns: IndexMap<String, CooldownState>,
    pub statuses: IndexMap<String, StatusState>,
    pub dots: IndexMap<String, DotState>,
    /// 历史条目不可变；写入列表时通过 Arc::make_mut 保持分支隔离。
    pub history: Arc<Vec<Arc<ActionHistoryEntry>>>,
    pub last_decision_id: Option<u64>,
    pub last_decision_after: Option<Arc<ModelStateFrame>>,
    pub last_completed_decision_after: Option<Arc<ModelStateFrame>>,
    pub last_history_sequence: u64,
}

impl Default for CombatState {
    fn default() -> Self {
        Self {
            time: 0.0,
            gcd_index: 0,
            base_gcd: None,
            fight_ends_at: 600.0,
            next_downtime_starts_at: None,
            downtime_ends_at: 0.0,
            downtime_duration: 0.0,
            mp: 10_000,
            max_mp: 10_000,
            is_dead: false,
            natural_mp_last_tick_at: 0.0,
            cast_ends_at: 0.0,
            gcd_ready_at: 0.0,
            weave_ends_at: 0.0,
            ogcds_weaved: 0,
            max_ogcd_per_window: 3,
            is_moving: false,
            boss_targetable: true,
            target_count: 1,
            cumulative_potency: 0.0,
            cumulative_dot_potency: 0.0,
            current_potency: 0.0,
            current_gcd_dot_potency: 0.0,
            job: JobState::default(),
            cooldowns: IndexMap::new(),
            statuses: IndexMap::new(),
            dots: IndexMap::new(),
            history: Arc::new(Vec::new()),
            last_decision_id: None,
            last_decision_after: None,
            last_completed_decision_after: None,
            last_history_sequence: 0,
        }
    }
}

impl CombatState {
    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn fight_remaining(&self) -> f64 {
        (self.fight_ends_at - self.time).max(0.0)
    }

    pub fn next_downtime_eta(&self) -> Option<f64> {
        self.next_downtime_starts_at
            .map(|deadline| (deadline - self.time).max(0.0))
    }

    pub fn downtime_remaining(&self) -> f64 {
        if self.boss_targetable {
            self.downtime_duration
        } else {
            (self.downtime_ends_at - self.time).max(0.0)
        }
    }

    pub fn set_downtime_remaining(&mut self, duration: f64) {
        self.downtime_duration = duration;
        self.downtime_ends_at = self.time + duration;
    }

    pub fn boss_targetable(&self) -> bool {
        self.boss_targetable
    }

    pub fn set_boss_targetable(&mut self, value: bool) {
        if self.boss_targetable && !value {
            self.downtime_ends_at = self.time + self.downtime_duration;
        }
        self.boss_targetable = value;
    }

    pub fn natural_mp_tick_progress(&self) -> f64 {
        (self.time - self.natural_mp_last_tick_at).max(0.0)
    }

    pub fn cast_remaining(&self) -> f64 {
        (self.cast_ends_at - self.time).max(0.0)
    }

    pub fn gcd_remaining(&self) -> f64 {
        (self.gcd_ready_at - self.time).max(0.0)
    }

    pub fn weave_window_remaining(&self) -> f64 {
        (self.weave_ends_at - self.time).max(0.0)
    }

    pub fn has_status(&self, key: &str) -> bool {
        self.statuses
            .get(key)
            .is_some_and(|status| status.remaining(self.time) > 0.0 && status.stacks > 0)
    }

    pub fn status_stacks(&self, key: &str) -> i32 {
        self.statuses
            .get(key)
            .filter(|status| status.remaining(self.time) > 0.0)
            .map_or(0, |status| status.stacks)
    }

    pub fn dot_remaining(&self, key: &str) -> f64 {
        self.dots
            .get(key)
            .map_or(0.0, |dot| dot.remaining(self.time))
    }

    pub fn clone_without_history(&self) -> Self {
        Self {
            history: Arc::new(Vec::new()),
            ..self.clone()
        }
    }

    pub fn reserve_history_sequence(&mut self) -> Result<u64, String> {
        self.last_history_sequence = self
            .last_history_sequence
            .checked_add(1)
            .ok_or_else(|| "历史序号溢出".to_owned())?;
        Ok(self.last_history_sequence)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlayerContext {
    pub time_seconds: f64,
    pub mp: i32,
    pub max_mp: i32,
    pub mp_ratio: f64,
    pub gcd_index: i32,
    pub current_gcd_seconds: f64,
    pub fight_remaining_seconds: f64,
    pub gcd_remaining_seconds: f64,
    pub cast_remaining_seconds: f64,
    pub gcd_remaining_gcds: f64,
    pub weave_window_seconds: f64,
    pub weave_window_gcds: f64,
    pub ogcd_window_seconds: f64,
    pub ogcd_window_gcds: f64,
    pub boss_targetable: bool,
    pub next_untargetable_in_seconds: Option<f64>,
    pub next_untargetable_in_gcds: Option<f64>,
    pub downtime_remaining_seconds: f64,
    pub downtime_remaining_gcds: f64,
    pub is_moving: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TargetContext {
    pub cumulative_potency: f64,
    pub cumulative_dot_potency: f64,
    pub current_potency: f64,
    pub current_gcd_dot_potency: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BuffContextEntry {
    pub source: String,
    pub status_key: String,
    pub status_id: i32,
    pub remaining_seconds: f64,
    pub remaining_gcds: f64,
    pub stacks: i32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DotContextEntry {
    pub dot_key: String,
    pub remaining_seconds: f64,
    pub remaining_gcds: f64,
    pub tick_interval_seconds: f64,
    pub potency_per_tick: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StateContext {
    pub job_tag: String,
    pub job_name: String,
    pub player: PlayerContext,
    pub target: TargetContext,
    pub resources: BTreeMap<String, ResourceValue>,
    pub buffs: Vec<BuffContextEntry>,
    pub dots: Vec<DotContextEntry>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelStateFrame {
    pub state: StateContext,
    pub skill_availability: BTreeMap<String, bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelStateSnapshot {
    pub previous_action_after: Arc<ModelStateFrame>,
    pub request_state: Arc<ModelStateFrame>,
}

impl ModelStateSnapshot {
    pub fn capture(
        previous_action_after: Option<Arc<ModelStateFrame>>,
        request_state: Arc<ModelStateFrame>,
    ) -> Self {
        Self {
            previous_action_after: previous_action_after.unwrap_or_else(|| request_state.clone()),
            request_state,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionHistoryEntry {
    pub skill_key: String,
    pub skill_id: i32,
    pub skill_name: String,
    pub skill_kind: String,
    pub potency: f64,
    pub value: f64,
    pub mp_before: i32,
    pub mp_after: i32,
    pub cast_time_seconds: f64,
    pub cast_time_gcds: f64,
    pub gcd_window_seconds: f64,
    pub gcd_window_gcds: f64,
    pub is_legal: bool,
    pub invalid_reason: String,
    pub next_cooldown_seconds: f64,
    pub available_charges: i32,
    pub max_charges: i32,
    pub job_resources_before: BTreeMap<String, ResourceValue>,
    pub job_resources_after: BTreeMap<String, ResourceValue>,
    pub job_resources_consumed: BTreeMap<String, ResourceValue>,
    pub time_seconds: f64,
    pub gcd_index: i32,
    pub state_before: StateContext,
    pub state_after: StateContext,
    pub request_timestamp: Option<f64>,
    pub cast_completed_timestamp: Option<f64>,
    pub effect_timestamp: Option<f64>,
    pub action_instance_id: Option<u64>,
    pub model_state: Option<ModelStateSnapshot>,
    pub history_sequence: u64,
}

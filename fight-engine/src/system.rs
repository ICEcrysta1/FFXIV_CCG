// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// 附加许可：FightEngine 链接例外 1.1 版，见 LICENSE-FightEngine-Linking-Exception。

//! 公共玩家、MP、冷却、状态和目标 DoT 规则；职业规则由调用方独立组合。

use std::collections::BTreeSet;

use indexmap::IndexMap;

use crate::config::{
    ActionKind, EngineTimingConfig, SkillDefinition, StatusDefinition, SystemConfig,
};
use crate::state::status::StatusRules;
use crate::state::{
    CombatState, CooldownState, DotState, ExternalCombatEvent, ExternalEventKind, ValidationResult,
};

pub mod timeline;

use timeline::{
    TimelineEvent, TimelineEventKind as Kind, TimelineEventPriority as Priority,
    TimelineEventScope as Scope,
};

pub const ZERO_EPSILON: f64 = 0.0001;

/// 装配完成后只读共享；每条队列的可变值均保存在 CombatState。
#[derive(Clone, Debug)]
pub struct SystemRules {
    timing: EngineTimingConfig,
    config: SystemConfig,
    statuses: StatusRules,
    target_dot_keys: BTreeSet<String>,
}

impl SystemRules {
    pub fn new(timing: EngineTimingConfig, config: SystemConfig) -> Result<Self, String> {
        let interval = config.mp_recovery.tick_interval_seconds;
        if !interval.is_finite() || interval <= 0.0 {
            return Err("MP tick interval must be finite and positive".into());
        }
        Ok(Self {
            timing,
            statuses: StatusRules::new(config.statuses.clone()),
            config,
            target_dot_keys: BTreeSet::new(),
        })
    }

    pub fn timing(&self) -> &EngineTimingConfig {
        &self.timing
    }

    pub fn config(&self) -> &SystemConfig {
        &self.config
    }

    pub fn status_definitions(&self) -> &IndexMap<String, StatusDefinition> {
        self.statuses.definitions()
    }

    pub fn register_target_dots<'a>(
        &mut self,
        skills: impl IntoIterator<Item = &'a SkillDefinition>,
    ) {
        self.target_dot_keys = skills
            .into_iter()
            .filter(|skill| skill.dot_duration > 0.0)
            .map(|skill| skill.dot_key.as_ref().unwrap_or(&skill.key).clone())
            .collect();
    }

    pub fn registered_target_dot_keys(&self) -> &BTreeSet<String> {
        &self.target_dot_keys
    }

    pub fn initial_state(
        &self,
        fight_remaining: f64,
        max_mp: i32,
        start_time: f64,
    ) -> Result<CombatState, String> {
        if !start_time.is_finite()
            || !fight_remaining.is_finite()
            || fight_remaining < 0.0
            || !(start_time + fight_remaining).is_finite()
        {
            return Err("initial time and fight duration must be finite and valid".into());
        }
        let mut state = CombatState::default();
        state.time = start_time;
        state.fight_ends_at = start_time + fight_remaining;
        state.mp = max_mp;
        state.max_mp = max_mp;
        state.cast_ends_at = start_time;
        state.gcd_ready_at = start_time;
        state.weave_ends_at = start_time;
        state.natural_mp_last_tick_at = start_time;
        state.downtime_ends_at = start_time;
        Ok(state)
    }

    pub fn supports_behavior(&self, behavior: &str) -> bool {
        behavior == "grant_status"
    }

    pub fn validate_common_action(
        &self,
        state: &CombatState,
        skill: &SkillDefinition,
        is_instant: bool,
        has_available_charge: bool,
        include_temporal_locks: bool,
    ) -> ValidationResult {
        let reason = if state.is_dead {
            Some("player_dead")
        } else if !state.boss_targetable() && skill.requires_target {
            Some("boss_untargetable")
        } else if include_temporal_locks && state.cast_remaining() > ZERO_EPSILON {
            Some("cast_locked")
        } else if include_temporal_locks
            && skill.kind == ActionKind::Gcd
            && state.gcd_remaining() > ZERO_EPSILON
        {
            Some("gcd_locked")
        } else if skill.kind == ActionKind::Ogcd
            && state.gcd_remaining() > ZERO_EPSILON
            && state.ogcds_weaved >= state.max_ogcd_per_window
        {
            Some("ogcd_limit")
        } else if state.is_moving && !is_instant {
            Some("movement_locked")
        } else if include_temporal_locks && !has_available_charge {
            Some("cooldown_locked")
        } else {
            None
        };
        reason.map_or_else(ValidationResult::valid, ValidationResult::invalid)
    }

    pub fn spend_mp(&self, state: &mut CombatState, amount: i32) {
        state.mp = (state.mp - amount).max(0);
    }

    pub fn recover_mp(&self, state: &mut CombatState, amount: i32) {
        state.mp = (state.mp + amount).min(state.max_mp);
    }

    pub fn restore_mp(&self, state: &mut CombatState) {
        state.mp = state.max_mp;
    }

    pub fn apply_action_timing(
        &self,
        state: &mut CombatState,
        skill: &SkillDefinition,
        effective_gcd: f64,
        actual_cast_seconds: f64,
        gcd_start_timestamp: Option<f64>,
    ) {
        if skill.kind == ActionKind::Gcd {
            state.gcd_index += 1;
            state.cast_ends_at = state.time() + actual_cast_seconds;
            state.gcd_ready_at = state
                .time()
                .max(gcd_start_timestamp.unwrap_or(state.time()) + effective_gcd);
            state.weave_ends_at = state.time() + (effective_gcd - actual_cast_seconds).max(0.0);
            state.ogcds_weaved = 0;
        } else {
            state.ogcds_weaved += 1;
        }
    }

    pub fn validate_status(
        &self,
        state: &CombatState,
        key: &str,
    ) -> Result<ValidationResult, String> {
        self.statuses.validate(state.time(), &state.statuses, key)
    }

    pub fn grant_status(
        &self,
        state: &mut CombatState,
        key: &str,
        remaining: Option<f64>,
        stacks: Option<i32>,
    ) -> Result<(), String> {
        self.statuses
            .grant(state.time(), &mut state.statuses, key, remaining, stacks)
    }

    pub fn clear_status(&self, state: &mut CombatState, key: &str) {
        self.statuses.clear(&mut state.statuses, key);
    }

    pub fn consume_status_stack(&self, state: &mut CombatState, key: &str) {
        self.statuses.consume_stack(&mut state.statuses, key);
    }

    pub fn register_expiry_hook(
        &mut self,
        key: impl Into<String>,
        hook: impl Fn(&mut CombatState) -> Result<(), String> + Send + Sync + 'static,
    ) -> Result<(), String> {
        self.statuses.register_expiry_hook(key, hook)
    }

    pub fn has_available_charge(&self, state: &CombatState, skill: &SkillDefinition) -> bool {
        skill.cooldown <= 0.0 && skill.charges <= 1
            || state
                .cooldowns
                .get(&skill.key)
                .is_none_or(|bucket| bucket.available_charges > 0)
    }

    pub fn next_charge_ready_at(
        &self,
        state: &CombatState,
        skill: &SkillDefinition,
    ) -> Option<f64> {
        state.cooldowns.get(&skill.key).and_then(|bucket| {
            bucket
                .recharge_ready_at
                .iter()
                .copied()
                .min_by(f64::total_cmp)
        })
    }

    pub fn build_cooldown_snapshot(
        &self,
        state: &CombatState,
        skill: &SkillDefinition,
    ) -> (f64, i32, i32) {
        let Some(bucket) = state.cooldowns.get(&skill.key) else {
            return (0.0, skill.charges, skill.charges);
        };
        let remaining = self
            .next_charge_ready_at(state, skill)
            .map_or(0.0, |deadline| (deadline - state.time()).max(0.0));
        (remaining, bucket.available_charges, skill.charges)
    }

    pub fn consume_cooldown(&self, state: &mut CombatState, skill: &SkillDefinition) {
        if skill.cooldown <= 0.0 && skill.charges <= 1 {
            return;
        }
        let now = state.time();
        let bucket = state
            .cooldowns
            .entry(skill.key.clone())
            .or_insert_with(|| CooldownState::new(now, skill.charges, Vec::new()));
        release_ready_charges(bucket, now, skill.charges);
        if bucket.available_charges > 0 {
            bucket.available_charges -= 1;
            bucket.recharge_ready_at.push(now + skill.cooldown);
            bucket.recharge_ready_at.sort_by(f64::total_cmp);
        }
    }

    pub fn apply_cooldown_reduction(
        &self,
        state: &mut CombatState,
        skill: &SkillDefinition,
        seconds: f64,
    ) {
        let now = state.time();
        let Some(bucket) = state.cooldowns.get_mut(&skill.key) else {
            return;
        };
        for deadline in &mut bucket.recharge_ready_at {
            *deadline = now.max(*deadline - seconds.max(0.0));
        }
        bucket.recharge_ready_at.sort_by(f64::total_cmp);
        release_ready_charges(bucket, now, skill.charges);
        if bucket.available_charges >= skill.charges && bucket.recharge_ready_at.is_empty() {
            state.cooldowns.shift_remove(&skill.key);
        }
    }

    pub fn grant_target_dot(
        &self,
        state: &mut CombatState,
        key: &str,
        remaining: f64,
        potency_per_tick: f64,
        tick_interval: f64,
    ) -> Result<(), String> {
        if !self.target_dot_keys.contains(key) {
            return Err(format!("unregistered dot key: {key}"));
        }
        if !tick_interval.is_finite() || tick_interval <= 0.0 {
            return Err("DoT tick interval must be finite and positive".into());
        }
        state.dots.insert(
            key.into(),
            DotState::new(
                state.time(),
                remaining,
                potency_per_tick,
                tick_interval,
                tick_interval,
            ),
        );
        Ok(())
    }

    /// 只描述公共资源，不读取或改写职业的状态容器。
    pub fn describe_events<P>(&self, state: &CombatState) -> Vec<TimelineEvent<P>> {
        let mut events = self.describe_player_events(state);
        events.extend(self.describe_status_events(state));
        events.extend(self.describe_dot_events(state));
        events.extend(self.describe_cooldown_events(state));
        events.push(self.describe_mp_event(state));
        events
    }

    pub fn describe_player_events<P>(&self, state: &CombatState) -> Vec<TimelineEvent<P>> {
        let now = state.time();
        let mut events = Vec::new();
        if state.gcd_ready_at > now
            || state.ogcds_weaved > 0
            || state.weave_window_remaining() > 0.0
        {
            events.push(TimelineEvent::new(
                now.max(state.gcd_ready_at),
                Priority::DecisionBoundary,
                Kind::GcdReady,
            ));
        }
        if state.boss_targetable()
            && let Some(start) = state.next_downtime_starts_at
            && state.downtime_remaining() > 0.0
        {
            events.push(owned_event(
                now.max(start),
                Priority::ExternalScene,
                Kind::SceneChanged,
                "downtime_start",
            ));
        } else if !state.boss_targetable() && state.downtime_ends_at > now + ZERO_EPSILON {
            events.push(owned_event(
                now.max(state.downtime_ends_at),
                Priority::ExternalScene,
                Kind::SceneChanged,
                "downtime_end",
            ));
        }
        events
    }

    pub fn describe_status_events<P>(&self, state: &CombatState) -> Vec<TimelineEvent<P>> {
        state
            .statuses
            .iter()
            .map(|(key, status)| {
                owned_event(
                    state.time().max(status.expires_at),
                    Priority::ExpirationAndCooldown,
                    Kind::StatusExpired,
                    key,
                )
            })
            .collect()
    }

    pub fn describe_dot_events<P>(&self, state: &CombatState) -> Vec<TimelineEvent<P>> {
        state
            .dots
            .iter()
            .map(|(key, dot)| {
                owned_event(
                    state.time().max(dot.next_tick_at.min(dot.expires_at)),
                    Priority::PeriodicSettlement,
                    Kind::DotTick,
                    key,
                )
            })
            .collect()
    }

    pub fn describe_cooldown_events<P>(&self, state: &CombatState) -> Vec<TimelineEvent<P>> {
        state
            .cooldowns
            .iter()
            .flat_map(|(key, bucket)| {
                bucket.recharge_ready_at.iter().map(move |&deadline| {
                    owned_event(
                        state.time().max(deadline),
                        Priority::ExpirationAndCooldown,
                        Kind::CooldownChargeReady,
                        key,
                    )
                })
            })
            .collect()
    }

    pub fn describe_mp_event<P>(&self, state: &CombatState) -> TimelineEvent<P> {
        TimelineEvent::new(
            state
                .time()
                .max(state.natural_mp_last_tick_at + self.config.mp_recovery.tick_interval_seconds),
            Priority::PeriodicSettlement,
            Kind::MpTick,
        )
    }

    pub fn settle_gcd_ready<P>(&self, state: &mut CombatState, event: &TimelineEvent<P>) {
        if state.gcd_ready_at <= event.timestamp + ZERO_EPSILON {
            state.ogcds_weaved = 0;
            state.weave_ends_at = state.time();
        }
    }

    pub fn apply_scene_deadline<P>(&self, state: &mut CombatState, event: &TimelineEvent<P>) {
        match event.owner_key.as_deref() {
            Some("downtime_start") => state.set_boss_targetable(false),
            Some("downtime_end") => {
                state.set_boss_targetable(true);
                state.set_downtime_remaining(0.0);
                state.next_downtime_starts_at = None;
            }
            _ => {}
        }
    }

    pub fn settle_status_expired<P>(
        &self,
        state: &mut CombatState,
        event: &TimelineEvent<P>,
    ) -> Result<(), String> {
        self.statuses.settle_expired(state, event, Scope::System)
    }

    pub fn settle_cooldown_ready<P>(
        &self,
        state: &mut CombatState,
        event: &TimelineEvent<P>,
        max_charges: i32,
    ) {
        let now = state.time();
        let Some(key) = &event.owner_key else {
            return;
        };
        let Some(bucket) = state.cooldowns.get_mut(key) else {
            return;
        };
        if let Some(index) = bucket
            .recharge_ready_at
            .iter()
            .position(|&time| time <= now)
        {
            bucket.recharge_ready_at.remove(index);
            bucket.available_charges = max_charges.min(bucket.available_charges + 1);
            if bucket.available_charges >= max_charges && bucket.recharge_ready_at.is_empty() {
                state.cooldowns.shift_remove(key);
            }
        }
    }

    pub fn settle_dot_tick<P>(&self, state: &mut CombatState, event: &TimelineEvent<P>) {
        let now = state.time();
        let targetable = state.boss_targetable();
        let Some(key) = &event.owner_key else {
            return;
        };
        let Some(dot) = state.dots.get_mut(key) else {
            return;
        };
        if dot.next_tick_at <= now && dot.next_tick_at <= dot.expires_at {
            if targetable {
                state.current_gcd_dot_potency += dot.potency_per_tick;
                state.cumulative_dot_potency += dot.potency_per_tick;
            }
            dot.next_tick_at += dot.tick_interval;
        }
        if dot.expires_at <= now {
            state.dots.shift_remove(key);
        }
    }

    pub fn settle_mp_tick<P>(
        &self,
        state: &mut CombatState,
        event: &TimelineEvent<P>,
        resolved_amount: i32,
    ) {
        if !state.is_dead {
            self.recover_mp(state, resolved_amount.max(0));
        }
        state.natural_mp_last_tick_at = event.timestamp;
    }

    pub fn apply_death(&self, state: &mut CombatState) {
        state.is_dead = true;
        state.mp = 0;
        state.statuses.clear();
    }

    pub fn validate_external_event(&self, external: &ExternalCombatEvent) -> Result<(), String> {
        use ExternalEventKind as External;
        let no_scene_fields = external.value.is_none()
            && external.target_count.is_none()
            && external.remaining_seconds.is_none();
        let supported = match external.kind {
            External::BossTargetableChanged | External::MovementChanged => {
                external.value.is_some()
                    && external.target_count.is_none()
                    && external.remaining_seconds.is_none()
            }
            External::RaidBuffWindowChanged => {
                self.config.statuses.contains_key("raid_buff_window")
                    && external.target_count.is_none()
                    && match external.value {
                        Some(true) => external
                            .remaining_seconds
                            .is_none_or(|remaining| remaining > 0.0),
                        Some(false) => external.remaining_seconds.is_none(),
                        None => false,
                    }
            }
            External::TargetCountChanged => {
                external.value.is_none()
                    && external.target_count.is_some_and(|count| count >= 0)
                    && external.remaining_seconds.is_none()
            }
            External::PlayerDeath | External::PlayerRevivedLb3 => no_scene_fields,
            External::PlayerRevived => no_scene_fields && external.mp.is_none_or(|mp| mp >= 0),
        };
        if !supported
            || !external.timestamp.is_finite()
            || external
                .remaining_seconds
                .is_some_and(|remaining| !remaining.is_finite())
            || external.mp.is_some() && external.kind != External::PlayerRevived
        {
            return Err(format!(
                "unsupported external scene event: {:?}",
                external.kind
            ));
        }
        Ok(())
    }

    /// 批次原子性、生死转换顺序及未完成动作取消由模拟游标负责。
    pub fn apply_external_event(
        &self,
        state: &mut CombatState,
        external: &ExternalCombatEvent,
    ) -> Result<(), String> {
        use ExternalEventKind as External;
        self.validate_external_event(external)?;
        match external.kind {
            External::BossTargetableChanged => state.set_boss_targetable(external.value.unwrap()),
            External::MovementChanged => state.is_moving = external.value.unwrap(),
            External::RaidBuffWindowChanged => {
                if external.value == Some(true) {
                    self.grant_status(state, "raid_buff_window", external.remaining_seconds, None)?;
                } else {
                    self.clear_status(state, "raid_buff_window");
                }
            }
            External::TargetCountChanged => state.target_count = external.target_count.unwrap(),
            External::PlayerDeath => self.apply_death(state),
            External::PlayerRevived => {
                state.is_dead = false;
                state.mp = state.max_mp.min(external.mp.unwrap_or(
                    (f64::from(state.max_mp) * self.config.mp_recovery.revival_ratio) as i32,
                ));
            }
            External::PlayerRevivedLb3 => {
                state.is_dead = false;
                state.mp = state.max_mp;
            }
        }
        Ok(())
    }

    pub fn resolve_aoe_target_multiplier(skill: &SkillDefinition, target_count: i32) -> f64 {
        if skill.max_targets <= 1 || target_count <= 1 {
            1.0
        } else {
            1.0 + (1.0 - skill.aoe_secondary_reduction).max(0.0)
                * f64::from(target_count.min(skill.max_targets) - 1)
        }
    }

    pub fn resolve_applied_potency(
        &self,
        state: &CombatState,
        base_potency: f64,
        skill: Option<&SkillDefinition>,
    ) -> f64 {
        let mut multiplier = skill.map_or(1.0, |skill| {
            Self::resolve_aoe_target_multiplier(skill, state.target_count)
        });
        if state.has_status("burst_potion") {
            multiplier *= self.config.potency.burst_potion_multiplier;
        }
        if state.has_status("raid_buff_window") {
            multiplier *= self.config.potency.raid_buff_window_multiplier;
        }
        base_potency.max(0.0) * multiplier
    }

    pub fn resolve_applied_potency_at_offset(
        &self,
        state: &CombatState,
        base_potency: f64,
        offset_seconds: f64,
    ) -> f64 {
        let offset = offset_seconds.max(0.0);
        let mut multiplier = 1.0;
        for (key, value) in [
            ("burst_potion", self.config.potency.burst_potion_multiplier),
            (
                "raid_buff_window",
                self.config.potency.raid_buff_window_multiplier,
            ),
        ] {
            if state
                .statuses
                .get(key)
                .is_some_and(|status| status.remaining(state.time()) - offset > 0.0)
            {
                multiplier *= value;
            }
        }
        base_potency.max(0.0) * multiplier
    }

    pub fn record_target_potency(
        &self,
        state: &mut CombatState,
        source: &CombatState,
        skill: &SkillDefinition,
        direct_potency: f64,
    ) {
        let potency = self.resolve_applied_potency(source, direct_potency, Some(skill));
        state.current_potency = potency;
        state.current_gcd_dot_potency = 0.0;
        state.cumulative_potency += potency;
    }

    pub fn record_delayed_potency(&self, state: &mut CombatState, resolved_potency: f64) -> f64 {
        let potency = resolved_potency.max(0.0);
        state.cumulative_potency += potency;
        potency
    }
}

fn release_ready_charges(bucket: &mut CooldownState, now: f64, max_charges: i32) {
    let before = bucket.recharge_ready_at.len();
    bucket.recharge_ready_at.retain(|&time| time > now);
    bucket.available_charges = max_charges
        .min(bucket.available_charges + (before - bucket.recharge_ready_at.len()) as i32);
}

fn owned_event<P>(timestamp: f64, priority: Priority, kind: Kind, key: &str) -> TimelineEvent<P> {
    TimelineEvent {
        owner_key: Some(key.into()),
        ..TimelineEvent::new(timestamp, priority, kind)
    }
}

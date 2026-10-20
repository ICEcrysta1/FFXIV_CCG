// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

//! 装配共享规则，解析动作校验、时序、效果与历史；运行游标由模拟器持有。

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::config::{ActionKind, ProjectConfig, SchemaConfig, SkillBook, SkillDefinition};
use crate::contracts::{JobRegistry, JobStateMachine};
use crate::outputs::StateContextBuilder;
use crate::state::{
    ActionHistoryEntry, CombatState, ModelStateFrame, ModelStateSnapshot, ValidationResult,
};
use crate::system::timeline::{TIME_EPSILON, TimelineEvent};
use crate::system::{SystemRules, ZERO_EPSILON};

pub mod replay;
pub mod sessions;
pub mod simulator;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionRequest {
    pub timestamp: f64,
    pub skill_key: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionTimingPlan {
    pub gcd_unit_seconds: f64,
    pub actual_cast_seconds: f64,
    pub effect_delay_seconds: f64,
    pub effective_gcd_seconds: f64,
    pub actual_occupancy_seconds: f64,
    pub actual_occupancy_source: String,
    pub gcd_window_seconds: f64,
    pub next_gcd_window_seconds: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionSubmissionPlan {
    pub accepted: bool,
    pub queued: bool,
    pub reason: String,
    pub accepted_timestamp: f64,
}

pub struct ActionHistoryMetadata {
    pub action_instance_id: u64,
    pub request_timestamp: f64,
    pub cast_completed_timestamp: f64,
    pub effect_timestamp: f64,
    pub model_state: ModelStateSnapshot,
}

/// 规则装配后只读共享，具体职业由宿主注入。
pub struct EngineRules {
    pub(crate) project: Arc<ProjectConfig>,
    pub(crate) system: Arc<SystemRules>,
    pub(crate) job: Arc<dyn JobStateMachine>,
    pub(crate) skills: SkillBook,
    pub(crate) schema: Arc<SchemaConfig>,
    pub(crate) max_history: Option<usize>,
    action_keys: Vec<String>,
    action_skills: Vec<SkillDefinition>,
    context_builder: StateContextBuilder,
}

impl EngineRules {
    pub fn from_project(
        project: Arc<ProjectConfig>,
        schema: Arc<SchemaConfig>,
        registry: &JobRegistry,
        max_history: Option<usize>,
    ) -> Result<Self, String> {
        let mut system = SystemRules::new(project.engine_timing.clone(), project.system.clone())?;
        system.register_target_dots(
            project
                .system
                .skills
                .iter()
                .chain(&project.job.skills)
                .filter(|skill| skill.enabled),
        );
        let system = Arc::new(system);
        let job = registry
            .get(&project.runtime.job_tag, project.clone(), system.clone())?
            .ok_or_else(|| {
                format!(
                    "unsupported job tag: {}; supported={}",
                    project.runtime.job_tag,
                    registry.registered_tags().join(", ")
                )
            })?;
        Self::new(project, system, job, schema, max_history)
    }

    pub fn new(
        project: Arc<ProjectConfig>,
        system: Arc<SystemRules>,
        job: Arc<dyn JobStateMachine>,
        schema: Arc<SchemaConfig>,
        max_history: Option<usize>,
    ) -> Result<Self, String> {
        let job_tag = job.runtime().resources.job_tag();
        if project.job.key != job_tag {
            return Err(format!(
                "loaded config job does not match requested tag: config={}, requested={job_tag}",
                project.job.key
            ));
        }
        let skills = SkillBook::from_project_config(&project)?;
        for key in skills.keys() {
            let skill = skills.get(key)?;
            if !system.supports_behavior(&skill.behavior) && !job.supports_behavior(&skill.behavior)
            {
                return Err(format!(
                    "unsupported behavior for skill {}: {}",
                    skill.key, skill.behavior
                ));
            }
        }
        let mut action_skills: Vec<_> = skills.enabled_skills(None).into_iter().cloned().collect();
        action_skills.sort_by(|left, right| left.key.cmp(&right.key));
        let action_keys = action_skills
            .iter()
            .map(|skill| skill.key.clone())
            .collect();
        let context_builder =
            StateContextBuilder::new(project.clone(), system.clone(), job.clone());
        Ok(Self {
            project,
            system,
            job,
            skills,
            schema,
            max_history,
            action_keys,
            action_skills,
            context_builder,
        })
    }

    pub fn project(&self) -> &Arc<ProjectConfig> {
        &self.project
    }

    pub fn system(&self) -> &Arc<SystemRules> {
        &self.system
    }

    pub fn job(&self) -> &Arc<dyn JobStateMachine> {
        &self.job
    }

    pub fn job_tag(&self) -> &str {
        self.job.runtime().resources.job_tag()
    }

    pub fn initial_state(
        &self,
        fight_remaining: Option<f64>,
        max_mp: Option<i32>,
        start_time: f64,
    ) -> Result<CombatState, String> {
        let duration = match fight_remaining {
            Some(duration) => duration,
            None => self.job.default_fight_remaining()?,
        };
        let mut state = self.system.initial_state(
            duration,
            max_mp.unwrap_or(self.project.engine_resources.max_mp),
            start_time,
        )?;
        self.job.runtime().initialize(&mut state)?;
        Ok(state)
    }

    pub fn resolve_skill(&self, key: &str) -> Result<&SkillDefinition, String> {
        self.skills.get(key)
    }

    pub fn validate_action(
        &self,
        state: &CombatState,
        skill: &SkillDefinition,
    ) -> Result<ValidationResult, String> {
        self.validate_action_core(state, skill, true, None)
    }

    fn validate_action_core(
        &self,
        state: &CombatState,
        skill: &SkillDefinition,
        include_temporal_locks: bool,
        actual_cast_seconds_override: Option<f64>,
    ) -> Result<ValidationResult, String> {
        if !skill.enabled {
            return Ok(ValidationResult::invalid("disabled"));
        }
        if !self.system.supports_behavior(&skill.behavior)
            && !self.job.supports_behavior(&skill.behavior)
        {
            return Ok(ValidationResult::invalid("unsupported_behavior"));
        }
        let is_instant = match actual_cast_seconds_override {
            Some(observed) => observed <= ZERO_EPSILON,
            None => self.job.is_instant(state, skill)?,
        };
        let common = self.system.validate_common_action(
            state,
            skill,
            is_instant,
            self.system.has_available_charge(state, skill),
            include_temporal_locks,
        );
        if !common.ok {
            return Ok(common);
        }
        if skill.behavior == "grant_status" {
            for key in &skill.applies_statuses {
                let validation = if self.system.status_definitions().contains_key(key) {
                    self.system.validate_status(state, key)?
                } else {
                    self.job.runtime().validate_status(state, key)?
                };
                if !validation.ok {
                    return Ok(validation);
                }
            }
        }
        self.job.validate_action(state, skill)
    }

    /// 只有时间锁允许进入容量一队列，非时间资源必须在请求时已经满足。
    pub fn evaluate_action_submission(
        &self,
        state: &CombatState,
        skill: &SkillDefinition,
        queue_occupied: bool,
        actual_cast_seconds_override: Option<f64>,
    ) -> Result<ActionSubmissionPlan, String> {
        let validation =
            self.validate_action_core(state, skill, true, actual_cast_seconds_override)?;
        if validation.ok {
            return Ok(ActionSubmissionPlan {
                accepted: true,
                queued: false,
                reason: String::new(),
                accepted_timestamp: state.time(),
            });
        }
        let rejected = |reason, timestamp| ActionSubmissionPlan {
            accepted: false,
            queued: false,
            reason,
            accepted_timestamp: timestamp,
        };
        if !matches!(
            validation.reason.as_str(),
            "cast_locked" | "gcd_locked" | "cooldown_locked"
        ) {
            return Ok(rejected(validation.reason, state.time()));
        }
        let non_temporal =
            self.validate_action_core(state, skill, false, actual_cast_seconds_override)?;
        if !non_temporal.ok {
            return Ok(rejected(non_temporal.reason, state.time()));
        }
        let mut ready_at = state.time();
        if state.cast_remaining() > ZERO_EPSILON {
            ready_at = ready_at.max(state.cast_ends_at);
        }
        if skill.kind == ActionKind::Gcd && state.gcd_remaining() > ZERO_EPSILON {
            ready_at = ready_at.max(state.gcd_ready_at);
        }
        if !self.system.has_available_charge(state, skill) {
            let Some(cooldown_ready_at) = self.system.next_charge_ready_at(state, skill) else {
                return Ok(rejected("cooldown_locked".into(), state.time()));
            };
            ready_at = ready_at.max(cooldown_ready_at);
        }
        if queue_occupied {
            return Ok(rejected("action_queue_occupied".into(), state.time()));
        }
        let queue_window = self
            .project
            .engine_timing
            .action_queue_window_seconds
            .max(0.0);
        if ready_at - state.time() > queue_window + ZERO_EPSILON {
            return Ok(rejected(validation.reason, ready_at));
        }
        Ok(ActionSubmissionPlan {
            accepted: true,
            queued: true,
            reason: String::new(),
            accepted_timestamp: ready_at,
        })
    }

    pub fn build_action_timing_plan(
        &self,
        state: &CombatState,
        skill: &SkillDefinition,
        actual_cast_seconds_override: Option<f64>,
    ) -> Result<ActionTimingPlan, String> {
        let gcd_unit_seconds = self.job.current_gcd_duration(state)?;
        let is_instant = self.job.is_instant(state, skill)?;
        let actual_cast_seconds = match actual_cast_seconds_override {
            Some(observed) => observed.max(0.0),
            None => self.build_actual_cast_seconds(
                skill,
                is_instant,
                gcd_unit_seconds,
                self.job.cast_time_multiplier(state, skill)?,
            )?,
        };
        let effect_delay_seconds = if actual_cast_seconds > TIME_EPSILON {
            (actual_cast_seconds - self.schema.slidecast_window_seconds).max(0.0)
        } else {
            0.0
        };
        let effective_gcd_seconds = self.build_effective_gcd_seconds(skill, gcd_unit_seconds)?;
        let (actual_occupancy_seconds, actual_occupancy_source, gcd_window_seconds) =
            self.build_action_metrics(skill, actual_cast_seconds, effective_gcd_seconds);
        let next_gcd_window_seconds = if skill.kind == ActionKind::Gcd {
            actual_occupancy_seconds.max(gcd_window_seconds)
        } else {
            state.gcd_remaining().max(0.0)
        };
        Ok(ActionTimingPlan {
            gcd_unit_seconds,
            actual_cast_seconds,
            effect_delay_seconds,
            effective_gcd_seconds,
            actual_occupancy_seconds,
            actual_occupancy_source,
            gcd_window_seconds,
            next_gcd_window_seconds,
        })
    }

    pub fn build_actual_cast_seconds(
        &self,
        skill: &SkillDefinition,
        is_instant: bool,
        gcd_unit_seconds: f64,
        cast_time_multiplier: f64,
    ) -> Result<f64, String> {
        if is_instant || skill.cast_time <= 0.0 {
            Ok(0.0)
        } else {
            Ok(
                self.scale_nominal_gcd_seconds(skill.cast_time, gcd_unit_seconds)?
                    * cast_time_multiplier.max(0.0),
            )
        }
    }

    pub fn build_effective_gcd_seconds(
        &self,
        skill: &SkillDefinition,
        gcd_unit_seconds: f64,
    ) -> Result<f64, String> {
        if skill.kind == ActionKind::Gcd {
            self.scale_nominal_gcd_seconds(skill.recast_time, gcd_unit_seconds)
        } else {
            Ok(skill.recast_time)
        }
    }

    pub fn build_action_metrics(
        &self,
        skill: &SkillDefinition,
        actual_cast_seconds: f64,
        effective_gcd: f64,
    ) -> (f64, String, f64) {
        if skill.kind == ActionKind::Gcd {
            if actual_cast_seconds <= 0.0 {
                (0.0, "instant".into(), effective_gcd)
            } else {
                (actual_cast_seconds, "cast_time".into(), effective_gcd)
            }
        } else {
            (0.0, "instant".into(), 0.0)
        }
    }

    pub fn apply_action_timing(
        &self,
        state: &mut CombatState,
        skill: &SkillDefinition,
        effective_gcd: f64,
        actual_cast_seconds: f64,
        gcd_start_timestamp: Option<f64>,
    ) -> Result<(), String> {
        self.system.apply_action_timing(
            state,
            skill,
            effective_gcd,
            actual_cast_seconds,
            gcd_start_timestamp,
        );
        self.system.consume_cooldown(state, skill);
        if skill.kind == ActionKind::Gcd {
            self.job.consume_timing_resources(state, skill)?;
        }
        Ok(())
    }

    pub fn apply_action_effect(
        &self,
        source: &CombatState,
        target: &mut CombatState,
        skill: &SkillDefinition,
    ) -> Result<(), String> {
        if !skill.applies_statuses.is_empty() {
            for key in &skill.applies_statuses {
                if self.system.status_definitions().contains_key(key) {
                    self.system.grant_status(target, key, None, None)?;
                } else {
                    self.job.runtime().grant_status(target, key, None, None)?;
                }
            }
        } else if self.job.supports_behavior(&skill.behavior) {
            self.job.apply_action(source, target, skill)?;
        }
        let potency = self
            .job
            .resolve_potency(source, skill, f64::from(skill.potency))?;
        self.system
            .record_target_potency(target, source, skill, potency);
        Ok(())
    }

    pub fn apply_death(&self, state: &mut CombatState) -> Result<(), String> {
        self.system.apply_death(state);
        self.job.on_death(state)
    }

    /// 跨域同刻事件按资源拥有者的固定装配顺序领取稳定序号。
    pub fn describe_events<P>(&self, state: &CombatState) -> Vec<TimelineEvent<P>> {
        let mut events = self.system.describe_player_events(state);
        events.extend(self.system.describe_status_events(state));
        events.extend(self.job.runtime().describe_status_events(state));
        events.extend(self.system.describe_dot_events(state));
        events.extend(self.system.describe_cooldown_events(state));
        events.extend(self.job.runtime().timeline.describe_events(state));
        events.push(self.system.describe_mp_event(state));
        events
    }

    pub fn action_keys(&self) -> &[String] {
        &self.action_keys
    }

    pub fn skill_availability(
        &self,
        state: &CombatState,
        queue_occupied: bool,
    ) -> Result<BTreeMap<String, bool>, String> {
        self.action_skills
            .iter()
            .map(|skill| {
                Ok((
                    skill.key.clone(),
                    self.evaluate_action_submission(state, skill, queue_occupied, None)?
                        .accepted,
                ))
            })
            .collect()
    }

    pub fn action_values(&self, state: &CombatState) -> Result<Vec<f64>, String> {
        self.action_skills
            .iter()
            .map(|skill| self.job.resolve_action_value(state, skill, skill.value))
            .collect()
    }

    pub fn capture_model_frame(
        &self,
        state: &CombatState,
        queue_occupied: bool,
    ) -> Result<Arc<ModelStateFrame>, String> {
        Ok(Arc::new(ModelStateFrame {
            state: self.context_builder.build(state)?,
            skill_availability: self.skill_availability(state, queue_occupied)?,
        }))
    }

    pub fn capture_model_state(
        &self,
        state: &CombatState,
        queue_occupied: bool,
    ) -> Result<ModelStateSnapshot, String> {
        Ok(ModelStateSnapshot::capture(
            state.last_decision_after.clone(),
            self.capture_model_frame(state, queue_occupied)?,
        ))
    }

    pub fn record_timeline_action_history(
        &self,
        request: &CombatState,
        effect: &mut CombatState,
        skill: &SkillDefinition,
        timing: &ActionTimingPlan,
        metadata: ActionHistoryMetadata,
    ) -> Result<(), String> {
        let state_before = self.context_builder.build(request)?;
        let state_after = self.context_builder.build(effect)?;
        let (next_cooldown_seconds, available_charges, max_charges) =
            self.system.build_cooldown_snapshot(effect, skill);
        let potency = self
            .job
            .resolve_potency(request, skill, f64::from(skill.potency))?;
        let value = self.job.resolve_action_value(request, skill, skill.value)?;
        let transition = self.job.runtime().resource_transition(request, effect)?;
        let history_sequence = effect.reserve_history_sequence()?;
        let history = ActionHistoryEntry {
            skill_key: skill.key.clone(),
            skill_id: skill.game_id,
            skill_name: skill.name.clone(),
            skill_kind: match skill.kind {
                ActionKind::Gcd => "gcd",
                ActionKind::Ogcd => "ogcd",
            }
            .into(),
            potency,
            value,
            mp_before: request.mp,
            mp_after: effect.mp,
            cast_time_seconds: timing.actual_cast_seconds,
            cast_time_gcds: gcd_units(timing.actual_cast_seconds, timing.gcd_unit_seconds),
            gcd_window_seconds: timing.gcd_window_seconds,
            gcd_window_gcds: gcd_units(timing.gcd_window_seconds, timing.gcd_unit_seconds),
            is_legal: true,
            invalid_reason: String::new(),
            next_cooldown_seconds,
            available_charges,
            max_charges,
            job_resources_before: transition.before,
            job_resources_after: transition.after,
            job_resources_consumed: transition.consumed,
            time_seconds: effect.time(),
            gcd_index: effect.gcd_index,
            state_before,
            state_after,
            request_timestamp: Some(metadata.request_timestamp),
            cast_completed_timestamp: Some(metadata.cast_completed_timestamp),
            effect_timestamp: Some(metadata.effect_timestamp),
            action_instance_id: Some(metadata.action_instance_id),
            model_state: Some(metadata.model_state),
            history_sequence,
        };
        let entries = Arc::make_mut(&mut effect.history);
        entries.push(Arc::new(history));
        Ok(())
    }

    fn scale_nominal_gcd_seconds(
        &self,
        nominal: f64,
        gcd_unit_seconds: f64,
    ) -> Result<f64, String> {
        let base = self.project.system.skill_table_base_gcd;
        if base <= 0.0 {
            Err("skill_table_base_gcd must be > 0".into())
        } else {
            Ok(nominal * gcd_unit_seconds / base)
        }
    }
}

fn gcd_units(seconds: f64, gcd_unit_seconds: f64) -> f64 {
    if gcd_unit_seconds <= 0.0 {
        0.0
    } else {
        seconds / gcd_unit_seconds
    }
}

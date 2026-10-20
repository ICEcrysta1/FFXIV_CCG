// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

//! 引擎差分测试共用合成职业，不依赖任何真实职业 crate。

use fight_engine::config::{
    ActionKind, JobStateRegistration, ProjectConfig, ResourceValue, SchemaConfig, SkillDefinition,
    StatusDefinition,
};
use fight_engine::contracts::{JobRegistry, JobRuntime, JobStateMachine};
use fight_engine::engine::EngineRules;
use fight_engine::state::{CombatState, CooldownState, ValidationResult};
use fight_engine::system::{SystemRules, timeline::TimelineMutation};
use indexmap::IndexMap;
use serde_json::Value;
use std::{path::Path, sync::Arc};

struct ProbeJob {
    project: Arc<ProjectConfig>,
    system: Arc<SystemRules>,
    runtime: JobRuntime,
}

impl ProbeJob {
    fn new(
        project: Arc<ProjectConfig>,
        system: Arc<SystemRules>,
        registration: &JobStateRegistration,
    ) -> Result<Self, String> {
        let mut runtime = JobRuntime::new(registration, &project.job.resource_limits)?;
        let resources = runtime.resources.clone();
        runtime.timeline.register_periodic(
            "periodic",
            3.0,
            "timer",
            |state| state.job.timeline_deadlines.get("p").copied(),
            move |event, state| {
                let mut next = state.clone();
                resources.set(
                    &mut next,
                    "periodic_seen_mp",
                    ResourceValue::Integer(state.mp),
                )?;
                next.job
                    .timeline_deadlines
                    .insert("p".into(), event.timestamp + 3.0);
                Ok(TimelineMutation {
                    state: Some(next),
                    ..TimelineMutation::default()
                })
            },
        )?;
        let resources = runtime.resources.clone();
        runtime.register_expiry_hook("z_job", move |state| {
            let charges = state
                .cooldowns
                .get("charge")
                .map_or(2, |bucket| bucket.available_charges);
            resources.set(state, "expiry_seen_charge", ResourceValue::Integer(charges))
        })?;
        Ok(Self {
            project,
            system,
            runtime,
        })
    }

    fn gauge(&self, state: &CombatState) -> Result<i32, String> {
        match self.runtime.resources.get(state, "gauge")? {
            ResourceValue::Integer(value) => Ok(value),
            _ => Err("探针 gauge 应为整数".into()),
        }
    }
}

impl JobStateMachine for ProbeJob {
    fn runtime(&self) -> &JobRuntime {
        &self.runtime
    }

    fn default_fight_remaining(&self) -> Result<f64, String> {
        Ok(600.0)
    }

    fn current_gcd_duration(&self, state: &CombatState) -> Result<f64, String> {
        Ok(state.base_gcd.unwrap_or(self.project.system.base_gcd))
    }

    fn is_instant(&self, _: &CombatState, skill: &SkillDefinition) -> Result<bool, String> {
        Ok(skill.kind == ActionKind::Ogcd || skill.cast_time <= 0.0)
    }

    fn supports_behavior(&self, behavior: &str) -> bool {
        matches!(behavior, "standard" | "spend" | "drain")
    }

    fn validate_action(
        &self,
        state: &CombatState,
        skill: &SkillDefinition,
    ) -> Result<ValidationResult, String> {
        Ok(if skill.key == "deny" {
            ValidationResult::invalid("probe_denied")
        } else if skill.behavior == "spend" && self.gauge(state)? <= 0 {
            ValidationResult::invalid("resource_missing")
        } else if state.mp < self.actual_mp_cost(state, skill)? {
            ValidationResult::invalid("not_enough_mp")
        } else {
            ValidationResult::valid()
        })
    }

    fn apply_action(
        &self,
        previous: &CombatState,
        next: &mut CombatState,
        skill: &SkillDefinition,
    ) -> Result<(), String> {
        self.system
            .spend_mp(next, self.actual_mp_cost(previous, skill)?);
        self.runtime.resources.set(
            next,
            "request_gauge",
            ResourceValue::Integer(self.gauge(previous)?),
        )?;
        if skill.behavior == "drain" {
            self.runtime
                .resources
                .set(next, "gauge", ResourceValue::Integer(0))?;
        }
        if skill.dot_duration > 0.0 {
            self.system.grant_target_dot(
                next,
                skill.dot_key.as_deref().unwrap_or(&skill.key),
                skill.dot_duration,
                f64::from(skill.dot_potency),
                3.0,
            )?;
        }
        Ok(())
    }

    fn consume_timing_resources(
        &self,
        state: &mut CombatState,
        skill: &SkillDefinition,
    ) -> Result<(), String> {
        if skill.behavior == "spend" {
            let value = self.gauge(state)? - 1;
            self.runtime
                .resources
                .set(state, "gauge", ResourceValue::Integer(value))?;
        }
        Ok(())
    }

    fn on_death(&self, state: &mut CombatState) -> Result<(), String> {
        self.runtime
            .resources
            .set(state, "gauge", ResourceValue::Integer(0))?;
        state.job.statuses.clear();
        state.job.timeline_deadlines.clear();
        Ok(())
    }
}

fn ordered_statuses(
    definitions: &IndexMap<String, StatusDefinition>,
    order: &Value,
) -> IndexMap<String, StatusDefinition> {
    order
        .as_array()
        .unwrap()
        .iter()
        .map(|key| {
            let key = key.as_str().unwrap();
            (key.to_owned(), definitions[key].clone())
        })
        .collect()
}

pub fn rules(baseline: &Value, max_history: Option<usize>) -> Arc<EngineRules> {
    let mut project: ProjectConfig = serde_json::from_value(baseline["project"].clone()).unwrap();
    project.system.statuses = ordered_statuses(
        &project.system.statuses,
        &baseline["status_order"]["system"],
    );
    project.job.statuses =
        ordered_statuses(&project.job.statuses, &baseline["status_order"]["job"]);
    let mut registration: JobStateRegistration =
        serde_json::from_value(baseline["registration"].clone()).unwrap();
    registration.statuses =
        ordered_statuses(&registration.statuses, &baseline["status_order"]["job"]);
    let mut registry = JobRegistry::default();
    registry
        .register_factory(project.runtime.job_tag.clone(), move |project, system| {
            Ok(Arc::new(ProbeJob::new(project, system, &registration)?))
        })
        .unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let schema = Arc::new(SchemaConfig::load(root).unwrap());
    Arc::new(EngineRules::from_project(Arc::new(project), schema, &registry, max_history).unwrap())
}

pub fn initial_state(rules: &EngineRules, input: &Value) -> CombatState {
    let now = input["timestamp"].as_f64().unwrap();
    let mut state = rules.initial_state(None, None, now).unwrap();
    state.mp = input["mp"].as_i64().unwrap() as i32;
    rules
        .job()
        .runtime()
        .resources
        .set(
            &mut state,
            "gauge",
            ResourceValue::Integer(input["gauge"].as_i64().unwrap() as i32),
        )
        .unwrap();
    if let Some(deadlines) = input["job_deadlines"].as_object() {
        for (key, value) in deadlines {
            state
                .job
                .timeline_deadlines
                .insert(key.clone(), value.as_f64().unwrap());
        }
    }
    if let Some(statuses) = input["job_statuses"].as_object() {
        for (key, value) in statuses {
            rules
                .job()
                .runtime()
                .grant_status(
                    &mut state,
                    key,
                    value["remaining"].as_f64(),
                    value["stacks"].as_i64().map(|value| value as i32),
                )
                .unwrap();
        }
    }
    if let Some(cooldowns) = input["cooldowns"].as_object() {
        for (key, value) in cooldowns {
            state.cooldowns.insert(
                key.clone(),
                CooldownState::new(
                    now,
                    value["available_charges"].as_i64().unwrap() as i32,
                    value["recharge_ready_at"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|timer| timer.as_f64().unwrap() - now)
                        .collect(),
                ),
            );
        }
    }
    state
}

// Copyright (C) 2026 ICE_crystal
// Copyright (C) 2026 SpikeHS (original Machinist implementation)
// SPDX-License-Identifier: GPL-3.0-only

//! 机工资源、连击、过热、整备和野火规则；公共时间线与战斗状态由核心持有。

use std::sync::Arc;

use fight_engine::config::{
    ActionKind, JobResourceDefinition, JobStateRegistration, ProjectConfig, ResourceValue,
    SkillDefinition,
};
use fight_engine::contracts::{JobFactory, JobRuntime, JobStateMachine};
use fight_engine::state::resources::ResourceRegistry;
use fight_engine::state::{CombatState, ValidationResult};
use fight_engine::system::timeline::TimelineMutation;
use fight_engine::system::{SystemRules, ZERO_EPSILON};

const COMBO_NONE: i32 = 0;
const COMBO_SPLIT: i32 = 1;
const COMBO_SLUG: i32 = 2;
const COMBO_TIMER: &str = "machinist.combo";
const WILDFIRE_TIMER: &str = "machinist.wildfire";

enum Behavior {
    ComboOne,
    ComboTwo,
    ComboThree,
    Tool,
    AirAnchor,
    ChainSaw,
    Excavator,
    BlazingShot,
    FullMetalField,
    Hypercharge,
    Wildfire,
    Detonator,
    BarrelStabilizer,
    Reassemble,
    DirectAbility,
}

impl Behavior {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "mch_combo_one" => Self::ComboOne,
            "mch_combo_two" => Self::ComboTwo,
            "mch_combo_three" => Self::ComboThree,
            "mch_tool" => Self::Tool,
            "mch_air_anchor" => Self::AirAnchor,
            "mch_chain_saw" => Self::ChainSaw,
            "mch_excavator" => Self::Excavator,
            "mch_blazing_shot" => Self::BlazingShot,
            "mch_full_metal_field" => Self::FullMetalField,
            "mch_hypercharge" => Self::Hypercharge,
            "mch_wildfire" => Self::Wildfire,
            "mch_detonator" => Self::Detonator,
            "mch_barrel_stabilizer" => Self::BarrelStabilizer,
            "mch_reassemble" => Self::Reassemble,
            "mch_direct_ability" => Self::DirectAbility,
            _ => return None,
        })
    }
}

pub struct Machinist {
    project: Arc<ProjectConfig>,
    system: Arc<SystemRules>,
    runtime: JobRuntime,
}

impl JobFactory for Machinist {
    const TAG: &'static str = "machinist";

    fn create(project: Arc<ProjectConfig>, system: Arc<SystemRules>) -> Result<Self, String> {
        if let (Some(duration), Some(maximum)) = (
            project.job.timing.get("wildfire_duration"),
            project.job.resource_limits.get("wildfire_remaining"),
        ) && (duration - maximum).abs() > 1e-9
        {
            return Err(format!(
                "job config wildfire_duration and resources.wildfire_remaining.max_value must match, got {duration} and {maximum}"
            ));
        }
        let resources = [
            ("heat", "int", ResourceValue::Integer(0), "count"),
            ("battery", "int", ResourceValue::Integer(0), "count"),
            (
                "combo_stage",
                "int",
                ResourceValue::Integer(COMBO_NONE),
                "count",
            ),
            (
                "combo_remaining",
                "float",
                ResourceValue::Float(0.0),
                "seconds",
            ),
            ("wasted_heat", "int", ResourceValue::Integer(0), "count"),
            ("wasted_battery", "int", ResourceValue::Integer(0), "count"),
            (
                "wasted_overheated_stacks",
                "int",
                ResourceValue::Integer(0),
                "count",
            ),
            (
                "wildfire_active",
                "bool",
                ResourceValue::Bool(false),
                "count",
            ),
            (
                "wildfire_remaining",
                "float",
                ResourceValue::Float(0.0),
                "seconds",
            ),
            ("wildfire_hits", "int", ResourceValue::Integer(0), "count"),
            (
                "wildfire_missed_weaponskills",
                "int",
                ResourceValue::Integer(0),
                "count",
            ),
            (
                "wildfire_resolved_potency",
                "float",
                ResourceValue::Float(0.0),
                "potency",
            ),
            (
                "wildfire_lost_potency",
                "float",
                ResourceValue::Float(0.0),
                "potency",
            ),
        ]
        .into_iter()
        .map(|(key, resource_type, default_value, display_unit)| {
            (
                key.to_owned(),
                JobResourceDefinition {
                    key: key.to_owned(),
                    resource_type: resource_type.to_owned(),
                    default_value,
                    vector_group: "resource".to_owned(),
                    display_unit: display_unit.to_owned(),
                    max_value: None,
                },
            )
        })
        .collect();
        let mut runtime = JobRuntime::new(
            &JobStateRegistration {
                job_tag: Self::TAG.to_owned(),
                resources,
                statuses: project.job.statuses.clone(),
            },
            &project.job.resource_limits,
        )?;
        let combo_resources = runtime.resources.clone();
        runtime.timeline.register_countdown(
            COMBO_TIMER,
            "combo_remaining",
            |state| state.job.timeline_deadlines.get(COMBO_TIMER).copied(),
            move |_, state, _| {
                let mut next = state.clone();
                clear_combo(&combo_resources, &mut next)?;
                Ok(TimelineMutation {
                    state: Some(next),
                    ..TimelineMutation::default()
                })
            },
        )?;
        let expiry_resources = runtime.resources.clone();
        runtime.register_expiry_hook("overheated", move |state| {
            if let Some(status) = state.job.statuses.get("overheated") {
                let wasted = expiry_resources
                    .get(state, "wasted_overheated_stacks")?
                    .as_integer()
                    + status.stacks.max(0);
                expiry_resources.set(
                    state,
                    "wasted_overheated_stacks",
                    ResourceValue::Integer(wasted),
                )?;
            }
            Ok(())
        })?;
        let deadline_resources = runtime.resources.clone();
        let wildfire_resources = runtime.resources.clone();
        let wildfire_project = project.clone();
        let wildfire_system = system.clone();
        runtime.timeline.register_countdown(
            WILDFIRE_TIMER,
            "wildfire_remaining",
            move |state| {
                if deadline_resources
                    .get(state, "wildfire_active")
                    .expect("机工注册表始终包含野火布尔资源")
                    .as_bool()
                {
                    state.job.timeline_deadlines.get(WILDFIRE_TIMER).copied()
                } else {
                    None
                }
            },
            move |event, state, targetable_at| {
                if !wildfire_resources.get(state, "wildfire_active")?.as_bool() {
                    return Ok(TimelineMutation::default());
                }
                let base = f64::from(wildfire_resources.get(state, "wildfire_hits")?.as_integer())
                    * wildfire_project
                        .job
                        .timing_value("wildfire_potency_per_hit")?;
                let potential = wildfire_system.resolve_applied_potency_at_offset(state, base, 0.0);
                let targetable = targetable_at
                    .map_or_else(|| state.boss_targetable(), |check| check(event.timestamp));
                let mut next = state.clone();
                let resolved = if targetable {
                    wildfire_system.record_delayed_potency(&mut next, potential)
                } else {
                    0.0
                };
                finish_wildfire(
                    &wildfire_resources,
                    &mut next,
                    resolved,
                    if targetable { 0.0 } else { potential },
                )?;
                Ok(TimelineMutation {
                    state: Some(next),
                    ..TimelineMutation::default()
                })
            },
        )?;
        Ok(Self {
            project,
            system,
            runtime,
        })
    }
}

impl JobStateMachine for Machinist {
    fn runtime(&self) -> &JobRuntime {
        &self.runtime
    }

    fn default_fight_remaining(&self) -> Result<f64, String> {
        self.project.job.default_fight_remaining()
    }

    fn current_gcd_duration(&self, state: &CombatState) -> Result<f64, String> {
        Ok(state.base_gcd.unwrap_or(self.project.system.base_gcd))
    }

    fn is_instant(&self, _state: &CombatState, skill: &SkillDefinition) -> Result<bool, String> {
        Ok(skill.kind == ActionKind::Ogcd || skill.cast_time <= 0.0)
    }

    fn supports_behavior(&self, behavior: &str) -> bool {
        Behavior::parse(behavior).is_some()
    }

    fn validate_action(
        &self,
        state: &CombatState,
        skill: &SkillDefinition,
    ) -> Result<ValidationResult, String> {
        let active = |key| state.job.has_status(state.time(), key);
        let failure = match Behavior::parse(&skill.behavior) {
            Some(Behavior::Excavator) if !active("excavator_ready") => {
                Some("requires_excavator_ready")
            }
            Some(Behavior::BlazingShot) if !active("overheated") => Some("requires_overheated"),
            Some(Behavior::FullMetalField) if !active("full_metal_machinist") => {
                Some("requires_full_metal_machinist")
            }
            Some(Behavior::Hypercharge) => {
                if active("overheated") {
                    Some("already_overheated")
                } else if active("free_hypercharge")
                    || self.runtime.resources.get(state, "heat")?.as_integer() >= 50
                {
                    None
                } else {
                    Some("not_enough_heat")
                }
            }
            Some(Behavior::Wildfire)
                if self
                    .runtime
                    .resources
                    .get(state, "wildfire_active")?
                    .as_bool() =>
            {
                Some("wildfire_already_active")
            }
            Some(Behavior::Detonator)
                if !self
                    .runtime
                    .resources
                    .get(state, "wildfire_active")?
                    .as_bool() =>
            {
                Some("requires_wildfire")
            }
            Some(Behavior::Reassemble) if active("reassemble") => Some("status_already_active"),
            _ => None,
        };
        Ok(failure.map_or_else(ValidationResult::valid, ValidationResult::invalid))
    }

    fn apply_action(
        &self,
        previous: &CombatState,
        next: &mut CombatState,
        skill: &SkillDefinition,
    ) -> Result<(), String> {
        let resources = &self.runtime.resources;
        match Behavior::parse(&skill.behavior)
            .ok_or_else(|| format!("machinist does not support behavior: {}", skill.behavior))?
        {
            Behavior::ComboOne => {
                gain_resource(resources, next, "heat", "wasted_heat", 5)?;
                self.set_combo(next, COMBO_SPLIT)?;
            }
            Behavior::ComboTwo => {
                if self.combo_matches(previous, COMBO_SPLIT)? {
                    gain_resource(resources, next, "heat", "wasted_heat", 5)?;
                    self.set_combo(next, COMBO_SLUG)?;
                } else {
                    clear_combo(resources, next)?;
                }
            }
            Behavior::ComboThree => {
                if self.combo_matches(previous, COMBO_SLUG)? {
                    gain_resource(resources, next, "heat", "wasted_heat", 5)?;
                    gain_resource(resources, next, "battery", "wasted_battery", 10)?;
                }
                clear_combo(resources, next)?;
            }
            Behavior::AirAnchor => gain_resource(resources, next, "battery", "wasted_battery", 20)?,
            Behavior::ChainSaw => {
                gain_resource(resources, next, "battery", "wasted_battery", 20)?;
                self.runtime
                    .grant_status(next, "excavator_ready", None, None)?;
            }
            Behavior::Excavator => {
                gain_resource(resources, next, "battery", "wasted_battery", 20)?;
                self.runtime.clear_status(next, "excavator_ready");
            }
            Behavior::BlazingShot => {
                self.runtime.consume_status_stack(next, "overheated");
                for key in ["double_check", "checkmate"] {
                    self.system
                        .apply_cooldown_reduction(next, self.require_skill(key)?, 15.0);
                }
            }
            Behavior::FullMetalField => self.runtime.clear_status(next, "full_metal_machinist"),
            Behavior::Hypercharge => {
                if next.job.has_status(next.time(), "free_hypercharge") {
                    self.runtime.clear_status(next, "free_hypercharge");
                } else {
                    let heat = resources.get(next, "heat")?.as_integer() - 50;
                    resources.set(next, "heat", ResourceValue::Integer(heat))?;
                }
                self.runtime
                    .grant_status(next, "overheated", None, Some(5))?;
            }
            Behavior::Wildfire => {
                resources.set(next, "wildfire_active", ResourceValue::Bool(true))?;
                next.job.timeline_deadlines.insert(
                    WILDFIRE_TIMER.to_owned(),
                    next.time() + self.project.job.timing_value("wildfire_duration")?,
                );
                resources.set(next, "wildfire_hits", ResourceValue::Integer(0))?;
            }
            Behavior::Detonator => {
                let base = self.wildfire_base_potency(previous)?;
                let resolved = self.system.resolve_applied_potency(previous, base, None);
                finish_wildfire(resources, next, resolved, 0.0)?;
            }
            Behavior::BarrelStabilizer => {
                self.runtime
                    .grant_status(next, "free_hypercharge", None, None)?;
                self.runtime
                    .grant_status(next, "full_metal_machinist", None, None)?;
            }
            Behavior::Reassemble => self.runtime.grant_status(next, "reassemble", None, None)?,
            Behavior::Tool | Behavior::DirectAbility => {}
        }
        if skill.kind == ActionKind::Gcd && has_tag(skill, "weaponskill") {
            if resources.get(next, "wildfire_active")?.as_bool() {
                let hits = (resources.get(next, "wildfire_hits")?.as_integer() + 1)
                    .min(resource_max(resources, "wildfire_hits")?);
                resources.set(next, "wildfire_hits", ResourceValue::Integer(hits))?;
            }
            if next.job.has_status(next.time(), "reassemble")
                && skill.potency > 0
                && !has_tag(skill, "reassemble_immune")
            {
                self.runtime.clear_status(next, "reassemble");
            }
        }
        Ok(())
    }

    fn consume_timing_resources(
        &self,
        _state: &mut CombatState,
        _skill: &SkillDefinition,
    ) -> Result<(), String> {
        Ok(())
    }

    fn actual_mp_cost(
        &self,
        _state: &CombatState,
        _skill: &SkillDefinition,
    ) -> Result<i32, String> {
        Ok(0)
    }

    fn resolve_potency(
        &self,
        state: &CombatState,
        skill: &SkillDefinition,
        base: f64,
    ) -> Result<f64, String> {
        let mut potency = base.max(0.0);
        if skill.key == "heated_slug_shot" && self.combo_matches(state, COMBO_SPLIT)? {
            potency = 320.0;
        } else if skill.key == "heated_clean_shot" && self.combo_matches(state, COMBO_SLUG)? {
            potency = 420.0;
        } else if skill.key == "detonator"
            && self
                .runtime
                .resources
                .get(state, "wildfire_active")?
                .as_bool()
        {
            potency = self.wildfire_base_potency(state)?;
        }
        if state.job.has_status(state.time(), "overheated")
            && has_tag(skill, "single_target_weaponskill")
        {
            potency += 20.0;
        }
        Ok(potency)
    }

    fn resolve_action_value(
        &self,
        state: &CombatState,
        skill: &SkillDefinition,
        base: f64,
    ) -> Result<f64, String> {
        let value = base.max(0.0);
        let applies = state.job.has_status(state.time(), "reassemble")
            && skill.kind == ActionKind::Gcd
            && skill.potency > 0
            && has_tag(skill, "weaponskill")
            && !has_tag(skill, "reassemble_immune");
        Ok(if applies { value.max(2.0) } else { value })
    }
}

impl Machinist {
    fn set_combo(&self, state: &mut CombatState, stage: i32) -> Result<(), String> {
        self.runtime
            .resources
            .set(state, "combo_stage", ResourceValue::Integer(stage))?;
        state.job.timeline_deadlines.insert(
            COMBO_TIMER.to_owned(),
            state.time() + self.project.job.timing_value("combo_duration")?,
        );
        Ok(())
    }

    fn combo_matches(&self, state: &CombatState, stage: i32) -> Result<bool, String> {
        Ok(self
            .runtime
            .resources
            .get(state, "combo_stage")?
            .as_integer()
            == stage
            && state
                .job
                .timeline_deadlines
                .get(COMBO_TIMER)
                .is_some_and(|deadline| *deadline > state.time() + ZERO_EPSILON))
    }

    fn wildfire_base_potency(&self, state: &CombatState) -> Result<f64, String> {
        Ok(f64::from(
            self.runtime
                .resources
                .get(state, "wildfire_hits")?
                .as_integer(),
        ) * self.project.job.timing_value("wildfire_potency_per_hit")?)
    }

    fn require_skill(&self, key: &str) -> Result<&SkillDefinition, String> {
        self.project
            .job
            .skills
            .iter()
            .find(|skill| skill.key == key)
            .ok_or_else(|| format!("machinist skill {key} is missing from job config"))
    }
}

fn resource_max(resources: &ResourceRegistry, key: &str) -> Result<i32, String> {
    resources
        .max(key)?
        .map(|value| value as i32)
        .ok_or_else(|| format!("job resource {key} is missing max_value"))
}

fn clear_combo(resources: &ResourceRegistry, state: &mut CombatState) -> Result<(), String> {
    resources.set(state, "combo_stage", ResourceValue::Integer(COMBO_NONE))?;
    state.job.timeline_deadlines.remove(COMBO_TIMER);
    Ok(())
}

fn gain_resource(
    resources: &ResourceRegistry,
    state: &mut CombatState,
    key: &str,
    wasted_key: &str,
    amount: i32,
) -> Result<(), String> {
    let raw = resources.get(state, key)?.as_integer() + amount.max(0);
    let maximum = resource_max(resources, key)?;
    if raw > maximum {
        let wasted = resources.get(state, wasted_key)?.as_integer() + raw - maximum;
        resources.set(state, wasted_key, ResourceValue::Integer(wasted))?;
    }
    resources.set(state, key, ResourceValue::Integer(raw.min(maximum)))
}

fn finish_wildfire(
    resources: &ResourceRegistry,
    state: &mut CombatState,
    resolved: f64,
    lost: f64,
) -> Result<(), String> {
    let hits = resources.get(state, "wildfire_hits")?.as_integer();
    for (key, potency) in [
        ("wildfire_resolved_potency", resolved),
        ("wildfire_lost_potency", lost),
    ] {
        let total = resources.get(state, key)?.as_float() + potency.max(0.0);
        resources.set(state, key, ResourceValue::Float(total))?;
    }
    let missed = resources
        .get(state, "wildfire_missed_weaponskills")?
        .as_integer()
        + (resource_max(resources, "wildfire_hits")? - hits).max(0);
    resources.set(
        state,
        "wildfire_missed_weaponskills",
        ResourceValue::Integer(missed),
    )?;
    resources.set(state, "wildfire_active", ResourceValue::Bool(false))?;
    state.job.timeline_deadlines.remove(WILDFIRE_TIMER);
    resources.set(state, "wildfire_hits", ResourceValue::Integer(0))
}

fn has_tag(skill: &SkillDefinition, tag: &str) -> bool {
    skill.tags.iter().any(|value| value == tag)
}

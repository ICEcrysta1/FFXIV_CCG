// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// 附加许可：FightEngine 链接例外 1.1 版，见 LICENSE-FightEngine-Linking-Exception。

//! 黑魔职业规则；公共时钟、MP、冷却和目标结算由核心系统持有。

use std::collections::BTreeMap;
use std::sync::Arc;

use fight_engine::config::{
    ActionKind, JobResourceDefinition, JobStateRegistration, ProjectConfig, ResourceValue,
    SkillDefinition,
};
use fight_engine::contracts::{JobFactory, JobRuntime, JobStateMachine};
use fight_engine::state::{CombatState, ValidationResult};
use fight_engine::system::timeline::TimelineMutation;
use fight_engine::system::{SystemRules, ZERO_EPSILON};

const POLYGLOT_TIMER: &str = "black_mage.polyglot";
const AF_FIRE_MULTIPLIERS: [f64; 4] = [1.0, 1.4, 1.6, 1.8];
const OPPOSITE_MULTIPLIERS: [f64; 4] = [1.0, 0.9, 0.8, 0.7];
const LUCID_TICK_MP: i32 = 550;
const UI_MP_PER_STACK: i32 = 2500;

#[derive(Clone, Copy)]
enum Aspect {
    Fire,
    Ice,
    None,
}

pub struct BlackMage {
    project: Arc<ProjectConfig>,
    system: Arc<SystemRules>,
    runtime: JobRuntime,
}

impl JobFactory for BlackMage {
    const TAG: &'static str = "black_mage";

    fn create(project: Arc<ProjectConfig>, system: Arc<SystemRules>) -> Result<Self, String> {
        let definitions = [
            (
                "astral_fire",
                "int",
                ResourceValue::Integer(0),
                "resource",
                "count",
            ),
            (
                "umbral_ice",
                "int",
                ResourceValue::Integer(0),
                "resource",
                "count",
            ),
            (
                "umbral_hearts",
                "int",
                ResourceValue::Integer(0),
                "resource",
                "count",
            ),
            (
                "astral_soul",
                "int",
                ResourceValue::Integer(0),
                "resource",
                "count",
            ),
            (
                "polyglot",
                "int",
                ResourceValue::Integer(0),
                "resource",
                "count",
            ),
            (
                "polyglot_timer",
                "float",
                ResourceValue::Float(0.0),
                "resource",
                "seconds",
            ),
            (
                "paradox_ready",
                "bool",
                ResourceValue::Bool(false),
                "resource",
                "count",
            ),
            (
                "thundercloud_ready",
                "bool",
                ResourceValue::Bool(false),
                "buff",
                "count",
            ),
            (
                "firestarter_ready",
                "bool",
                ResourceValue::Bool(false),
                "buff",
                "count",
            ),
        ];
        let registration = JobStateRegistration {
            job_tag: Self::TAG.into(),
            resources: definitions
                .into_iter()
                .map(
                    |(key, resource_type, default_value, vector_group, display_unit)| {
                        (
                            key.into(),
                            JobResourceDefinition {
                                key: key.into(),
                                resource_type: resource_type.into(),
                                default_value,
                                vector_group: vector_group.into(),
                                display_unit: display_unit.into(),
                                max_value: None,
                            },
                        )
                    },
                )
                .collect::<BTreeMap<_, _>>(),
            statuses: project.job.statuses.clone(),
        };
        let mut runtime = JobRuntime::new(&registration, &project.job.resource_limits)?;
        let interval = project.job.timing_value("polyglot_interval")?;
        let resources = runtime.resources.clone();
        runtime.timeline.register_periodic(
            POLYGLOT_TIMER,
            interval,
            "polyglot_timer",
            |state| {
                elemental_active(state)
                    .then(|| state.job.timeline_deadlines.get(POLYGLOT_TIMER).copied())
                    .flatten()
            },
            move |event, state| {
                let Some(&deadline) = state.job.timeline_deadlines.get(POLYGLOT_TIMER) else {
                    return Ok(TimelineMutation::default());
                };
                if !elemental_active(state) || state.time().max(deadline) != event.timestamp {
                    return Ok(TimelineMutation::default());
                }
                let mut next = state.clone();
                let current = resources.get(&next, "polyglot")?.as_integer();
                let maximum = resources
                    .max("polyglot")?
                    .ok_or("job resource polyglot is missing max_value")?
                    as i32;
                resources.set(
                    &mut next,
                    "polyglot",
                    ResourceValue::Integer(maximum.min(current + 1)),
                )?;
                next.job
                    .timeline_deadlines
                    .insert(POLYGLOT_TIMER.into(), next.time() + interval);
                Ok(TimelineMutation {
                    state: Some(next),
                    ..TimelineMutation::default()
                })
            },
        )?;
        runtime
            .timeline
            .register_mp_modifier(|state, natural_amount| {
                if resource_integer(state, "astral_fire") > 0 {
                    return 0;
                }
                let lucid = state
                    .job
                    .statuses
                    .get("lucid_dreaming")
                    .is_some_and(|status| status.remaining(state.time()) > ZERO_EPSILON);
                natural_amount.max(0) + if lucid { LUCID_TICK_MP } else { 0 }
            })?;
        Ok(Self {
            project,
            system,
            runtime,
        })
    }
}

impl JobStateMachine for BlackMage {
    fn runtime(&self) -> &JobRuntime {
        &self.runtime
    }

    fn default_fight_remaining(&self) -> Result<f64, String> {
        self.project.job.default_fight_remaining()
    }

    fn current_gcd_duration(&self, state: &CombatState) -> Result<f64, String> {
        let gcd = state.base_gcd.unwrap_or(self.project.system.base_gcd);
        Ok(if state.job.has_status(state.time(), "ley_lines") {
            gcd * self
                .project
                .job
                .timing_value("ley_lines_haste_multiplier")?
        } else {
            gcd
        })
    }

    fn is_instant(&self, state: &CombatState, skill: &SkillDefinition) -> Result<bool, String> {
        Ok(skill.kind == ActionKind::Ogcd
            || skill.cast_time <= 0.0
            || skill.key == "fire_iii" && self.flag(state, "firestarter_ready")?
            || state.job.has_status(state.time(), "swiftcast")
            || state.job.has_status(state.time(), "triplecast"))
    }

    fn supports_behavior(&self, behavior: &str) -> bool {
        matches!(
            behavior,
            "fire_iii"
                | "high_fire_ii"
                | "fire_iv"
                | "despair"
                | "flare"
                | "flare_star"
                | "blizzard_iii"
                | "blizzard_iv"
                | "freeze"
                | "high_blizzard_ii"
                | "paradox"
                | "polyglot_spender"
                | "thundercloud_dot"
                | "transpose"
                | "umbral_soul"
                | "manafont"
                | "amplifier"
        )
    }

    fn validate_action(
        &self,
        state: &CombatState,
        skill: &SkillDefinition,
    ) -> Result<ValidationResult, String> {
        let reason = match skill.behavior.as_str() {
            "fire_iv" | "despair" | "flare" | "manafont" if !self.in_fire(state)? => {
                Some("requires_af")
            }
            "blizzard_iv" | "freeze" | "umbral_soul" if !self.in_ice(state)? => Some("requires_ui"),
            "flare_star" if self.int(state, "astral_soul")? < self.maximum("astral_soul")? => {
                Some("insufficient_astral_soul")
            }
            "paradox" if !self.flag(state, "paradox_ready")? => Some("requires_paradox"),
            "polyglot_spender" if self.int(state, "polyglot")? < 1 => Some("requires_polyglot"),
            "thundercloud_dot" if !self.flag(state, "thundercloud_ready")? => {
                Some("requires_thundercloud")
            }
            "transpose" | "amplifier" if !self.in_fire(state)? && !self.in_ice(state)? => {
                Some("requires_elemental_state")
            }
            "despair" | "flare" if state.mp < skill.mp_cost_floor => Some("not_enough_mp"),
            "fire_iii" | "high_fire_ii" | "fire_iv" | "blizzard_iii" | "high_blizzard_ii"
            | "blizzard_iv" | "freeze" | "paradox"
                if state.mp < self.actual_mp_cost(state, skill)? =>
            {
                Some("not_enough_mp")
            }
            _ => None,
        };
        Ok(reason.map_or_else(ValidationResult::valid, ValidationResult::invalid))
    }

    fn apply_action(
        &self,
        previous: &CombatState,
        next: &mut CombatState,
        skill: &SkillDefinition,
    ) -> Result<(), String> {
        match skill.behavior.as_str() {
            "fire_iii" | "high_fire_ii" => {
                self.system
                    .spend_mp(next, self.actual_mp_cost(previous, skill)?);
                self.set_fire(next, self.maximum("astral_fire")?)?;
                self.set(next, "thundercloud_ready", ResourceValue::Bool(true))?;
                if self.int(previous, "umbral_ice")? == self.maximum("umbral_ice")?
                    && self.int(previous, "umbral_hearts")? >= self.maximum("umbral_hearts")?
                {
                    self.set(next, "paradox_ready", ResourceValue::Bool(true))?;
                }
            }
            "fire_iv" => {
                self.system
                    .spend_mp(next, self.actual_mp_cost(previous, skill)?);
                if self.int(next, "umbral_hearts")? > 0 {
                    self.set(
                        next,
                        "umbral_hearts",
                        ResourceValue::Integer(self.int(next, "umbral_hearts")? - 1),
                    )?;
                }
                self.set(
                    next,
                    "astral_soul",
                    ResourceValue::Integer(
                        self.maximum("astral_soul")?
                            .min(self.int(next, "astral_soul")? + 1),
                    ),
                )?;
            }
            "despair" => {
                self.system.spend_mp(next, next.mp);
                self.set_fire(next, self.maximum("astral_fire")?)?;
            }
            "flare" => {
                self.system.spend_mp(next, self.flare_cost(next)?);
                self.set_fire(next, self.maximum("astral_fire")?)?;
                if self.int(next, "umbral_hearts")? > 0 {
                    self.set(next, "umbral_hearts", ResourceValue::Integer(0))?;
                }
                self.set(
                    next,
                    "astral_soul",
                    ResourceValue::Integer(
                        self.maximum("astral_soul")?
                            .min(self.int(next, "astral_soul")? + 3),
                    ),
                )?;
            }
            "flare_star" => self.set(next, "astral_soul", ResourceValue::Integer(0))?,
            "blizzard_iii" | "high_blizzard_ii" => {
                self.system
                    .spend_mp(next, self.actual_mp_cost(previous, skill)?);
                self.set_ice(next, self.maximum("umbral_ice")?)?;
                self.set(next, "thundercloud_ready", ResourceValue::Bool(true))?;
                if self.int(previous, "astral_fire")? == self.maximum("astral_fire")? {
                    self.set(next, "paradox_ready", ResourceValue::Bool(true))?;
                }
                let ui = self.int(previous, "umbral_ice")?;
                self.system.recover_mp(
                    next,
                    if ui >= self.maximum("umbral_ice")? {
                        next.max_mp
                    } else {
                        UI_MP_PER_STACK * ui.max(1)
                    },
                );
            }
            "blizzard_iv" | "freeze" => {
                self.system
                    .spend_mp(next, self.actual_mp_cost(previous, skill)?);
                self.set(
                    next,
                    "umbral_hearts",
                    ResourceValue::Integer(self.maximum("umbral_hearts")?),
                )?;
                self.recover_ice_mp(next)?;
            }
            "paradox" => {
                if self.in_fire(previous)? {
                    self.system
                        .spend_mp(next, self.actual_mp_cost(previous, skill)?);
                    self.set(next, "firestarter_ready", ResourceValue::Bool(true))?;
                }
                self.set(next, "paradox_ready", ResourceValue::Bool(false))?;
            }
            "polyglot_spender" => self.set(
                next,
                "polyglot",
                ResourceValue::Integer((self.int(next, "polyglot")? - 1).max(0)),
            )?,
            "thundercloud_dot" => {
                self.set(next, "thundercloud_ready", ResourceValue::Bool(false))?;
                let potency = self.system.resolve_applied_potency(
                    previous,
                    self.resolve_potency(previous, skill, f64::from(skill.dot_potency))?,
                    Some(skill),
                );
                self.system.grant_target_dot(
                    next,
                    skill.dot_key.as_deref().unwrap_or(&skill.key),
                    skill.dot_duration,
                    potency,
                    3.0,
                )?;
            }
            "transpose" => {
                if self.in_fire(previous)? {
                    self.set_ice(next, 1)?;
                    if self.int(previous, "astral_fire")? == self.maximum("astral_fire")? {
                        self.set(next, "paradox_ready", ResourceValue::Bool(true))?;
                    }
                } else if self.in_ice(previous)? {
                    self.set_fire(next, 1)?;
                    if self.int(previous, "umbral_ice")? == self.maximum("umbral_ice")?
                        && self.int(previous, "umbral_hearts")? >= self.maximum("umbral_hearts")?
                    {
                        self.set(next, "paradox_ready", ResourceValue::Bool(true))?;
                    }
                } else {
                    self.set_fire(next, 1)?;
                }
                self.set(next, "thundercloud_ready", ResourceValue::Bool(true))?;
            }
            "umbral_soul" => {
                let current = self.int(next, "umbral_ice")?;
                self.set_ice(
                    next,
                    if current > 0 {
                        self.maximum("umbral_ice")?.min(current + 1)
                    } else {
                        1
                    },
                )?;
                self.set(
                    next,
                    "umbral_hearts",
                    ResourceValue::Integer(
                        self.maximum("umbral_hearts")?
                            .min(self.int(next, "umbral_hearts")? + 1),
                    ),
                )?;
                self.recover_ice_mp(next)?;
            }
            "manafont" => {
                self.system.restore_mp(next);
                self.set_fire(next, self.maximum("astral_fire")?)?;
                self.set(
                    next,
                    "umbral_hearts",
                    ResourceValue::Integer(self.maximum("umbral_hearts")?),
                )?;
                self.set(next, "paradox_ready", ResourceValue::Bool(true))?;
                self.set(next, "thundercloud_ready", ResourceValue::Bool(true))?;
            }
            "amplifier" => self.set(
                next,
                "polyglot",
                ResourceValue::Integer(
                    self.maximum("polyglot")?
                        .min(self.int(next, "polyglot")? + 1),
                ),
            )?,
            behavior => return Err(format!("black mage does not support behavior: {behavior}")),
        }
        Ok(())
    }

    fn consume_timing_resources(
        &self,
        state: &mut CombatState,
        skill: &SkillDefinition,
    ) -> Result<(), String> {
        if skill.kind != ActionKind::Gcd || skill.cast_time <= 0.0 {
            return Ok(());
        }
        if skill.key == "fire_iii" && self.flag(state, "firestarter_ready")? {
            self.set(state, "firestarter_ready", ResourceValue::Bool(false))?;
        } else if state.job.has_status(state.time(), "swiftcast") {
            self.runtime.clear_status(state, "swiftcast");
        } else if state.job.has_status(state.time(), "triplecast") {
            self.runtime.consume_status_stack(state, "triplecast");
        }
        Ok(())
    }

    fn on_death(&self, state: &mut CombatState) -> Result<(), String> {
        self.runtime.initialize(state)?;
        state.job.statuses.clear();
        state.job.timeline_deadlines.clear();
        Ok(())
    }

    fn cast_time_multiplier(
        &self,
        state: &CombatState,
        skill: &SkillDefinition,
    ) -> Result<f64, String> {
        if skill.kind != ActionKind::Gcd || skill.cast_time <= 0.0 {
            return Ok(1.0);
        }
        Ok(
            if self.int(state, "astral_fire")? >= self.maximum("astral_fire")?
                && has_tag(skill, "ice")
                || self.int(state, "umbral_ice")? >= self.maximum("umbral_ice")?
                    && has_tag(skill, "fire")
            {
                0.5
            } else {
                1.0
            },
        )
    }

    fn actual_mp_cost(&self, state: &CombatState, skill: &SkillDefinition) -> Result<i32, String> {
        if skill.key == "fire_iii" && self.flag(state, "firestarter_ready")? {
            return Ok(0);
        }
        match skill.key.as_str() {
            "despair" => Ok(state.mp),
            "flare" => self.flare_cost(state),
            "paradox" => self.elemental_cost(
                state,
                skill,
                if self.in_fire(state)? { 800 } else { 0 },
                false,
            ),
            _ if skill.mp_cost_is_full => Ok(0),
            _ => self.elemental_cost(state, skill, skill.mp_cost, skill.key == "fire_iv"),
        }
    }

    fn resolve_potency(
        &self,
        state: &CombatState,
        skill: &SkillDefinition,
        base: f64,
    ) -> Result<f64, String> {
        let base = base.max(0.0);
        if base <= 0.0 || skill.key == "paradox" {
            return Ok(base);
        }
        let af = self.int(state, "astral_fire")?;
        let ui = self.int(state, "umbral_ice")?;
        let multiplier = match self.aspect(state, skill)? {
            Aspect::Fire if af > 0 => multiplier(&AF_FIRE_MULTIPLIERS, af)?,
            Aspect::Fire if ui > 0 => multiplier(&OPPOSITE_MULTIPLIERS, ui)?,
            Aspect::Ice if af > 0 => multiplier(&OPPOSITE_MULTIPLIERS, af)?,
            _ => 1.0,
        };
        Ok(base * multiplier)
    }
}

impl BlackMage {
    fn int(&self, state: &CombatState, key: &str) -> Result<i32, String> {
        Ok(self.runtime.resources.get(state, key)?.as_integer())
    }

    fn flag(&self, state: &CombatState, key: &str) -> Result<bool, String> {
        Ok(self.runtime.resources.get(state, key)?.as_bool())
    }

    fn set(&self, state: &mut CombatState, key: &str, value: ResourceValue) -> Result<(), String> {
        self.runtime.resources.set(state, key, value)
    }

    fn maximum(&self, key: &str) -> Result<i32, String> {
        self.runtime
            .resources
            .max(key)?
            .map(|value| value as i32)
            .ok_or_else(|| format!("job resource {key} is missing max_value"))
    }

    fn in_fire(&self, state: &CombatState) -> Result<bool, String> {
        Ok(self.int(state, "astral_fire")? > 0)
    }
    fn in_ice(&self, state: &CombatState) -> Result<bool, String> {
        Ok(self.int(state, "umbral_ice")? > 0)
    }

    fn set_fire(&self, state: &mut CombatState, stacks: i32) -> Result<(), String> {
        let was_elemental = self.in_fire(state)? || self.in_ice(state)?;
        self.set(
            state,
            "astral_fire",
            ResourceValue::Integer(stacks.min(self.maximum("astral_fire")?).max(0)),
        )?;
        self.set(state, "umbral_ice", ResourceValue::Integer(0))?;
        self.start_cycle(state, was_elemental)
    }

    fn set_ice(&self, state: &mut CombatState, stacks: i32) -> Result<(), String> {
        let was_elemental = self.in_fire(state)? || self.in_ice(state)?;
        self.set(
            state,
            "umbral_ice",
            ResourceValue::Integer(stacks.min(self.maximum("umbral_ice")?).max(0)),
        )?;
        self.set(state, "astral_fire", ResourceValue::Integer(0))?;
        self.set(state, "astral_soul", ResourceValue::Integer(0))?;
        self.start_cycle(state, was_elemental)
    }

    fn start_cycle(&self, state: &mut CombatState, was_elemental: bool) -> Result<(), String> {
        if !was_elemental {
            state.job.timeline_deadlines.insert(
                POLYGLOT_TIMER.into(),
                state.time() + self.project.job.timing_value("polyglot_interval")?,
            );
        }
        Ok(())
    }

    fn flare_cost(&self, state: &CombatState) -> Result<i32, String> {
        Ok(if self.int(state, "umbral_hearts")? > 0 {
            state.mp / 2
        } else {
            state.mp
        })
    }

    fn recover_ice_mp(&self, state: &mut CombatState) -> Result<(), String> {
        let ui = self.int(state, "umbral_ice")?;
        if ui > 0 {
            self.system.recover_mp(
                state,
                if ui >= self.maximum("umbral_ice")? {
                    10_000
                } else {
                    UI_MP_PER_STACK * ui
                },
            );
        }
        Ok(())
    }

    fn aspect(&self, state: &CombatState, skill: &SkillDefinition) -> Result<Aspect, String> {
        if skill.key == "paradox" {
            return Ok(if self.in_fire(state)? {
                Aspect::Fire
            } else if self.in_ice(state)? {
                Aspect::Ice
            } else {
                Aspect::None
            });
        }
        Ok(match (has_tag(skill, "fire"), has_tag(skill, "ice")) {
            (true, false) => Aspect::Fire,
            (false, true) => Aspect::Ice,
            _ => Aspect::None,
        })
    }

    fn elemental_cost(
        &self,
        state: &CombatState,
        skill: &SkillDefinition,
        base: i32,
        heart_discount: bool,
    ) -> Result<i32, String> {
        let base = base.max(0);
        if base <= 0 {
            return Ok(0);
        }
        match self.aspect(state, skill)? {
            Aspect::Fire if self.in_ice(state)? => Ok(0),
            Aspect::Fire => {
                let mut cost = f64::from(base);
                if self.in_fire(state)? {
                    cost *= 2.0;
                    if heart_discount && self.int(state, "umbral_hearts")? > 0 {
                        cost *= 0.5;
                    }
                }
                Ok(cost as i32)
            }
            Aspect::Ice if self.in_fire(state)? || self.in_ice(state)? => Ok(0),
            _ => Ok(base),
        }
    }
}

fn has_tag(skill: &SkillDefinition, tag: &str) -> bool {
    skill.tags.iter().any(|value| value == tag)
}

fn resource_integer(state: &CombatState, key: &str) -> i32 {
    state
        .job
        .resources
        .get(key)
        .copied()
        .unwrap_or(ResourceValue::Integer(0))
        .as_integer()
}

fn elemental_active(state: &CombatState) -> bool {
    resource_integer(state, "astral_fire") > 0 || resource_integer(state, "umbral_ice") > 0
}

fn multiplier(values: &[f64], stacks: i32) -> Result<f64, String> {
    values
        .get(stacks as usize)
        .copied()
        .ok_or_else(|| format!("unsupported elemental stack count: {stacks}"))
}

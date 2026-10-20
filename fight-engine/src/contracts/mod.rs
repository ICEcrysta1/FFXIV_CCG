//! 职业扩展契约与注册；核心不依赖任何具体职业实现。

use std::any::TypeId;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::config::{JobStateRegistration, ProjectConfig, SkillDefinition};
use crate::state::resources::{ResourceRegistry, ResourceSnapshot, ResourceTransition};
use crate::state::status::StatusRules;
use crate::state::{CombatState, ValidationResult};
use crate::system::SystemRules;
use crate::system::timeline::{
    TimelineEvent, TimelineEventKind, TimelineEventPriority, TimelineEventScope, TimelineMutation,
};

/// 职业实例在装配后共享，只能修改传入的队列状态。
pub trait JobStateMachine: Send + Sync {
    fn runtime(&self) -> &JobRuntime;
    fn default_fight_remaining(&self) -> Result<f64, String>;
    fn current_gcd_duration(&self, state: &CombatState) -> Result<f64, String>;
    fn is_instant(&self, state: &CombatState, skill: &SkillDefinition) -> Result<bool, String>;
    fn supports_behavior(&self, behavior: &str) -> bool;
    fn validate_action(
        &self,
        state: &CombatState,
        skill: &SkillDefinition,
    ) -> Result<ValidationResult, String>;
    fn apply_action(
        &self,
        previous: &CombatState,
        next: &mut CombatState,
        skill: &SkillDefinition,
    ) -> Result<(), String>;
    fn consume_timing_resources(
        &self,
        state: &mut CombatState,
        skill: &SkillDefinition,
    ) -> Result<(), String>;

    fn on_death(&self, _state: &mut CombatState) -> Result<(), String> {
        Ok(())
    }

    fn cast_time_multiplier(
        &self,
        _state: &CombatState,
        _skill: &SkillDefinition,
    ) -> Result<f64, String> {
        Ok(1.0)
    }

    fn actual_mp_cost(&self, state: &CombatState, skill: &SkillDefinition) -> Result<i32, String> {
        Ok(if skill.mp_cost_is_full {
            state.mp.max(0)
        } else {
            skill.mp_cost.max(0)
        })
    }

    fn resolve_potency(
        &self,
        _state: &CombatState,
        _skill: &SkillDefinition,
        base: f64,
    ) -> Result<f64, String> {
        Ok(base.max(0.0))
    }

    fn resolve_action_value(
        &self,
        _state: &CombatState,
        _skill: &SkillDefinition,
        base: f64,
    ) -> Result<f64, String> {
        Ok(base.max(0.0))
    }
}

/// 构造与注册分开；注册读取标签，不创建职业实例。
pub trait JobFactory: JobStateMachine + Sized + 'static {
    const TAG: &'static str;

    fn create(project: Arc<ProjectConfig>, system: Arc<SystemRules>) -> Result<Self, String>;
}

type Factory = dyn Fn(Arc<ProjectConfig>, Arc<SystemRules>) -> Result<Arc<dyn JobStateMachine>, String>
    + Send
    + Sync;

struct Registration {
    type_id: Option<TypeId>,
    factory: Arc<Factory>,
}

#[derive(Default)]
pub struct JobRegistry {
    registrations: BTreeMap<String, Registration>,
}

impl JobRegistry {
    pub fn register<T: JobFactory>(&mut self) -> Result<(), String> {
        let tag = T::TAG;
        if tag.is_empty() {
            return Err("job state machine must define a non-empty tag".into());
        }
        if let Some(existing) = self.registrations.get(tag) {
            return if existing.type_id == Some(TypeId::of::<T>()) {
                Ok(())
            } else {
                Err(format!("duplicate job state machine tag: {tag}"))
            };
        }
        self.registrations.insert(
            tag.to_owned(),
            Registration {
                type_id: Some(TypeId::of::<T>()),
                factory: Arc::new(|project, system| {
                    T::create(project, system).map(|job| Arc::new(job) as Arc<dyn JobStateMachine>)
                }),
            },
        );
        Ok(())
    }

    pub fn register_factory(
        &mut self,
        tag: impl Into<String>,
        factory: impl Fn(
            Arc<ProjectConfig>,
            Arc<SystemRules>,
        ) -> Result<Arc<dyn JobStateMachine>, String>
        + Send
        + Sync
        + 'static,
    ) -> Result<(), String> {
        let tag = tag.into();
        if self.registrations.contains_key(&tag) {
            return Err(format!("duplicate job state machine tag: {tag}"));
        }
        self.registrations.insert(
            tag,
            Registration {
                type_id: None,
                factory: Arc::new(factory),
            },
        );
        Ok(())
    }

    pub fn get(
        &self,
        tag: &str,
        project: Arc<ProjectConfig>,
        system: Arc<SystemRules>,
    ) -> Result<Option<Arc<dyn JobStateMachine>>, String> {
        self.registrations
            .get(tag)
            .map(|registration| (registration.factory)(project, system))
            .transpose()
    }

    pub fn registered_tags(&self) -> Vec<&str> {
        self.registrations.keys().map(String::as_str).collect()
    }
}

type Deadline = dyn Fn(&CombatState) -> Option<f64> + Send + Sync;
type Tick = dyn Fn(&TimelineEvent, &CombatState) -> Result<TimelineMutation, String> + Send + Sync;
type Expiry = dyn Fn(
        &TimelineEvent,
        &CombatState,
        Option<&dyn Fn(f64) -> bool>,
    ) -> Result<TimelineMutation, String>
    + Send
    + Sync;
type MpModifier = dyn Fn(&CombatState, i32) -> i32 + Send + Sync;

#[derive(Clone)]
struct PeriodicRegistration {
    output_key: String,
    interval: f64,
    deadline: Arc<Deadline>,
    tick: Arc<Tick>,
}

#[derive(Clone)]
struct CountdownRegistration {
    output_key: String,
    deadline: Arc<Deadline>,
    expiry: Arc<Expiry>,
}

/// 职业声明绝对截止时刻；中心时间线负责实际推进和事件派发。
#[derive(Clone, Default)]
pub struct JobTimelineRegistry {
    periodic: BTreeMap<String, PeriodicRegistration>,
    countdowns: BTreeMap<String, CountdownRegistration>,
    mp_modifier: Option<Arc<MpModifier>>,
}

impl JobTimelineRegistry {
    pub fn register_periodic(
        &mut self,
        key: impl Into<String>,
        interval: f64,
        output_key: impl Into<String>,
        deadline: impl Fn(&CombatState) -> Option<f64> + Send + Sync + 'static,
        tick: impl Fn(&TimelineEvent, &CombatState) -> Result<TimelineMutation, String>
        + Send
        + Sync
        + 'static,
    ) -> Result<(), String> {
        let key = key.into();
        let output_key = output_key.into();
        self.validate_registration(&key, &output_key)?;
        if !interval.is_finite() || interval <= 0.0 {
            return Err("job periodic interval must be positive and finite".into());
        }
        self.periodic.insert(
            key,
            PeriodicRegistration {
                output_key,
                interval,
                deadline: Arc::new(deadline),
                tick: Arc::new(tick),
            },
        );
        Ok(())
    }

    pub fn register_countdown(
        &mut self,
        key: impl Into<String>,
        output_key: impl Into<String>,
        deadline: impl Fn(&CombatState) -> Option<f64> + Send + Sync + 'static,
        expiry: impl Fn(
            &TimelineEvent,
            &CombatState,
            Option<&dyn Fn(f64) -> bool>,
        ) -> Result<TimelineMutation, String>
        + Send
        + Sync
        + 'static,
    ) -> Result<(), String> {
        let key = key.into();
        let output_key = output_key.into();
        self.validate_registration(&key, &output_key)?;
        self.countdowns.insert(
            key,
            CountdownRegistration {
                output_key,
                deadline: Arc::new(deadline),
                expiry: Arc::new(expiry),
            },
        );
        Ok(())
    }

    fn validate_registration(&self, key: &str, output_key: &str) -> Result<(), String> {
        if key.trim().is_empty() || output_key.trim().is_empty() {
            return Err("job timer key and output key must not be empty".into());
        }
        if self.periodic.contains_key(key) || self.countdowns.contains_key(key) {
            return Err(format!("job timer already registered: {key}"));
        }
        Ok(())
    }

    pub fn register_mp_modifier(
        &mut self,
        modifier: impl Fn(&CombatState, i32) -> i32 + Send + Sync + 'static,
    ) -> Result<(), String> {
        if self.mp_modifier.is_some() {
            return Err("job MP tick modifier already registered".into());
        }
        self.mp_modifier = Some(Arc::new(modifier));
        Ok(())
    }

    pub fn resolve_mp_tick(&self, state: &CombatState, natural_amount: i32) -> i32 {
        self.mp_modifier
            .as_ref()
            .map_or(natural_amount, |modifier| modifier(state, natural_amount))
            .max(0)
    }

    pub fn describe_events<P>(&self, state: &CombatState) -> Vec<TimelineEvent<P>> {
        let mut events = Vec::new();
        for (key, registration) in &self.periodic {
            if let Some(deadline) = (registration.deadline)(state) {
                events.push(TimelineEvent {
                    owner_key: Some(key.clone()),
                    scope: TimelineEventScope::Job,
                    ..TimelineEvent::new(
                        state.time().max(deadline),
                        TimelineEventPriority::PeriodicSettlement,
                        TimelineEventKind::JobPeriodicTick,
                    )
                });
            }
        }
        for (key, registration) in &self.countdowns {
            if let Some(deadline) = (registration.deadline)(state) {
                events.push(TimelineEvent {
                    owner_key: Some(key.clone()),
                    scope: TimelineEventScope::Job,
                    ..TimelineEvent::new(
                        state.time().max(deadline),
                        TimelineEventPriority::ExpirationAndCooldown,
                        TimelineEventKind::JobTimerExpired,
                    )
                });
            }
        }
        events
    }

    pub fn handle_periodic(
        &self,
        event: &TimelineEvent,
        state: &CombatState,
    ) -> Result<TimelineMutation, String> {
        if let Some(registration) = event
            .owner_key
            .as_deref()
            .and_then(|key| self.periodic.get(key))
        {
            return (registration.tick)(event, state);
        }
        Ok(TimelineMutation::default())
    }

    pub fn handle_countdown(
        &self,
        event: &TimelineEvent,
        state: &CombatState,
        targetable_at: Option<&dyn Fn(f64) -> bool>,
    ) -> Result<TimelineMutation, String> {
        if let Some(registration) = event
            .owner_key
            .as_deref()
            .and_then(|key| self.countdowns.get(key))
        {
            return (registration.expiry)(event, state, targetable_at);
        }
        Ok(TimelineMutation::default())
    }

    pub fn timer_views(&self, state: &CombatState) -> Vec<(String, f64)> {
        let mut views = Vec::new();
        for registration in self.periodic.values() {
            let value = (registration.deadline)(state).map_or(0.0, |deadline| {
                (registration.interval - (deadline - state.time())).max(0.0)
            });
            views.push((registration.output_key.clone(), value));
        }
        for registration in self.countdowns.values() {
            let value = (registration.deadline)(state)
                .map_or(0.0, |deadline| (deadline - state.time()).max(0.0));
            views.push((registration.output_key.clone(), value));
        }
        views
    }
}

/// 职业规则持有注册定义，运行值只放在 CombatState.job。
pub struct JobRuntime {
    pub resources: Arc<ResourceRegistry>,
    pub statuses: StatusRules,
    pub timeline: JobTimelineRegistry,
}

impl JobRuntime {
    pub fn new(
        registration: &JobStateRegistration,
        limits: &BTreeMap<String, f64>,
    ) -> Result<Self, String> {
        Ok(Self {
            resources: Arc::new(ResourceRegistry::new(registration, limits)?),
            statuses: StatusRules::new(registration.statuses.clone()),
            timeline: JobTimelineRegistry::default(),
        })
    }

    pub fn initialize(&self, state: &mut CombatState) -> Result<(), String> {
        self.resources.initialize(state)
    }

    pub fn resource_snapshot(&self, state: &CombatState) -> Result<ResourceSnapshot, String> {
        let mut snapshot = self.resources.snapshot(state)?;
        for (key, value) in self.timeline.timer_views(state) {
            if snapshot.contains_key(&key) {
                snapshot.insert(
                    key.clone(),
                    self.resources
                        .normalize(&key, crate::config::ResourceValue::Float(value))?,
                );
            }
        }
        Ok(snapshot)
    }

    pub fn resource_transition(
        &self,
        previous: &CombatState,
        next: &CombatState,
    ) -> Result<ResourceTransition, String> {
        self.resources.transition(
            self.resource_snapshot(previous)?,
            self.resource_snapshot(next)?,
        )
    }

    pub fn validate_status(
        &self,
        state: &CombatState,
        key: &str,
    ) -> Result<ValidationResult, String> {
        self.statuses
            .validate(state.time(), &state.job.statuses, key)
    }

    pub fn grant_status(
        &self,
        state: &mut CombatState,
        key: &str,
        remaining: Option<f64>,
        stacks: Option<i32>,
    ) -> Result<(), String> {
        self.statuses.grant(
            state.time(),
            &mut state.job.statuses,
            key,
            remaining,
            stacks,
        )
    }

    pub fn clear_status(&self, state: &mut CombatState, key: &str) {
        self.statuses.clear(&mut state.job.statuses, key);
    }

    pub fn consume_status_stack(&self, state: &mut CombatState, key: &str) {
        self.statuses.consume_stack(&mut state.job.statuses, key);
    }

    pub fn register_expiry_hook(
        &mut self,
        key: impl Into<String>,
        hook: impl Fn(&mut CombatState) -> Result<(), String> + Send + Sync + 'static,
    ) -> Result<(), String> {
        self.statuses.register_expiry_hook(key, hook)
    }

    pub fn describe_status_events<P>(&self, state: &CombatState) -> Vec<TimelineEvent<P>> {
        state
            .job
            .statuses
            .iter()
            .map(|(key, status)| TimelineEvent {
                owner_key: Some(key.clone()),
                scope: TimelineEventScope::Job,
                ..TimelineEvent::new(
                    state.time().max(status.expires_at),
                    TimelineEventPriority::ExpirationAndCooldown,
                    TimelineEventKind::StatusExpired,
                )
            })
            .collect()
    }

    pub fn handle_status_expired<P>(
        &self,
        event: &TimelineEvent<P>,
        state: &mut CombatState,
    ) -> Result<(), String> {
        self.statuses
            .settle_expired(state, event, TimelineEventScope::Job)
    }
}

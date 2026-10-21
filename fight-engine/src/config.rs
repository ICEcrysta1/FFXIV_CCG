// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// 附加许可：FightEngine 链接例外 1.1 版，见 LICENSE-FightEngine-Linking-Exception。

//! 读取共享 YAML，装配系统、职业、技能与输出 schema；不执行职业规则。

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_yaml_ng::{Mapping, Value};

mod yaml;

use yaml::load_mapping;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    #[default]
    Gcd,
    Ogcd,
}

impl ActionKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "gcd" => Ok(Self::Gcd),
            "ogcd" => Ok(Self::Ogcd),
            _ => Err(format!("unsupported action kind: {value}")),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SkillDefinition {
    pub key: String,
    pub game_id: i32,
    pub name: String,
    pub kind: ActionKind,
    pub behavior: String,
    pub potency: i32,
    pub value: f64,
    pub cast_time: f64,
    pub recast_time: f64,
    pub mp_cost_is_full: bool,
    pub mp_cost: i32,
    pub mp_cost_floor: i32,
    pub cooldown: f64,
    pub charges: i32,
    pub enabled: bool,
    pub max_targets: i32,
    pub aoe_secondary_reduction: f64,
    pub dot_potency: i32,
    pub dot_duration: f64,
    pub dot_key: Option<String>,
    pub requires_target: bool,
    pub applies_statuses: Vec<String>,
    pub tags: Vec<String>,
}

impl Default for SkillDefinition {
    fn default() -> Self {
        Self {
            key: String::new(),
            game_id: 0,
            name: String::new(),
            kind: ActionKind::Gcd,
            behavior: String::new(),
            potency: 0,
            value: 1.0,
            cast_time: 0.0,
            recast_time: 2.5,
            mp_cost_is_full: false,
            mp_cost: 0,
            mp_cost_floor: 0,
            cooldown: 0.0,
            charges: 1,
            enabled: true,
            max_targets: 1,
            aoe_secondary_reduction: 1.0,
            dot_potency: 0,
            dot_duration: 0.0,
            dot_key: None,
            requires_target: false,
            applies_statuses: Vec::new(),
            tags: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StatusDefinition {
    pub key: String,
    pub game_id: i32,
    pub duration: f64,
    pub max_stacks: i32,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ResourceValue {
    Integer(i32),
    Float(f64),
    Bool(bool),
}

impl ResourceValue {
    pub fn as_integer(self) -> i32 {
        match self {
            Self::Integer(value) => value,
            Self::Float(value) => value as i32,
            Self::Bool(value) => i32::from(value),
        }
    }

    pub fn as_float(self) -> f64 {
        match self {
            Self::Integer(value) => f64::from(value),
            Self::Float(value) => value,
            Self::Bool(value) => f64::from(u8::from(value)),
        }
    }

    pub fn as_bool(self) -> bool {
        match self {
            Self::Integer(value) => value != 0,
            Self::Float(value) => value != 0.0,
            Self::Bool(value) => value,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct JobResourceDefinition {
    pub key: String,
    pub resource_type: String,
    pub default_value: ResourceValue,
    pub vector_group: String,
    pub display_unit: String,
    pub max_value: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct JobStateRegistration {
    pub job_tag: String,
    pub resources: BTreeMap<String, JobResourceDefinition>,
    pub statuses: IndexMap<String, StatusDefinition>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RuntimeConfig {
    pub job_tag: String,
    pub system_config_path: PathBuf,
    pub job_config_path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EngineTimingConfig {
    pub action_queue_window_seconds: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EngineResourceConfig {
    pub max_mp: i32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SystemPotencyConfig {
    pub burst_potion_multiplier: f64,
    pub raid_buff_window_multiplier: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SystemMpRecoveryConfig {
    pub tick_interval_seconds: f64,
    pub in_combat_amount: i32,
    pub revival_ratio: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SystemConfig {
    pub base_gcd: f64,
    pub skill_table_base_gcd: f64,
    pub mp_recovery: SystemMpRecoveryConfig,
    pub potency: SystemPotencyConfig,
    pub raid_buff_window_marker_skills: Vec<String>,
    pub raid_buff_window_duration: f64,
    pub statuses: IndexMap<String, StatusDefinition>,
    pub skills: Vec<SkillDefinition>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct JobConfig {
    pub key: String,
    pub name: String,
    pub timing: BTreeMap<String, f64>,
    pub resource_limits: BTreeMap<String, f64>,
    pub statuses: IndexMap<String, StatusDefinition>,
    pub skills: Vec<SkillDefinition>,
}

impl JobConfig {
    pub fn timing_value(&self, key: &str) -> Result<f64, String> {
        self.timing
            .get(key)
            .copied()
            .ok_or_else(|| format!("job timing is missing {key}"))
    }

    pub fn default_fight_remaining(&self) -> Result<f64, String> {
        self.timing_value("default_fight_remaining")
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ProjectConfig {
    pub name: String,
    pub version: String,
    pub runtime: RuntimeConfig,
    pub engine_timing: EngineTimingConfig,
    pub engine_resources: EngineResourceConfig,
    pub system: SystemConfig,
    pub job: JobConfig,
}

impl ProjectConfig {
    pub fn load(project_root: impl AsRef<Path>, job_tag: &str) -> Result<Self, String> {
        Self::from_path(project_root.as_ref().join("config/default.yaml"), job_tag)
    }

    pub fn from_path(config_path: impl AsRef<Path>, job_tag: &str) -> Result<Self, String> {
        let config_path = absolute_path(config_path.as_ref())?;
        let context = config_path.display().to_string();
        let root = config_path
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| format!("invalid project config path: {context}"))?;
        let data = load_mapping(&config_path)?;
        let runtime = require_mapping(&data, "runtime", &context)?;
        let timing = require_mapping(&data, "engine_timing", &context)?;
        let resources = require_mapping(&data, "engine_resources", &context)?;
        let configured_paths = require_mapping(runtime, "job_configs", &context)?;
        let configured_job = get(configured_paths, job_tag).ok_or_else(|| {
            let mut supported: Vec<_> = configured_paths.keys().map(to_text).collect();
            supported.sort();
            format!(
                "no job config registered for {job_tag}; supported={}",
                supported.join(", ")
            )
        })?;
        let system_path =
            resolve_project_path(&to_text(require(runtime, "system_config", &context)?), root)?;
        let job_path = resolve_project_path(&to_text(configured_job), root)?;
        let system = SystemConfig::load(&system_path)?;
        let job =
            JobConfig::from_mapping(&load_mapping(&job_path)?, &job_path.display().to_string())?;
        validate_definitions(&system, &job)?;
        let project = require_mapping(&data, "project", &context)?;
        Ok(Self {
            name: to_text(require(project, "name", &context)?),
            version: to_text(require(project, "version", &context)?),
            runtime: RuntimeConfig {
                job_tag: job_tag.to_owned(),
                system_config_path: system_path,
                job_config_path: job_path,
            },
            engine_timing: EngineTimingConfig {
                action_queue_window_seconds: number(
                    timing,
                    "action_queue_window_seconds",
                    &context,
                )?,
            },
            engine_resources: EngineResourceConfig {
                max_mp: integer(resources, "max_mp", &context)?,
            },
            system,
            job,
        })
    }
}

impl SystemConfig {
    pub fn load(config_path: impl AsRef<Path>) -> Result<Self, String> {
        let path = config_path.as_ref();
        Self::from_mapping(&load_mapping(path)?, &path.display().to_string())
    }

    fn from_mapping(data: &Mapping, context: &str) -> Result<Self, String> {
        let timing = require_mapping(data, "timing", context)?;
        let potency = optional_mapping(data, "potency", context)?;
        let raid = require_mapping(data, "raid_buff_window", context)?;
        let marker_skills = string_list(require(raid, "marker_skills", context)?)?;
        if marker_skills.is_empty() {
            return Err("raid_buff_window.marker_skills must not be empty".to_owned());
        }
        if marker_skills.iter().collect::<BTreeSet<_>>().len() != marker_skills.len() {
            return Err("raid_buff_window.marker_skills must not contain duplicates".to_owned());
        }
        let raid_duration = number(raid, "duration_seconds", context)?;
        if raid_duration <= 0.0 {
            return Err("raid_buff_window.duration_seconds must be positive".to_owned());
        }
        let recovery = require_mapping(data, "mp_recovery", context)?;
        let revival_ratio = number(
            recovery,
            "revival_ratio",
            &format!("{context}: mp_recovery.revival_ratio"),
        )?;
        if !revival_ratio.is_finite() || !(0.0..=1.0).contains(&revival_ratio) {
            return Err(
                "mp_recovery.revival_ratio must be a finite number between 0 and 1".to_owned(),
            );
        }
        Ok(Self {
            base_gcd: number(timing, "base_gcd", context)?,
            skill_table_base_gcd: number(timing, "skill_table_base_gcd", context)?,
            mp_recovery: SystemMpRecoveryConfig {
                tick_interval_seconds: number(recovery, "tick_interval_seconds", context)?,
                in_combat_amount: integer(recovery, "in_combat_amount", context)?,
                revival_ratio,
            },
            potency: SystemPotencyConfig {
                burst_potion_multiplier: optional_number(
                    potency,
                    "burst_potion_multiplier",
                    1.05,
                    context,
                )?,
                raid_buff_window_multiplier: optional_number(
                    potency,
                    "raid_buff_window_multiplier",
                    1.10,
                    context,
                )?,
            },
            raid_buff_window_marker_skills: marker_skills,
            raid_buff_window_duration: raid_duration,
            statuses: build_statuses(optional_mapping(data, "statuses", context)?, context)?,
            skills: build_skills(optional_mapping(data, "skills", context)?, context)?,
        })
    }
}

impl JobConfig {
    fn from_mapping(data: &Mapping, context: &str) -> Result<Self, String> {
        let job = require_mapping(data, "job", context)?;
        let timing = require_mapping(data, "timing", context)?;
        let mut resource_limits = BTreeMap::new();
        if let Some(resources) = optional_mapping(data, "resources", context)? {
            for (key, value) in resources {
                let key = to_text(key);
                let resource = as_mapping(value, &format!("job resource {key}"), context)?;
                let value = get(resource, "max_value")
                    .ok_or_else(|| format!("job resource {key} is missing max_value"))?;
                let limit = to_number(value, context)?;
                if !limit.is_finite() || limit <= 0.0 {
                    return Err(format!(
                        "job resource {key}: max_value must be positive and finite, got {limit}"
                    ));
                }
                resource_limits.insert(key, limit);
            }
        }
        Ok(Self {
            key: to_text(require(job, "key", context)?),
            name: to_text(require(job, "name", context)?),
            timing: timing
                .iter()
                .map(|(key, value)| Ok((to_text(key), to_number(value, context)?)))
                .collect::<Result<_, String>>()?,
            resource_limits,
            statuses: build_statuses(optional_mapping(data, "statuses", context)?, context)?,
            skills: build_skills(optional_mapping(data, "skills", context)?, context)?,
        })
    }
}

fn build_statuses(
    data: Option<&Mapping>,
    context: &str,
) -> Result<IndexMap<String, StatusDefinition>, String> {
    let mut statuses = IndexMap::new();
    for (key, value) in data.into_iter().flatten() {
        let key = to_text(key);
        let payload = as_mapping(value, &format!("status {key}"), context)?;
        statuses.insert(
            key.clone(),
            StatusDefinition {
                key,
                game_id: integer(payload, "game_id", context)?,
                duration: number(payload, "duration", context)?,
                max_stacks: optional_integer(payload, "max_stacks", 1, context)?,
            },
        );
    }
    Ok(statuses)
}

fn build_skills(data: Option<&Mapping>, context: &str) -> Result<Vec<SkillDefinition>, String> {
    let mut skills = Vec::new();
    for (key, value) in data.into_iter().flatten() {
        let key = to_text(key);
        let payload = as_mapping(value, &format!("skill {key}"), context)?;
        let mp_cost_is_full = matches!(get(payload, "mp_cost"), Some(Value::String(_)));
        let mp_cost = match get(payload, "mp_cost") {
            Some(Value::String(text)) if text == "full" => 0,
            Some(Value::String(text)) => {
                return Err(format!(
                    "unsupported mp_cost semantic for skill {key}: {text}"
                ));
            }
            Some(value) => to_integer(value, context)?,
            None => 0,
        };
        let aoe_secondary_reduction =
            optional_number(Some(payload), "aoe_secondary_reduction", 1.0, context)?;
        if !(0.0..=1.0).contains(&aoe_secondary_reduction) {
            return Err(format!(
                "skill {key}: aoe_secondary_reduction must be between 0 and 1, got {aoe_secondary_reduction}"
            ));
        }
        let value = optional_number(Some(payload), "value", 1.0, context)?;
        if !value.is_finite() || value <= 0.0 {
            return Err(format!(
                "skill {key}: value must be a positive finite number, got {value}"
            ));
        }
        let potency = optional_integer(payload, "potency", 0, context)?;
        let dot_potency = optional_integer(payload, "dot_potency", 0, context)?;
        let requires_target = get(payload, "requires_target")
            .filter(|value| !value.is_null())
            .map_or(potency > 0 || dot_potency > 0, to_bool);
        skills.push(SkillDefinition {
            key,
            game_id: integer(payload, "game_id", context)?,
            name: to_text(require(payload, "name", context)?),
            kind: ActionKind::parse(&to_text(require(payload, "kind", context)?))?,
            behavior: to_text(require(payload, "behavior", context)?),
            potency,
            value,
            cast_time: optional_number(Some(payload), "cast_time", 0.0, context)?,
            recast_time: optional_number(Some(payload), "recast_time", 2.5, context)?,
            mp_cost_is_full,
            mp_cost,
            mp_cost_floor: optional_integer(payload, "mp_cost_floor", mp_cost, context)?,
            cooldown: optional_number(Some(payload), "cooldown", 0.0, context)?,
            charges: optional_integer(payload, "charges", 1, context)?,
            enabled: get(payload, "enabled").is_none_or(to_bool),
            max_targets: optional_integer(payload, "max_targets", 1, context)?,
            aoe_secondary_reduction,
            dot_potency,
            dot_duration: optional_number(Some(payload), "dot_duration", 0.0, context)?,
            dot_key: get(payload, "dot_key")
                .filter(|value| !value.is_null())
                .map(to_text),
            requires_target,
            applies_statuses: optional_string_list(payload, "applies_statuses")?,
            tags: optional_string_list(payload, "tags")?,
        });
    }
    Ok(skills)
}

fn validate_definitions(system: &SystemConfig, job: &JobConfig) -> Result<(), String> {
    let mut system_status_ids = BTreeMap::new();
    for (key, status) in &system.statuses {
        if system_status_ids.insert(status.game_id, key).is_some() {
            return Err(format!(
                "duplicate system status game_id: {}",
                status.game_id
            ));
        }
    }
    for (key, status) in &job.statuses {
        if system.statuses.contains_key(key) {
            return Err(format!(
                "duplicate status keys across system/job config: {key}"
            ));
        }
        if let Some(system_key) = system_status_ids.get(&status.game_id) {
            return Err(format!(
                "duplicate status game_id across system/job config: {} ({system_key}, {key})",
                status.game_id
            ));
        }
    }
    let system_skill_keys: BTreeSet<_> = system.skills.iter().map(|skill| &skill.key).collect();
    let mut system_skill_ids = BTreeMap::new();
    for skill in &system.skills {
        if system_skill_ids.insert(skill.game_id, &skill.key).is_some() {
            return Err(format!("duplicate system skill game_id: {}", skill.game_id));
        }
    }
    for skill in &job.skills {
        if system_skill_keys.contains(&skill.key) {
            return Err(format!(
                "duplicate skill keys across system/job config: {}",
                skill.key
            ));
        }
        if let Some(system_key) = system_skill_ids.get(&skill.game_id) {
            return Err(format!(
                "duplicate skill game_id across system/job config: {} ({system_key}, {})",
                skill.game_id, skill.key
            ));
        }
    }
    for (source, statuses, skills) in [
        ("system", &system.statuses, &system.skills),
        ("job", &job.statuses, &job.skills),
    ] {
        for (key, status) in statuses {
            if status.max_stacks < 1 {
                return Err(format!(
                    "{source} status {key}: max_stacks must be >= 1, got {}",
                    status.max_stacks
                ));
            }
        }
        for skill in skills {
            if skill.behavior == "grant_status" && skill.applies_statuses.is_empty() {
                return Err(format!(
                    "{source} skill {} uses grant_status but applies_statuses is empty",
                    skill.key
                ));
            }
            for key in &skill.applies_statuses {
                if !system.statuses.contains_key(key) && !job.statuses.contains_key(key) {
                    return Err(format!(
                        "{source} skill {} references unregistered status: {key}",
                        skill.key
                    ));
                }
            }
        }
    }
    Ok(())
}

/// 按系统技能、职业技能的 YAML 顺序保留动作空间，同时建立查询索引。
#[derive(Clone, Debug)]
pub struct SkillBook {
    ordered: Vec<SkillDefinition>,
    by_key: BTreeMap<String, usize>,
    by_game_id: BTreeMap<i32, usize>,
}

impl SkillBook {
    pub fn new(skills: Vec<SkillDefinition>) -> Result<Self, String> {
        let mut by_key = BTreeMap::new();
        let mut by_game_id = BTreeMap::new();
        for (index, skill) in skills.iter().enumerate() {
            if by_key.insert(skill.key.clone(), index).is_some() {
                return Err(format!("duplicate skill key: {}", skill.key));
            }
            if by_game_id.insert(skill.game_id, index).is_some() {
                return Err(format!("duplicate skill game_id: {}", skill.game_id));
            }
        }
        Ok(Self {
            ordered: skills,
            by_key,
            by_game_id,
        })
    }

    pub fn from_project_config(config: &ProjectConfig) -> Result<Self, String> {
        Self::new(
            config
                .system
                .skills
                .iter()
                .chain(&config.job.skills)
                .cloned()
                .collect(),
        )
    }

    pub fn get(&self, key: &str) -> Result<&SkillDefinition, String> {
        self.by_key
            .get(key)
            .map(|index| &self.ordered[*index])
            .ok_or_else(|| format!("unknown skill key: {key}"))
    }

    pub fn by_game_id(&self, game_id: i32) -> Result<&SkillDefinition, String> {
        self.by_game_id
            .get(&game_id)
            .map(|index| &self.ordered[*index])
            .ok_or_else(|| format!("unknown skill game_id: {game_id}"))
    }

    pub fn enabled_skills(&self, kind: Option<ActionKind>) -> Vec<&SkillDefinition> {
        self.ordered
            .iter()
            .filter(|skill| skill.enabled && kind.is_none_or(|kind| skill.kind == kind))
            .collect()
    }

    pub fn keys(&self) -> Vec<&str> {
        self.ordered
            .iter()
            .map(|skill| skill.key.as_str())
            .collect()
    }

    pub fn contains(&self, key: &str) -> bool {
        self.by_key.contains_key(key)
    }

    pub fn contains_game_id(&self, game_id: i32) -> bool {
        self.by_game_id.contains_key(&game_id)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PolicyActionDefinition {
    pub key: String,
    pub raw_id: i32,
    pub name: String,
    pub kind: String,
    pub behavior: String,
    pub value: f64,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PolicyActionRegistry {
    ordered: Vec<PolicyActionDefinition>,
    by_key: BTreeMap<String, usize>,
}

impl PolicyActionRegistry {
    pub fn new(
        actions: Vec<PolicyActionDefinition>,
        skill_book: &SkillBook,
    ) -> Result<Self, String> {
        let mut by_key = BTreeMap::new();
        let mut ids = BTreeSet::new();
        for (index, action) in actions.iter().enumerate() {
            if by_key.insert(action.key.clone(), index).is_some() {
                return Err(format!("duplicate policy action key: {}", action.key));
            }
            if !ids.insert(action.raw_id) {
                return Err(format!("duplicate policy action raw_id: {}", action.raw_id));
            }
            if skill_book.contains(&action.key) || skill_book.contains_game_id(action.raw_id) {
                return Err(format!(
                    "policy action conflicts with game skill: {}",
                    action.key
                ));
            }
            if !matches!(action.kind.as_str(), "gcd" | "ogcd") {
                return Err(format!(
                    "unsupported policy action kind for {}: {}",
                    action.key, action.kind
                ));
            }
        }
        Ok(Self {
            ordered: actions,
            by_key,
        })
    }

    pub fn load(project_root: impl AsRef<Path>, skill_book: &SkillBook) -> Result<Self, String> {
        let path = project_root.as_ref().join("config/policy_actions.yaml");
        let context = path.display().to_string();
        let data = load_mapping(&path)?;
        let actions = require_mapping(&data, "policy_actions", &context)?;
        let mut definitions = Vec::new();
        for (key, value) in actions {
            let key = to_text(key);
            let action = as_mapping(value, &format!("policy action {key}"), &context)?;
            definitions.push(PolicyActionDefinition {
                key,
                raw_id: integer(action, "raw_id", &context)?,
                name: to_text(require(action, "name", &context)?),
                kind: to_text(require(action, "kind", &context)?),
                behavior: to_text(require(action, "behavior", &context)?),
                value: optional_number(Some(action), "value", 1.0, &context)?,
                tags: optional_string_list(action, "tags")?,
            });
        }
        Self::new(definitions, skill_book)
    }

    pub fn actions(&self) -> &[PolicyActionDefinition] {
        &self.ordered
    }

    pub fn get(&self, key: &str) -> Result<&PolicyActionDefinition, String> {
        self.by_key
            .get(key)
            .map(|index| &self.ordered[*index])
            .ok_or_else(|| format!("unknown policy action key: {key}"))
    }

    pub fn contains(&self, key: &str) -> bool {
        self.by_key.contains_key(key)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StateVectorGroupSchema {
    pub group_key: String,
    pub feature_keys_field: String,
    pub context_key: Option<String>,
    pub encoding: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SchemaConfig {
    pub slidecast_window_seconds: f64,
    pub scene_epsilon: f64,
    pub sidecar_contract_version: i32,
    pub scene_context_absolute_mode: String,
    pub scene_context_keys: BTreeMap<String, String>,
    pub window_time_fields: Vec<String>,
    pub targetable_segment_kinds: Vec<String>,
    pub raid_buff_window_source_keys: Vec<String>,
    pub window_extra_fields: BTreeMap<String, Vec<String>>,
    pub canonical_schema_version: i32,
    pub token_key: String,
    pub canonical_top_level_keys: Vec<String>,
    pub state_vector_groups: Vec<StateVectorGroupSchema>,
    pub state_snapshots: Vec<String>,
    pub state_vector_fields: BTreeMap<String, Vec<String>>,
}

impl SchemaConfig {
    pub fn load(project_root: impl AsRef<Path>) -> Result<Self, String> {
        let path = project_root.as_ref().join("config/schema.yaml");
        let context = path.display().to_string();
        let data = load_mapping(&path)?;
        let contracts = require_mapping(&data, "contracts", &context)?;
        let scene = require_mapping(&data, "scene_context", &context)?;
        let output = require_mapping(&data, "canonical_output", &context)?;
        let groups = require_mapping(output, "state_vector_groups", &context)?;
        let mut state_vector_groups = Vec::new();
        for (key, value) in groups {
            let key = to_text(key);
            let group = as_mapping(value, &key, &context)?;
            let encoding = to_text(require(group, "encoding", &context)?);
            if !matches!(encoding.as_str(), "anchored_delta" | "absolute_numeric") {
                return Err(format!("{context}: unsupported state encoding {encoding}"));
            }
            let expected = if key == "skill_state" {
                "absolute_numeric"
            } else {
                "anchored_delta"
            };
            if encoding != expected {
                return Err(format!(
                    "{context}: state group {key} has incompatible encoding {encoding}"
                ));
            }
            state_vector_groups.push(StateVectorGroupSchema {
                group_key: key,
                feature_keys_field: to_text(require(group, "feature_keys_field", &context)?),
                context_key: get(group, "context_key")
                    .filter(|value| !value.is_null())
                    .map(to_text),
                encoding,
            });
        }
        let state_snapshots = string_list(require(output, "state_snapshots", &context)?)?;
        if state_snapshots != ["previous_action_after", "request_state"] {
            return Err(format!("{context}: unsupported state snapshot order"));
        }
        Ok(Self {
            slidecast_window_seconds: number(contracts, "slidecast_window_seconds", &context)?,
            scene_epsilon: number(contracts, "scene_epsilon", &context)?,
            sidecar_contract_version: integer(contracts, "sidecar_contract_version", &context)?,
            scene_context_absolute_mode: to_text(require(
                contracts,
                "scene_context_absolute_mode",
                &context,
            )?),
            scene_context_keys: require_mapping(contracts, "scene_context_keys", &context)?
                .iter()
                .map(|(key, value)| (to_text(key), to_text(value)))
                .collect(),
            window_time_fields: string_list(require(scene, "window_time_fields", &context)?)?,
            targetable_segment_kinds: string_list(require(
                scene,
                "targetable_segment_kinds",
                &context,
            )?)?,
            raid_buff_window_source_keys: string_list(require(
                scene,
                "raid_buff_window_source_keys",
                &context,
            )?)?,
            window_extra_fields: string_list_mapping(require_mapping(
                scene,
                "window_extra_fields",
                &context,
            )?)?,
            canonical_schema_version: integer(output, "schema_version", &context)?,
            token_key: to_text(require(output, "token_key", &context)?),
            canonical_top_level_keys: string_list(require(output, "top_level_keys", &context)?)?,
            state_vector_groups,
            state_snapshots,
            state_vector_fields: string_list_mapping(require_mapping(
                &data,
                "state_vector_fields",
                &context,
            )?)?,
        })
    }

    pub fn require_state_vector_fields(&self, group: &str) -> Result<&[String], String> {
        self.state_vector_fields
            .get(group)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("schema.yaml 缺少 state_vector_fields.{group}"))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TensorDtype {
    Int32,
    Int64,
    Float32,
    Float16,
    Bfloat16,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PrecisionConfig {
    pub int_dtype: TensorDtype,
    pub float_dtype: TensorDtype,
}

impl PrecisionConfig {
    pub fn load(project_root: impl AsRef<Path>) -> Result<Self, String> {
        let path = project_root.as_ref().join("config/precision.yaml");
        let context = path.display().to_string();
        let data = load_mapping(&path)?;
        let precision = optional_mapping(&data, "precision", &context)?;
        let int_dtype = match precision
            .and_then(|data| get(data, "int_dtype"))
            .map_or_else(|| "int32".to_owned(), to_text)
            .to_lowercase()
            .as_str()
        {
            "int32" => TensorDtype::Int32,
            "int64" => TensorDtype::Int64,
            value => return Err(format!("precision.yaml: invalid int_dtype {value}")),
        };
        let float_dtype = match precision
            .and_then(|data| get(data, "float_dtype"))
            .map_or_else(|| "float32".to_owned(), to_text)
            .to_lowercase()
            .as_str()
        {
            "float32" | "fp32" => TensorDtype::Float32,
            "float16" | "fp16" => TensorDtype::Float16,
            "bfloat16" | "bf16" => TensorDtype::Bfloat16,
            value => return Err(format!("precision.yaml: invalid float_dtype {value}")),
        };
        Ok(Self {
            int_dtype,
            float_dtype,
        })
    }
}

fn get<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Value> {
    mapping.get(Value::String(key.to_owned()))
}

fn require<'a>(mapping: &'a Mapping, key: &str, context: &str) -> Result<&'a Value, String> {
    get(mapping, key).ok_or_else(|| format!("{context}: missing key {key}"))
}

fn as_mapping<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Mapping, String> {
    value
        .as_mapping()
        .ok_or_else(|| format!("{key} must be a mapping: {context}"))
}

fn require_mapping<'a>(
    mapping: &'a Mapping,
    key: &str,
    context: &str,
) -> Result<&'a Mapping, String> {
    as_mapping(require(mapping, key, context)?, key, context)
}

fn optional_mapping<'a>(
    mapping: &'a Mapping,
    key: &str,
    context: &str,
) -> Result<Option<&'a Mapping>, String> {
    get(mapping, key)
        .filter(|value| !value.is_null())
        .map(|value| as_mapping(value, key, context))
        .transpose()
}

fn to_integer(value: &Value, context: &str) -> Result<i32, String> {
    let converted = match value {
        Value::Bool(value) => Some(i64::from(*value)),
        Value::String(value) => value.trim().parse::<i64>().ok(),
        Value::Number(value) => value.as_i64().or_else(|| {
            value.as_f64().and_then(|value| {
                (value.is_finite()
                    && value.trunc() >= f64::from(i32::MIN)
                    && value.trunc() <= f64::from(i32::MAX))
                .then_some(value.trunc() as i64)
            })
        }),
        _ => None,
    };
    converted
        .and_then(|value| i32::try_from(value).ok())
        .ok_or_else(|| format!("{context}: invalid integer value {}", to_text(value)))
}

fn to_number(value: &Value, context: &str) -> Result<f64, String> {
    let converted = match value {
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::String(value) => value.trim().parse().ok(),
        Value::Number(value) => value.as_f64(),
        _ => None,
    };
    converted.ok_or_else(|| format!("{context}: invalid float value {}", to_text(value)))
}

fn to_bool(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64() != Some(0.0),
        Value::String(value) => !value.is_empty(),
        _ => true,
    }
}

fn to_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        _ => serde_yaml_ng::to_string(value)
            .unwrap_or_default()
            .trim()
            .to_owned(),
    }
}

fn integer(mapping: &Mapping, key: &str, context: &str) -> Result<i32, String> {
    to_integer(require(mapping, key, context)?, context)
}

fn optional_integer(
    mapping: &Mapping,
    key: &str,
    default: i32,
    context: &str,
) -> Result<i32, String> {
    get(mapping, key).map_or(Ok(default), |value| to_integer(value, context))
}

fn number(mapping: &Mapping, key: &str, context: &str) -> Result<f64, String> {
    to_number(require(mapping, key, context)?, context)
}

fn optional_number(
    mapping: Option<&Mapping>,
    key: &str,
    default: f64,
    context: &str,
) -> Result<f64, String> {
    mapping
        .and_then(|mapping| get(mapping, key))
        .map_or(Ok(default), |value| to_number(value, context))
}

fn string_list(value: &Value) -> Result<Vec<String>, String> {
    match value {
        Value::Null => Ok(Vec::new()),
        Value::Sequence(items) => Ok(items.iter().map(to_text).collect()),
        _ => Err(format!("expected a sequence, got {}", to_text(value))),
    }
}

fn optional_string_list(mapping: &Mapping, key: &str) -> Result<Vec<String>, String> {
    get(mapping, key).map_or_else(|| Ok(Vec::new()), string_list)
}

fn string_list_mapping(mapping: &Mapping) -> Result<BTreeMap<String, Vec<String>>, String> {
    mapping
        .iter()
        .map(|(key, value)| Ok((to_text(key), string_list(value)?)))
        .collect()
}

fn resolve_project_path(value: &str, root: &Path) -> Result<PathBuf, String> {
    absolute_path(&root.join(value))
}

fn absolute_path(path: &Path) -> Result<PathBuf, String> {
    let path = std::path::absolute(path)
        .map_err(|error| format!("invalid config path {}: {error}", path.display()))?;
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            component => normalized.push(component.as_os_str()),
        }
    }
    Ok(normalized)
}

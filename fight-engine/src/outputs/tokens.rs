// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// 附加许可：FightEngine 链接例外 1.1 版，见 LICENSE-FightEngine-Linking-Exception。

//! 技能 token 与按 schema 排列的两段冻结状态向量。

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde_json::{Map, Value, json};

use crate::config::SchemaConfig;
use crate::contracts::JobStateMachine;
use crate::state::resources::ResourceSnapshot;
use crate::state::{ActionHistoryEntry, ModelStateSnapshot, PlayerContext, StateContext};
use crate::system::SystemRules;

pub struct SkillTokenInput<'a> {
    pub skill_id: i32,
    pub skill_key: &'a str,
    pub skill_name: &'a str,
    pub potency: f64,
    pub value: f64,
    pub kind: &'a str,
    pub actual_mp_cost: i32,
    pub cast_time_seconds: f64,
    pub gcd_window_seconds: f64,
    pub is_legal: bool,
    pub invalid_reason: &'a str,
    pub next_cooldown_seconds: f64,
    pub available_charges: i32,
    pub max_charges: i32,
    pub job_resources_consumed: &'a ResourceSnapshot,
}

pub fn build_skill_token(input: SkillTokenInput<'_>) -> Result<Value, String> {
    let kind = match input.kind {
        "gcd" => 1_i32,
        "ogcd" => 0_i32,
        kind => return Err(format!("unsupported skill kind: {kind}")),
    };
    Ok(json!({
        "skill_id":input.skill_id,"skill_key":input.skill_key,"skill_name":input.skill_name,
        "potency":input.potency,"value":input.value,"kind":kind,"actual_mp_cost":input.actual_mp_cost,
        "cast_time":{"seconds":input.cast_time_seconds},"gcd_window":{"seconds":input.gcd_window_seconds},
        "is_legal":input.is_legal,"invalid_reason":input.invalid_reason,
        "next_cooldown_seconds":input.next_cooldown_seconds,
        "available_charges":input.available_charges,"max_charges":input.max_charges,
        "job_resources_consumed":input.job_resources_consumed,
    }))
}

pub(crate) fn history_skill_token(entry: &ActionHistoryEntry) -> Result<Value, String> {
    build_skill_token(SkillTokenInput {
        skill_id: entry.skill_id,
        skill_key: &entry.skill_key,
        skill_name: &entry.skill_name,
        potency: entry.potency,
        value: entry.value,
        kind: &entry.skill_kind,
        actual_mp_cost: (entry.mp_before - entry.mp_after).max(0),
        cast_time_seconds: entry.cast_time_seconds,
        gcd_window_seconds: entry.gcd_window_seconds,
        is_legal: entry.is_legal,
        invalid_reason: &entry.invalid_reason,
        next_cooldown_seconds: entry.next_cooldown_seconds,
        available_charges: entry.available_charges,
        max_charges: entry.max_charges,
        job_resources_consumed: &entry.job_resources_consumed,
    })
}

pub(crate) struct StateTokens {
    schema: Arc<SchemaConfig>,
    statuses: Vec<(String, String)>,
    buff_resources: Vec<String>,
    resources: Vec<String>,
    dots: Vec<String>,
}

impl StateTokens {
    pub fn new(
        schema: Arc<SchemaConfig>,
        system: &SystemRules,
        job: &dyn JobStateMachine,
    ) -> Result<Self, String> {
        for group in [
            "player_state",
            "buff_state",
            "target_buff_dot_state",
            "target_buff_fixed",
        ] {
            schema.require_state_vector_fields(group)?;
        }
        let mut statuses = Vec::new();
        for (source, definitions) in [
            ("system", system.status_definitions()),
            ("job", job.runtime().statuses.definitions()),
        ] {
            let mut keys: Vec<_> = definitions.keys().cloned().collect();
            keys.sort();
            statuses.extend(keys.into_iter().map(|key| (source.to_owned(), key)));
        }
        Ok(Self {
            schema,
            statuses,
            buff_resources: job.runtime().resources.vector_keys("buff").to_vec(),
            resources: job.runtime().resources.vector_keys("resource").to_vec(),
            dots: system
                .registered_target_dot_keys()
                .iter()
                .cloned()
                .collect(),
        })
    }

    pub fn metadata(&self, actions: &[String]) -> Result<Map<String, Value>, String> {
        self.schema
            .state_vector_groups
            .iter()
            .map(|group| {
                let base = self.feature_keys(&group.group_key, actions)?;
                let keys: Vec<_> = self
                    .schema
                    .state_snapshots
                    .iter()
                    .flat_map(|snapshot| base.iter().map(move |key| format!("{snapshot}.{key}")))
                    .collect();
                Ok((group.feature_keys_field.clone(), json!(keys)))
            })
            .collect()
    }

    fn feature_keys(&self, group: &str, actions: &[String]) -> Result<Vec<String>, String> {
        Ok(match group {
            "player_state" => self.fields("player_state")?.to_vec(),
            "buff_state" => self
                .statuses
                .iter()
                .flat_map(|(source, key)| {
                    self.schema.state_vector_fields["buff_state"]
                        .iter()
                        .map(move |field| format!("{source}.{key}.{field}"))
                })
                .chain(
                    self.buff_resources
                        .iter()
                        .map(|key| format!("resource.{key}")),
                )
                .collect(),
            "target_buff_state" => self
                .dots
                .iter()
                .flat_map(|key| {
                    self.schema.state_vector_fields["target_buff_dot_state"]
                        .iter()
                        .map(move |field| format!("target.{key}.{field}"))
                })
                .chain(self.fields("target_buff_fixed")?.iter().cloned())
                .collect(),
            "resource_state" => self.resources.clone(),
            "skill_availability" => actions.to_vec(),
            group => return Err(format!("unsupported state group: {group}")),
        })
    }

    fn fields(&self, group: &str) -> Result<&[String], String> {
        self.schema.require_state_vector_fields(group)
    }

    pub fn build(
        &self,
        snapshot: &ModelStateSnapshot,
        actions: &[String],
    ) -> Result<Value, String> {
        let frames = [&snapshot.previous_action_after, &snapshot.request_state];
        if actions.iter().collect::<BTreeSet<_>>().len() != actions.len()
            || frames.iter().any(|frame| {
                frame.skill_availability.len() != actions.len()
                    || actions
                        .iter()
                        .any(|key| !frame.skill_availability.contains_key(key))
            })
        {
            return Err("frozen skill availability must match the complete action layout".into());
        }
        self.schema
            .state_vector_groups
            .iter()
            .map(|group| {
                let mut vector = Vec::new();
                for frame in frames {
                    vector.extend(match group.group_key.as_str() {
                        "player_state" => self
                            .fields("player_state")?
                            .iter()
                            .map(|field| player_value(&frame.state.player, field))
                            .collect::<Result<Vec<_>, _>>()?,
                        "buff_state" => self.buff_vector(&frame.state)?,
                        "target_buff_state" => self.target_vector(&frame.state)?,
                        "resource_state" => {
                            resource_vector(&frame.state.resources, &self.resources)?
                        }
                        "skill_availability" => actions
                            .iter()
                            .map(|key| f64::from(frame.skill_availability[key]))
                            .collect(),
                        group => return Err(format!("unsupported state group: {group}")),
                    });
                }
                Ok((group.group_key.clone(), json!(vector)))
            })
            .collect::<Result<Map<_, _>, String>>()
            .map(Value::Object)
    }

    fn buff_vector(&self, state: &StateContext) -> Result<Vec<f64>, String> {
        let mut active = BTreeMap::new();
        for buff in &state.buffs {
            if active
                .insert((&buff.source, &buff.status_key), buff)
                .is_some()
            {
                return Err("state context has duplicate buffs".into());
            }
        }
        let mut vector = Vec::new();
        for (source, key) in &self.statuses {
            let buff = active.get(&(source, key));
            for field in self.fields("buff_state")? {
                vector.push(match field.as_str() {
                    "active" => f64::from(buff.is_some()),
                    "remaining_seconds" => buff.map_or(0.0, |buff| buff.remaining_seconds),
                    "stacks" => buff.map_or(0.0, |buff| f64::from(buff.stacks)),
                    field => return Err(format!("schema.yaml 中未知的 Buff 状态字段: {field}")),
                });
            }
        }
        vector.extend(resource_vector(&state.resources, &self.buff_resources)?);
        Ok(vector)
    }

    fn target_vector(&self, state: &StateContext) -> Result<Vec<f64>, String> {
        let mut active = BTreeMap::new();
        for dot in &state.dots {
            if active.insert(&dot.dot_key, dot).is_some() {
                return Err("state context has duplicate DoTs".into());
            }
        }
        let mut vector = Vec::new();
        for key in &self.dots {
            let dot = active.get(key);
            for field in self.fields("target_buff_dot_state")? {
                vector.push(match field.as_str() {
                    "active" | "stacks" => f64::from(dot.is_some()),
                    "remaining_seconds" => dot.map_or(0.0, |dot| dot.remaining_seconds),
                    field => return Err(format!("schema.yaml 中未知的目标 DoT 字段: {field}")),
                });
            }
        }
        for field in self.fields("target_buff_fixed")? {
            vector.push(match field.as_str() {
                "target.current_potency" => state.target.current_potency,
                "target.current_gcd_dot_potency" => state.target.current_gcd_dot_potency,
                field => return Err(format!("schema.yaml 中未知的目标威力字段: {field}")),
            });
        }
        Ok(vector)
    }
}

fn resource_vector(resources: &ResourceSnapshot, keys: &[String]) -> Result<Vec<f64>, String> {
    keys.iter()
        .map(|key| {
            resources
                .get(key)
                .map(|value| value.as_float())
                .ok_or_else(|| format!("resource snapshot is missing {key}"))
        })
        .collect()
}

fn player_value(player: &PlayerContext, field: &str) -> Result<f64, String> {
    Ok(match field {
        "mp" => f64::from(player.mp),
        "max_mp" => f64::from(player.max_mp),
        "mp_ratio" => player.mp_ratio,
        "time_seconds" => player.time_seconds,
        "current_gcd_seconds" => player.current_gcd_seconds,
        "boss_targetable" => f64::from(player.boss_targetable),
        "next_untargetable_in_seconds" => player.next_untargetable_in_seconds.unwrap_or(0.0),
        "downtime_remaining_seconds" => player.downtime_remaining_seconds,
        "is_moving" => f64::from(player.is_moving),
        field => return Err(format!("schema.yaml 中未知的玩家状态字段: {field}")),
    })
}

pub fn build_death_token(timestamp: f64, after_history_length: u64) -> Value {
    json!({"token":"<dead/>","timestamp":timestamp,"after_history_length":after_history_length})
}

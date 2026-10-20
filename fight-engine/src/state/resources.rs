// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

//! 职业无关的资源类型、上限、快照和消耗统计；时间投影由职业运行层负责。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::config::{JobResourceDefinition, JobStateRegistration, ResourceValue};
use crate::state::CombatState;

pub type ResourceSnapshot = BTreeMap<String, ResourceValue>;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResourceTransition {
    pub before: ResourceSnapshot,
    pub after: ResourceSnapshot,
    pub consumed: ResourceSnapshot,
}

/// 规则只读共享；每个队列的实际资源值保存在自己的 CombatState 中。
#[derive(Clone, Debug)]
pub struct ResourceRegistry {
    job_tag: String,
    definitions: BTreeMap<String, JobResourceDefinition>,
    snapshot_keys: Vec<String>,
    vector_keys: BTreeMap<String, Vec<String>>,
}

impl ResourceRegistry {
    pub fn new(
        registration: &JobStateRegistration,
        resource_limits: &BTreeMap<String, f64>,
    ) -> Result<Self, String> {
        let unknown: Vec<_> = resource_limits
            .keys()
            .filter(|key| !registration.resources.contains_key(*key))
            .map(String::as_str)
            .collect();
        if !unknown.is_empty() {
            return Err(format!(
                "{} resources contain unknown definitions: {}",
                registration.job_tag,
                unknown.join(", ")
            ));
        }
        let mut definitions = registration.resources.clone();
        for (key, limit) in resource_limits {
            if let Some(definition) = definitions.get_mut(key) {
                definition.max_value = Some(*limit);
            }
        }
        let mut snapshot_keys = Vec::new();
        let mut vector_keys: BTreeMap<String, Vec<String>> = BTreeMap::new();
        // 快照是对象；使用字典序稳定导出。模型向量与原规则相同，也按键排序。
        for (key, definition) in &definitions {
            if definition.vector_group != "internal" {
                snapshot_keys.push(key.clone());
                vector_keys
                    .entry(definition.vector_group.clone())
                    .or_default()
                    .push(key.clone());
            }
        }
        Ok(Self {
            job_tag: registration.job_tag.clone(),
            definitions,
            snapshot_keys,
            vector_keys,
        })
    }

    pub fn definitions(&self) -> &BTreeMap<String, JobResourceDefinition> {
        &self.definitions
    }

    pub fn job_tag(&self) -> &str {
        &self.job_tag
    }

    pub fn snapshot_keys(&self) -> &[String] {
        &self.snapshot_keys
    }

    pub fn vector_keys(&self, group: &str) -> &[String] {
        self.vector_keys.get(group).map_or(&[], Vec::as_slice)
    }

    pub fn max(&self, key: &str) -> Result<Option<f64>, String> {
        Ok(self.definition(key)?.max_value)
    }

    pub fn initialize(&self, state: &mut CombatState) -> Result<(), String> {
        let resources = self
            .definitions
            .iter()
            .map(|(key, definition)| {
                Ok((
                    key.clone(),
                    normalize_value(definition, definition.default_value)?,
                ))
            })
            .collect::<Result<_, String>>()?;
        state.job.resources = resources;
        Ok(())
    }

    pub fn get(&self, state: &CombatState, key: &str) -> Result<ResourceValue, String> {
        let definition = self.definition(key)?;
        let value = state
            .job
            .resources
            .get(key)
            .copied()
            .unwrap_or(definition.default_value);
        normalize_value(definition, value)
    }

    pub fn set(
        &self,
        state: &mut CombatState,
        key: &str,
        value: ResourceValue,
    ) -> Result<(), String> {
        let value = self.normalize(key, value)?;
        state.job.resources.insert(key.to_owned(), value);
        Ok(())
    }

    /// 统一外部计算结果的类型与上限，保留 F64 精度。
    pub fn normalize(&self, key: &str, value: ResourceValue) -> Result<ResourceValue, String> {
        normalize_value(self.definition(key)?, value)
    }

    pub fn snapshot(&self, state: &CombatState) -> Result<ResourceSnapshot, String> {
        self.snapshot_keys
            .iter()
            .map(|key| Ok((key.clone(), self.get(state, key)?)))
            .collect()
    }

    /// 接受已应用时间视图投影的快照，避免资源层另存计时或重复构建快照。
    pub fn transition(
        &self,
        before: ResourceSnapshot,
        after: ResourceSnapshot,
    ) -> Result<ResourceTransition, String> {
        let mut consumed = BTreeMap::new();
        for key in &self.snapshot_keys {
            let previous = before
                .get(key)
                .copied()
                .ok_or_else(|| format!("resource snapshot is missing {key}"))?;
            let next = after
                .get(key)
                .copied()
                .ok_or_else(|| format!("resource snapshot is missing {key}"))?;
            let value = match self.definitions[key].resource_type.as_str() {
                "int" => ResourceValue::Integer(
                    as_integer(previous).wrapping_sub(as_integer(next)).max(0),
                ),
                "float" => {
                    let difference = as_float(previous) - as_float(next);
                    ResourceValue::Float(if difference < 0.0 { 0.0 } else { difference })
                }
                "bool" => ResourceValue::Bool(as_bool(previous) && !as_bool(next)),
                value => return Err(format!("unsupported job resource type: {value}")),
            };
            consumed.insert(key.clone(), value);
        }
        Ok(ResourceTransition {
            before,
            after,
            consumed,
        })
    }

    fn definition(&self, key: &str) -> Result<&JobResourceDefinition, String> {
        self.definitions
            .get(key)
            .ok_or_else(|| format!("unregistered job resource: {key}"))
    }
}

fn normalize_value(
    definition: &JobResourceDefinition,
    value: ResourceValue,
) -> Result<ResourceValue, String> {
    match definition.resource_type.as_str() {
        "int" => {
            let mut value = as_integer(value);
            if let Some(maximum) = definition.max_value {
                value = value.min(maximum as i32).max(0);
            }
            Ok(ResourceValue::Integer(value))
        }
        "float" => {
            let mut value = as_float(value);
            if let Some(maximum) = definition.max_value {
                // 比较式保留原有 NaN 传播语义，避免 min/max 把它替换成普通数值。
                if value > maximum || maximum.is_nan() {
                    value = maximum;
                }
                if value < 0.0 {
                    value = 0.0;
                }
            }
            Ok(ResourceValue::Float(value))
        }
        "bool" => Ok(ResourceValue::Bool(as_bool(value))),
        value => Err(format!("unsupported job resource type: {value}")),
    }
}

fn as_integer(value: ResourceValue) -> i32 {
    match value {
        ResourceValue::Integer(value) => value,
        ResourceValue::Float(value) => value as i32,
        ResourceValue::Bool(value) => i32::from(value),
    }
}

fn as_float(value: ResourceValue) -> f64 {
    match value {
        ResourceValue::Integer(value) => f64::from(value),
        ResourceValue::Float(value) => value,
        ResourceValue::Bool(value) => f64::from(u8::from(value)),
    }
}

fn as_bool(value: ResourceValue) -> bool {
    match value {
        ResourceValue::Integer(value) => value != 0,
        ResourceValue::Float(value) => value != 0.0,
        ResourceValue::Bool(value) => value,
    }
}

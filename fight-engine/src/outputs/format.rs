// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

//! 秒制、GCD 制、场景 schema 和模型载荷的末端格式化。

use std::sync::Arc;

use serde_json::{Map, Number, Value, json};

use super::{ModelHistoryRow, OutputContextBuilder};
use crate::config::{ProjectConfig, SchemaConfig};
use crate::contracts::JobStateMachine;
use crate::state::{CombatState, ModelStateSnapshot};
use crate::system::SystemRules;

pub struct OutputFormatter {
    context: OutputContextBuilder,
}

impl OutputFormatter {
    pub fn new(
        project: Arc<ProjectConfig>,
        system: Arc<SystemRules>,
        job: Arc<dyn JobStateMachine>,
        schema: Arc<SchemaConfig>,
        history_limit: Option<usize>,
    ) -> Result<Self, String> {
        Ok(Self {
            context: OutputContextBuilder::new(project, system, job, schema, history_limit)?,
        })
    }

    pub fn context_builder(&self) -> &OutputContextBuilder {
        &self.context
    }

    pub fn format(&self, state: &CombatState, mode: &str) -> Result<Value, String> {
        let context = self.context.build_state_context(state)?;
        let player = &context.player;
        let gcd_units = |seconds: f64| {
            if player.current_gcd_seconds <= 0.0 {
                0.0
            } else {
                seconds / player.current_gcd_seconds
            }
        };
        let mut payload = match mode {
            "seconds" => json!({
                "mode":mode,"time_seconds":player.time_seconds,"gcd_index":player.gcd_index,
                "current_gcd_seconds":player.current_gcd_seconds,"fight_remaining_seconds":player.fight_remaining_seconds,
                "gcd_remaining_seconds":player.gcd_remaining_seconds,"cast_remaining_seconds":player.cast_remaining_seconds,
                "weave_window_seconds":player.weave_window_seconds,"next_downtime_eta_seconds":player.next_untargetable_in_seconds,
                "downtime_remaining_seconds":player.downtime_remaining_seconds,"boss_targetable":player.boss_targetable,
                "job_tag":context.job_tag,"job_name":context.job_name,"mp":player.mp,"max_mp":player.max_mp,
                "mp_ratio":player.mp_ratio,"execution_metrics":{
                    "cumulative_potency":context.target.cumulative_potency,
                    "cumulative_dot_potency":context.target.cumulative_dot_potency,
                },
            }),
            "gcd" => json!({
                "mode":mode,"gcd_index":player.gcd_index,"current_gcd_seconds":player.current_gcd_seconds,
                "fight_remaining_gcd":gcd_units(player.fight_remaining_seconds),
                "gcd_remaining_gcd":gcd_units(player.gcd_remaining_seconds),
                "weave_window_gcd":gcd_units(player.weave_window_seconds),
                "next_downtime_eta_gcd":player.next_untargetable_in_seconds.map(gcd_units),
                "downtime_remaining_gcd":gcd_units(player.downtime_remaining_seconds),
                "boss_targetable":player.boss_targetable,"job_tag":context.job_tag,"job_name":context.job_name,
                "mp":player.mp,"max_mp":player.max_mp,"mp_ratio":player.mp_ratio,
            }),
            mode => {
                return Err(format!(
                    "unsupported output mode: {mode}; supported=gcd, seconds"
                ));
            }
        };
        let fields = payload.as_object_mut().expect("状态载荷始终是对象");
        for (key, value) in context.resources {
            let definition = &self.context.job.runtime().resources.definitions()[&key];
            if mode == "gcd" && definition.display_unit == "seconds" {
                fields.insert(format!("{key}_gcd"), json!(gcd_units(value.as_float())));
            } else {
                fields.insert(key, json!(value));
            }
        }
        Ok(payload)
    }

    pub fn format_vectors(
        &self,
        state: &CombatState,
        action_keys: &[String],
        action_values: &[f64],
        current: &ModelStateSnapshot,
        history: Option<&[Arc<ModelHistoryRow>]>,
    ) -> Result<Value, String> {
        self.context
            .build_context(state, action_keys, action_values, current, history)
    }

    pub fn format_tensors(
        &self,
        state: &CombatState,
        action_keys: &[String],
        action_values: &[f64],
        current: &ModelStateSnapshot,
        history: Option<&[Arc<ModelHistoryRow>]>,
    ) -> Result<Value, String> {
        pack_tensor_payload(&self.format_vectors(
            state,
            action_keys,
            action_values,
            current,
            history,
        )?)
    }
}

pub fn scene_context(schema: &SchemaConfig) -> Result<Value, String> {
    let mut context = Map::new();
    for window in ["targetable", "forced_movement", "raid_buff", "target_count"] {
        let key = schema
            .scene_context_keys
            .get(window)
            .ok_or_else(|| format!("schema.yaml 缺少 scene_context_keys.{window}"))?;
        let extra = schema.window_extra_fields.get(window).ok_or_else(|| {
            format!("schema.yaml 缺少 scene_context.window_extra_fields.{window}")
        })?;
        let mut features = schema.window_time_fields.clone();
        if window == "raid_buff" && schema.raid_buff_window_source_keys.is_empty() {
            return Err("raid buff window source keys must not be empty".into());
        }
        for field in extra {
            if window == "raid_buff" && field.contains("{key}") {
                features.extend(
                    schema
                        .raid_buff_window_source_keys
                        .iter()
                        .map(|key| field.replace("{key}", key)),
                );
            } else {
                features.push(field.clone());
            }
        }
        if window == "targetable" {
            features.extend(
                schema
                    .targetable_segment_kinds
                    .iter()
                    .map(|kind| format!("segment_kind.{kind}")),
            );
        }
        context.insert(key.clone(), json!({"feature_keys":features,"tokens":[]}));
    }
    Ok(Value::Object(context))
}

pub fn output_schema_metadata(output: &Value, schema: &SchemaConfig) -> Result<Value, String> {
    let keys = &schema.canonical_top_level_keys;
    let key = |index| {
        keys.get(index)
            .ok_or_else(|| "canonical schema has incomplete top-level keys".to_owned())
    };
    let history = output
        .get(key(4)?)
        .ok_or_else(|| "state history context is missing".to_owned())?;
    let current = output
        .get(key(5)?)
        .ok_or_else(|| "current state context is missing".to_owned())?;
    let feature_keys = |context: &Value| {
        schema
            .state_vector_groups
            .iter()
            .map(|group| {
                let value = context
                    .get(&group.feature_keys_field)
                    .and_then(Value::as_array)
                    .filter(|keys| keys.iter().all(Value::is_string))
                    .ok_or_else(|| {
                        format!("{} must be a list of strings", group.feature_keys_field)
                    })?;
                Ok((group.group_key.clone(), json!(value)))
            })
            .collect::<Result<Map<_, _>, String>>()
    };
    let schema_version = output
        .get("schema_version")
        .and_then(Value::as_i64)
        .ok_or_else(|| "schema_version must be an int".to_owned())?;
    Ok(json!({
        "schema_version":schema_version,"top_level_keys":keys,
        "scene_context_key":key(2)?,"skill_history_context_key":key(3)?,
        "state_history_context_key":key(4)?,"current_state_context_key":key(5)?,
        "action_keys_key":key(6)?,"action_legal_mask_key":key(7)?,"action_values_key":key(8)?,
        "state_context_token_key":schema.token_key,
        "state_vector_group_keys":schema.state_vector_groups.iter().map(|group| &group.group_key).collect::<Vec<_>>(),
        "state_vector_feature_key_fields":schema.state_vector_groups.iter().map(|group| (group.group_key.clone(),json!(group.feature_keys_field))).collect::<Map<_,_>>(),
        "state_history_feature_keys":feature_keys(history)?,"current_state_feature_keys":feature_keys(current)?,
    }))
}

#[derive(Default)]
struct NumericLayout {
    shape: Vec<usize>,
    has_null: bool,
    has_float: bool,
    all_bool: bool,
}

fn numeric_layout(value: &Value) -> Option<NumericLayout> {
    match value {
        Value::Array(values) if !values.is_empty() => {
            let children = values
                .iter()
                .map(numeric_layout)
                .collect::<Option<Vec<_>>>()?;
            let shape = &children[0].shape;
            if children.iter().any(|child| child.shape != *shape) {
                return None;
            }
            Some(NumericLayout {
                shape: std::iter::once(values.len())
                    .chain(shape.iter().copied())
                    .collect(),
                has_null: children.iter().any(|child| child.has_null),
                has_float: children.iter().any(|child| child.has_float),
                all_bool: children.iter().all(|child| child.all_bool),
            })
        }
        Value::Null => Some(NumericLayout {
            has_null: true,
            ..NumericLayout::default()
        }),
        Value::Bool(_) => Some(NumericLayout {
            all_bool: true,
            ..NumericLayout::default()
        }),
        Value::Number(number) => Some(NumericLayout {
            has_float: number.is_f64(),
            ..NumericLayout::default()
        }),
        _ => None,
    }
}

fn map_nested(
    value: &Value,
    convert: &impl Fn(&Value) -> Result<Value, String>,
) -> Result<Value, String> {
    if let Value::Array(values) = value {
        values
            .iter()
            .map(|value| map_nested(value, convert))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array)
    } else {
        convert(value)
    }
}

/// 纯数值矩阵保持形状；浮点/null 附缺失掩码，同键对象列表转为列式结构。
/// 仅此末端执行 FP32 舍入；模型输入的组装和所有引擎计算保持 F64。
pub fn pack_tensor_payload(payload: &Value) -> Result<Value, String> {
    match payload {
        Value::Object(object) => object
            .iter()
            .map(|(key, value)| Ok((key.clone(), pack_tensor_payload(value)?)))
            .collect::<Result<Map<_, _>, String>>()
            .map(Value::Object),
        Value::Array(values) => {
            if let Some(layout) = numeric_layout(payload) {
                if layout.has_null || layout.has_float {
                    let values = map_nested(payload, &|value| {
                        let raw = match value {
                            Value::Null => 0.0,
                            Value::Bool(value) => f64::from(*value),
                            Value::Number(number) => number
                                .as_f64()
                                .ok_or_else(|| "invalid tensor number".to_owned())?,
                            _ => unreachable!("数值布局只包含数值、布尔和 null"),
                        };
                        Number::from_f64(f64::from(raw as f32))
                            .map(Value::Number)
                            .ok_or_else(|| "tensor FP32 value is not finite".to_owned())
                    })?;
                    return Ok(
                        json!({"values":values,"is_null":map_nested(payload,&|value| Ok(json!(value.is_null())))?}),
                    );
                }
                if layout.all_bool {
                    return Ok(payload.clone());
                }
                return map_nested(payload, &|value| {
                    Ok(match value {
                        Value::Bool(value) => json!(i32::from(*value)),
                        value => value.clone(),
                    })
                });
            }
            if let Some(first) = values.first().and_then(Value::as_object)
                && values.iter().all(|value| {
                    value
                        .as_object()
                        .is_some_and(|object| object.keys().eq(first.keys()))
                })
            {
                return first
                    .keys()
                    .map(|key| {
                        let column =
                            Value::Array(values.iter().map(|value| value[key].clone()).collect());
                        Ok((key.clone(), pack_tensor_payload(&column)?))
                    })
                    .collect::<Result<Map<_, _>, String>>()
                    .map(Value::Object);
            }
            values
                .iter()
                .map(pack_tensor_payload)
                .collect::<Result<Vec<_>, _>>()
                .map(Value::Array)
        }
        value => Ok(value.clone()),
    }
}

// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

use serde_json::{Value, json};

/// 新技能列单独验收，旧基线移除 MP 比例列；继续比较所有未变字段，不重写历史 fixture。
#[allow(dead_code)] // 各集成测试独立编译此共享模块，仅输出测试使用该投影。
pub fn legacy_model_state(mut output: Value) -> Value {
    fn column(values: Value) -> Value {
        let values = values.as_array().expect("技能状态必须是向量");
        assert_eq!(values.len() % 3, 0);
        json!(
            values
                .as_chunks::<3>()
                .0
                .iter()
                .map(|fields| {
                    if let Some(key) = fields[0].as_str() {
                        let prefix = key.strip_suffix(".available").unwrap();
                        assert_eq!(fields[1], format!("{prefix}.current_charges"));
                        assert_eq!(fields[2], format!("{prefix}.next_cooldown_seconds"));
                        json!(prefix)
                    } else {
                        let available = fields[0].as_f64().unwrap();
                        let charges = fields[1].as_f64().unwrap();
                        let seconds = fields[2].as_f64().unwrap();
                        assert!(matches!(available, 0.0 | 1.0));
                        assert!(charges.is_finite() && charges >= 0.0 && charges.fract() == 0.0);
                        assert!(seconds.is_finite() && seconds >= 0.0);
                        fields[0].clone()
                    }
                })
                .collect::<Vec<_>>()
        )
    }
    match &mut output {
        Value::Object(object) => {
            if let Some(keys) = object.get("player_state_feature_keys") {
                let removed = keys
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .filter_map(|(index, key)| {
                        key.as_str()
                            .unwrap()
                            .ends_with(".mp_ratio")
                            .then_some(index)
                    })
                    .collect::<Vec<_>>();
                let remove_columns = |values: &mut Value| {
                    let values = values.as_array_mut().unwrap();
                    for &index in removed.iter().rev() {
                        values.remove(index);
                    }
                };
                if let Some(tokens) = object.get_mut("tokens") {
                    if let Some(tokens) = tokens.as_array_mut() {
                        for token in tokens {
                            remove_columns(&mut token["player_state"]);
                        }
                    } else {
                        for field in ["values", "is_null"] {
                            for row in tokens["player_state"][field].as_array_mut().unwrap() {
                                remove_columns(row);
                            }
                        }
                    }
                }
            }
            if let Some(fields) = object.remove("skill_state_feature_keys") {
                object.insert("skill_availability_feature_keys".into(), column(fields));
            }
            if let Some(fields) = object.remove("skill_state") {
                let projected = if fields == "skill_state_feature_keys" {
                    json!("skill_availability_feature_keys")
                } else {
                    column(fields)
                };
                object.insert("skill_availability".into(), projected);
            }
            if object.get("schema_version") == Some(&json!(17))
                && (object.contains_key("action_keys") || object.contains_key("top_level_keys"))
            {
                object.insert("schema_version".into(), json!(16));
            }
            for (key, value) in object.iter_mut() {
                if key == "state_vector_group_keys" {
                    for group in value.as_array_mut().unwrap() {
                        if group == "skill_state" {
                            *group = json!("skill_availability");
                        }
                    }
                } else {
                    *value = legacy_model_state(std::mem::take(value));
                }
            }
        }
        Value::Array(values) => {
            values.retain(|value| {
                !value
                    .as_str()
                    .is_some_and(|key| key == "mp_ratio" || key.ends_with(".mp_ratio"))
            });
            for value in values {
                *value = legacy_model_state(std::mem::take(value));
            }
        }
        _ => {}
    }
    output
}

/// 离散值和顺序严格比较；浮点保留 FP32 级误差边界并报告字段路径。
pub fn assert_equivalent(expected: &Value, actual: &Value, path: &str) {
    match (expected, actual) {
        (Value::Object(expected), Value::Object(actual)) => {
            assert_eq!(expected.len(), actual.len(), "{path}: 字段数量不同");
            for (key, expected) in expected {
                let next = format!("{path}.{key}");
                let actual = actual
                    .get(key)
                    .unwrap_or_else(|| panic!("{next}: 缺少字段"));
                assert_equivalent(expected, actual, &next);
            }
        }
        (Value::Array(expected), Value::Array(actual)) => {
            assert_eq!(expected.len(), actual.len(), "{path}: 数组长度不同");
            for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                assert_equivalent(expected, actual, &format!("{path}[{index}]"));
            }
        }
        (Value::Number(expected), Value::Number(actual)) => {
            if let (Some(expected), Some(actual)) = (expected.as_i64(), actual.as_i64()) {
                assert_eq!(expected, actual, "{path}: 整数不同");
            } else {
                let expected = expected.as_f64().unwrap();
                let actual = actual.as_f64().unwrap();
                assert!(
                    (expected - actual).abs() <= expected.abs().max(1.0) * 1e-6,
                    "{path}: C#={expected}, Rust={actual}"
                );
            }
        }
        _ => assert_eq!(expected, actual, "{path}: 值或类型不同"),
    }
}

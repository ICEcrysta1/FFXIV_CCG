use serde_json::Value;

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

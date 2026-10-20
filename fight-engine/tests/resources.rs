//! 通用资源规则对照固定基线，浮点输出允许保留原先四位舍入之前的精度。

use std::collections::BTreeMap;

use fight_engine::config::{JobStateRegistration, ResourceValue};
use fight_engine::state::CombatState;
use fight_engine::state::resources::{ResourceRegistry, ResourceSnapshot};
use serde_json::Value;

fn assert_values(expected: &Value, actual: &ResourceSnapshot, tolerance: f64) {
    let expected = expected.as_object().unwrap();
    assert_eq!(expected.len(), actual.len());
    for (key, value) in actual {
        let expected = &expected[key];
        match value {
            ResourceValue::Integer(value) => assert_eq!(expected.as_i64(), Some(i64::from(*value))),
            ResourceValue::Bool(value) => assert_eq!(expected.as_bool(), Some(*value)),
            ResourceValue::Float(value) => assert!(
                (expected.as_f64().unwrap() - value).abs() <= tolerance,
                "{key}: 基线={expected}, Rust={value}"
            ),
        }
    }
}

#[test]
fn resource_types_limits_snapshots_and_consumption_match_baseline() {
    let baseline: Value = serde_json::from_str(include_str!("fixtures/resources.json")).unwrap();
    assert_eq!(
        baseline["source_commit"],
        "837d841ca49d504bbf7de996f90af195fc3cdb7d"
    );
    let registration: JobStateRegistration =
        serde_json::from_value(baseline["registration"].clone()).unwrap();
    let limits: BTreeMap<String, f64> = serde_json::from_value(baseline["limits"].clone()).unwrap();
    let registry = ResourceRegistry::new(&registration, &limits).unwrap();
    assert_eq!(registry.job_tag(), baseline["job_tag"].as_str().unwrap());
    assert_eq!(
        registry.definitions(),
        &serde_json::from_value(baseline["definitions"].clone()).unwrap()
    );
    assert_eq!(
        registry.snapshot_keys(),
        serde_json::from_value::<Vec<String>>(baseline["snapshot_keys"].clone()).unwrap()
    );
    for (group, expected) in baseline["vector_keys"].as_object().unwrap() {
        assert_eq!(
            registry.vector_keys(group),
            serde_json::from_value::<Vec<String>>(expected.clone()).unwrap()
        );
    }
    for (key, expected) in baseline["max"].as_object().unwrap() {
        assert_eq!(registry.max(key).unwrap(), expected.as_f64());
    }
    let mut initial = CombatState::default();
    registry.initialize(&mut initial).unwrap();
    assert_values(&baseline["initial"], &initial.job.resources, 1e-12);

    for scenario in baseline["scenarios"].as_array().unwrap() {
        let mut state = CombatState::default();
        registry.initialize(&mut state).unwrap();
        let mut normalized = BTreeMap::new();
        for input in scenario["sets"].as_array().unwrap() {
            let key = input["key"].as_str().unwrap();
            let value: ResourceValue = serde_json::from_value(input["value"].clone()).unwrap();
            registry.set(&mut state, key, value).unwrap();
            normalized.insert(key.to_owned(), registry.normalize(key, value).unwrap());
        }
        let gets = registry
            .definitions()
            .keys()
            .map(|key| (key.clone(), registry.get(&state, key).unwrap()))
            .collect();
        assert_values(&scenario["gets"], &gets, 1e-12);
        // 旧输出舍入到四位；Rust 直接保留 F64，最多相差半个末位单位。
        assert_values(
            &scenario["snapshot"],
            &registry.snapshot(&state).unwrap(),
            5.0001e-5,
        );
        assert_values(&scenario["normalized"], &normalized, 5.0001e-5);
    }

    let mut before = initial.clone();
    let mut after = initial.clone();
    for (name, state) in [("before", &mut before), ("after", &mut after)] {
        for input in baseline["transition_inputs"][name].as_array().unwrap() {
            let key = input["key"].as_str().unwrap();
            let value: ResourceValue = serde_json::from_value(input["value"].clone()).unwrap();
            registry.set(state, key, value).unwrap();
        }
    }
    let transition = registry
        .transition(
            registry.snapshot(&before).unwrap(),
            registry.snapshot(&after).unwrap(),
        )
        .unwrap();
    assert_values(
        &baseline["transition"]["before"],
        &transition.before,
        5.0001e-5,
    );
    assert_values(
        &baseline["transition"]["after"],
        &transition.after,
        5.0001e-5,
    );
    // 消耗是两份快照的差，各自旧舍入的误差最多累积一个末位单位。
    assert_values(
        &baseline["transition"]["consumed"],
        &transition.consumed,
        1.00001e-4,
    );

    let unknown_limits = BTreeMap::from([("missing".to_owned(), 1.0)]);
    assert_eq!(
        ResourceRegistry::new(&registration, &unknown_limits).is_err(),
        baseline["errors"]["unknown_limit"].as_bool().unwrap()
    );
    assert_eq!(
        registry.get(&initial, "missing").is_err(),
        baseline["errors"]["unknown_resource"].as_bool().unwrap()
    );
    let mut invalid = registration;
    invalid.resources.values_mut().next().unwrap().resource_type = "unsupported".to_owned();
    let invalid_registry = ResourceRegistry::new(&invalid, &BTreeMap::new()).unwrap();
    assert_eq!(
        invalid_registry.initialize(&mut initial).is_err(),
        baseline["errors"]["unsupported_type"].as_bool().unwrap()
    );
}

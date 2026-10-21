// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

//! 固定基线来自 fixture 中 source_commit 指定的 C# 实现；Rust 测试不依赖 .NET。

mod common;

use std::path::{Path, PathBuf};

use fight_engine::config::{
    ActionKind, PolicyActionRegistry, PrecisionConfig, ProjectConfig, ResourceValue, SchemaConfig,
    SkillBook, SystemConfig,
};
use fight_engine::state::{CombatState, CooldownState, DotState, StatusState};
use fight_engine::system::timeline::{Timeline, TimelineMutation};
use serde_json::{Value, json};

use common::assert_equivalent;

fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("Rust 工程应位于仓库根目录下")
        .to_owned()
}

// 整数、布尔、对象键和数组顺序严格比较，浮点允许 FP32 级误差。
#[test]
fn configuration_and_skill_lookup_match_csharp() {
    let baseline: Value = serde_json::from_str(include_str!("fixtures/foundation.json"))
        .expect("C# 基线 fixture 应为有效 JSON");
    assert_eq!(
        baseline["source_commit"],
        "815e71b439cc1820fbaa9aca07a110b4a155d0e9"
    );
    let root = project_root();
    assert_equivalent(
        &baseline["jobs"]["black_mage"]["project"]["system"],
        &serde_json::to_value(SystemConfig::load(root.join("config/system.yaml")).unwrap())
            .unwrap(),
        "independent_system_config",
    );
    let mut schema = serde_json::to_value(SchemaConfig::load(&root).unwrap()).unwrap();
    assert_eq!(schema["sidecar_contract_version"], 20);
    assert_eq!(schema["canonical_schema_version"], 17);
    assert_eq!(
        schema["state_vector_fields"]["skill_state"],
        json!(["available", "current_charges", "next_cooldown_seconds"])
    );
    schema["sidecar_contract_version"] = baseline["schema"]["sidecar_contract_version"].clone();
    schema["canonical_schema_version"] = baseline["schema"]["canonical_schema_version"].clone();
    schema["state_vector_fields"]
        .as_object_mut()
        .unwrap()
        .remove("skill_state");
    for group in schema["state_vector_groups"].as_array_mut().unwrap() {
        if group["group_key"] == "skill_state" {
            assert_eq!(group["encoding"], "absolute_numeric");
            assert_eq!(group["feature_keys_field"], "skill_state_feature_keys");
            group["group_key"] = json!("skill_availability");
            group["encoding"] = json!("absolute_binary");
            group["feature_keys_field"] = json!("skill_availability_feature_keys");
        }
    }
    assert_equivalent(&baseline["schema"], &schema, "schema");
    assert_equivalent(
        &baseline["precision"],
        &serde_json::to_value(PrecisionConfig::load(&root).unwrap()).unwrap(),
        "precision",
    );
    for job in ["black_mage", "machinist"] {
        let config = ProjectConfig::load(&root, job).expect("正式职业配置应加载成功");
        let skills = SkillBook::from_project_config(&config).expect("技能表应构建成功");
        let policy = PolicyActionRegistry::load(&root, &skills).expect("策略动作索引应加载成功");
        let mut project = serde_json::to_value(&config).expect("配置应可序列化");
        // 基线不携带机器目录；同时比较两边从正式配置解析出的路径。
        for key in ["system_config_path", "job_config_path"] {
            let path = Path::new(project["runtime"][key].as_str().unwrap());
            project["runtime"][key] = json!(path.strip_prefix(&root).unwrap());
        }
        project["job"]["default_fight_remaining"] =
            project["job"]["timing"]["default_fight_remaining"].clone();
        let enabled = |kind| {
            skills
                .enabled_skills(kind)
                .into_iter()
                .map(|skill| skill.key.as_str())
                .collect::<Vec<_>>()
        };
        let actual = json!({
            "project": project,
            "keys": skills.keys(),
            "enabled": enabled(None),
            "gcd": enabled(Some(ActionKind::Gcd)),
            "ogcd": enabled(Some(ActionKind::Ogcd)),
            "by_game_id": skills.keys().into_iter().map(|key| {
                let skill = skills.get(key).expect("按 key 查询应命中");
                &skills.by_game_id(skill.game_id).expect("按游戏 id 查询应命中").key
            }).collect::<Vec<_>>(),
            "contains": [skills.contains("missing_skill"), skills.contains_game_id(-1)],
            "policy": policy.actions(),
        });
        assert_equivalent(&baseline["jobs"][job], &actual, job);
    }
}

#[test]
fn resource_deadlines_and_clone_isolation_match_csharp() {
    let baseline: Value = serde_json::from_str(include_str!("fixtures/foundation.json"))
        .expect("C# 基线 fixture 应为有效 JSON");
    let mut state = CombatState::default();
    state
        .statuses
        .insert("probe".into(), StatusState::new(100.0, 12.5, 2));
    state
        .job
        .statuses
        .insert("job_probe".into(), StatusState::new(100.0, 5.0, 1));
    state.job.resources.extend([
        ("heat".into(), ResourceValue::Integer(50)),
        ("active".into(), ResourceValue::Bool(true)),
        ("timer".into(), ResourceValue::Float(12.5)),
    ]);
    state
        .dots
        .insert("dot".into(), DotState::new(100.0, 15.0, 40.0, 3.0, 3.0));
    state.cooldowns.insert(
        "cooldown".into(),
        CooldownState::new(100.0, 1, vec![5.0, 10.0]),
    );
    let mut timeline = Timeline::<()>::new(state).expect("初始时间应有效");
    let mut resources = Vec::new();
    for time in [100.0, 103.0, 120.0] {
        // 这里只推进观测时钟；到期资源的结算留给后续时间线场景。
        let state = timeline
            .advance_to(time, |_| Vec::new(), |_, _| Ok(TimelineMutation::default()))
            .expect("时钟应单调推进");
        let status = &state.statuses["probe"];
        let job_status = &state.job.statuses["job_probe"];
        let dot = &state.dots["dot"];
        let cooldown = &state.cooldowns["cooldown"];
        resources.push(json!({
            "time": state.time(),
            "status_remaining": status.remaining(time),
            "status_expires_at": status.expires_at,
            "status_stacks": state.status_stacks("probe"),
            "has_status": state.has_status("probe"),
            "job_remaining": job_status.remaining(time),
            "job_stacks": state.job.status_stacks(time, "job_probe"),
            "has_job_status": state.job.has_status(time, "job_probe"),
            "dot_remaining": state.dot_remaining("dot"),
            "dot_expires_at": dot.expires_at,
            "dot_next_tick_at": dot.next_tick_at,
            "dot_next_tick_in_seconds": dot.next_tick_in_seconds(time),
            "potency_per_tick": dot.potency_per_tick,
            "tick_interval": dot.tick_interval,
            "available_charges": cooldown.available_charges,
            "recharge_ready_at": cooldown.recharge_ready_at,
            "recharge_timers": cooldown.recharge_timers(time),
        }));
    }
    assert_equivalent(&baseline["resources"], &json!(resources), "resources");

    let state = timeline.state();
    let mut clone = state.clone();
    clone.statuses.get_mut("probe").unwrap().stacks = 99;
    clone.job.statuses.get_mut("job_probe").unwrap().stacks = 99;
    clone
        .job
        .resources
        .insert("heat".into(), ResourceValue::Integer(99));
    clone.dots.get_mut("dot").unwrap().potency_per_tick = 99.0;
    clone
        .cooldowns
        .get_mut("cooldown")
        .unwrap()
        .recharge_ready_at
        .push(999.0);
    let isolated = state.statuses["probe"].stacks == 2
        && state.job.statuses["job_probe"].stacks == 1
        && state.job.resources["heat"] == ResourceValue::Integer(50)
        && state.dots["dot"].potency_per_tick == 40.0
        && state.cooldowns["cooldown"].recharge_ready_at.len() == 2;
    assert_eq!(baseline["clone_isolated"], json!(isolated));
}

#[test]
fn yaml_scalars_aliases_and_documents_match_csharp() {
    let baseline: Value = serde_json::from_str(include_str!("fixtures/yaml.json")).unwrap();
    let root = project_root();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let scratch =
        std::env::temp_dir().join(format!("fight-engine-yaml-{}-{unique}", std::process::id()));
    let originals: Vec<_> = [
        "config/default.yaml",
        "config/system.yaml",
        "config/skills/black_mage.yaml",
        "config/skills/machinist.yaml",
    ]
    .into_iter()
    .map(|path| (path, std::fs::read_to_string(root.join(path)).unwrap()))
    .collect();
    for case in baseline["cases"].as_array().unwrap() {
        for (path, original) in &originals {
            let destination = scratch.join(path);
            std::fs::create_dir_all(destination.parent().unwrap()).unwrap();
            let contents = if *path == case["path"].as_str().unwrap() {
                format!(
                    "{}{}{}",
                    case["prefix"].as_str().unwrap(),
                    original.replace(
                        case["search"].as_str().unwrap(),
                        case["replacement"].as_str().unwrap()
                    ),
                    case["suffix"].as_str().unwrap()
                )
            } else {
                original.clone()
            };
            std::fs::write(destination, contents).unwrap();
        }
        let actual = match ProjectConfig::load(&scratch, "black_mage") {
            Ok(project) => json!({
                "accepted": true,
                "max_mp": project.engine_resources.max_mp,
                "base_gcd": project.system.base_gcd,
                "potion_enabled": project.system.skills.iter().find(|skill| skill.key == "potion").unwrap().enabled,
            }),
            Err(_) => json!({"accepted": false}),
        };
        assert_equivalent(&case["result"], &actual, case["name"].as_str().unwrap());
    }
    std::fs::remove_dir_all(scratch).unwrap();
}

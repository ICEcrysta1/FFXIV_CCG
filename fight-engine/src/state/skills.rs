// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// 附加许可：FightEngine 链接例外 1.1 版，见 LICENSE-FightEngine-Linking-Exception。

//! 技能状态定义、稳定布局与冻结快照；实际冷却仍由共享系统维护。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::config::{SkillBook, SkillDefinition};
use crate::state::CombatState;
use crate::system::SystemRules;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkillState {
    pub available: bool,
    pub current_charges: i32,
    pub next_cooldown_seconds: f64,
}

pub type SkillStateSnapshot = BTreeMap<String, SkillState>;

/// 注册启用技能的只读定义，不持有各队列的运行值。
#[derive(Clone, Debug)]
pub struct SkillStateRegistry {
    definitions: Vec<SkillDefinition>,
    keys: Vec<String>,
}

impl SkillStateRegistry {
    pub fn new(skills: &SkillBook) -> Self {
        let mut definitions: Vec<_> = skills.enabled_skills(None).into_iter().cloned().collect();
        definitions.sort_by(|left, right| left.key.cmp(&right.key));
        let keys = definitions.iter().map(|skill| skill.key.clone()).collect();
        Self { definitions, keys }
    }

    pub fn definitions(&self) -> &[SkillDefinition] {
        &self.definitions
    }

    pub fn keys(&self) -> &[String] {
        &self.keys
    }

    pub fn snapshot(
        &self,
        state: &CombatState,
        system: &SystemRules,
        mut available: impl FnMut(&SkillDefinition) -> Result<bool, String>,
    ) -> Result<SkillStateSnapshot, String> {
        self.definitions
            .iter()
            .map(|skill| {
                let (next_cooldown_seconds, current_charges, _) =
                    system.build_cooldown_snapshot(state, skill);
                Ok((
                    skill.key.clone(),
                    SkillState {
                        available: available(skill)?,
                        current_charges,
                        next_cooldown_seconds,
                    },
                ))
            })
            .collect()
    }
}

// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// 附加许可：FightEngine 链接例外 1.1 版，见 LICENSE-FightEngine-Linking-Exception。

//! 冻结状态、历史 token 与 canonical 上下文；格式翻译留在输出模块内部。

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde_json::{Map, Value, json};

use crate::config::{ProjectConfig, SchemaConfig};
use crate::contracts::JobStateMachine;
use crate::state::resources::ResourceSnapshot;
use crate::state::{
    ActionHistoryEntry, BuffContextEntry, CombatState, DotContextEntry, ModelStateSnapshot,
    PlayerContext, StateContext, TargetContext,
};
use crate::system::SystemRules;

mod format;
mod tokens;

pub use format::{OutputFormatter, output_schema_metadata, pack_tensor_payload, scene_context};
pub use tokens::{SkillTokenInput, build_death_token, build_skill_token};

pub struct StateContextBuilder {
    project: Arc<ProjectConfig>,
    system: Arc<SystemRules>,
    job: Arc<dyn JobStateMachine>,
}

/// 一行冻结模型输入；Arc 身份用于复用 token，序号只用于历史排序和增量游标。
#[derive(Clone, Debug)]
pub struct ModelHistoryRow {
    pub sequence: u64,
    pub model_state: ModelStateSnapshot,
    pub skill: Value,
    pub execution_metrics: BTreeMap<String, f64>,
}

#[derive(Default)]
struct HistoryCache {
    // 缓存同时持有源 Arc，防止快照恢复后相同序号或分配地址被错误复用。
    entries: BTreeMap<usize, (Arc<ActionHistoryEntry>, Arc<ModelHistoryRow>)>,
    tokens: BTreeMap<usize, (Arc<ModelHistoryRow>, Value)>,
    action_keys: Vec<String>,
}

pub struct OutputContextBuilder {
    state: StateContextBuilder,
    job: Arc<dyn JobStateMachine>,
    schema: Arc<SchemaConfig>,
    history_limit: Option<usize>,
    tokens: tokens::StateTokens,
    cache: Mutex<HistoryCache>,
}

impl OutputContextBuilder {
    pub fn new(
        project: Arc<ProjectConfig>,
        system: Arc<SystemRules>,
        job: Arc<dyn JobStateMachine>,
        schema: Arc<SchemaConfig>,
        history_limit: Option<usize>,
    ) -> Result<Self, String> {
        Ok(Self {
            tokens: tokens::StateTokens::new(schema.clone(), &system, job.as_ref())?,
            state: StateContextBuilder::new(project, system, job.clone()),
            job,
            schema,
            history_limit,
            cache: Mutex::new(HistoryCache::default()),
        })
    }

    pub fn build_state_context(&self, state: &CombatState) -> Result<StateContext, String> {
        self.state.build(state)
    }

    pub fn build_noop_resource_transition(
        &self,
        state: &CombatState,
    ) -> Result<ResourceSnapshot, String> {
        Ok(self
            .job
            .runtime()
            .resource_transition(state, state)?
            .consumed)
    }

    pub fn collect_history(
        &self,
        state: &CombatState,
    ) -> Result<Vec<Arc<ModelHistoryRow>>, String> {
        let first = self
            .history_limit
            .map_or(0, |limit| state.history.len().saturating_sub(limit));
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| "output history cache is poisoned")?;
        let mut retained = BTreeMap::new();
        let mut rows = Vec::new();
        for entry in &state.history[first..] {
            let identity = Arc::as_ptr(entry) as usize;
            let row = if let Some((original, row)) = cache.entries.get(&identity)
                && Arc::ptr_eq(original, entry)
            {
                row.clone()
            } else {
                Arc::new(ModelHistoryRow {
                    sequence: entry.history_sequence,
                    model_state: entry
                        .model_state
                        .clone()
                        .ok_or_else(|| "history is missing its frozen model state".to_owned())?,
                    skill: tokens::history_skill_token(entry)?,
                    execution_metrics: BTreeMap::from([
                        (
                            "cumulative_potency".into(),
                            entry.state_after.target.cumulative_potency,
                        ),
                        (
                            "cumulative_dot_potency".into(),
                            entry.state_after.target.cumulative_dot_potency,
                        ),
                    ]),
                })
            };
            retained.insert(identity, (entry.clone(), row.clone()));
            rows.push(row);
        }
        cache.entries = retained;
        Ok(rows)
    }

    pub fn build_context(
        &self,
        state: &CombatState,
        action_keys: &[String],
        action_values: &[f64],
        current: &ModelStateSnapshot,
        history: Option<&[Arc<ModelHistoryRow>]>,
    ) -> Result<Value, String> {
        if action_keys.len() != action_values.len() {
            return Err("action output arrays must have matching lengths".into());
        }
        let owned;
        let history = if let Some(history) = history {
            history
        } else {
            owned = self.collect_history(state)?;
            &owned
        };
        let mut history_context = self.tokens.metadata(action_keys)?;
        let mut current_context = history_context.clone();
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| "output history cache is poisoned")?;
        if cache.action_keys != action_keys {
            cache.tokens.clear();
            cache.action_keys = action_keys.to_vec();
        }
        let mut retained = BTreeMap::new();
        let mut history_tokens = Vec::new();
        for row in history {
            let identity = Arc::as_ptr(row) as usize;
            let token = if let Some((original, token)) = cache.tokens.get(&identity)
                && Arc::ptr_eq(original, row)
            {
                token.clone()
            } else {
                self.tokens.build(&row.model_state, action_keys)?
            };
            retained.insert(identity, (row.clone(), token.clone()));
            history_tokens.push(token);
        }
        cache.tokens = retained;
        drop(cache);
        history_context.insert(self.schema.token_key.clone(), json!(history_tokens));
        history_context.insert(
            "execution_metrics".into(),
            json!(
                history
                    .iter()
                    .map(|row| &row.execution_metrics)
                    .collect::<Vec<_>>()
            ),
        );
        current_context.insert(
            self.schema.token_key.clone(),
            json!([self.tokens.build(current, action_keys)?]),
        );
        let legal = action_keys
            .iter()
            .map(|key| {
                current
                    .request_state
                    .skill_state
                    .get(key)
                    .map(|skill| skill.available)
                    .ok_or_else(|| format!("frozen skill state is missing {key}"))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let context = json!({
            "job_tag":self.job.runtime().resources.job_tag(),"schema_version":self.schema.canonical_schema_version,
            "history_cursor":state.last_history_sequence,"scene_context":scene_context(&self.schema)?,
            "skill_history_context":history.iter().map(|row| &row.skill).collect::<Vec<_>>(),
            "state_history_context":history_context,"current_state_context":current_context,
            "action_keys":action_keys,"action_legal_mask":legal,"action_values":action_values,
        });
        self.schema
            .canonical_top_level_keys
            .iter()
            .map(|key| {
                context
                    .get(key)
                    .cloned()
                    .map(|value| (key.clone(), value))
                    .ok_or_else(|| format!("canonical output is missing {key}"))
            })
            .collect::<Result<Map<_, _>, _>>()
            .map(Value::Object)
    }
}

impl StateContextBuilder {
    pub fn new(
        project: Arc<ProjectConfig>,
        system: Arc<SystemRules>,
        job: Arc<dyn JobStateMachine>,
    ) -> Self {
        Self {
            project,
            system,
            job,
        }
    }

    pub fn build(&self, state: &CombatState) -> Result<StateContext, String> {
        let now = state.time();
        let current_gcd = self.job.current_gcd_duration(state)?;
        let gcd_units = |seconds: f64| {
            if current_gcd <= 0.0 {
                0.0
            } else {
                seconds / current_gcd
            }
        };
        let gcd_remaining = state.gcd_remaining();
        let weave_window = state.weave_window_remaining();
        let next_untargetable = state.next_downtime_eta();
        let downtime_remaining = state.downtime_remaining();
        let player = PlayerContext {
            time_seconds: now,
            mp: state.mp,
            max_mp: state.max_mp,
            mp_ratio: if state.max_mp <= 0 {
                0.0
            } else {
                f64::from(state.mp) / f64::from(state.max_mp)
            },
            gcd_index: state.gcd_index,
            current_gcd_seconds: current_gcd,
            fight_remaining_seconds: state.fight_remaining(),
            gcd_remaining_seconds: gcd_remaining,
            cast_remaining_seconds: state.cast_remaining(),
            gcd_remaining_gcds: gcd_units(gcd_remaining),
            weave_window_seconds: weave_window,
            weave_window_gcds: gcd_units(weave_window),
            ogcd_window_seconds: weave_window,
            ogcd_window_gcds: gcd_units(weave_window),
            boss_targetable: state.boss_targetable(),
            next_untargetable_in_seconds: next_untargetable,
            next_untargetable_in_gcds: next_untargetable.map(gcd_units),
            downtime_remaining_seconds: downtime_remaining,
            downtime_remaining_gcds: gcd_units(downtime_remaining),
            is_moving: state.is_moving,
        };
        let mut buffs = Vec::new();
        for (source, definitions, statuses) in [
            ("system", self.system.status_definitions(), &state.statuses),
            (
                "job",
                self.job.runtime().statuses.definitions(),
                &state.job.statuses,
            ),
        ] {
            // Buff 上下文遵循定义注册顺序，与实际授予顺序独立。
            for (key, definition) in definitions {
                let Some(status) = statuses.get(key) else {
                    continue;
                };
                let remaining = status.remaining(now);
                if remaining <= 0.0 || status.stacks <= 0 {
                    continue;
                }
                buffs.push(BuffContextEntry {
                    source: source.to_owned(),
                    status_key: key.clone(),
                    status_id: definition.game_id,
                    remaining_seconds: remaining,
                    remaining_gcds: gcd_units(remaining),
                    stacks: status.stacks,
                });
            }
        }
        let mut dots = Vec::new();
        for (key, dot) in &state.dots {
            let remaining = dot.remaining(now);
            if remaining <= 0.0 {
                continue;
            }
            dots.push(DotContextEntry {
                dot_key: key.clone(),
                remaining_seconds: remaining,
                remaining_gcds: gcd_units(remaining),
                tick_interval_seconds: dot.tick_interval,
                potency_per_tick: dot.potency_per_tick,
            });
        }
        dots.sort_by(|left, right| left.dot_key.cmp(&right.dot_key));
        Ok(StateContext {
            job_tag: self.job.runtime().resources.job_tag().to_owned(),
            job_name: self.project.job.name.clone(),
            player,
            target: TargetContext {
                cumulative_potency: state.cumulative_potency,
                cumulative_dot_potency: state.cumulative_dot_potency,
                current_potency: state.current_potency,
                current_gcd_dot_potency: state.current_gcd_dot_potency,
            },
            resources: self.job.runtime().resource_snapshot(state)?,
            buffs,
            dots,
        })
    }
}

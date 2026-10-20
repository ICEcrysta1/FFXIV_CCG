// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

//! 为执行历史和模型快照装配同一份当前状态上下文。

use std::sync::Arc;

use crate::config::ProjectConfig;
use crate::contracts::JobStateMachine;
use crate::state::{
    BuffContextEntry, CombatState, DotContextEntry, PlayerContext, StateContext, TargetContext,
};
use crate::system::SystemRules;

pub struct StateContextBuilder {
    project: Arc<ProjectConfig>,
    system: Arc<SystemRules>,
    job: Arc<dyn JobStateMachine>,
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

// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

//! Buff 定义与层数操作；调用方明确传入自己拥有的状态容器。

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use indexmap::IndexMap;

use crate::config::StatusDefinition;
use crate::system::timeline::{TimelineEvent, TimelineEventScope};

use super::{CombatState, StatusState, ValidationResult};

type ExpiryHook = dyn Fn(&mut CombatState) -> Result<(), String> + Send + Sync;

#[derive(Clone)]
pub struct StatusRules {
    definitions: IndexMap<String, StatusDefinition>,
    expiry_hooks: BTreeMap<String, Arc<ExpiryHook>>,
}

impl fmt::Debug for StatusRules {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StatusRules")
            .field("definitions", &self.definitions)
            .field(
                "expiry_hooks",
                &self.expiry_hooks.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl StatusRules {
    pub fn new(definitions: IndexMap<String, StatusDefinition>) -> Self {
        Self {
            definitions,
            expiry_hooks: BTreeMap::new(),
        }
    }

    pub fn definitions(&self) -> &IndexMap<String, StatusDefinition> {
        &self.definitions
    }

    pub fn definition(&self, key: &str) -> Result<&StatusDefinition, String> {
        self.definitions
            .get(key)
            .ok_or_else(|| format!("unregistered status: {key}"))
    }

    pub fn validate(
        &self,
        now: f64,
        statuses: &IndexMap<String, StatusState>,
        key: &str,
    ) -> Result<ValidationResult, String> {
        let definition = self.definition(key)?;
        Ok(
            if statuses.get(key).is_some_and(|status| {
                status.remaining(now) > 0.0 && status.stacks >= definition.max_stacks
            }) {
                ValidationResult::invalid("status_already_active")
            } else {
                ValidationResult::valid()
            },
        )
    }

    pub fn grant(
        &self,
        now: f64,
        statuses: &mut IndexMap<String, StatusState>,
        key: &str,
        remaining: Option<f64>,
        stacks: Option<i32>,
    ) -> Result<(), String> {
        let definition = self.definition(key)?;
        let remaining = remaining.unwrap_or(definition.duration).max(0.0);
        let stacks = stacks
            .unwrap_or(definition.max_stacks)
            .clamp(0, definition.max_stacks);
        if remaining <= 0.0 || stacks <= 0 {
            statuses.shift_remove(key);
        } else {
            statuses.insert(key.to_owned(), StatusState::new(now, remaining, stacks));
        }
        Ok(())
    }

    pub fn clear(&self, statuses: &mut IndexMap<String, StatusState>, key: &str) {
        statuses.shift_remove(key);
    }

    pub fn consume_stack(&self, statuses: &mut IndexMap<String, StatusState>, key: &str) {
        if let Some(status) = statuses.get_mut(key) {
            status.stacks -= 1;
            if status.stacks <= 0 {
                statuses.shift_remove(key);
            }
        }
    }

    pub fn register_expiry_hook(
        &mut self,
        key: impl Into<String>,
        hook: impl Fn(&mut CombatState) -> Result<(), String> + Send + Sync + 'static,
    ) -> Result<(), String> {
        let key = key.into();
        if key.trim().is_empty() || self.expiry_hooks.contains_key(&key) {
            return Err(format!("invalid or duplicate status expiry hook: {key}"));
        }
        self.expiry_hooks.insert(key, Arc::new(hook));
        Ok(())
    }

    /// 到期钩子先读取残余层数，再在同一次结算中移除所属状态。
    pub fn settle_expired<P>(
        &self,
        state: &mut CombatState,
        event: &TimelineEvent<P>,
        scope: TimelineEventScope,
    ) -> Result<(), String> {
        if event.scope != scope {
            return Err("status expiration belongs to another scope".into());
        }
        let key = event
            .owner_key
            .as_deref()
            .ok_or("status expiration is missing its key")?;
        let statuses = match scope {
            TimelineEventScope::System => &state.statuses,
            TimelineEventScope::Job => &state.job.statuses,
        };
        if statuses
            .get(key)
            .is_some_and(|status| status.expires_at <= state.time())
        {
            if let Some(hook) = self.expiry_hooks.get(key) {
                hook(state)?;
            }
            let statuses = match scope {
                TimelineEventScope::System => &mut state.statuses,
                TimelineEventScope::Job => &mut state.job.statuses,
            };
            statuses.shift_remove(key);
        }
        Ok(())
    }
}

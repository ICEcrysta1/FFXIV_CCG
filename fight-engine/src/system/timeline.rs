// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only
// 附加许可：FightEngine 链接例外 1.1 版，见 LICENSE-FightEngine-Linking-Exception。

//! 唯一逻辑时钟与事件内核；领域模块只描述截止时刻并处理单次事实。

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::state::CombatState;

pub const TIME_EPSILON: f64 = 0.000_000_1;
const MAX_REPEATED_DISPATCH: usize = 5;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum TimelineEventKind {
    ActionAccepted,
    CastCompleted,
    ActionEffect,
    GcdReady,
    CooldownChargeReady,
    StatusExpired,
    DotTick,
    MpTick,
    JobPeriodicTick,
    JobTimerExpired,
    SceneChanged,
    DecisionBoundary,
    PlayerDeath,
    PlayerRevived,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum TimelineEventPriority {
    ExternalScene,
    ConfirmedActionEffect,
    PeriodicSettlement,
    ExpirationAndCooldown,
    PlayerReset,
    ActionAccepted,
    DecisionBoundary,
}

impl PartialEq for TimelineEventPriority {
    fn eq(&self, other: &Self) -> bool {
        self.rank() == other.rank()
    }
}

impl Eq for TimelineEventPriority {}

impl TimelineEventPriority {
    pub const fn rank(self) -> u8 {
        match self {
            Self::ExternalScene => 10,
            Self::ConfirmedActionEffect => 20,
            Self::PeriodicSettlement => 30,
            Self::ExpirationAndCooldown => 40,
            Self::PlayerReset => 45,
            Self::ActionAccepted | Self::DecisionBoundary => 50,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum TimelineEventScope {
    #[default]
    System,
    Job,
}

/// 载荷由宿主定义，内核无需依赖动作、输出或具体职业。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimelineEvent<P = ()> {
    pub timestamp: f64,
    pub priority: TimelineEventPriority,
    pub kind: TimelineEventKind,
    pub owner_key: Option<String>,
    pub action_instance_id: Option<u64>,
    pub payload: Option<P>,
    pub sequence: u64,
    pub scope: TimelineEventScope,
}

impl<P> TimelineEvent<P> {
    pub fn new(timestamp: f64, priority: TimelineEventPriority, kind: TimelineEventKind) -> Self {
        Self {
            timestamp,
            priority,
            kind,
            owner_key: None,
            action_instance_id: None,
            payload: None,
            sequence: 0,
            scope: TimelineEventScope::System,
        }
    }
}

/// 处理器返回完整的新状态，内核校验时钟未被领域逻辑改写后才应用。
#[derive(Clone, Debug)]
pub struct TimelineMutation<P = ()> {
    pub state: Option<CombatState>,
    pub cancel: Vec<u64>,
    pub schedule: Vec<TimelineEvent<P>>,
}

impl<P> Default for TimelineMutation<P> {
    fn default() -> Self {
        Self {
            state: None,
            cancel: Vec::new(),
            schedule: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimulationSnapshot<P = ()> {
    pub state: CombatState,
    pub pending_events: Vec<TimelineEvent<P>>,
    pub pending_settlements: Vec<TimelineEvent<P>>,
    pub next_sequence: u64,
    /// 显式保存资源事件归属，恢复时无需猜测哪些事件来自资源描述。
    pub resource_sequences: BTreeSet<u64>,
}

/// 有限时间戳按数值比较，正负零视为同一个时刻。
#[derive(Clone, Copy, Debug)]
struct EventTime(f64);

impl PartialEq for EventTime {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for EventTime {}

impl PartialOrd for EventTime {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EventTime {
    fn cmp(&self, other: &Self) -> Ordering {
        if self == other {
            Ordering::Equal
        } else {
            self.0.total_cmp(&other.0)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct EventOrder {
    timestamp: EventTime,
    priority: u8,
    sequence: u64,
}

impl EventOrder {
    fn of<P>(event: &TimelineEvent<P>) -> Self {
        Self {
            timestamp: EventTime(event.timestamp),
            priority: event.priority.rank(),
            sequence: event.sequence,
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SettlementTarget {
    kind: TimelineEventKind,
    scope: TimelineEventScope,
    owner_key: Option<String>,
    timestamp: EventTime,
}

impl SettlementTarget {
    fn of<P>(event: &TimelineEvent<P>) -> Self {
        Self {
            kind: event.kind,
            scope: event.scope,
            owner_key: event.owner_key.clone(),
            timestamp: EventTime(event.timestamp),
        }
    }
}

/// 有序表直接移除取消项，不保留堆中的失效载荷。
#[derive(Clone, Debug)]
pub struct Timeline<P = ()> {
    state: CombatState,
    events: BTreeMap<EventOrder, TimelineEvent<P>>,
    event_orders: BTreeMap<u64, EventOrder>,
    settlements: BTreeMap<EventOrder, TimelineEvent<P>>,
    settlement_targets: BTreeSet<SettlementTarget>,
    resource_sequences: BTreeSet<u64>,
    next_sequence: u64,
}

impl<P: Clone + PartialEq> Timeline<P> {
    pub fn new(state: CombatState) -> Result<Self, String> {
        validate_time(state.time)?;
        Ok(Self {
            state,
            events: BTreeMap::new(),
            event_orders: BTreeMap::new(),
            settlements: BTreeMap::new(),
            settlement_targets: BTreeSet::new(),
            resource_sequences: BTreeSet::new(),
            next_sequence: 1,
        })
    }

    pub fn current_time(&self) -> f64 {
        self.state.time
    }

    pub fn state(&self) -> &CombatState {
        &self.state
    }

    pub fn get_state(&self) -> CombatState {
        self.state.clone()
    }

    pub fn pending_event_count(&self) -> usize {
        self.events.len()
    }

    pub fn pending_settlement_count(&self) -> usize {
        self.settlements.len()
    }

    pub fn pending_events(&self) -> Vec<TimelineEvent<P>> {
        self.events.values().cloned().collect()
    }

    pub(crate) fn iter_pending_events(&self) -> impl Iterator<Item = &TimelineEvent<P>> {
        self.events.values()
    }

    pub fn next_scheduled_event_time(&self) -> Option<f64> {
        self.events
            .first_key_value()
            .map(|(order, _)| order.timestamp.0)
    }

    pub fn schedule(&mut self, mut event: TimelineEvent<P>) -> Result<TimelineEvent<P>, String> {
        validate_time(event.timestamp)?;
        if event.timestamp < self.state.time - TIME_EPSILON {
            return Err("cannot schedule an event in the past".into());
        }
        if event.sequence != 0 {
            return Err("new events must not provide a sequence".into());
        }
        event.sequence = self.reserve_sequence()?;
        self.insert_event(event.clone())?;
        Ok(event)
    }

    pub fn cancel(&mut self, sequence: u64) -> bool {
        self.resource_sequences.remove(&sequence);
        let Some(order) = self.event_orders.remove(&sequence) else {
            return false;
        };
        self.events.remove(&order);
        true
    }

    pub fn create_snapshot(&self) -> SimulationSnapshot<P> {
        SimulationSnapshot {
            state: self.state.clone(),
            pending_events: self.pending_events(),
            pending_settlements: self.settlements.values().cloned().collect(),
            next_sequence: self.next_sequence,
            resource_sequences: self.resource_sequences.clone(),
        }
    }

    pub fn restore_snapshot(&mut self, snapshot: &SimulationSnapshot<P>) -> Result<(), String> {
        if snapshot.next_sequence == 0 {
            return Err("next sequence must be positive".into());
        }
        let mut restored = Self::new(snapshot.state.clone())?;
        restored.next_sequence = snapshot.next_sequence;
        for event in &snapshot.pending_events {
            restored.validate_restored_event(event)?;
            restored.insert_event(event.clone())?;
            restored.advance_sequence_past(event.sequence)?;
        }
        for settlement in &snapshot.pending_settlements {
            restored.validate_restored_event(settlement)?;
            restored.insert_settlement(settlement.clone());
            restored.advance_sequence_past(settlement.sequence)?;
        }
        restored.resource_sequences = snapshot
            .resource_sequences
            .iter()
            .copied()
            .filter(|sequence| restored.event_orders.contains_key(sequence))
            .collect();
        *self = restored;
        Ok(())
    }

    pub fn fork(&self) -> Self {
        self.clone()
    }

    /// 已到期资源先转为不可撤销的事实，再协调尚未到期的资源事件。
    pub fn synchronize_resources<D>(&mut self, describe: &mut D) -> Result<(), String>
    where
        D: FnMut(&CombatState) -> Vec<TimelineEvent<P>>,
    {
        let due: Vec<_> = self
            .resource_sequences
            .iter()
            .filter_map(|sequence| self.event_orders.get(sequence))
            .filter_map(|order| self.events.get(order))
            .filter(|event| event.timestamp <= self.state.time + TIME_EPSILON)
            .cloned()
            .collect();
        for event in due {
            self.cancel(event.sequence);
            self.insert_settlement(event);
        }

        let mut requested = describe(&self.state);
        for event in &requested {
            validate_time(event.timestamp)?;
            if event.sequence != 0 {
                return Err("resource descriptions must not provide a sequence".into());
            }
        }
        let resources: Vec<_> = self.resource_sequences.iter().copied().collect();
        for sequence in resources {
            let Some(existing) = self
                .event_orders
                .get(&sequence)
                .and_then(|order| self.events.get(order))
            else {
                self.resource_sequences.remove(&sequence);
                continue;
            };
            let mut unnumbered = existing.clone();
            unnumbered.sequence = 0;
            if let Some(index) = requested.iter().position(|event| event == &unnumbered) {
                requested.remove(index);
            } else {
                self.cancel(sequence);
            }
        }

        // 同次描述里的到期项从后向前领取稳定序号。
        for index in (0..requested.len()).rev() {
            if requested[index].timestamp <= self.state.time + TIME_EPSILON {
                let mut settlement = requested.remove(index);
                settlement.sequence = self.reserve_sequence()?;
                self.insert_settlement(settlement);
            }
        }
        for event in requested {
            let sequence = self.schedule(event)?.sequence;
            self.resource_sequences.insert(sequence);
        }
        Ok(())
    }

    pub fn apply_mutation<D>(
        &mut self,
        mutation: TimelineMutation<P>,
        describe: &mut D,
    ) -> Result<(), String>
    where
        D: FnMut(&CombatState) -> Vec<TimelineEvent<P>>,
    {
        if let Some(state) = mutation.state {
            if state.time != self.state.time {
                return Err("timeline mutation cannot change logical time".into());
            }
            self.state = state;
        }
        for sequence in mutation.cancel {
            self.cancel(sequence);
        }
        for event in mutation.schedule {
            self.schedule(event)?;
        }
        self.synchronize_resources(describe)
    }

    pub fn advance_to<D, H>(
        &mut self,
        timestamp: f64,
        mut describe: D,
        mut handle: H,
    ) -> Result<CombatState, String>
    where
        D: FnMut(&CombatState) -> Vec<TimelineEvent<P>>,
        H: FnMut(&TimelineEvent<P>, &Self) -> Result<TimelineMutation<P>, String>,
    {
        self.advance_clock_to(timestamp, &mut describe, &mut handle)?;
        Ok(self.state.clone())
    }

    pub fn advance_clock_to<D, H>(
        &mut self,
        timestamp: f64,
        describe: &mut D,
        handle: &mut H,
    ) -> Result<(), String>
    where
        D: FnMut(&CombatState) -> Vec<TimelineEvent<P>>,
        H: FnMut(&TimelineEvent<P>, &Self) -> Result<TimelineMutation<P>, String>,
    {
        validate_time(timestamp)?;
        if timestamp < self.state.time - TIME_EPSILON {
            return Err(format!(
                "timeline cannot move backwards from {}",
                self.state.time
            ));
        }
        let mut previous: Option<TimelineEvent<P>> = None;
        let mut repeated = 0;
        while let Some(event) = self.dequeue_next_dispatch(timestamp, describe)? {
            repeated = if previous
                .as_ref()
                .is_some_and(|last| same_dispatch(last, &event))
            {
                repeated + 1
            } else {
                0
            };
            if repeated > MAX_REPEATED_DISPATCH {
                return Err(format!(
                    "timeline event {:?}({:?}) at {} did not consume its fact or advance its deadline",
                    event.kind, event.owner_key, event.timestamp
                ));
            }
            self.state.time = self.state.time.max(event.timestamp);
            let mutation = handle(&event, self)?;
            previous = Some(event);
            self.apply_mutation(mutation, describe)?;
        }
        self.state.time = self.state.time.max(timestamp);
        self.synchronize_resources(describe)
    }

    fn dequeue_next_dispatch<D>(
        &mut self,
        timestamp: f64,
        describe: &mut D,
    ) -> Result<Option<TimelineEvent<P>>, String>
    where
        D: FnMut(&CombatState) -> Vec<TimelineEvent<P>>,
    {
        self.synchronize_resources(describe)?;
        let settlement_order = self.settlements.first_key_value().map(|(order, _)| *order);
        let event_order = self.events.first_key_value().map(|(order, _)| *order);
        match (settlement_order, event_order) {
            (Some(order), other) if other.is_none_or(|event| order <= event) => {
                if order.timestamp.0 > timestamp {
                    return Ok(None);
                }
                let event = self.settlements.pop_first().expect("peeked settlement").1;
                self.settlement_targets
                    .remove(&SettlementTarget::of(&event));
                Ok(Some(event))
            }
            (_, Some(order)) if order.timestamp.0 <= timestamp => {
                let event = self.events.pop_first().expect("peeked event").1;
                self.event_orders.remove(&event.sequence);
                self.resource_sequences.remove(&event.sequence);
                Ok(Some(event))
            }
            _ => Ok(None),
        }
    }

    fn insert_event(&mut self, event: TimelineEvent<P>) -> Result<(), String> {
        let order = EventOrder::of(&event);
        if self.event_orders.contains_key(&event.sequence) {
            return Err(format!("duplicate timeline sequence: {}", event.sequence));
        }
        self.event_orders.insert(event.sequence, order);
        self.events.insert(order, event);
        Ok(())
    }

    fn insert_settlement(&mut self, event: TimelineEvent<P>) {
        if self.settlement_targets.insert(SettlementTarget::of(&event)) {
            self.settlements.insert(EventOrder::of(&event), event);
        }
    }

    fn reserve_sequence(&mut self) -> Result<u64, String> {
        let sequence = self.next_sequence;
        self.next_sequence = sequence
            .checked_add(1)
            .ok_or("timeline sequence overflow")?;
        Ok(sequence)
    }

    fn advance_sequence_past(&mut self, sequence: u64) -> Result<(), String> {
        if sequence >= self.next_sequence {
            self.next_sequence = sequence
                .checked_add(1)
                .ok_or("timeline sequence overflow")?;
        }
        Ok(())
    }

    fn validate_restored_event(&self, event: &TimelineEvent<P>) -> Result<(), String> {
        validate_time(event.timestamp)?;
        if event.sequence == 0 {
            return Err("snapshot events must have positive sequences".into());
        }
        if event.timestamp < self.state.time - TIME_EPSILON {
            return Err("snapshot cannot contain an event in the past".into());
        }
        Ok(())
    }
}

fn same_dispatch<P>(left: &TimelineEvent<P>, right: &TimelineEvent<P>) -> bool {
    left.kind == right.kind
        && left.scope == right.scope
        && left.owner_key == right.owner_key
        && (left.timestamp - right.timestamp).abs() <= TIME_EPSILON
}

fn validate_time(timestamp: f64) -> Result<(), String> {
    if timestamp.is_finite() {
        Ok(())
    } else {
        Err("timeline timestamp must be finite".into())
    }
}

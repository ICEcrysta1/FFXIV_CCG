// Copyright (C) 2026 ICE_crystal
// SPDX-License-Identifier: GPL-3.0-only

//! Python 原生句柄只装配职业和转换边界值；队列执行与编码均释放 GIL。

use std::sync::Arc;

use fight_engine::config::{PolicyActionRegistry, ProjectConfig, SchemaConfig, SkillBook};
use fight_engine::contracts::JobRegistry;
use fight_engine::engine::EngineRules;
use fight_engine::engine::sessions::{
    MAX_CAPACITY, SessionOptions, SessionResult, SimulationEngine, SimulationSession,
};
use fight_engine::outputs::build_death_token;
use fight_engine::state::ExternalCombatEvent;
use fight_engine_black_mage::BlackMage;
use fight_engine_machinist::Machinist;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::Value;

mod packet;

type Submission = (
    bool,
    bool,
    String,
    Option<String>,
    f64,
    Option<f64>,
    Option<f64>,
    Option<f64>,
);
type ExternalInput = (
    f64,
    String,
    Option<bool>,
    Option<i32>,
    Option<f64>,
    Option<i32>,
);
type ExternalReceipt = (bool, String, f64, Option<f64>, Option<String>, Option<u64>);
type Observation<'py> = (f64, Option<f64>, Bound<'py, PyBytes>);

#[pyclass(name = "SimulationEngine", module = "fight_engine_native", frozen)]
struct PythonEngine {
    inner: Arc<SimulationEngine>,
}

#[pymethods]
impl PythonEngine {
    #[new]
    fn new(
        py: Python<'_>,
        project_root: String,
        job_tag: String,
        capacity: usize,
    ) -> PyResult<Self> {
        let inner = py
            .detach(|| {
                let project = Arc::new(ProjectConfig::load(&project_root, &job_tag)?);
                let schema = Arc::new(SchemaConfig::load(&project_root)?);
                let skills = SkillBook::from_project_config(&project)?;
                let policy = Arc::new(PolicyActionRegistry::load(&project_root, &skills)?);
                let mut registry = JobRegistry::default();
                registry.register::<BlackMage>()?;
                registry.register::<Machinist>()?;
                let rules = Arc::new(EngineRules::from_project(project, schema, &registry, None)?);
                SimulationEngine::new(rules, policy, capacity)
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn active_count(&self, py: Python<'_>) -> PyResult<usize> {
        py.detach(|| self.inner.active_count())
            .map_err(PyRuntimeError::new_err)
    }

    #[pyo3(signature = (max_history, actual_base_gcd, initial_timestamp, fight_remaining))]
    fn create_session(
        &self,
        py: Python<'_>,
        max_history: Option<usize>,
        actual_base_gcd: Option<f64>,
        initial_timestamp: f64,
        fight_remaining: Option<f64>,
    ) -> PyResult<PythonSession> {
        let inner = py
            .detach(|| {
                self.inner.create_session(SessionOptions {
                    history_limit: max_history,
                    actual_base_gcd,
                    initial_timestamp,
                    fight_remaining,
                })
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok(PythonSession { inner })
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.close())
            .map_err(PyRuntimeError::new_err)
    }
}

#[pyclass(name = "SimulationSession", module = "fight_engine_native", frozen)]
struct PythonSession {
    inner: Arc<SimulationSession>,
}

#[pymethods]
impl PythonSession {
    #[getter]
    fn id(&self, py: Python<'_>) -> u64 {
        py.detach(|| self.inner.id())
    }

    #[pyo3(signature = (max_history, actual_base_gcd, initial_timestamp, fight_remaining))]
    fn reset(
        &self,
        py: Python<'_>,
        max_history: Option<usize>,
        actual_base_gcd: Option<f64>,
        initial_timestamp: f64,
        fight_remaining: Option<f64>,
    ) -> PyResult<()> {
        py.detach(|| {
            self.inner
                .reset(SessionOptions {
                    history_limit: max_history,
                    actual_base_gcd,
                    initial_timestamp,
                    fight_remaining,
                })
                .map(|_| ())
        })
        .map_err(PyRuntimeError::new_err)
    }

    fn advance_to(&self, py: Python<'_>, timestamp: f64) -> PyResult<(f64, Option<f64>)> {
        py.detach(|| {
            self.inner
                .advance_to(timestamp)
                .map(|result| (result.timestamp, result.next_scheduled_event_time))
        })
        .map_err(PyRuntimeError::new_err)
    }

    #[pyo3(signature = (timestamp, action, actual_cast_seconds))]
    fn submit_action(
        &self,
        py: Python<'_>,
        timestamp: f64,
        action: String,
        actual_cast_seconds: Option<f64>,
    ) -> PyResult<Submission> {
        py.detach(|| {
            self.inner
                .submit_action(timestamp, &action, actual_cast_seconds)
                .map(|result| {
                    let value = result.value;
                    (
                        value.accepted,
                        value.queued,
                        value.reason,
                        value.action_instance_id.map(|id| id.to_string()),
                        value.request_timestamp,
                        value.accepted_timestamp,
                        value.effect_timestamp,
                        value.next_scheduled_event_time,
                    )
                })
        })
        .map_err(PyRuntimeError::new_err)
    }

    fn validate_at(
        &self,
        py: Python<'_>,
        timestamp: f64,
        action: String,
    ) -> PyResult<(bool, String, f64, Option<f64>)> {
        py.detach(|| {
            self.inner
                .validate_action_at(timestamp, &action)
                .map(|result| {
                    (
                        result.value.ok,
                        result.value.reason,
                        result.timestamp,
                        result.next_scheduled_event_time,
                    )
                })
        })
        .map_err(PyRuntimeError::new_err)
    }

    fn record_policy_action(
        &self,
        py: Python<'_>,
        timestamp: f64,
        action: String,
        next_observation_timestamp: f64,
    ) -> PyResult<(String, f64, f64, i32)> {
        py.detach(|| {
            self.inner
                .record_policy_action(timestamp, &action, next_observation_timestamp)
                .map(|result| {
                    (
                        result.value.action.key,
                        result.value.timestamp,
                        next_observation_timestamp,
                        result.value.gcd_index,
                    )
                })
        })
        .map_err(PyRuntimeError::new_err)
    }

    fn apply_external_events(
        &self,
        py: Python<'_>,
        events: Vec<ExternalInput>,
    ) -> PyResult<ExternalReceipt> {
        py.detach(|| {
            let events = events
                .into_iter()
                .map(
                    |(timestamp, kind, value, target_count, remaining_seconds, mp)| {
                        Ok(ExternalCombatEvent {
                            timestamp,
                            kind: serde_json::from_value(Value::String(kind))
                                .map_err(|error| error.to_string())?,
                            value,
                            target_count,
                            remaining_seconds,
                            mp,
                        })
                    },
                )
                .collect::<Result<Vec<_>, String>>()?;
            self.inner.apply_external_events(&events).map(|result| {
                (
                    result.value.accepted,
                    result.value.reason,
                    result.value.timestamp,
                    result.next_scheduled_event_time,
                    result.value.token,
                    result.value.history_cursor,
                )
            })
        })
        .map_err(PyRuntimeError::new_err)
    }

    #[pyo3(signature = (timestamp, format, next_observation_timestamp))]
    fn observe_at<'py>(
        &self,
        py: Python<'py>,
        timestamp: f64,
        format: String,
        next_observation_timestamp: Option<f64>,
    ) -> PyResult<Observation<'py>> {
        let (timestamp, next, packet) = py
            .detach(|| {
                encode_observation(self.inner.observe_at(
                    timestamp,
                    &format,
                    next_observation_timestamp,
                )?)
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok((timestamp, next, PyBytes::new(py, &packet)))
    }

    fn observe_vector_delta_at<'py>(
        &self,
        py: Python<'py>,
        timestamp: f64,
        next_observation_timestamp: f64,
        after_sequence: u64,
    ) -> PyResult<Observation<'py>> {
        let (timestamp, next, packet) = py
            .detach(|| {
                encode_observation(self.inner.observe_vector_delta_at(
                    timestamp,
                    next_observation_timestamp,
                    after_sequence,
                )?)
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok((timestamp, next, PyBytes::new(py, &packet)))
    }

    fn replay_policy_sequence<'py>(
        &self,
        py: Python<'py>,
        requests_json: String,
        facts_json: String,
        fight_duration: f64,
        policy_action_key: String,
    ) -> PyResult<Observation<'py>> {
        let (timestamp, next, packet) = py
            .detach(|| {
                encode_observation(self.inner.replay_policy_sequence(
                    &requests_json,
                    &facts_json,
                    fight_duration,
                    &policy_action_key,
                )?)
            })
            .map_err(PyRuntimeError::new_err)?;
        Ok((timestamp, next, PyBytes::new(py, &packet)))
    }

    fn statistics(&self, py: Python<'_>) -> PyResult<(f64, usize, usize, usize, usize, usize)> {
        py.detach(|| {
            self.inner.statistics().map(|result| {
                (
                    result.timestamp,
                    result.action_history_count,
                    result.policy_history_count,
                    result.pending_event_count,
                    result.queue_entry_count,
                    result.pending_settlement_count,
                )
            })
        })
        .map_err(PyRuntimeError::new_err)
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.close())
            .map_err(PyRuntimeError::new_err)
    }
}

fn encode_observation(result: SessionResult<Value>) -> Result<(f64, Option<f64>, Vec<u8>), String> {
    Ok((
        result.timestamp,
        result.next_scheduled_event_time,
        packet::encode(&result.value)?,
    ))
}

#[pyfunction]
fn death_token_packet(
    py: Python<'_>,
    timestamp: f64,
    history_cursor: u64,
) -> PyResult<Bound<'_, PyBytes>> {
    let packet = py
        .detach(|| packet::encode(&build_death_token(timestamp, history_cursor)))
        .map_err(PyRuntimeError::new_err)?;
    Ok(PyBytes::new(py, &packet))
}

#[pymodule]
fn fight_engine_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    // 内容由编译器嵌入，旧轮子不会读取运行目录的新 schema 冒充新契约。
    let contract_version = module
        .py()
        .detach(|| {
            let schema: serde_yaml_ng::Value =
                serde_yaml_ng::from_str(include_str!("../../../config/schema.yaml"))
                    .map_err(|error| error.to_string())?;
            schema["contracts"]["sidecar_contract_version"]
                .as_i64()
                .ok_or_else(|| "编译时 schema 缺少整数 sidecar_contract_version".to_owned())
        })
        .map_err(PyRuntimeError::new_err)?;
    module.add("CONTRACT_VERSION", contract_version)?;
    module.add("MAX_QUEUE_CAPACITY", MAX_CAPACITY)?;
    module.add_class::<PythonEngine>()?;
    module.add_class::<PythonSession>()?;
    module.add_function(wrap_pyfunction!(death_token_packet, module)?)?;
    Ok(())
}

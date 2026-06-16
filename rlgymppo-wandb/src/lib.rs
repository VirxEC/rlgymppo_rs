//! Sends metrics to Weights & Biases via an embedded Python interpreter.
//!
//! Mirrors the C++ `MetricSender` which uses pybind11 to call into Python's
//! `wandb` library.  This crate wraps `wandb.init()` and `wandb.log()` through
//! pyo3 so users don't need a separate Python script.

use std::collections::HashMap;

use pyo3::{prelude::*, types::PyDict};

/// A handle to an active wandb run.
///
/// Created by [`MetricSender::new`], which calls `wandb.init(...)`.
/// Call [`send`](Self::send) each iteration to log scalar metrics.
pub struct MetricSender {
    run_id: String,
    /// The Python `wandb.Run` object returned by `wandb.init()`.
    py_run: Py<PyAny>,
}

impl MetricSender {
    /// Initialise a new wandb run.
    ///
    /// `project_name`, `group_name`, `run_name` map directly to `wandb.init()`.
    /// If `run_id` is non-empty the run is **resumed** (wandb `id=` +
    /// `resume="allow"`), which lets you continue a crashed / interrupted
    /// training run.
    ///
    /// Returns an error if the Python `wandb` module cannot be imported or
    /// `wandb.init()` fails.
    pub fn new(
        project_name: &str,
        group_name: &str,
        run_name: &str,
        run_id: &str,
    ) -> PyResult<Self> {
        Python::attach(|py| {
            let wandb = py.import("wandb")?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("project", project_name)?;
            kwargs.set_item("group", group_name)?;
            kwargs.set_item("name", run_name)?;

            if !run_id.is_empty() {
                kwargs.set_item("id", run_id)?;
                kwargs.set_item("resume", "allow")?;
            }

            let run = wandb.call_method("init", (), Some(&kwargs))?;
            let rid: String = run.getattr("id")?.extract()?;

            Ok(MetricSender {
                run_id: rid,
                py_run: run.into(),
            })
        })
    }

    /// The run ID returned by wandb (used for checkpointing / resume).
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Send a flat dictionary of scalar metrics to wandb.
    ///
    /// Equivalent to calling `wandb.log(metrics)` in Python.
    pub fn send(&self, metrics: &HashMap<String, f64>) -> PyResult<()> {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            for (key, val) in metrics {
                dict.set_item(key.as_str(), *val)?;
            }
            self.py_run.as_ref().call_method(py, "log", (dict,), None)?;
            Ok(())
        })
    }
}

impl Drop for MetricSender {
    fn drop(&mut self) {
        // Attempt to close the wandb run nicely.
        let _ = Python::attach(|py| self.py_run.as_ref().call_method0(py, "finish").map(|_| ()));
    }
}

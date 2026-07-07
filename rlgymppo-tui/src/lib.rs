//! Terminal-based metrics dashboard for rlgymppo — "local-only wandb".
//!
//! This crate provides a [`TuiDisplay`] that renders training metrics in a
//! ratatui alternate-screen TUI, updating after each iteration.  It is
//! designed to run alongside the cloud-based wandb logger, or standalone.

mod app;
mod format;
mod render;

pub use app::{TuiDisplay, TuiHandle, TuiNotifier};

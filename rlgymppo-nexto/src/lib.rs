//! Nexto policy as a self-contained crate with dynamic player/entity counts.
//!
//! This crate bundles three pieces that work together:
//!
//! - [`model`] — [`NextoModel`], a backend-generic Burn inference model with
//!   pre-generated weights committed in `nexto/nexto-model.bpk`.
//! - [`obs`] — [`NextoObsBuilder`], which reads an `rlgym::GameState` and
//!   produces the q/kv/mask tensors the model consumes.
//! - [`action`] — the fixed 90-action table (`[throttle, steer, pitch, yaw,
//!   roll, jump, boost, handbrake]`) as the [`NextoAction`] value type, plus
//!   the [`previous_action`] conversion.
//!
//! The crate does not depend on the other `rlgymppo-*` crates. It depends on
//! `burn`, `burn-store`, `rlgym`, and `thiserror`, and re-exports `rlgym` for
//! convenient access to the RocketSim types used by the observation builder.

pub mod action;
pub mod model;
pub mod obs;

pub use action::{ACTION_COUNT, NextoAction, previous_action};
pub use model::{NextoError, NextoModel, NextoOutput};
pub use obs::{
    ACTIONS, ANG_VEL, BIG_PAD_BOOST, BOOST, DEFAULT_NORM, DEMO, FW, HAS_FLIP, IS_BALL, IS_BOOST,
    IS_MATE, IS_OPP, IS_SELF, KV_LEN, LIN_VEL, NextoObs, NextoObsBuilder, NextoObsConfig,
    ON_GROUND, POS, Q_LEN, SMALL_PAD_BOOST, UP,
};
pub use rlgym;

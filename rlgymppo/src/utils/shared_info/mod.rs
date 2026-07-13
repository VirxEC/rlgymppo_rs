pub use rlgymppo_utils::shared_info::SharedInfoRng;

use crate::utils::Report;

/// A trait for shared information that provides access to a report.
pub trait SharedInfoReport {
    fn report(&mut self) -> &mut Report;
}

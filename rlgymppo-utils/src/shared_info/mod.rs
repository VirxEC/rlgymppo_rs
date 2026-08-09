use crate::report::Report;
use rand::RngExt;

/// A trait for shared information that provides access to a report.
pub trait SharedInfoReport {
    fn report(&mut self) -> &mut Report;
}

/// Shared information that provides access to a random number generator.
pub trait SharedInfoRng {
    type Rng: RngExt;

    fn rng(&mut self) -> &mut Self::Rng;
}

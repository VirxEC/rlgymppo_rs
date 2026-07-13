use rand::RngExt;

/// Shared information that provides access to a random number generator.
pub trait SharedInfoRng {
    type Rng: RngExt;

    fn rng(&mut self) -> &mut Self::Rng;
}

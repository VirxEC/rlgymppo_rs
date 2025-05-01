pub trait Agent {
    fn react(&self, state: &[f32]) -> usize;
}

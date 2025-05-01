use burn::prelude::Backend;

pub trait Agent<B: Backend> {
    fn react(&self, state: &[f32], device: &B::Device) -> usize;
}

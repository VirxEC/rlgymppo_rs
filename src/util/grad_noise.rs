use tch::Tensor;

pub struct GradNoiseTracker {
    step_count: u64,
    batch_size: u32,
    update_interval: u32,
    average_decay: f32,
    batch_big: f32,
    batch_small: f32,
    moving_avg_scale: f32,
    moving_avg_noise: f32,
    batches_grad: Vec<Tensor>,
    last_noise_scale: f32,
}

impl GradNoiseTracker {
    pub fn new(batch_size: u32, update_interval: u32, beta: f32) -> Self {
        Self {
            step_count: 0,
            batch_size,
            update_interval,
            average_decay: beta,
            batch_big: (batch_size * update_interval) as f32,
            batch_small: batch_size as f32,
            moving_avg_scale: 0.0,
            moving_avg_noise: 0.0,
            batches_grad: Vec::new(),
            last_noise_scale: 0.0,
        }
    }
}

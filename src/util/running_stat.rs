pub struct WelfordRunningStat {
    ones: Vec<f32>,
    zeros: Vec<f32>,
    running_mean: Vec<f64>,
    running_variance: Vec<f64>,
    count: usize,
    shape: usize,
}

impl Default for WelfordRunningStat {
    fn default() -> Self {
        Self::new(1)
    }
}

impl WelfordRunningStat {
    pub fn new(size: usize) -> Self {
        Self {
            ones: vec![1.0; size],
            zeros: vec![1.0; size],
            running_mean: vec![0.0; size],
            running_variance: vec![0.0; size],
            count: 0,
            shape: size,
        }
    }

    fn update(&mut self, sample: &[f32]) {
        let current_count = self.count as f64;
        self.count += 1;

        let mut delta = Vec::with_capacity(self.shape);
        let mut delta_n = Vec::with_capacity(self.shape);

        for i in 0..self.shape {
            delta.push(sample[i] as f64 - self.running_mean[i]);
            delta_n.push(delta[i] / self.count as f64);
        }

        for i in 0..self.shape {
            self.running_mean[i] += delta_n[i];
            self.running_variance[i] += delta[i] * delta_n[i] * current_count;
        }
    }

    pub fn increment(&mut self, samples: &[f32], num: usize) {
        for sample in samples.chunks(1).take(num) {
            self.update(sample);
        }
    }

    pub fn get_std(&self) -> Vec<f32> {
        if self.count < 2 {
            return self.ones.clone();
        }

        self.running_variance
            .iter()
            .map(|var| {
                let cur_var = var / (self.count - 1) as f64;
                if cur_var == 0.0 {
                    1.0
                } else {
                    cur_var.sqrt() as f32
                }
            })
            .collect()
    }
}

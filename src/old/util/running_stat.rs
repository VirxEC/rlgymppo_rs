use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Default, Clone, Copy)]
pub struct WelfordRunningStat {
    running_mean: f64,
    running_variance: f64,
    count: usize,
}

impl WelfordRunningStat {
    fn update(&mut self, sample: f64) {
        let current_count = self.count as f64;
        self.count += 1;

        let delta = sample - self.running_mean;
        let delta_n = delta / self.count as f64;

        self.running_mean += delta_n;
        self.running_variance += delta * delta_n * current_count;
    }

    pub fn increment(&mut self, samples: Vec<f32>, num: usize) {
        for sample in samples.into_iter().take(num) {
            self.update(sample as f64);
        }
    }

    pub fn get_std(&self) -> f32 {
        if self.count < 2 {
            return 1.0;
        }

        let cur_var = self.running_variance / (self.count - 1) as f64;
        if cur_var == 0.0 {
            1.0
        } else {
            cur_var.sqrt() as f32
        }
    }
}

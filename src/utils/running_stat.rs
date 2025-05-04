use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Stats {
    pub cumulative_timesteps: u64,
    pub cumulative_model_updates: u64,
    pub cumulative_epochs: u64,
    pub return_stat: WelfordRunningStat,
    // todo, for wandb metrics reporting
    // run_id: String,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone, Copy)]
pub struct WelfordRunningStat {
    mean: f64,
    n: usize,
    m2: f64,
}

impl WelfordRunningStat {
    fn update(&mut self, sample: f64) {
        self.n += 1;
        let delta = sample - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = sample - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn increment(&mut self, samples: Vec<f32>) {
        for sample in samples {
            self.update(sample as f64);
        }
    }

    pub fn get_std(&self) -> f32 {
        if self.n < 2 {
            return 1.0;
        }

        let cur_var = self.m2 / (self.n - 1) as f64;
        if cur_var == 0.0 {
            1.0
        } else {
            cur_var.sqrt() as f32
        }
    }
}

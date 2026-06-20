use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Saved wandb run ID for checkpoint/resume.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct WandbRun {
    pub run_id: String,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Stats {
    pub cumulative_timesteps: u64,
    pub cumulative_model_updates: u64,
    pub cumulative_epochs: u64,
    pub return_stat: WelfordRunningStat,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wandb_run: Option<WandbRun>,
    /// Per-mode Elo ratings from the skill tracker (e.g. `"1v1"`, `"2v2"`).
    /// `None` when the skill tracker is disabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skill_ratings: Option<HashMap<String, f32>>,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone, Copy)]
pub struct WelfordRunningStat {
    mean: f64,
    mean_sqr: f64,
    count: u64,
}

impl WelfordRunningStat {
    fn update(&mut self, sample: f64) {
        self.count += 1;
        let delta = sample - self.mean;
        self.mean += delta / self.count as f64;
        let delta_2 = sample - self.mean;
        self.mean_sqr += delta * delta_2;
    }

    pub fn increment(&mut self, samples: Vec<f32>) {
        for sample in samples {
            self.update(sample as f64);
        }
    }

    pub fn get_std(&self) -> f32 {
        if self.count < 2 {
            return 1.0;
        }

        let cur_var = self.mean_sqr / (self.count - 1) as f64;
        if cur_var == 0.0 {
            1.0
        } else {
            cur_var.sqrt() as f32
        }
    }
}

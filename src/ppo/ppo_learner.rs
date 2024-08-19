use super::{discrete::DiscretePolicy, exp_buf::ExperienceBuffer, value_est::ValueEstimator};
use crate::util::{compute::NonBlockingTransfer, report::Report};
use std::{
    sync::Arc,
    thread::{self, available_parallelism},
    time::Instant,
};
use tch::{
    nn::{self, OptimizerConfig},
    no_grad_guard, Device, Kind, Reduction, Tensor,
};

fn var_store_to_tensor(vs: &nn::VarStore) -> Tensor {
    let mut vars: Vec<_> = vs.variables().into_iter().collect();
    // Ensure deterministic order
    vars.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let tensors: Vec<_> = vars.iter().map(|(_, t)| t.flatten(0, -1)).collect();
    Tensor::cat(&tensors, 0)
}

#[derive(Default)]
struct TrainMetrics {
    entropy: f64,
    divergence: f64,
    val_loss: f64,
    ratio: f64,
    clip_fraction: Option<f64>,
    backprop_time: f64,
    value_est_time: f64,
    gradient_time: f64,
}

#[derive(Debug, Clone)]
pub struct PPOLearnerConfig {
    pub policy_layer_sizes: Vec<i64>,
    pub critic_layer_sizes: Vec<i64>,
    pub batch_size: u64,
    pub epochs: u8,
    /// Policy learning rate
    pub policy_lr: f64,
    /// Critic learning rate
    pub critic_lr: f64,
    /// Entropy coefficient
    pub ent_coef: f64,
    pub clip_range: f64,
    /// Set to 0 to just use batchSize
    pub mini_batch_size: u64,
    // /// https://openai.com/index/how-ai-training-scales/
    // /// Measures the noise of both policy and critic gradients every epoch
    // pub measure_gradient_noise: bool,
    // pub gradient_noise_update_interval: i64,
    // pub gradient_noise_avg_decay: f32,
}

impl Default for PPOLearnerConfig {
    fn default() -> Self {
        Self {
            policy_layer_sizes: vec![256, 256, 256],
            critic_layer_sizes: vec![256, 256, 256],
            batch_size: 50_000,
            epochs: 10,
            policy_lr: 3e-4,
            critic_lr: 3e-4,
            ent_coef: 0.005,
            clip_range: 0.2,
            mini_batch_size: 0,
            // measure_gradient_noise: false,
            // gradient_noise_update_interval: 10,
            // gradient_noise_avg_decay: 0.9925,
        }
    }
}

pub struct PPOLearner {
    policy: Arc<DiscretePolicy>,
    value_net: ValueEstimator,
    policy_optimizer: nn::Optimizer,
    value_optimizer: nn::Optimizer,
    policy_store: nn::VarStore,
    value_store: nn::VarStore,
    // noiseTrackerPolicy: GradNoiseTracker,
    // noiseTrackerValueNet: GradNoiseTracker,
    value_loss_fn: Reduction,
    config: PPOLearnerConfig,
    cumulative_model_updates: u64,
    device: Device,
}

impl PPOLearner {
    pub fn new(
        obs_space_size: usize,
        action_space_size: usize,
        mut config: PPOLearnerConfig,
        device: Device,
    ) -> Self {
        if config.mini_batch_size == 0 {
            config.mini_batch_size = config.batch_size;
        }

        if config.batch_size % config.mini_batch_size != 0 {
            panic!("Batch size must be divisible by mini batch size");
        }

        let policy_store = nn::VarStore::new(device);
        let policy = DiscretePolicy::new(
            obs_space_size as i64,
            action_space_size as i64,
            &config.policy_layer_sizes,
            policy_store.root(),
            device,
        );
        let policy_optimizer = nn::Adam::default()
            .build(&policy_store, config.policy_lr)
            .unwrap();

        let value_store = nn::VarStore::new(device);
        let value_net = ValueEstimator::new(
            obs_space_size as i64,
            &config.critic_layer_sizes,
            value_store.root(),
            device,
        );
        let value_optimizer = nn::Adam::default()
            .build(&value_store, config.critic_lr)
            .unwrap();

        // grad noise stuff?

        Self {
            policy: Arc::new(policy),
            value_net,
            policy_optimizer,
            value_optimizer,
            policy_store,
            value_store,
            value_loss_fn: Reduction::Mean,
            config,
            cumulative_model_updates: 0,
            device,
        }
    }

    pub fn get_policy(&self) -> Arc<DiscretePolicy> {
        self.policy.clone()
    }

    pub fn get_value_net(&self) -> &ValueEstimator {
        &self.value_net
    }

    pub fn learn(&mut self, exp_buffer: &mut ExperienceBuffer, mut report: &mut Report) {
        let mut num_mini_batch_iterations = 0;
        let mut mean_entropy: f64 = 0.;
        let mut mean_divergence: f64 = 0.;
        let mut mean_val_loss: f64 = 0.;
        let mut mean_ratio: f64 = 0.;
        let mut clip_fractions: Vec<f64> = Vec::new();

        let policy_before = var_store_to_tensor(&self.policy_store);
        let critic_before = var_store_to_tensor(&self.value_store);

        let train_policy = self.config.policy_lr > 0.0;
        let train_critic = self.config.critic_lr > 0.0;

        let train_time_start = Instant::now();
        for _ in 0..self.config.epochs {
            // Get randomly-ordered timesteps for PPO
            let batches = exp_buffer.get_all_batches_shuffled(self.config.batch_size);

            for batch in batches {
                let batch_acts = batch.actions.view([self.config.batch_size as i64, -1]);
                self.policy_optimizer.zero_grad();
                self.value_optimizer.zero_grad();

                let run_minibatch = |batch_size_ratio: f64,
                                     acts: Tensor,
                                     obs: Tensor,
                                     advantages: Tensor,
                                     old_probs: Tensor,
                                     target_values: Tensor| {
                    let mut metrics = TrainMetrics::default();

                    let forward_time_start = Instant::now();
                    let mut vals = self.value_net.forward(&obs, true);
                    let forward_time_elapsed = forward_time_start.elapsed();
                    metrics.value_est_time = forward_time_elapsed.as_secs_f64();

                    let backward_time_start = Instant::now();

                    let mut train_policy_data = None;
                    if train_policy {
                        // Get policy log probs & entropy
                        let bp_result = self.policy.get_backprop_data(&obs, &acts);

                        let mut log_probs = bp_result.action_log_probs;
                        let entropy = bp_result.entropy;

                        log_probs = log_probs.view_as(&old_probs);

                        let backward_time_elapsed = backward_time_start.elapsed();
                        metrics.backprop_time = backward_time_elapsed.as_secs_f64();

                        // Compute PPO loss
                        let ratio = (&log_probs - &old_probs).exp();
                        metrics.ratio = ratio
                            .mean(Kind::Float)
                            .detach()
                            .to(Device::Cpu)
                            .double_value(&[]);
                        let clipped =
                            ratio.clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range);

                        // Compute policy loss
                        let policy_loss = -(&ratio * &advantages)
                            .minimum(&(&clipped * &advantages))
                            .mean(Kind::Float);
                        let ppo_loss =
                            (&policy_loss - &entropy * self.config.ent_coef) * batch_size_ratio;
                        metrics.entropy = entropy.detach().to(Device::Cpu).double_value(&[]);

                        train_policy_data = Some((log_probs, ratio, ppo_loss));
                    }

                    let mut value_loss = None;
                    if train_critic {
                        // Compute value loss
                        vals = vals.view_as(&target_values);
                        value_loss = Some(
                            vals.mse_loss(&target_values, self.value_loss_fn) * batch_size_ratio,
                        );
                    }

                    if let Some((log_probs, ratio, _)) = &train_policy_data {
                        // Compute KL divergence & clip fraction using SB3 method for reporting
                        let _no_grad = no_grad_guard();

                        let log_ratio = log_probs - old_probs;
                        let kl_tensor = (&log_ratio.exp() - 1.0) - &log_ratio;
                        let kl = kl_tensor
                            .mean(Kind::Float)
                            .detach()
                            .to(Device::Cpu)
                            .double_value(&[]);
                        metrics.divergence = kl;

                        let clip_fraction = (ratio - 1.0)
                            .abs()
                            .gt(self.config.clip_range)
                            .to_kind(Kind::Float)
                            .mean(Kind::Float)
                            .to(Device::Cpu)
                            .double_value(&[]);
                        metrics.clip_fraction = Some(clip_fraction);
                    }

                    if let Some((_, _, ppo_loss)) = train_policy_data {
                        ppo_loss.backward();
                    }

                    if let Some(value_loss) = value_loss {
                        value_loss.backward();
                        metrics.val_loss = value_loss.detach().to(Device::Cpu).double_value(&[]);
                    }

                    let gradient_time_elapsed = backward_time_start.elapsed();
                    metrics.gradient_time = gradient_time_elapsed.as_secs_f64();

                    metrics
                };

                let reporter = &mut report;
                if self.device.is_cuda() {
                    for start in (0..self.config.batch_size as i64)
                        .step_by(self.config.mini_batch_size as usize)
                    {
                        let stop = start + self.config.mini_batch_size as i64;
                        let batch_size_ratio =
                            (stop - start) as f64 / self.config.batch_size as f64;

                        // Send everything to the device and enforce correct shapes
                        let acts = batch_acts.slice(0, start, stop, 1).no_block_to(self.device);
                        let obs = batch
                            .states
                            .slice(0, start, stop, 1)
                            .no_block_to(self.device);

                        let advantages = batch
                            .advantages
                            .slice(0, start, stop, 1)
                            .no_block_to(self.device);
                        let old_probs = batch
                            .log_probs
                            .slice(0, start, stop, 1)
                            .no_block_to(self.device);
                        let target_values = batch
                            .values
                            .slice(0, start, stop, 1)
                            .no_block_to(self.device);

                        let metrics = run_minibatch(
                            batch_size_ratio,
                            acts,
                            obs,
                            advantages,
                            old_probs,
                            target_values,
                        );

                        // Update metrics
                        mean_entropy += metrics.entropy;
                        mean_divergence += metrics.divergence;
                        mean_val_loss += metrics.val_loss;
                        mean_ratio += metrics.ratio;

                        if let Some(clip_fraction) = metrics.clip_fraction {
                            clip_fractions.push(clip_fraction);
                        }

                        reporter["PPO Backprop Time"] += metrics.backprop_time;
                        reporter["PPO Value Estimate Time"] += metrics.value_est_time;
                        reporter["PPO Gradient Time"] += metrics.gradient_time;

                        num_mini_batch_iterations += 1;
                    }
                } else {
                    thread::scope(|s| {
                        let mut handles = Vec::new();
                        let real_batch_size = self.config.batch_size as i64
                            / available_parallelism().unwrap().get() as i64;

                        for start in
                            (0..self.config.batch_size as i64).step_by(real_batch_size as usize)
                        {
                            let stop = start + real_batch_size;
                            let batch_size_ratio =
                                (stop - start) as f64 / self.config.batch_size as f64;

                            // Send everything to the device and enforce correct shapes
                            let acts = batch_acts.slice(0, start, stop, 1).no_block_to(self.device);
                            let obs = batch
                                .states
                                .slice(0, start, stop, 1)
                                .no_block_to(self.device);

                            let advantages = batch
                                .advantages
                                .slice(0, start, stop, 1)
                                .no_block_to(self.device);
                            let old_probs = batch
                                .log_probs
                                .slice(0, start, stop, 1)
                                .no_block_to(self.device);
                            let target_values = batch
                                .values
                                .slice(0, start, stop, 1)
                                .no_block_to(self.device);

                            handles.push(s.spawn(move || {
                                run_minibatch(
                                    batch_size_ratio,
                                    acts,
                                    obs,
                                    advantages,
                                    old_probs,
                                    target_values,
                                )
                            }));
                        }

                        for handle in handles {
                            let metrics = handle.join().unwrap();

                            // Update metrics
                            mean_entropy += metrics.entropy;
                            mean_divergence += metrics.divergence;
                            mean_val_loss += metrics.val_loss;
                            mean_ratio += metrics.ratio;

                            if let Some(clip_fraction) = metrics.clip_fraction {
                                clip_fractions.push(clip_fraction);
                            }

                            reporter["PPO Backprop Time"] += metrics.backprop_time;
                            reporter["PPO Value Estimate Time"] += metrics.value_est_time;
                            reporter["PPO Gradient Time"] += metrics.gradient_time;

                            num_mini_batch_iterations += 1;
                        }
                    })
                }
            }

            if train_policy {
                self.policy_optimizer.clip_grad_norm(0.5);
                self.policy_optimizer.step();
            }

            if train_critic {
                self.value_optimizer.clip_grad_norm(0.5);
                self.value_optimizer.step();
            }
        }

        let num_iterations = self.config.epochs.max(1) as f64;
        let num_mini_batch_iterations = num_mini_batch_iterations.max(1) as f64;

        mean_entropy /= num_mini_batch_iterations;
        mean_divergence /= num_mini_batch_iterations;
        mean_val_loss /= num_mini_batch_iterations;
        mean_ratio /= num_mini_batch_iterations;

        let mean_clip = if clip_fractions.is_empty() {
            0.
        } else {
            clip_fractions.iter().sum::<f64>() / clip_fractions.len() as f64
        };

        let policy_after = var_store_to_tensor(&self.policy_store);
        let critic_after = var_store_to_tensor(&self.value_store);

        let policy_update_magnitude = (policy_after - policy_before).norm().double_value(&[]);
        let critic_update_magnitude = (critic_after - critic_before).norm().double_value(&[]);

        let total_time_elapsed = train_time_start.elapsed();

        self.cumulative_model_updates += u64::from(self.config.epochs);
        report["PPO Batch Consumption Time"] = total_time_elapsed.as_secs_f64() / num_iterations;
        report["Cumulative Model Updates"] = self.cumulative_model_updates as f64;
        report["Policy Entropy"] += mean_entropy;
        report["Mean KL Divergence"] += mean_divergence;
        report["Mean Ratio"] += mean_ratio;
        report["Value Function Loss"] += mean_val_loss;
        report["SB3 Clip Fraction"] += mean_clip;
        report["Policy Update Magnitude"] += policy_update_magnitude;
        report["Value Function Update Magnitude"] += critic_update_magnitude;
        report["PPO Learn Time"] += total_time_elapsed.as_secs_f64();

        self.policy_optimizer.zero_grad();
        self.value_optimizer.zero_grad();
    }
}

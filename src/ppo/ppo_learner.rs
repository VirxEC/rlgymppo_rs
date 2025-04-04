use super::{discrete::DiscretePolicy, exp_buf::ExperienceBuffer, value_est::ValueEstimator};
use crate::util::{compute::NonBlockingTransfer, report::Report};
use std::{
    fs::{create_dir, File},
    io::Write,
    path::Path,
    sync::Arc,
    thread::{self, available_parallelism},
    time::Instant,
};
use tch::{
    nn::{self, OptimizerConfig},
    no_grad, no_grad_guard, with_grad, Device, Kind, Reduction, Tensor,
};

fn var_store_to_tensor(vs: &nn::VarStore) -> Tensor {
    let mut vars: Vec<_> = vs.variables().into_iter().collect();
    // Ensure deterministic order
    vars.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let tensors: Vec<_> = vars.iter().map(|(_, t)| t.flatten(0, -1)).collect();
    Tensor::cat(&tensors, 0)
}

pub fn store_vs(vs: &nn::VarStore, file_name: &str) {
    let mut flat = Vec::new();

    let mut vars: Vec<_> = vs.variables().into_iter().collect();
    // Ensure deterministic order
    vars.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    for (_, t) in vars {
        flat.push(Vec::<f32>::try_from(t.flatten(0, -1)).unwrap());
    }

    let _ = create_dir("out");
    let mut file = File::create(format!("out/{}.txt", file_name)).unwrap();

    for tensor in flat {
        for val in tensor {
            write!(file, "{} ", val).unwrap();
        }
        writeln!(file).unwrap();
    }
}

pub fn store_vs_grad(vs: &nn::VarStore, file_name: &str) {
    let mut flat = Vec::new();

    let mut vars: Vec<_> = vs.variables().into_iter().collect();
    // Ensure deterministic order
    vars.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    for (_, t) in vars {
        let mut grad = t.grad();

        if grad.defined() {
            grad = t
                .grad()
                .to(Device::Cpu)
                .detach()
                .ravel()
                .to_kind(Kind::Float);
            flat.push(Vec::<f32>::try_from(grad).unwrap());
        } else {
            let zeros = Tensor::zeros_like(&t);
            flat.push(Vec::<f32>::try_from(zeros.ravel()).unwrap());
        }
    }

    let _ = create_dir("out");
    let mut file = File::create(format!("out/{}.txt", file_name)).unwrap();

    for grad in flat {
        for val in grad {
            write!(file, "{} ", val).unwrap();
        }
        writeln!(file).unwrap();
    }
}

#[track_caller]
pub fn store_t_1d(t: &Tensor, file_name: &str) {
    let cpu_t = t.to(Device::Cpu).detach().to_kind(Kind::Float);

    let _ = create_dir("out");
    let mut file = File::create(format!("out/{}.txt", file_name)).unwrap();

    for val in Vec::<f32>::try_from(cpu_t).unwrap() {
        write!(file, "{:.18} ", val).unwrap();
    }
    writeln!(file).unwrap();
}

#[track_caller]
pub fn store_t_2d(t: &Tensor, file_name: &str) {
    let cpu_t = t.to(Device::Cpu).detach();

    let _ = create_dir("out");
    let mut file = File::create(format!("out/{}.txt", file_name)).unwrap();

    for row in 0..cpu_t.size()[0] {
        let row = cpu_t.get(row);
        for val in Vec::<f32>::try_from(row).unwrap() {
            write!(file, "{:.18} ", val).unwrap();
        }
        writeln!(file).unwrap();
    }
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
    backwards_time: f64,
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
    config: PPOLearnerConfig,
    pub cumulative_model_updates: u64,
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

        // print the number of parameters in the policy network
        let num_params = policy_store
            .variables()
            .values()
            .map(Tensor::numel)
            .sum::<usize>();
        println!("\tPolicy network has {num_params} parameters");

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

        let num_params = value_store
            .variables()
            .values()
            .map(Tensor::numel)
            .sum::<usize>();
        println!("\tCritic network has {num_params} parameters");

        let value_optimizer = nn::Adam::default()
            .build(&value_store, config.critic_lr)
            .unwrap();

        Self {
            policy: Arc::new(policy),
            value_net,
            policy_optimizer,
            value_optimizer,
            policy_store,
            value_store,
            config,
            cumulative_model_updates: 0,
            device,
        }
    }

    /// https://github.com/LaurentMazare/tch-rs?tab=readme-ov-file#importing-pre-trained-weights-from-pytorch-using-safetensors
    /// > `safetensors` is a new simple format by HuggingFace for storing tensors.
    /// > It does not rely on Python's pickle module,
    /// > and therefore the tensors are not bound to the specific classes
    /// > and the exact directory structure used when the model is saved.
    /// > It is also zero-copy, which means that reading the file will
    /// > require no more memory than the original file.
    ///
    /// The link above shows how to load a safetensors file in Python.
    pub fn load<P: AsRef<Path>>(&mut self, path: P) {
        let path = path.as_ref();
        self.policy_store
            .load(path.join("policy.safetensors"))
            .unwrap();
        self.value_store
            .load(path.join("critic.safetensors"))
            .unwrap();
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) {
        let path = path.as_ref();
        self.policy_store
            .save(path.join("policy.safetensors"))
            .unwrap();
        self.value_store
            .save(path.join("critic.safetensors"))
            .unwrap();
    }

    pub fn get_policy(&self) -> Arc<DiscretePolicy> {
        self.policy.clone()
    }

    pub fn get_value_net(&self) -> &ValueEstimator {
        &self.value_net
    }

    pub fn learn(&mut self, exp_buffer: &mut ExperienceBuffer, report: &mut Report) {
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
                    let mut vals = with_grad(|| self.value_net.forward(&obs, true));
                    let forward_time_elapsed = forward_time_start.elapsed();
                    metrics.value_est_time = forward_time_elapsed.as_secs_f64();

                    let backward_time_start = Instant::now();

                    let mut train_policy_data = None;
                    if train_policy {
                        train_policy_data = Some(with_grad(|| {
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
                            let clipped = ratio
                                .clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range);

                            // Compute KL divergence & clip fraction using SB3 method for reporting
                            no_grad(|| {
                                let log_ratio = log_probs - old_probs;
                                let kl_tensor = (&log_ratio.exp() - 1.0) - &log_ratio;
                                let kl = kl_tensor
                                    .mean(Kind::Float)
                                    .detach()
                                    .to(Device::Cpu)
                                    .double_value(&[]);
                                metrics.divergence = kl;

                                let clip_fraction = (&ratio - 1.0)
                                    .abs()
                                    .gt(self.config.clip_range)
                                    .to_kind(Kind::Float)
                                    .mean(None)
                                    .to(Device::Cpu)
                                    .double_value(&[]);
                                metrics.clip_fraction = Some(clip_fraction);
                            });

                            // Compute policy loss
                            let policy_loss = -(ratio * &advantages)
                                .minimum(&(clipped * advantages))
                                .mean(None);

                            let ppo_loss =
                                (&policy_loss - &entropy * self.config.ent_coef) * batch_size_ratio;
                            metrics.entropy = entropy.detach().to(Device::Cpu).double_value(&[]);

                            ppo_loss
                        }));
                    }

                    let mut value_loss = None;
                    if train_critic {
                        // Compute value loss
                        value_loss = Some(with_grad(|| {
                            vals = vals.view_as(&target_values);
                            vals.mse_loss(&target_values, Reduction::Mean)
                        }));
                    }

                    let gradient_time_elapsed = backward_time_start.elapsed();
                    metrics.gradient_time = gradient_time_elapsed.as_secs_f64();
                    let backwards_start = Instant::now();

                    with_grad(|| {
                        if let Some(ppo_loss) = train_policy_data {
                            ppo_loss.backward();
                        }

                        if let Some(value_loss) = &value_loss {
                            value_loss.backward();
                            metrics.val_loss =
                                value_loss.detach().to(Device::Cpu).double_value(&[]);
                        }
                    });

                    let backward_time_elapsed = backwards_start.elapsed();
                    metrics.backwards_time = backward_time_elapsed.as_secs_f64();

                    metrics
                };

                if self.device.is_cuda() {
                    let mini_batch_size = self.config.mini_batch_size as i64;

                    for start in (0..self.config.batch_size as i64)
                        .step_by(self.config.mini_batch_size as usize)
                    {
                        let stop = start + mini_batch_size;
                        let batch_size_ratio =
                            (stop - start) as f64 / self.config.batch_size as f64;

                        let acts = batch_acts.slice(0, start, stop, 1).to_device_(
                            self.device,
                            Kind::Float,
                            true,
                            true,
                        );
                        let obs = batch.states.slice(0, start, stop, 1).to_device_(
                            self.device,
                            Kind::Float,
                            true,
                            true,
                        );
                        let advantages = batch.advantages.slice(0, start, stop, 1).to_device_(
                            self.device,
                            Kind::Float,
                            true,
                            true,
                        );
                        let old_probs = batch.log_probs.slice(0, start, stop, 1).to_device_(
                            self.device,
                            Kind::Float,
                            true,
                            true,
                        );
                        let target_values = batch.values.slice(0, start, stop, 1).to_device_(
                            self.device,
                            Kind::Float,
                            true,
                            true,
                        );

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

                        report["PPO Backward Time"] += metrics.backwards_time.into();
                        report["PPO Backprop Time"] += metrics.backprop_time.into();
                        report["PPO Value Estimate Time"] += metrics.value_est_time.into();
                        report["PPO Gradient Time"] += metrics.gradient_time.into();

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

                            report["PPO Backprop Time"] += metrics.backprop_time.into();
                            report["PPO Value Estimate Time"] += metrics.value_est_time.into();
                            report["PPO Gradient Time"] += metrics.gradient_time.into();

                            num_mini_batch_iterations += 1;
                        }
                    })
                }

                if train_policy {
                    self.policy_optimizer.clip_grad_norm(0.5);
                }

                if train_critic {
                    self.value_optimizer.clip_grad_norm(0.5);
                }

                if train_policy {
                    self.policy_optimizer.step();
                }

                if train_critic {
                    self.value_optimizer.step();
                }
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
        report["PPO Batch Consumption Time"] =
            (total_time_elapsed.as_secs_f64() / num_iterations).into();
        report["Cumulative Model Updates"] = self.cumulative_model_updates.into();
        report["Policy Entropy"] += mean_entropy.into();
        report["Mean KL Divergence"] += mean_divergence.into();
        report["Mean Ratio"] += mean_ratio.into();
        report["Value Function Loss"] += mean_val_loss.into();
        report["SB3 Clip Fraction"] += mean_clip.into();
        report["Policy Update Magnitude"] += policy_update_magnitude.into();
        report["Value Function Update Magnitude"] += critic_update_magnitude.into();
        report["PPO Learn Time"] += total_time_elapsed.as_secs_f64().into();

        self.policy_optimizer.zero_grad();
        self.value_optimizer.zero_grad();
    }
}

use std::path::PathBuf;
use std::time::{Duration, Instant};

use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{DType, FloatDType, Transaction};
use rlgymppo_model::Policy;
use rlgymppo_utils::Report;

use super::{flatten_net, l2_diff};
use crate::NormSelection;
use crate::agent::Ppo;
use crate::agent::config::{LrSchedule, RegenConfig};
use crate::agent::model::{Actic, Net};
use crate::base::{Memory, get_states_batch_range};

/// What the teacher (old, larger) policy is: its architecture and where its
/// checkpoints live. Passed to `Learner::transfer_learn` alongside a
/// [`TransferLearnConfig`] holding the distillation hyperparameters.
///
/// The teacher and student must share the same action space, but the
/// observation space may differ when the learner was initialized with an old
/// (teacher) obs builder via [`LearnerConfig::init_with_old_obs`] — that
/// builder runs alongside the student's and scores the same states with its
/// own layout.
#[derive(Clone, Debug)]
pub struct TeacherConfig {
    /// Path to the directory containing the teacher model checkpoint(s).
    ///
    /// May point directly at a checkpoint folder containing `actor.mpk.gz`, or
    /// at a parent directory of timestamped checkpoints (the latest one is
    /// used). Do not point this at the directory this run saves its own
    /// checkpoints to, or a later run may pick the student up as the teacher.
    pub models_path: PathBuf,
    /// Layer sizes of the teacher's actor network (the old, larger model).
    pub policy_layer_sizes: Vec<usize>,
    /// Layer sizes of the teacher's shared feature head (empty if it had none).
    pub shared_head_layer_sizes: Vec<usize>,
    /// Normalization used by the teacher network.
    pub norm: NormSelection,
}

impl Default for TeacherConfig {
    fn default() -> Self {
        Self {
            models_path: PathBuf::new(),
            policy_layer_sizes: Vec::new(),
            shared_head_layer_sizes: Vec::new(),
            norm: NormSelection::RmsNorm,
        }
    }
}

impl TeacherConfig {
    pub fn validate(&self) {
        assert!(
            !self.policy_layer_sizes.is_empty(),
            "policy_layer_sizes must not be empty"
        );
    }
}

/// Hyperparameters for the distillation phase of transfer learning: training
/// the student's actor and shared head to match a frozen, already-trained
/// teacher policy (see [`TeacherConfig`]).
///
/// The student acts in the environment while the teacher, frozen, scores the
/// same states; the student's actor and shared head are trained to match the
/// teacher's action distribution. The critic is left untouched.
#[derive(Clone, Debug)]
pub struct TransferLearnConfig {
    /// Learning rate for the distillation optimizer steps.
    pub lr: f32,
    /// Timesteps collected from the environment before each distillation
    /// iteration.
    pub batch_size: usize,
    /// Timesteps staged on the GPU at once during teacher inference and
    /// student gradient accumulation. Must divide `batch_size`.
    pub mini_batch_size: usize,
    /// Distillation epochs per collected batch.
    pub epochs: usize,
    /// Use forward KL divergence instead of mean absolute difference.
    ///
    /// When enabled the loss is the per-state forward KL
    /// `sum_a q(a) * (log q(a) - log p(a))` averaged over states (then scaled
    /// by `loss_scale` and `teacher_temperature^2`, the standard
    /// temperature-scaled KD rule). When disabled (the default) the loss is
    /// the mean absolute difference over all state-action entries.
    pub use_kl_div: bool,
    /// Scale of the distillation loss. The natural loss is very small, so this
    /// prevents the optimizers from effectively dying.
    pub loss_scale: f32,
    /// Exponent applied to each sample's loss before averaging.
    pub loss_exponent: f32,
    /// Learning-rate schedule applied per distillation round.
    /// `Constant` (the default) preserves the historical flat rate.
    pub lr_schedule: LrSchedule,
    /// Regenerative regularization toward initialization, applied to the
    /// student actor and shared head only (the critic is never touched by
    /// distillation). `None` (the default) disables it.
    pub regen: Option<RegenConfig>,
    /// Temperature applied to both the teacher targets and the student's
    /// training distribution. `1.0` (the default) preserves the historical
    /// raw distributions; higher values (e.g. `2.0`) soften both sides so
    /// small students get denser gradients. The student still acts and reports
    /// entropy at temperature `1.0` (deployment distribution); only the
    /// training loss uses the softened pair. Must be positive and finite.
    ///
    /// For forward KL the loss is additionally scaled by `temperature^2`
    /// (standard temperature-scaled KD); the L1 path is not rescaled.
    pub teacher_temperature: f32,
}

impl Default for TransferLearnConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            batch_size: 50_000,
            mini_batch_size: 10_000,
            epochs: 2,
            use_kl_div: false,
            loss_scale: 500.0,
            loss_exponent: 1.0,
            lr_schedule: LrSchedule::Constant,
            regen: None,
            teacher_temperature: 1.0,
        }
    }
}

impl TransferLearnConfig {
    pub fn validate(&self) {
        assert!(self.batch_size > 0, "batch size must be greater than zero");
        assert!(
            self.mini_batch_size > 0,
            "mini batch size must be greater than zero"
        );
        assert!(self.epochs > 0, "epochs must be greater than zero");
        assert!(self.lr > 0.0, "learning rate must be greater than zero");
        assert!(
            self.loss_scale > 0.0,
            "loss scale must be greater than zero"
        );
        assert!(
            self.loss_exponent >= 0.0 && self.loss_exponent.is_finite(),
            "loss exponent must be finite and non-negative"
        );
        assert!(
            self.teacher_temperature > 0.0 && self.teacher_temperature.is_finite(),
            "teacher temperature must be positive and finite"
        );
        assert_eq!(
            self.batch_size % self.mini_batch_size,
            0,
            "batch size must be divisible by mini batch size"
        );
    }
}

/// Distillation loss for one chunk (`[batch, actions]` probability tensors).
///
/// - L1 (default, `use_kl_div = false`): mean absolute difference over all
///   state-action entries, then `* loss_scale`. Preserves the historical
///   reduction and has no temperature rescaling.
/// - Forward KL (`use_kl_div = true`): true per-state
///   `sum_a q * (log q - log p)` averaged over states (stable log-difference
///   form; inference clamps both sides to `[1e-11, 1]` so logs are finite),
///   clamped at `0` before `loss_exponent` so float rounding (`-1e-9`) with a
///   fractional exponent cannot produce `NaN`. Standard `T^2` KD scaling.
///   An exponent of `1.0` skips the extra pow kernel.
///
/// Masked-out actions contribute ~`0` on both sides (same masks), so the KL
/// stays nonnegative and finite there.
pub(crate) fn distillation_loss<B: AutodiffBackend>(
    teacher: Tensor<B, 2>,
    student: Tensor<B, 2>,
    use_kl_div: bool,
    loss_scale: f32,
    loss_exponent: f32,
    temperature: f32,
) -> Tensor<B, 1> {
    if use_kl_div {
        let per_state = (teacher.clone() * (teacher.log() - student.log()))
            .sum_dim(1)
            .clamp(0.0, f32::MAX);
        let averaged = if loss_exponent == 1.0 {
            per_state.mean()
        } else {
            per_state.powf_scalar(loss_exponent).mean()
        };
        averaged * loss_scale * temperature * temperature
    } else {
        let per_sample = (teacher - student).abs();
        if loss_exponent == 1.0 {
            per_sample.mean() * loss_scale
        } else {
            per_sample.powf_scalar(loss_exponent).mean() * loss_scale
        }
    }
}

impl<B: AutodiffBackend, O: Optimizer<Net<B>, B>> Ppo<B, O> {
    /// Distill the frozen teacher policy's action distribution into the
    /// student `net`.
    ///
    /// The teacher's softmax action probabilities are computed once per
    /// collected batch (no gradients), then `epochs` of gradient descent train
    /// the student's actor and shared head to match them. The critic and its
    /// optimizer are untouched.
    ///
    /// States upload once as `f16` and stay resident on the GPU (same as PPO
    /// training): teacher inference and every distillation epoch slice them
    /// with `narrow` and upcast on the device. Masks upload once as `u8` and
    /// upcast per slice. Chunks are contiguous, so no index build, no
    /// `select` gather, and no re-upload per epoch.
    ///
    /// Returns the updated network and the number of timesteps consumed.
    pub fn transfer_learn(
        &mut self,
        mut net: Actic<B>,
        teacher: &Policy<B::InnerBackend>,
        memory: &Memory,
        tl: &TransferLearnConfig,
        metrics: &mut Report,
    ) -> (Actic<B>, usize) {
        let n = memory.len();
        assert!(n > 0, "Cannot distill from an empty memory");
        tl.validate();
        memory
            .validate()
            .unwrap_or_else(|error| panic!("Invalid learner memory: {error}"));
        let mb = tl.mini_batch_size;

        // Fix the regen anchor to the pre-update weights before any optimizer
        // step (preserved across checkpoints; `Learner::transfer_learn` warns
        // about legacy resumes with stats context).
        self.ensure_regen_snapshot(&net, tl.regen);

        // Snapshot actor + shared-head parameters before training for
        // update-magnitude computation.
        let snapshot_start = Instant::now();
        let actor_params_before = flatten_net(&net.actor);
        let head_params_before = net.shared_head.as_ref().map(flatten_net);
        metrics["Transfer/param snapshot time"] = snapshot_start.elapsed().as_secs_f64().into();
        // Effective LR for this round (schedule runs on completed rounds).
        let effective_lr = (f64::from(tl.lr) * tl.lr_schedule.scale(self.transfer_rounds)) as f32;
        metrics["Transfer/lr"] = f64::from(effective_lr).into();

        // Teacher observations: the old (teacher) obs builder's states when one
        // is configured (different observation space), otherwise the student's
        // own states (same observation space).
        let (teacher_states, teacher_width) = if memory.old_state_width() > 0 {
            (memory.old_states(), memory.old_state_width())
        } else {
            (memory.states(), memory.state_width())
        };

        // ── Resident upload (timed as staging) ────────────────────────
        // One `f32` to `f16` conversion plus one upload per state buffer, one
        // `u8` upload for the masks. Everything below slices these with
        // `narrow`: no per-chunk and no per-epoch re-upload.
        let resident_start = Instant::now();
        let resident_teacher = get_states_batch_range::<B::InnerBackend>(
            teacher_states,
            teacher_width,
            0,
            n,
            &self.device,
        );
        // Same obs space: share the teacher storage for the student (no copy,
        // half the VRAM). Different spaces: upload the student buffer too.
        let resident_student_inner = if memory.old_state_width() > 0 {
            get_states_batch_range::<B::InnerBackend>(
                memory.states(),
                memory.state_width(),
                0,
                n,
                &self.device,
            )
        } else {
            resident_teacher.clone()
        };
        let mask_width = memory.action_mask_width();
        let resident_masks_inner = if memory.action_masks().is_empty() {
            None
        } else {
            let masks: Vec<u8> = memory.action_masks().to_vec();
            Some(Tensor::<B::InnerBackend, 2, Int>::from_data(
                TensorData::new(masks, [n, mask_width]),
                &self.device,
            ))
        };
        // Wrap once for the autodiff backend (no copy: the training graph
        // references the same device storage). Chunks narrow these views.
        let resident_student = Tensor::<B, 2>::from_inner(resident_student_inner);
        let resident_masks = resident_masks_inner
            .as_ref()
            .map(|masks| Tensor::<B, 2, Int>::from_inner(masks.clone()));
        metrics["Transfer/batch staging time"] = resident_start.elapsed().as_secs_f64().into();

        // ── Teacher probabilities (no gradients) ──────────────────────
        // One masked-logit forward per chunk; both the tempered training
        // targets and the `T=1` entropy baseline derive from those logits via
        // `probs_with_temperature` (no second forward). Targets are stored
        // once per batch and reused across all epochs. Entropy at
        // temperature 1.0 (`Transfer/teacher entropy`) stays comparable
        // across runs; the softened target entropy
        // (`Transfer/teacher target entropy`) is logged alongside.
        let teacher_start = Instant::now();
        let mut teacher_chunks = Vec::with_capacity(n.div_ceil(mb));
        let mut teacher_entropy: Option<Tensor<B::InnerBackend, 1>> = None;
        let mut teacher_target_entropy: Option<Tensor<B::InnerBackend, 1>> = None;
        let temp = tl.teacher_temperature;
        for start in (0..n).step_by(mb) {
            let chunk_len = (start + mb).min(n) - start;
            let weight = chunk_len as f32 / n as f32;
            let states = resident_teacher
                .clone()
                .narrow(0, start, chunk_len)
                .cast(DType::F32);
            let masks = resident_masks_inner.as_ref().map(|masks| {
                masks
                    .clone()
                    .narrow(0, start, chunk_len)
                    .cast(FloatDType::F32)
            });
            let logits = teacher.masked_logits(states, masks);
            let probs_target = Net::<B::InnerBackend>::probs_with_temperature(logits.clone(), temp);
            let chunk_target_entropy = -(probs_target.clone() * probs_target.clone().log())
                .sum_dim(1)
                .mean();
            let weighted_target = chunk_target_entropy * weight;
            teacher_target_entropy = Some(match teacher_target_entropy {
                None => weighted_target,
                Some(sum) => sum + weighted_target,
            });
            // T=1 entropy for comparable metrics; reuse the target when the
            // temperature is already 1.0 (identical distributions).
            let chunk_entropy = if temp == 1.0 {
                -(probs_target.clone() * probs_target.clone().log())
                    .sum_dim(1)
                    .mean()
            } else {
                let probs_plain = Net::<B::InnerBackend>::probs_with_temperature(logits, 1.0);
                -(probs_plain.clone() * probs_plain.clone().log())
                    .sum_dim(1)
                    .mean()
            };
            let weighted = chunk_entropy * weight;
            teacher_entropy = Some(match teacher_entropy {
                None => weighted,
                Some(sum) => sum + weighted,
            });
            teacher_chunks.push(probs_target);
        }
        let [teacher_entropy_sum, teacher_target_entropy_sum] =
            match (teacher_entropy, teacher_target_entropy) {
                (Some(plain), Some(target)) => Transaction::default()
                    .register(plain)
                    .register(target)
                    .execute()
                    .try_into()
                    .expect("teacher entropy transaction"),
                _ => unreachable!("non-empty memory always fills teacher entropies"),
            };
        let teacher_entropy_sum = teacher_entropy_sum.to_vec::<f32>().unwrap()[0] as f64;
        let teacher_target_entropy_sum =
            teacher_target_entropy_sum.to_vec::<f32>().unwrap()[0] as f64;
        metrics["Timing/teacher inference"] = teacher_start.elapsed().as_secs_f64().into();
        metrics["Transfer/teacher entropy"] = teacher_entropy_sum.into();
        metrics["Transfer/teacher target entropy"] = teacher_target_entropy_sum.into();

        let mut actor_gradients = GradientsAccumulator::new();
        let mut head_gradients = GradientsAccumulator::new();

        // First-epoch metrics accumulate on the GPU: one readback at the end
        // instead of one sync per chunk.
        let mut loss_acc: Option<Tensor<B, 1>> = None;
        let mut accuracy_acc: Option<Tensor<B, 1>> = None;
        let mut entropy_acc: Option<Tensor<B, 1>> = None;

        let mut upload_time = Duration::ZERO;
        let mut optim_time = Duration::ZERO;
        let training_start = Instant::now();
        for epoch in 0..tl.epochs {
            for (chunk_idx, teacher_chunk) in teacher_chunks.iter().enumerate() {
                let start = chunk_idx * mb;
                let end = (start + mb).min(n);
                let chunk_len = end - start;
                let weight = chunk_len as f32 / n as f32;

                let upload_start = Instant::now();
                let states = resident_student
                    .clone()
                    .narrow(0, start, chunk_len)
                    .cast(DType::F32);
                let masks = resident_masks.as_ref().map(|masks| {
                    masks
                        .clone()
                        .narrow(0, start, chunk_len)
                        .cast(FloatDType::F32)
                });
                upload_time += upload_start.elapsed();

                let features = net.apply_shared_head(states);
                // KD trains the student softened at the same temperature as
                // the teacher targets; deployment/entropy stays at T=1. One
                // masked-logit forward serves both (no second forward).
                let student_logits = net.actor.masked_logits(features, masks);
                let student_target = Net::<B>::probs_with_temperature(student_logits.clone(), temp);
                let teacher_probs = Tensor::<B, 2>::from_inner(teacher_chunk.clone());

                let loss = distillation_loss(
                    teacher_probs.clone(),
                    student_target.clone(),
                    tl.use_kl_div,
                    tl.loss_scale,
                    tl.loss_exponent,
                    temp,
                );

                if epoch == 0 {
                    // First-epoch metrics at deployment temperature (T=1) so
                    // runs with different softening stay comparable. Argmax
                    // order is temperature-invariant, so accuracy matches.
                    let student_plain = if temp == 1.0 {
                        student_target.clone()
                    } else {
                        Net::<B>::probs_with_temperature(student_logits, 1.0)
                    };
                    let matches = student_plain
                        .clone()
                        .detach()
                        .argmax(1)
                        .equal(teacher_probs.clone().argmax(1))
                        .float()
                        .mean();
                    let entropy = -(student_plain.clone() * student_plain.clone().log())
                        .sum_dim(1)
                        .mean();
                    let weighted_loss = loss.clone().detach() * weight;
                    let weighted_acc = matches * weight;
                    let weighted_entropy = entropy * weight;
                    loss_acc = Some(match loss_acc {
                        None => weighted_loss,
                        Some(sum) => sum + weighted_loss,
                    });
                    accuracy_acc = Some(match accuracy_acc {
                        None => weighted_acc,
                        Some(sum) => sum + weighted_acc,
                    });
                    entropy_acc = Some(match entropy_acc {
                        None => weighted_entropy,
                        Some(sum) => sum + weighted_entropy,
                    });
                }

                let mut grads = (loss * weight).backward();
                actor_gradients.accumulate(
                    &net.actor,
                    GradientsParams::from_module(&mut grads, &net.actor),
                );
                if let Some(head) = net.shared_head.as_ref() {
                    head_gradients.accumulate(head, GradientsParams::from_module(&mut grads, head));
                }
            }

            // One optimizer step per epoch over the whole batch. The LR
            // schedule runs on completed distillation rounds (logged once as
            // `Transfer/lr` before training). `grads()` drains the accumulator
            // after each epoch.
            let optim_start = Instant::now();
            let lr: f64 = effective_lr.into();
            net.actor = self
                .policy_optimizer
                .step(lr, net.actor.clone(), actor_gradients.grads());
            if let Some(head) = net.shared_head.take() {
                net.shared_head = Some(self.shared_head_optimizer.step(
                    lr,
                    head,
                    head_gradients.grads(),
                ));
            }
            self.apply_regen(&mut net, tl.regen, false);
            optim_time += optim_start.elapsed();
        }

        // Single readback for all first-epoch metrics.
        let [loss_sum, accuracy_sum, entropy_sum] = match (loss_acc, accuracy_acc, entropy_acc) {
            (Some(loss), Some(acc), Some(entropy)) => Transaction::default()
                .register(loss)
                .register(acc)
                .register(entropy)
                .execute()
                .try_into()
                .expect("Correct amount of tensor data"),
            _ => unreachable!("epochs > 0 always fills the metric accumulators"),
        };
        let loss_sum = loss_sum.to_vec::<f32>().unwrap()[0] as f64;
        let accuracy_sum = accuracy_sum.to_vec::<f32>().unwrap()[0] as f64;
        let entropy_sum = entropy_sum.to_vec::<f32>().unwrap()[0] as f64;

        metrics["Timing/distillation"] = training_start.elapsed().as_secs_f64().into();
        metrics["Transfer/upload time"] = upload_time.as_secs_f64().into();
        metrics["Transfer/optimizer step time"] = optim_time.as_secs_f64().into();
        metrics["Transfer/loss"] = loss_sum.into();
        metrics["Transfer/accuracy"] = accuracy_sum.into();
        metrics["Transfer/entropy"] = entropy_sum.into();

        // Total update magnitude covers every trained parameter (actor +
        // shared head); per-part magnitudes are logged alongside.
        let actor_params_after = flatten_net(&net.actor);
        let head_params_after = net.shared_head.as_ref().map(flatten_net);
        let actor_magnitude = l2_diff(&actor_params_before, &actor_params_after);
        let head_magnitude = match (head_params_before, head_params_after) {
            (Some(before), Some(after)) => l2_diff(&before, &after),
            _ => 0.0,
        };
        let total_magnitude = (actor_magnitude.powi(2) + head_magnitude.powi(2)).sqrt();
        metrics["Transfer/update magnitude"] = total_magnitude.into();
        metrics["Transfer/actor update magnitude"] = actor_magnitude.into();
        metrics["Transfer/head update magnitude"] = head_magnitude.into();

        // Plasticity diagnostics, same keys as the PPO path, re-emitted
        // every round from the cache. The critic is reported but never
        // modified by distillation.
        {
            let sample_len = n.min(2048);
            let sample = resident_student
                .clone()
                .narrow(0, 0, sample_len)
                .cast(DType::F32);
            let round = self.transfer_rounds;
            self.maybe_log_plasticity(
                round,
                &net.actor,
                &net.critic,
                net.shared_head.as_ref(),
                sample,
                metrics,
            );
        }

        self.transfer_rounds += 1;
        (net, n)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use burn::backend::{Autodiff, Flex};
    use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};

    use super::*;
    use crate::agent::config::PpoLearnerConfig;
    use crate::base::TerminalState;

    type B = Autodiff<Flex>;

    fn memory_batch(n: usize, width: usize, old_width: usize, n_actions: usize) -> Memory {
        let mut memory = Memory::with_capacity(n);
        let states = (0..n * width)
            .map(|i| ((i % 7) as f32) / 7.0 - 0.5)
            .collect();
        let old_states = (0..n * old_width)
            .map(|i| ((i % 5) as f32) / 5.0 - 0.5)
            .collect();
        let mut terminals = vec![TerminalState::None; n];
        *terminals.last_mut().unwrap() = TerminalState::Normal;
        memory.push_player(
            states,
            width,
            vec![0; n],
            vec![0.0; n],
            vec![0.0; n],
            terminals,
            vec![1u8; n * n_actions],
            n_actions,
            old_states,
            old_width,
            None,
        );
        memory
    }

    #[test]
    fn distillation_trains_student_policy_towards_teacher() {
        let device = Default::default();
        let obs = 8;
        let n_actions = 5;
        let n = 64;

        // Teacher: larger network with a shared head.
        let teacher =
            Actic::<Flex>::new(obs, n_actions, vec![16, 16], vec![16], &[16], &device, None);

        // Persist the teacher, then load it back as a `Policy` (the same path
        // `Learner::transfer_learn` takes).
        let root = std::env::temp_dir().join(format!(
            "rlgymppo-transfer-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
        teacher
            .actor
            .clone()
            .save_file(root.join("actor"), &recorder)
            .unwrap();
        teacher
            .shared_head
            .as_ref()
            .unwrap()
            .clone()
            .save_file(root.join("shared_head"), &recorder)
            .unwrap();
        let teacher = Policy::<Flex>::load(
            &root,
            &rlgymppo_model::PolicyConfig {
                input_size: obs,
                action_size: n_actions,
                actor_layer_sizes: vec![16, 16],
                shared_head_layer_sizes: vec![16],
                norm: rlgymppo_model::NormSelection::None,
            },
            &device,
        )
        .unwrap();
        fs::remove_dir_all(&root).unwrap();

        // Student: smaller network.
        let mut net = Actic::<B>::new(obs, n_actions, vec![8], vec![8], &[8], &device, None);

        let memory = memory_batch(n, obs, 0, n_actions);
        let tl = TransferLearnConfig {
            lr: 0.05,
            batch_size: n,
            mini_batch_size: 32,
            epochs: 10,
            loss_scale: 1.0,
            ..Default::default()
        };

        let mut ppo = PpoLearnerConfig {
            learning_rate: 0.05,
            ..Default::default()
        }
        .init(device);

        // Critic must stay untouched by distillation.
        let critic_before = flatten_net(&net.critic);

        let mut metrics = Report::default();
        (net, _) = ppo.transfer_learn(net, &teacher, &memory, &tl, &mut metrics);
        let first_loss = metrics["Transfer/loss"].as_float();

        let mut metrics = Report::default();
        (net, _) = ppo.transfer_learn(net, &teacher, &memory, &tl, &mut metrics);
        let second_loss = metrics["Transfer/loss"].as_float();

        assert!(
            first_loss.is_finite() && first_loss > 0.0,
            "first distillation loss should be finite and positive, got {first_loss}"
        );
        assert!(
            second_loss < first_loss,
            "distillation should reduce the loss ({first_loss} -> {second_loss})"
        );
        let accuracy = metrics["Transfer/accuracy"].as_float();
        assert!(
            (0.0..=1.0).contains(&accuracy),
            "accuracy should be a fraction, got {accuracy}"
        );
        assert!(
            metrics["Transfer/update magnitude"].as_float() > 0.0,
            "the student actor should have been updated"
        );
        assert_eq!(
            flatten_net(&net.critic),
            critic_before,
            "distillation must not touch the critic"
        );
        assert!(
            metrics["Transfer/teacher entropy"].as_float().is_finite()
                && metrics["Transfer/entropy"].as_float().is_finite(),
            "entropy metrics should be finite"
        );
    }

    #[test]
    fn distillation_supports_different_teacher_obs_space() {
        let device = Default::default();
        let n = 64;
        let n_actions = 5;
        let student_obs = 5;
        let teacher_obs = 8;

        // Teacher consumes the *old* (larger) observation space.
        let teacher = Actic::<Flex>::new(
            teacher_obs,
            n_actions,
            vec![16, 16],
            vec![16],
            &[16],
            &device,
            None,
        );
        let root = std::env::temp_dir().join(format!(
            "rlgymppo-transfer-diffobs-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
        teacher
            .actor
            .clone()
            .save_file(root.join("actor"), &recorder)
            .unwrap();
        teacher
            .shared_head
            .as_ref()
            .unwrap()
            .clone()
            .save_file(root.join("shared_head"), &recorder)
            .unwrap();
        let teacher = Policy::<Flex>::load(
            &root,
            &rlgymppo_model::PolicyConfig {
                input_size: teacher_obs,
                action_size: n_actions,
                actor_layer_sizes: vec![16, 16],
                shared_head_layer_sizes: vec![16],
                norm: rlgymppo_model::NormSelection::None,
            },
            &device,
        )
        .unwrap();
        fs::remove_dir_all(&root).unwrap();

        // Student consumes the *new* (smaller) observation space.
        let mut net = Actic::<B>::new(
            student_obs,
            n_actions,
            vec![8],
            vec![8],
            &[8],
            &device,
            None,
        );

        // Memory rows carry per-step old-obs rows of a different width.
        let memory = memory_batch(n, student_obs, teacher_obs, n_actions);
        assert_eq!(memory.state_width(), student_obs);
        assert_eq!(memory.old_state_width(), teacher_obs);
        assert!(memory.validate().is_ok());

        let tl = TransferLearnConfig {
            lr: 0.05,
            batch_size: n,
            mini_batch_size: 32,
            epochs: 10,
            loss_scale: 1.0,
            ..Default::default()
        };

        let mut ppo = PpoLearnerConfig {
            learning_rate: 0.05,
            ..Default::default()
        }
        .init(device);

        let critic_before = flatten_net(&net.critic);
        let mut metrics = Report::default();
        (net, _) = ppo.transfer_learn(net, &teacher, &memory, &tl, &mut metrics);
        let first_loss = metrics["Transfer/loss"].as_float();

        let mut metrics = Report::default();
        (net, _) = ppo.transfer_learn(net, &teacher, &memory, &tl, &mut metrics);
        let second_loss = metrics["Transfer/loss"].as_float();

        assert!(
            first_loss.is_finite() && first_loss > 0.0,
            "first distillation loss should be finite and positive, got {first_loss}"
        );
        assert!(
            second_loss < first_loss,
            "distillation should reduce the loss ({first_loss} -> {second_loss})"
        );
        assert_eq!(
            flatten_net(&net.critic),
            critic_before,
            "distillation must not touch the critic"
        );
    }

    #[test]
    fn distillation_supports_schedule_regen_and_plasticity_logging() {
        use crate::agent::config::{LrSchedule, RegenConfig};

        let device = Default::default();
        let obs = 8;
        let n_actions = 5;
        let n = 64;

        let teacher = Actic::<Flex>::new(obs, n_actions, vec![8], vec![8], &[8], &device, None);
        let root = std::env::temp_dir().join(format!(
            "rlgymppo-transfer-regen-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
        teacher
            .actor
            .clone()
            .save_file(root.join("actor"), &recorder)
            .unwrap();
        teacher
            .shared_head
            .as_ref()
            .unwrap()
            .clone()
            .save_file(root.join("shared_head"), &recorder)
            .unwrap();
        let teacher = Policy::<Flex>::load(
            &root,
            &rlgymppo_model::PolicyConfig {
                input_size: obs,
                action_size: n_actions,
                actor_layer_sizes: vec![8],
                shared_head_layer_sizes: vec![8],
                norm: rlgymppo_model::NormSelection::None,
            },
            &device,
        )
        .unwrap();
        fs::remove_dir_all(&root).unwrap();

        let mut net = Actic::<B>::new(obs, n_actions, vec![8], vec![8], &[8], &device, None);
        let memory = memory_batch(n, obs, 0, n_actions);
        let tl = TransferLearnConfig {
            lr: 0.05,
            batch_size: n,
            mini_batch_size: 32,
            epochs: 2,
            loss_scale: 1.0,
            lr_schedule: LrSchedule::Exponential {
                decay_per_iteration: 0.999,
            },
            regen: Some(RegenConfig {
                strength: 1e-3,
                noise_std: 0.0,
            }),
            ..Default::default()
        };

        let mut ppo = PpoLearnerConfig {
            learning_rate: 0.05,
            plasticity_log_interval: Some(1),
            ..Default::default()
        }
        .init(device);

        let critic_before = flatten_net(&net.critic);
        let mut metrics = Report::default();
        (net, _) = ppo.transfer_learn(net, &teacher, &memory, &tl, &mut metrics);
        let first_loss = metrics["Transfer/loss"].as_float();
        let first_lr = metrics["Transfer/lr"].as_float();
        assert!(
            ppo.has_regen_snapshot(),
            "regen anchor must exist after first round"
        );

        let mut metrics = Report::default();
        (net, _) = ppo.transfer_learn(net, &teacher, &memory, &tl, &mut metrics);
        let second_loss = metrics["Transfer/loss"].as_float();
        let second_lr = metrics["Transfer/lr"].as_float();

        assert!(
            first_loss.is_finite() && first_loss > 0.0,
            "first distillation loss should be finite and positive, got {first_loss}"
        );
        assert!(
            second_loss.is_finite() && second_loss > 0.0,
            "second distillation loss should stay finite and positive, got {second_loss}"
        );
        // Schedule invariant (deterministic for any random init): exponential
        // decay applies per completed round.
        let expected_first = 0.05
            * LrSchedule::Exponential {
                decay_per_iteration: 0.999,
            }
            .scale(0);
        let expected_second = 0.05
            * LrSchedule::Exponential {
                decay_per_iteration: 0.999,
            }
            .scale(1);
        assert!(
            (first_lr - expected_first).abs() < 1e-6,
            "round-0 LR should follow schedule (got {first_lr}, want {expected_first})"
        );
        assert!(
            (second_lr - expected_second).abs() < 1e-6,
            "round-1 LR should decay per schedule (got {second_lr}, want {expected_second})"
        );
        assert!(
            second_lr < first_lr,
            "LR schedule should decay ({first_lr} -> {second_lr})"
        );
        // Training happened despite regen: actor and head moved.
        assert!(metrics["Transfer/actor update magnitude"].as_float() > 0.0);
        assert!(metrics["Transfer/head update magnitude"].as_float() > 0.0);
        assert_eq!(
            flatten_net(&net.critic),
            critic_before,
            "distillation with regen must not touch the critic"
        );
        assert_eq!(
            ppo.transfer_rounds, 2,
            "each transfer_learn call counts one round"
        );
        for key in [
            "Update/actor weight norm",
            "Update/critic weight norm",
            "Update/actor stable rank",
            "Update/critic stable rank",
            "Update/actor dormant frac",
            "Update/critic dormant frac",
        ] {
            assert!(
                metrics[key].as_float().is_finite(),
                "plasticity metric {key} should be finite"
            );
        }
    }

    #[test]
    fn distillation_softens_teacher_targets_with_temperature() {
        let device = Default::default();
        let obs = 8;
        let n_actions = 5;
        let n = 64;

        let teacher = Actic::<Flex>::new(obs, n_actions, vec![8], vec![8], &[8], &device, None);
        let root = std::env::temp_dir().join(format!(
            "rlgymppo-transfer-temp-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
        teacher
            .actor
            .clone()
            .save_file(root.join("actor"), &recorder)
            .unwrap();
        teacher
            .shared_head
            .as_ref()
            .unwrap()
            .clone()
            .save_file(root.join("shared_head"), &recorder)
            .unwrap();
        let teacher = Policy::<Flex>::load(
            &root,
            &rlgymppo_model::PolicyConfig {
                input_size: obs,
                action_size: n_actions,
                actor_layer_sizes: vec![8],
                shared_head_layer_sizes: vec![8],
                norm: rlgymppo_model::NormSelection::None,
            },
            &device,
        )
        .unwrap();
        fs::remove_dir_all(&root).unwrap();

        let net = Actic::<B>::new(obs, n_actions, vec![8], vec![8], &[8], &device, None);
        let memory = memory_batch(n, obs, 0, n_actions);
        let base_tl = TransferLearnConfig {
            lr: 0.05,
            batch_size: n,
            mini_batch_size: 32,
            epochs: 1,
            loss_scale: 1.0,
            ..Default::default()
        };

        let mut ppo: crate::agent::Ppo<B> = PpoLearnerConfig {
            learning_rate: 0.05,
            ..Default::default()
        }
        .init(device);

        let mut metrics = Report::default();
        ppo.transfer_learn(net.clone(), &teacher, &memory, &base_tl, &mut metrics);
        let base_entropy = metrics["Transfer/teacher entropy"].as_float();
        let base_target = metrics["Transfer/teacher target entropy"].as_float();
        assert!(base_entropy.is_finite());
        assert!(
            (base_target - base_entropy).abs() < 1e-6,
            "T=1 target entropy should equal plain entropy ({base_entropy} vs {base_target})"
        );

        let hot_tl = TransferLearnConfig {
            teacher_temperature: 2.0,
            ..base_tl
        };
        let mut metrics = Report::default();
        ppo.transfer_learn(net, &teacher, &memory, &hot_tl, &mut metrics);
        let hot_loss = metrics["Transfer/loss"].as_float();
        let hot_entropy = metrics["Transfer/teacher entropy"].as_float();
        let hot_target = metrics["Transfer/teacher target entropy"].as_float();
        assert!(
            hot_loss.is_finite() && hot_loss > 0.0,
            "softened distillation loss should be finite and positive, got {hot_loss}"
        );
        assert!(
            (hot_entropy - base_entropy).abs() < 1e-5,
            "T=1 teacher entropy stays comparable across temperatures ({base_entropy} -> {hot_entropy})"
        );
        assert!(
            hot_target >= hot_entropy,
            "higher temperature should not sharpen the targets ({hot_entropy} -> {hot_target})"
        );
    }

    #[test]
    fn kl_loss_is_true_forward_kl() {
        use burn::tensor::TensorData;

        let device = Default::default();
        // Known distributions: row0 q=[0.9, 0.1] vs p=[0.5, 0.5], row1 equal.
        let teacher_data = vec![0.9_f32, 0.1, 0.5, 0.5];
        let student_data = vec![0.5_f32, 0.5, 0.5, 0.5];
        let teacher = Tensor::<B, 2>::from_data(TensorData::new(teacher_data, [2, 2]), &device);
        let student = Tensor::<B, 2>::from_data(TensorData::new(student_data, [2, 2]), &device);
        let loss = distillation_loss::<B>(teacher, student, true, 1.0, 1.0, 1.0);
        let loss_val: Vec<f32> = loss.into_data().into_vec().unwrap();
        let kl0 = 0.9_f32 * (0.9_f32 / 0.5_f32).ln() + 0.1_f32 * (0.1_f32 / 0.5_f32).ln();
        let expected = (kl0 + 0.0) / 2.0;
        assert!(
            (loss_val[0] - expected).abs() < 1e-5,
            "true forward KL should sum per action then mean per state (got {}, want {expected})",
            loss_val[0]
        );
        // Old `abs(q*log(q/p)).mean()` would give a different value.
        let old_style =
            (0.9_f32 * (0.9_f32 / 0.5_f32).ln()).abs() + (0.1_f32 * (0.1_f32 / 0.5_f32).ln()).abs();
        let old_style = old_style / 2.0;
        assert!(
            (old_style - expected).abs() > 1e-4,
            "sanity: old abs-mean differs from true KL"
        );
    }

    #[test]
    fn kl_loss_zero_equal_nonnegative_masked_and_scaled() {
        use burn::tensor::TensorData;

        let device = Default::default();
        // Zero when equal (both L1 and KL, T=1 and T>1 with same inputs).
        for use_kl in [false, true] {
            for temp in [1.0_f32, 2.0] {
                let probs = vec![0.7_f32, 0.2, 0.1, 0.4, 0.4, 0.2];
                let teacher =
                    Tensor::<B, 2>::from_data(TensorData::new(probs.clone(), [2, 3]), &device);
                let student = Tensor::<B, 2>::from_data(TensorData::new(probs, [2, 3]), &device);
                // KL at T>1 scales by T^2 even when inputs are identical
                // (still zero); L1 ignores temperature.
                let loss = distillation_loss::<B>(teacher, student, use_kl, 1.0, 1.0, temp);
                let val: Vec<f32> = loss.into_data().into_vec().unwrap();
                assert!(
                    val[0].abs() < 1e-6,
                    "equal distributions should give zero KD (use_kl={use_kl}, T={temp}), got {}",
                    val[0]
                );
            }
        }
        // Nonnegative, finite with masked actions (~1e-11 on both sides).
        let teacher = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.6_f32, 0.4, 1e-11, 0.5, 0.5, 1e-11], [2, 3]),
            &device,
        );
        let student = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.5_f32, 0.5, 1e-11, 0.7, 0.3, 1e-11], [2, 3]),
            &device,
        );
        let loss = distillation_loss::<B>(teacher, student, true, 1.0, 1.0, 1.0);
        let val: Vec<f32> = loss.into_data().into_vec().unwrap();
        assert!(
            val[0].is_finite() && val[0] >= -1e-6,
            "masked KL should be nonnegative finite, got {}",
            val[0]
        );
        // loss_exponent applies per state: mean of squared KLs.
        let teacher2 = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.9_f32, 0.1, 0.5, 0.5], [2, 2]),
            &device,
        );
        let student2 = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.5_f32, 0.5, 0.5, 0.5], [2, 2]),
            &device,
        );
        let loss1 = distillation_loss::<B>(teacher2.clone(), student2.clone(), true, 1.0, 1.0, 1.0);
        let loss2 = distillation_loss::<B>(teacher2, student2, true, 1.0, 2.0, 1.0);
        let v1: Vec<f32> = loss1.into_data().into_vec().unwrap();
        let v2: Vec<f32> = loss2.into_data().into_vec().unwrap();
        let kl0 = 0.9_f32 * (0.9_f32 / 0.5_f32).ln() + 0.1_f32 * (0.1_f32 / 0.5_f32).ln();
        let expected_sq = (kl0 * kl0) / 2.0;
        assert!(
            (v2[0] - expected_sq).abs() < 1e-5,
            "exponent 2 should square per-state KL (got {}, want {expected_sq})",
            v2[0]
        );
        assert!(v1[0] > 0.0 && v2[0] > 0.0);
        // T^2 scaling for KL only; L1 ignores temperature.
        let teacher3 =
            Tensor::<B, 2>::from_data(TensorData::new(vec![0.9_f32, 0.1], [1, 2]), &device);
        let student3 =
            Tensor::<B, 2>::from_data(TensorData::new(vec![0.5_f32, 0.5], [1, 2]), &device);
        let kl_t1 = distillation_loss::<B>(teacher3.clone(), student3.clone(), true, 1.0, 1.0, 1.0);
        let kl_t2 = distillation_loss::<B>(teacher3.clone(), student3.clone(), true, 1.0, 1.0, 2.0);
        let l1_t1 =
            distillation_loss::<B>(teacher3.clone(), student3.clone(), false, 1.0, 1.0, 1.0);
        let l1_t2 = distillation_loss::<B>(teacher3, student3, false, 1.0, 1.0, 2.0);
        let kl_t1: Vec<f32> = kl_t1.into_data().into_vec().unwrap();
        let kl_t2: Vec<f32> = kl_t2.into_data().into_vec().unwrap();
        let l1_t1: Vec<f32> = l1_t1.into_data().into_vec().unwrap();
        let l1_t2: Vec<f32> = l1_t2.into_data().into_vec().unwrap();
        assert!(
            (kl_t2[0] - 4.0 * kl_t1[0]).abs() < 1e-5,
            "KL should scale by T^2 ({} vs {})",
            kl_t2[0],
            kl_t1[0]
        );
        assert!(
            (l1_t2[0] - l1_t1[0]).abs() < 1e-6,
            "L1 must not scale by T^2 ({} vs {})",
            l1_t2[0],
            l1_t1[0]
        );
        // Near-equal distributions with a fractional exponent stay finite
        // (clamped per-state KL cannot go slightly negative into `NaN`).
        let teacher4 = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.500001_f32, 0.499999, 0.5, 0.5], [2, 2]),
            &device,
        );
        let student4 = Tensor::<B, 2>::from_data(
            TensorData::new(vec![0.5_f32, 0.5, 0.5, 0.5], [2, 2]),
            &device,
        );
        let frac = distillation_loss::<B>(teacher4, student4, true, 1.0, 0.5, 1.0);
        let frac: Vec<f32> = frac.into_data().into_vec().unwrap();
        assert!(
            frac[0].is_finite() && frac[0] >= 0.0,
            "fractional-exponent KL must be finite nonnegative, got {}",
            frac[0]
        );
    }

    #[test]
    fn temperature_applies_to_both_sides_and_metrics_stay_t1() {
        use burn::module::AutodiffModule;

        let device = Default::default();
        let obs = 6;
        let n_actions = 4;
        let n = 32;
        let memory = memory_batch(n, obs, 0, n_actions);
        let mut ppo: crate::agent::Ppo<B> = PpoLearnerConfig {
            learning_rate: 0.05,
            ..Default::default()
        }
        .init(device);
        let base_tl = TransferLearnConfig {
            lr: 0.05,
            batch_size: n,
            mini_batch_size: 16,
            epochs: 1,
            loss_scale: 1.0,
            ..Default::default()
        };
        // Identical teacher/student via `.valid()`: no checkpoint roundtrip.
        let net = Actic::<B>::new(obs, n_actions, vec![8], vec![8], &[8], &device, None);
        let valid = net.valid();
        let teacher = Policy::<Flex> {
            actor: valid.actor.clone(),
            shared_head: valid.shared_head.clone(),
        };
        for use_kl in [false, true] {
            let hot_tl = TransferLearnConfig {
                teacher_temperature: 2.0,
                use_kl_div: use_kl,
                ..base_tl.clone()
            };
            let mut metrics = Report::default();
            ppo.transfer_learn(net.clone(), &teacher, &memory, &hot_tl, &mut metrics);
            let loss = metrics["Transfer/loss"].as_float();
            assert!(
                loss.abs() < 1e-5,
                "identical teacher/student at T=2 should give zero KD (use_kl={use_kl}), got {loss}"
            );
            // Metrics stay at deployment temperature: student and teacher T=1
            // entropies match for identical models.
            let student_ent = metrics["Transfer/entropy"].as_float();
            let teacher_ent = metrics["Transfer/teacher entropy"].as_float();
            assert!(student_ent.is_finite() && teacher_ent.is_finite());
            assert!(
                (student_ent - teacher_ent).abs() < 1e-4,
                "T=1 entropies should match for identical models ({student_ent} vs {teacher_ent})"
            );
        }
        // Normal inference still defaults to T=1 (deployment distribution).
        let _ = teacher.infer(
            burn::tensor::Tensor::<Flex, 2>::zeros([2, obs], &device),
            None,
        );
    }

    #[test]
    fn transfer_rounds_persist_and_schedule_continues() {
        use crate::agent::config::LrSchedule;
        use crate::utils::running_stat::Stats;

        // Stats serde: present field round-trips, missing field defaults to 0.
        let stats = Stats {
            transfer_rounds: 7,
            ..Default::default()
        };
        let toml_str = toml::to_string(&stats).unwrap();
        let back: Stats = toml::from_str(&toml_str).unwrap();
        assert_eq!(back.transfer_rounds, 7);
        let full = toml::to_string(&Stats::default()).unwrap();
        let legacy_toml = full
            .lines()
            .filter(|line| !line.starts_with("transfer_rounds"))
            .collect::<Vec<_>>()
            .join("\n");
        let legacy: Stats = toml::from_str(&legacy_toml).unwrap();
        assert_eq!(legacy.transfer_rounds, 0);
        // Schedule continuation across a simulated restart.
        let device = Default::default();
        let obs = 6;
        let n_actions = 4;
        let n = 32;
        let teacher_actic =
            Actic::<Flex>::new(obs, n_actions, vec![8], vec![8], &[8], &device, None);
        let teacher = Policy::<Flex> {
            actor: teacher_actic.actor.clone(),
            shared_head: teacher_actic.shared_head.clone(),
        };
        let memory = memory_batch(n, obs, 0, n_actions);
        let tl = TransferLearnConfig {
            lr: 0.05,
            batch_size: n,
            mini_batch_size: 16,
            epochs: 1,
            loss_scale: 1.0,
            lr_schedule: LrSchedule::Cosine {
                total_iterations: 10,
                min_scale: 0.1,
            },
            ..Default::default()
        };
        let mut ppo: crate::agent::Ppo<B> = PpoLearnerConfig {
            learning_rate: 0.05,
            ..Default::default()
        }
        .init(device);
        assert_eq!(ppo.transfer_rounds(), 0);
        let net = Actic::<B>::new(obs, n_actions, vec![8], vec![8], &[8], &device, None);
        let mut metrics = Report::default();
        let (net, _) = ppo.transfer_learn(net, &teacher, &memory, &tl, &mut metrics);
        let lr0 = metrics["Transfer/lr"].as_float();
        assert!(
            (lr0 - 0.05).abs() < 1e-6,
            "round 0 LR should be base LR, got {lr0}"
        );
        assert_eq!(ppo.transfer_rounds(), 1);
        // Simulate checkpoint save/load: Stats carries the count.
        let saved = Stats {
            transfer_rounds: ppo.transfer_rounds(),
            ..Default::default()
        };
        let toml_str = toml::to_string(&saved).unwrap();
        let restored: Stats = toml::from_str(&toml_str).unwrap();
        let mut ppo2: crate::agent::Ppo<B> = PpoLearnerConfig {
            learning_rate: 0.05,
            ..Default::default()
        }
        .init(device);
        ppo2.set_transfer_rounds(restored.transfer_rounds);
        let mut metrics2 = Report::default();
        ppo2.transfer_learn(net, &teacher, &memory, &tl, &mut metrics2);
        let lr1 = metrics2["Transfer/lr"].as_float();
        let expected_scale = LrSchedule::Cosine {
            total_iterations: 10,
            min_scale: 0.1,
        }
        .scale(1);
        let expected = 0.05 * expected_scale;
        assert!(
            (lr1 - expected).abs() < 1e-6,
            "resumed round 1 LR should continue cosine (got {lr1}, want {expected})"
        );
        assert!(lr1 < lr0, "cosine should decay ({lr0} -> {lr1})");
        assert_eq!(ppo2.transfer_rounds(), 2);
    }

    #[test]
    fn regen_anchor_is_pre_update_and_survives_save_load() {
        use crate::agent::config::RegenConfig;

        let device = Default::default();
        let obs = 6;
        let n_actions = 4;
        let n = 32;
        let teacher_actic =
            Actic::<Flex>::new(obs, n_actions, vec![8], vec![8], &[8], &device, None);
        let teacher = Policy::<Flex> {
            actor: teacher_actic.actor.clone(),
            shared_head: teacher_actic.shared_head.clone(),
        };
        let memory = memory_batch(n, obs, 0, n_actions);
        let tl = TransferLearnConfig {
            lr: 0.05,
            batch_size: n,
            mini_batch_size: 16,
            epochs: 2,
            loss_scale: 1.0,
            regen: Some(RegenConfig {
                strength: 0.01,
                noise_std: 0.0,
            }),
            ..Default::default()
        };
        let mut ppo: crate::agent::Ppo<B> = PpoLearnerConfig {
            learning_rate: 0.05,
            ..Default::default()
        }
        .init(device);
        assert!(!ppo.has_regen_snapshot());
        let net = Actic::<B>::new(obs, n_actions, vec![8], vec![8], &[8], &device, None);
        let before_actor = super::flatten_net(&net.actor);
        let before_head = net.shared_head.as_ref().map(super::flatten_net);
        let (net_after, _) =
            ppo.transfer_learn(net, &teacher, &memory, &tl, &mut metrics_default());
        assert!(ppo.has_regen_snapshot());
        let after_actor = super::flatten_net(&net_after.actor);
        assert_ne!(before_actor, after_actor, "training should move the actor");
        // Anchor saved to disk must equal the pre-update weights, not post.
        let dir = std::env::temp_dir().join(format!(
            "rlgymppo-regen-anchor-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        ppo.save_regen_snapshot(&dir);
        assert!(dir.join("actor_init.mpk.gz").is_file());
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
        let loaded_actor = net_after
            .actor
            .clone()
            .load_file(dir.join("actor_init"), &recorder, &device)
            .unwrap();
        let loaded_flat = super::flatten_net(&loaded_actor);
        assert_eq!(
            loaded_flat, before_actor,
            "regen anchor must be pre-update weights"
        );
        if let Some(before_h) = before_head {
            let template = net_after.shared_head.clone().unwrap();
            let loaded_head = template
                .load_file(dir.join("shared_head_init"), &recorder, &device)
                .unwrap();
            assert_eq!(super::flatten_net(&loaded_head), before_h);
        }
        // Save/resume identity: a fresh PPO loading the same folder keeps it.
        let mut ppo2: crate::agent::Ppo<B> = PpoLearnerConfig {
            learning_rate: 0.05,
            ..Default::default()
        }
        .init(device);
        assert!(!ppo2.has_regen_snapshot());
        ppo2.load_regen_snapshot(&dir, &net_after);
        assert!(ppo2.has_regen_snapshot());
        let dir2 = std::env::temp_dir().join(format!(
            "rlgymppo-regen-anchor2-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir2).unwrap();
        ppo2.save_regen_snapshot(&dir2);
        let reloaded = net_after
            .actor
            .clone()
            .load_file(dir2.join("actor_init"), &recorder, &device)
            .unwrap();
        assert_eq!(super::flatten_net(&reloaded), before_actor);
        fs::remove_dir_all(&dir).unwrap();
        fs::remove_dir_all(&dir2).unwrap();
        // Legacy folder without anchor files is a silent no-op (caller warns).
        let empty = std::env::temp_dir().join(format!(
            "rlgymppo-regen-empty-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&empty).unwrap();
        let mut ppo3: crate::agent::Ppo<B> = PpoLearnerConfig {
            learning_rate: 0.05,
            ..Default::default()
        }
        .init(device);
        ppo3.load_regen_snapshot(&empty, &net_after);
        assert!(!ppo3.has_regen_snapshot());
        fs::remove_dir_all(&empty).unwrap();
    }

    fn metrics_default() -> Report {
        Report::default()
    }

    #[test]
    fn update_magnitude_includes_shared_head() {
        let device = Default::default();
        let obs = 6;
        let n_actions = 4;
        let n = 32;
        let teacher_actic =
            Actic::<Flex>::new(obs, n_actions, vec![8], vec![8], &[8], &device, None);
        let teacher = Policy::<Flex> {
            actor: teacher_actic.actor.clone(),
            shared_head: teacher_actic.shared_head.clone(),
        };
        let memory = memory_batch(n, obs, 0, n_actions);
        let tl = TransferLearnConfig {
            lr: 0.05,
            batch_size: n,
            mini_batch_size: 16,
            epochs: 2,
            loss_scale: 1.0,
            ..Default::default()
        };
        let mut ppo: crate::agent::Ppo<B> = PpoLearnerConfig {
            learning_rate: 0.05,
            ..Default::default()
        }
        .init(device);
        let net = Actic::<B>::new(obs, n_actions, vec![8], vec![8], &[8], &device, None);
        let mut metrics = Report::default();
        ppo.transfer_learn(net, &teacher, &memory, &tl, &mut metrics);
        let total = metrics["Transfer/update magnitude"].as_float();
        let actor = metrics["Transfer/actor update magnitude"].as_float();
        let head = metrics["Transfer/head update magnitude"].as_float();
        assert!(total.is_finite() && actor.is_finite() && head.is_finite());
        assert!(actor > 0.0, "actor should move");
        assert!(head > 0.0, "shared head should move and be counted");
        let recombined = (actor * actor + head * head).sqrt();
        assert!(
            (total - recombined).abs() < 1e-6,
            "total should be sqrt(actor^2 + head^2) (got {total}, want {recombined})"
        );
        assert!(total > actor, "total must exceed actor-only magnitude");
        assert!(metrics["Transfer/lr"].as_float() > 0.0);
    }
}

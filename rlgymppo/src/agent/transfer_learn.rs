use std::path::PathBuf;
use std::time::Instant;

use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::Transaction;
use burn::tensor::backend::AutodiffBackend;
use rlgymppo_model::Policy;
use rlgymppo_utils::Report;

use super::{flatten_net, l2_diff};
use crate::NormSelection;
use crate::agent::Ppo;
use crate::agent::model::{Actic, Net};
use crate::base::{Memory, get_action_masks_batch, get_states_batch_range};

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
    pub use_kl_div: bool,
    /// Scale of the distillation loss. The natural loss is very small, so this
    /// prevents the optimizers from effectively dying.
    pub loss_scale: f32,
    /// Exponent applied to each sample's loss before averaging.
    pub loss_exponent: f32,
}

impl Default for TransferLearnConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            batch_size: 50_000,
            mini_batch_size: 10_000,
            epochs: 5,
            use_kl_div: false,
            loss_scale: 500.0,
            loss_exponent: 1.0,
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
        assert_eq!(
            self.batch_size % self.mini_batch_size,
            0,
            "batch size must be divisible by mini batch size"
        );
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
        let mb = tl.mini_batch_size;

        // Teacher observations: the old (teacher) obs builder's states when one
        // is configured (different observation space), otherwise the student's
        // own states (same observation space).
        let (teacher_states, teacher_width) = if memory.old_state_width() > 0 {
            (memory.old_states(), memory.old_state_width())
        } else {
            (memory.states(), memory.state_width())
        };

        // ── Teacher probabilities (no gradients) ──────────────────────
        // Computed once per batch and reused across all epochs.
        let teacher_start = Instant::now();
        let mut teacher_chunks = Vec::with_capacity(n.div_ceil(mb));
        let mut teacher_entropy_sum = 0.0f64;
        for start in (0..n).step_by(mb) {
            let end = (start + mb).min(n);
            let chunk_len = end - start;
            let weight = chunk_len as f64 / n as f64;
            let indices = (start..end).collect::<Vec<_>>();
            let states = get_states_batch_range::<B::InnerBackend>(
                teacher_states,
                teacher_width,
                start,
                end,
                &self.device,
            );
            let masks = get_action_masks_batch::<B::InnerBackend>(
                memory.action_masks(),
                memory.action_mask_width(),
                &indices,
                &self.device,
            );
            let probs = teacher.infer(states, Some(masks));
            let chunk_entropy = -(probs.clone() * probs.clone().log()).sum_dim(1).mean();
            teacher_entropy_sum +=
                chunk_entropy.into_data().to_vec::<f32>().unwrap()[0] as f64 * weight;
            teacher_chunks.push(probs);
        }
        metrics["Timing/teacher inference"] = teacher_start.elapsed().as_secs_f64().into();
        metrics["Transfer/teacher entropy"] = teacher_entropy_sum.into();

        let actor_params_before = flatten_net(&net.actor);
        let mut actor_gradients = GradientsAccumulator::new();
        let mut head_gradients = GradientsAccumulator::new();

        let mut loss_sum = 0.0f64;
        let mut accuracy_sum = 0.0f64;
        let mut entropy_sum = 0.0f64;

        let training_start = Instant::now();
        for epoch in 0..tl.epochs {
            for (chunk_idx, teacher_chunk) in teacher_chunks.iter().enumerate() {
                let start = chunk_idx * mb;
                let end = (start + mb).min(n);
                let chunk_len = end - start;
                let weight = chunk_len as f32 / n as f32;
                let indices = (start..end).collect::<Vec<_>>();

                let states = get_states_batch_range::<B>(
                    memory.states(),
                    memory.state_width(),
                    start,
                    end,
                    &self.device,
                );
                let masks = get_action_masks_batch::<B>(
                    memory.action_masks(),
                    memory.action_mask_width(),
                    &indices,
                    &self.device,
                );

                let features = net.apply_shared_head(states);
                let student_probs = net.actor.infer(features, Some(masks));
                let teacher_probs = Tensor::<B, 2>::from_inner(teacher_chunk.clone());

                // Match GigaLearn's distillation loss:
                //   abs-diff or forward KL, per-sample, then mean * loss_scale.
                let loss = if tl.use_kl_div {
                    (teacher_probs.clone() * (teacher_probs.clone() / student_probs.clone()).log())
                        .abs()
                } else {
                    (teacher_probs.clone() - student_probs.clone()).abs()
                };
                let loss = loss.powf_scalar(tl.loss_exponent).mean() * tl.loss_scale;

                if epoch == 0 {
                    // First-epoch metrics, matching GigaLearn's reporting.
                    let matches = student_probs
                        .clone()
                        .detach()
                        .argmax(1)
                        .equal(teacher_probs.clone().argmax(1))
                        .float()
                        .mean();
                    let entropy = -(student_probs.clone() * student_probs.clone().log())
                        .sum_dim(1)
                        .mean();
                    let [loss_data, acc_data, entropy_data] = Transaction::default()
                        .register(loss.clone().detach())
                        .register(matches)
                        .register(entropy)
                        .execute()
                        .try_into()
                        .expect("Correct amount of tensor data");
                    loss_sum += loss_data.to_vec::<f32>().unwrap()[0] as f64 * weight as f64;
                    accuracy_sum += acc_data.to_vec::<f32>().unwrap()[0] as f64 * weight as f64;
                    entropy_sum += entropy_data.to_vec::<f32>().unwrap()[0] as f64 * weight as f64;
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

            // One optimizer step per epoch over the whole batch.
            let lr: f64 = tl.lr.into();
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
        }

        metrics["Timing/distillation"] = training_start.elapsed().as_secs_f64().into();
        metrics["Transfer/loss"] = loss_sum.into();
        metrics["Transfer/accuracy"] = accuracy_sum.into();
        metrics["Transfer/entropy"] = entropy_sum.into();

        let actor_params_after = flatten_net(&net.actor);
        metrics["Transfer/update magnitude"] =
            l2_diff(&actor_params_before, &actor_params_after).into();

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
            vec![true; n * n_actions],
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
}

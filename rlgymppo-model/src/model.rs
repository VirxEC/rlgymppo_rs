use burn::module::Param;
use burn::nn::modules::norm::{Normalization, NormalizationConfig};
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{log_softmax, relu, softmax};
use burn::tensor::{Distribution, Transaction};

use crate::tensor::{
    SampledActions, argmax_actions, sample_actions_from_logits, sample_actions_from_logits_tensor,
    sampled_actions_to_vec, to_mask_tensor_2d, to_mask_tensor_2d_flat, to_mask_tensor_2d_indexed,
    to_mask_tensor_2d_indexed_flat, to_state_tensor_2d, to_state_tensor_2d_flat,
    to_state_tensor_2d_indexed, to_state_tensor_2d_indexed_flat,
};

pub struct PPOOutput<B: Backend> {
    pub log_probs: Tensor<B, 2>,
    pub values: Tensor<B, 2>,
}

impl<B: Backend> PPOOutput<B> {
    pub fn new(log_probs: Tensor<B, 2>, values: Tensor<B, 2>) -> Self {
        Self { log_probs, values }
    }
}

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    linear_layers: Vec<Linear<B>>,
    /// Normalization applied after each hidden linear layer (before activation).
    /// Empty when normalization is disabled.
    layer_norms: Vec<Normalization<B>>,
    /// Whether the last linear layer is a dedicated output layer (no norm, no activation).
    /// `false` for the shared head, whose last layer IS a hidden layer.
    #[module(skip)]
    has_output_layer: bool,
}

unsafe impl<B: Backend> Sync for Net<B> {}

impl<B: Backend> Net<B> {
    /// `add_output_layer`: when `true` (default), a final Linear(last_hidden, output_size)
    /// is appended.  When `false` the network ends after the last hidden layer; `output_size`
    /// is ignored (used for the shared head, which produces features not logits/values).
    pub fn new(
        input_size: usize,
        output_size: usize,
        layer_sizes: Vec<usize>,
        device: &B::Device,
        norm_config: Option<NormalizationConfig>,
        add_output_layer: bool,
    ) -> Self {
        assert_ne!(layer_sizes.len(), 0);

        let initializer = Initializer::KaimingUniform {
            gain: 1.0 / 3.0f64.sqrt(),
            fan_out_only: false,
        };

        let num_linears = layer_sizes.len() + if add_output_layer { 1 } else { 0 };
        let mut linear_layers = Vec::with_capacity(num_linears);
        let mut layer_norms = if norm_config.is_some() {
            Vec::with_capacity(layer_sizes.len())
        } else {
            Vec::new()
        };

        linear_layers.push(
            LinearConfig::new(input_size, layer_sizes[0])
                .with_initializer(initializer.clone())
                .init(device),
        );

        for i in 1..layer_sizes.len() {
            linear_layers.push(
                LinearConfig::new(layer_sizes[i - 1], layer_sizes[i])
                    .with_initializer(initializer.clone())
                    .init(device),
            );
        }

        if add_output_layer {
            linear_layers.push(
                LinearConfig::new(layer_sizes[layer_sizes.len() - 1], output_size)
                    .with_initializer(initializer.clone())
                    .init(device),
            );
        }

        if let Some(ref base_config) = norm_config {
            for &size in &layer_sizes {
                let config = base_config.clone().with_num_features(size);
                layer_norms.push(config.init::<B>(device));
            }
        }

        Self {
            linear_layers,
            layer_norms,
            has_output_layer: add_output_layer,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut output = input;

        let num_hidden = if self.has_output_layer {
            self.linear_layers.len() - 1
        } else {
            self.linear_layers.len()
        };

        for (i, layer) in self.linear_layers[..num_hidden].iter().enumerate() {
            output = layer.forward(output);
            if let Some(norm) = self.layer_norms.get(i) {
                output = norm.forward(output);
            }
            output = relu(output);
        }

        if self.has_output_layer {
            self.linear_layers[num_hidden].forward(output)
        } else {
            output
        }
    }

    /// Mean absolute post-ReLU activation per hidden layer, in order.
    ///
    /// Mirrors [`Net::forward`] over the hidden stack only (no output layer).
    /// Entries near zero (below ~`1e-3`) mark dormant units. All scalars sync
    /// in one transaction; any transient autodiff graph is dropped with the
    /// returned values.
    pub fn hidden_dormancy(&self, input: Tensor<B, 2>) -> Vec<f32> {
        let mut output = input;

        let num_hidden = if self.has_output_layer {
            self.linear_layers.len() - 1
        } else {
            self.linear_layers.len()
        };

        let mut means = Vec::with_capacity(num_hidden);
        for (i, layer) in self.linear_layers[..num_hidden].iter().enumerate() {
            output = layer.forward(output);
            if let Some(norm) = self.layer_norms.get(i) {
                output = norm.forward(output);
            }
            output = relu(output);
            means.push(output.clone().abs().mean());
        }

        let mut transaction = Transaction::default();
        for mean in &means {
            transaction = transaction.register(mean.clone());
        }
        transaction
            .execute()
            .iter()
            .map(|data| data.to_vec::<f32>().unwrap()[0])
            .collect()
    }

    /// Blend every parameter toward a reference snapshot (regenerative
    /// regularization): `theta <- theta - strength * (theta - init)`.
    ///
    /// Optional isotropic Gaussian noise (`noise_std > 0.0`, applied to linear
    /// weights and biases only) makes this behave like soft
    /// shrink-and-perturb, except blending is toward the run's own
    /// initialization rather than freshly sampled weights. Normalization
    /// scales are shrunk without noise. Parameter IDs are preserved, so
    /// optimizer momentum state stays valid.
    pub fn regen_step(&mut self, init: &Self, strength: f64, noise_std: f64, device: &B::Device) {
        assert!(
            (0.0..1.0).contains(&strength),
            "regen strength must be in [0, 1), got {strength}"
        );
        assert!(noise_std >= 0.0, "regen noise_std must be non-negative");
        assert_eq!(
            self.linear_layers.len(),
            init.linear_layers.len(),
            "regen snapshot architecture mismatch"
        );

        for (layer, init_layer) in self.linear_layers.iter_mut().zip(init.linear_layers.iter()) {
            let id = layer.weight.id;
            layer.weight = Param::initialized(
                id,
                blend_param(
                    &layer.weight,
                    &init_layer.weight,
                    strength,
                    noise_std,
                    device,
                ),
            );
            if let (Some(bias), Some(init_bias)) = (layer.bias.as_mut(), init_layer.bias.as_ref()) {
                let id = bias.id;
                *bias = Param::initialized(
                    id,
                    blend_param(bias, init_bias, strength, noise_std, device),
                );
            }
        }

        for (norm, init_norm) in self.layer_norms.iter_mut().zip(init.layer_norms.iter()) {
            match (norm, init_norm) {
                (Normalization::Layer(ln), Normalization::Layer(init_ln)) => {
                    let id = ln.gamma.id;
                    ln.gamma = Param::initialized(
                        id,
                        blend_param(&ln.gamma, &init_ln.gamma, strength, 0.0, device),
                    );
                    if let (Some(beta), Some(init_beta)) = (ln.beta.as_mut(), init_ln.beta.as_ref())
                    {
                        let id = beta.id;
                        *beta = Param::initialized(
                            id,
                            blend_param(beta, init_beta, strength, 0.0, device),
                        );
                    }
                }
                (Normalization::Rms(rms), Normalization::Rms(init_rms)) => {
                    let id = rms.gamma.id;
                    rms.gamma = Param::initialized(
                        id,
                        blend_param(&rms.gamma, &init_rms.gamma, strength, 0.0, device),
                    );
                }
                _ => {
                    // Mismatched normalization variants: leave untouched.
                }
            }
        }
    }
    /// `mask` is an optional [N, n_actions] f32 tensor (1.0 = valid, 0.0 = invalid).
    /// Masks disabled actions by adding a large negative bias to their logits.
    pub fn linear_layers(&self) -> &[Linear<B>] {
        &self.linear_layers
    }

    pub fn layer_norms(&self) -> &[Normalization<B>] {
        &self.layer_norms
    }

    pub fn masked_logits(&self, input: Tensor<B, 2>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 2> {
        let logits = self.forward(input);
        if let Some(mask) = mask {
            logits - (1.0 - mask) * 1e10_f32
        } else {
            logits
        }
    }

    pub fn log_probs(&self, input: Tensor<B, 2>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 2> {
        log_softmax(self.masked_logits(input, mask), 1)
    }

    pub fn infer(&self, input: Tensor<B, 2>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 2> {
        self.infer_with_temperature(input, mask, 1.0)
    }

    /// Softmax action distribution with temperature scaling: `softmax(logits / temperature)`.
    ///
    /// `temperature == 1.0` matches [`Net::infer`] exactly; higher values
    /// soften the distribution (higher entropy targets for distillation).
    /// Must be positive and finite.
    pub fn infer_with_temperature(
        &self,
        input: Tensor<B, 2>,
        mask: Option<Tensor<B, 2>>,
        temperature: f32,
    ) -> Tensor<B, 2> {
        assert!(
            temperature > 0.0 && temperature.is_finite(),
            "temperature must be positive and finite, got {temperature}"
        );
        softmax(self.masked_logits(input, mask) / temperature, 1).clamp(1e-11, 1.0)
    }

    pub fn react_deterministic(
        &self,
        state: &[Vec<f32>],
        masks: &[Vec<bool>],
        device: &B::Device,
    ) -> Vec<usize> {
        let mask_tensor = (!masks.is_empty()).then(|| to_mask_tensor_2d(masks, device));
        argmax_actions(self.masked_logits(to_state_tensor_2d(state, device), mask_tensor))
    }

    pub fn react_deterministic_indexed(
        &self,
        state: &[Vec<f32>],
        masks: &[Vec<bool>],
        indices: &[usize],
        device: &B::Device,
    ) -> Vec<usize> {
        let mask_tensor =
            (!masks.is_empty()).then(|| to_mask_tensor_2d_indexed(masks, indices, device));
        argmax_actions(self.masked_logits(
            to_state_tensor_2d_indexed(state, indices, device),
            mask_tensor,
        ))
    }

    pub fn react(
        &self,
        state: &[Vec<f32>],
        masks: &[Vec<bool>],
        device: &B::Device,
    ) -> (Vec<usize>, Vec<f32>) {
        let mask_tensor = (!masks.is_empty()).then(|| to_mask_tensor_2d(masks, device));
        sample_actions_from_logits(
            self.masked_logits(to_state_tensor_2d(state, device), mask_tensor),
            device,
        )
    }

    pub fn react_indexed(
        &self,
        state: &[Vec<f32>],
        masks: &[Vec<bool>],
        indices: &[usize],
        device: &B::Device,
    ) -> (Vec<usize>, Vec<f32>) {
        let mask_tensor =
            (!masks.is_empty()).then(|| to_mask_tensor_2d_indexed(masks, indices, device));
        sample_actions_from_logits(
            self.masked_logits(
                to_state_tensor_2d_indexed(state, indices, device),
                mask_tensor,
            ),
            device,
        )
    }
}

/// Blend one parameter tensor toward its initialization snapshot:
/// `(1 - strength) * param + strength * init`, plus optional isotropic
/// Gaussian noise. See [`Net::regen_step`].
fn blend_param<B: Backend, const D: usize>(
    param: &Param<Tensor<B, D>>,
    init: &Param<Tensor<B, D>>,
    strength: f64,
    noise_std: f64,
    device: &B::Device,
) -> Tensor<B, D> {
    let blended = param
        .val()
        .clone()
        .mul_scalar(1.0 - strength)
        .add(init.val().clone().mul_scalar(strength));
    if noise_std > 0.0 {
        let shape = blended.shape().dims::<D>();
        blended.add(Tensor::random(
            shape,
            Distribution::Normal(0.0, noise_std),
            device,
        ))
    } else {
        blended
    }
}

/// Device-resident actor output. Calling [`Self::wait`] is the first point at
/// which sampled actions and log probabilities are transferred back to the CPU.
pub struct PendingActions<B: Backend> {
    sampled: SampledActions<B>,
}
impl<B: Backend> PendingActions<B> {
    pub fn wait(self) -> (Vec<usize>, Vec<f32>) {
        sampled_actions_to_vec(self.sampled)
    }
}

#[derive(Module, Debug)]
pub struct Actic<B: Backend> {
    /// Optional feature extractor shared between the actor and critic.
    /// When present, raw observations flow through `shared_head` before
    /// being passed to both `actor` and `critic`.
    pub shared_head: Option<Net<B>>,
    pub actor: Net<B>,
    pub critic: Net<B>,
}

impl<B: Backend> Actic<B> {
    /// When `shared_head_layer_sizes` is non-empty, a shared feature extractor is
    /// created.  The actor and critic then take the shared head's last hidden size
    /// as their input dimension instead of `input_size`.
    pub fn new(
        input_size: usize,
        output_size: usize,
        actor_layers: Vec<usize>,
        critic_layers: Vec<usize>,
        shared_head_layer_sizes: &[usize],
        device: &B::Device,
        norm_config: Option<NormalizationConfig>,
    ) -> Self {
        let (head, actor_input) = if shared_head_layer_sizes.is_empty() {
            (None, input_size)
        } else {
            let shared = Net::new(
                input_size,
                0,
                shared_head_layer_sizes.to_vec(),
                device,
                norm_config.clone(),
                false, // no output layer – just features
            );
            let feat_size = *shared_head_layer_sizes.last().unwrap();
            (Some(shared), feat_size)
        };

        Self {
            shared_head: head,
            actor: Net::new(
                actor_input,
                output_size,
                actor_layers,
                device,
                norm_config.clone(),
                true,
            ),
            critic: Net::new(actor_input, 1, critic_layers, device, norm_config, true),
        }
    }

    pub fn apply_shared_head(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        match self.shared_head {
            Some(ref head) => head.forward(input),
            None => input,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>, mask: Option<Tensor<B, 2>>) -> PPOOutput<B> {
        let features = self.apply_shared_head(input);
        let log_probs = self.actor.log_probs(features.clone(), mask);
        let values = self.critic.forward(features);

        PPOOutput::<B>::new(log_probs, values)
    }

    /// Dispatch stochastic action inference and retain the sampled output on
    /// the backend device until [`PendingActions::wait`] is called.
    pub fn submit_react(
        &self,
        state: &[Vec<f32>],
        masks: &[Vec<bool>],
        device: &B::Device,
    ) -> PendingActions<B> {
        let input = to_state_tensor_2d(state, device);
        let features = self.apply_shared_head(input);
        let mask_tensor = (!masks.is_empty()).then(|| to_mask_tensor_2d(masks, device));
        PendingActions {
            sampled: sample_actions_from_logits_tensor(
                self.actor.masked_logits(features, mask_tensor),
                device,
            ),
        }
    }

    pub fn react(
        &self,
        state: &[Vec<f32>],
        masks: &[Vec<bool>],
        device: &B::Device,
    ) -> (Vec<usize>, Vec<f32>) {
        self.submit_react(state, masks, device).wait()
    }

    /// Dispatch stochastic inference for selected rows and retain the sampled
    /// output on the backend device until [`PendingActions::wait`] is called.
    pub fn submit_react_indexed(
        &self,
        state: &[Vec<f32>],
        masks: &[Vec<bool>],
        indices: &[usize],
        device: &B::Device,
    ) -> PendingActions<B> {
        let input = to_state_tensor_2d_indexed(state, indices, device);
        let features = self.apply_shared_head(input);
        let mask_tensor =
            (!masks.is_empty()).then(|| to_mask_tensor_2d_indexed(masks, indices, device));
        PendingActions {
            sampled: sample_actions_from_logits_tensor(
                self.actor.masked_logits(features, mask_tensor),
                device,
            ),
        }
    }

    pub fn react_indexed(
        &self,
        state: &[Vec<f32>],
        masks: &[Vec<bool>],
        indices: &[usize],
        device: &B::Device,
    ) -> (Vec<usize>, Vec<f32>) {
        self.submit_react_indexed(state, masks, indices, device)
            .wait()
    }

    /// Dispatch stochastic inference for selected flat player rows and retain
    /// the sampled output on the backend device until
    /// [`PendingActions::wait`] is called. Used for old-model self-play
    /// inference over a subset of players.
    pub fn submit_react_indexed_flat(
        &self,
        state: &[f32],
        state_width: usize,
        masks: &[bool],
        mask_width: usize,
        indices: &[usize],
        device: &B::Device,
    ) -> PendingActions<B> {
        let input = to_state_tensor_2d_indexed_flat(state, state_width, indices, device);
        let features = self.apply_shared_head(input);
        let mask_tensor = (mask_width > 0)
            .then(|| to_mask_tensor_2d_indexed_flat(masks, mask_width, indices, device));
        PendingActions {
            sampled: sample_actions_from_logits_tensor(
                self.actor.masked_logits(features, mask_tensor),
                device,
            ),
        }
    }

    /// Dispatch stochastic inference for a flat, pre-sized obs batch and
    /// retain the sampled output on the backend device until
    /// [`PendingActions::wait`] is called. The collector stores all player
    /// obs in one contiguous buffer in global player order.
    pub fn submit_react_flat(
        &self,
        state: &[f32],
        rows: usize,
        state_width: usize,
        masks: &[bool],
        mask_width: usize,
        device: &B::Device,
    ) -> PendingActions<B> {
        let input = to_state_tensor_2d_flat(state, rows, state_width, device);
        let features = self.apply_shared_head(input);
        let mask_tensor =
            (mask_width > 0).then(|| to_mask_tensor_2d_flat(masks, rows, mask_width, device));
        PendingActions {
            sampled: sample_actions_from_logits_tensor(
                self.actor.masked_logits(features, mask_tensor),
                device,
            ),
        }
    }

    /// Synchronous stochastic inference for a flat, pre-sized obs batch.
    pub fn react_flat(
        &self,
        state: &[f32],
        rows: usize,
        state_width: usize,
        masks: &[bool],
        mask_width: usize,
        device: &B::Device,
    ) -> (Vec<usize>, Vec<f32>) {
        self.submit_react_flat(state, rows, state_width, masks, mask_width, device)
            .wait()
    }

    pub fn react_deterministic(
        &self,
        state: &[Vec<f32>],
        masks: &[Vec<bool>],
        device: &B::Device,
    ) -> Vec<usize> {
        let input = to_state_tensor_2d(state, device);
        let features = self.apply_shared_head(input);
        let mask_tensor = (!masks.is_empty()).then(|| to_mask_tensor_2d(masks, device));
        argmax_actions(self.actor.masked_logits(features, mask_tensor))
    }

    pub fn react_deterministic_indexed(
        &self,
        state: &[Vec<f32>],
        masks: &[Vec<bool>],
        indices: &[usize],
        device: &B::Device,
    ) -> Vec<usize> {
        let input = to_state_tensor_2d_indexed(state, indices, device);
        let features = self.apply_shared_head(input);
        let mask_tensor =
            (!masks.is_empty()).then(|| to_mask_tensor_2d_indexed(masks, indices, device));
        argmax_actions(self.actor.masked_logits(features, mask_tensor))
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Flex;
    use burn::prelude::Backend;

    use super::*;

    fn flat<B: Backend>(net: &Net<B>) -> Vec<f32> {
        let mut data = Vec::new();
        for layer in net.linear_layers() {
            data.extend(
                layer
                    .weight
                    .val()
                    .clone()
                    .into_data()
                    .into_vec::<f32>()
                    .unwrap(),
            );
        }
        data
    }

    #[test]
    fn regen_step_with_zero_strength_is_identity() {
        let device = Default::default();
        let mut net = Net::<Flex>::new(4, 2, vec![8, 8], &device, None, true);
        let init = net.clone();
        let before = flat(&net);
        net.regen_step(&init, 0.0, 0.0, &device);
        assert_eq!(flat(&net), before);
    }

    #[test]
    fn regen_step_pulls_toward_snapshot() {
        let device = Default::default();
        let mut net = Net::<Flex>::new(4, 2, vec![8, 8], &device, None, true);
        let init = net.clone();
        let init_flat = flat(&init);

        // Move away with pure noise, then pull halfway back.
        net.regen_step(&init, 0.0, 1.0, &device);
        let noisy = flat(&net);
        assert_ne!(noisy, init_flat);
        net.regen_step(&init, 0.5, 0.0, &device);
        let pulled = flat(&net);
        for ((p, n), i) in pulled.iter().zip(noisy.iter()).zip(init_flat.iter()) {
            assert!((p - (0.5 * n + 0.5 * i)).abs() < 1e-5);
        }
    }

    #[test]
    fn hidden_dormancy_reports_per_hidden_layer() {
        let device = Default::default();
        let net = Net::<Flex>::new(4, 2, vec![8, 8], &device, None, true);
        let input = Tensor::<Flex, 2>::zeros([16, 4], &device);
        let means = net.hidden_dormancy(input);
        assert_eq!(means.len(), 2);
        assert!(means.iter().all(|mean| mean.is_finite() && *mean >= 0.0));
    }

    #[test]
    fn temperature_one_matches_infer_and_softens() {
        let device = Default::default();
        let net = Net::<Flex>::new(4, 3, vec![8], &device, None, true);
        let input = Tensor::<Flex, 2>::ones([8, 4], &device);
        let base: Vec<f32> = net
            .infer(input.clone(), None)
            .into_data()
            .into_vec()
            .unwrap();
        let same: Vec<f32> = net
            .infer_with_temperature(input.clone(), None, 1.0)
            .into_data()
            .into_vec()
            .unwrap();
        assert_eq!(base, same);
        let soft: Vec<f32> = net
            .infer_with_temperature(input, None, 2.0)
            .into_data()
            .into_vec()
            .unwrap();
        let entropy =
            |probs: &[f32]| -probs.iter().map(|p| p * p.max(1e-11).ln()).sum::<f32>() / 8.0;
        assert!(entropy(&soft) >= entropy(&base));
    }
}

use burn::nn::modules::norm::{Normalization, NormalizationConfig};
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{relu, softmax};

use crate::utils::{argmax_actions, sample_actions, to_mask_tensor_2d, to_state_tensor_2d};

pub struct PPOOutput<B: Backend> {
    pub policies: Tensor<B, 2>,
    pub values: Tensor<B, 2>,
}

impl<B: Backend> PPOOutput<B> {
    pub fn new(policies: Tensor<B, 2>, values: Tensor<B, 2>) -> Self {
        Self { policies, values }
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
    fn new(
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

    /// `mask` is an optional [N, n_actions] f32 tensor (1.0 = valid, 0.0 = invalid).
    /// Masks disabled actions by adding a large negative bias to their logits.
    pub fn linear_layers(&self) -> &[Linear<B>] {
        &self.linear_layers
    }

    pub fn layer_norms(&self) -> &[Normalization<B>] {
        &self.layer_norms
    }

    pub fn infer(&self, input: Tensor<B, 2>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 2> {
        let logits = self.forward(input);
        let logits = if let Some(mask) = mask {
            logits - (1.0 - mask) * 1e10_f32
        } else {
            logits
        };
        softmax(logits, 1).clamp(1e-11, 1.0)
    }

    pub fn react_deterministic(
        &self,
        state: &[Vec<f32>],
        masks: &[Vec<bool>],
        device: &B::Device,
    ) -> Vec<usize> {
        let mask_tensor = (!masks.is_empty()).then(|| to_mask_tensor_2d(masks, device));
        argmax_actions(self.infer(to_state_tensor_2d(state, device), mask_tensor))
    }

    pub fn react(
        &self,
        state: &[Vec<f32>],
        masks: &[Vec<bool>],
        device: &B::Device,
    ) -> (Vec<usize>, Vec<f32>) {
        let mask_tensor = (!masks.is_empty()).then(|| to_mask_tensor_2d(masks, device));
        sample_actions(
            self.infer(to_state_tensor_2d(state, device), mask_tensor),
            device,
        )
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
        let policies = self.actor.infer(features.clone(), mask);
        let values = self.critic.forward(features);

        PPOOutput::<B>::new(policies, values)
    }

    pub fn react(
        &self,
        state: &[Vec<f32>],
        masks: &[Vec<bool>],
        device: &B::Device,
    ) -> (Vec<usize>, Vec<f32>) {
        let input = to_state_tensor_2d(state, device);
        let features = self.apply_shared_head(input);
        let mask_tensor = (!masks.is_empty()).then(|| to_mask_tensor_2d(masks, device));
        sample_actions(self.actor.infer(features, mask_tensor), device)
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
        argmax_actions(self.actor.infer(features, mask_tensor))
    }
}

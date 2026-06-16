use burn::{
    nn::{Initializer, LayerNormConfig, Linear, LinearConfig, norm::LayerNorm},
    prelude::*,
    tensor::activation::{relu, softmax},
};

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
    /// LayerNorm applied after each hidden linear layer (before activation).
    /// Empty when layer normalization is disabled.
    layer_norms: Vec<LayerNorm<B>>,
}

unsafe impl<B: Backend> Sync for Net<B> {}

impl<B: Backend> Net<B> {
    fn new(
        input_size: usize,
        output_size: usize,
        layer_sizes: Vec<usize>,
        device: &B::Device,
        add_layer_norm: bool,
    ) -> Self {
        assert_ne!(layer_sizes.len(), 0);

        let initializer = Initializer::XavierUniform { gain: 1.0 };

        let mut linear_layers = Vec::with_capacity(layer_sizes.len() + 2);
        let mut layer_norms = if add_layer_norm {
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

        linear_layers.push(
            LinearConfig::new(layer_sizes[layer_sizes.len() - 1], output_size)
                .with_initializer(initializer.clone())
                .init(device),
        );

        if add_layer_norm {
            for &size in &layer_sizes {
                layer_norms.push(LayerNormConfig::new(size).init(device));
            }
        }

        Self {
            linear_layers,
            layer_norms,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut output = input;
        let num_hidden = self.linear_layers.len() - 1;
        for (i, layer) in self.linear_layers[..num_hidden].iter().enumerate() {
            output = layer.forward(output);
            if let Some(norm) = self.layer_norms.get(i) {
                output = norm.forward(output);
            }
            output = relu(output);
        }

        self.linear_layers[num_hidden].forward(output)
    }

    /// `mask` is an optional [N, n_actions] f32 tensor (1.0 = valid, 0.0 = invalid).
    /// Masks disabled actions by adding a large negative bias to their logits.
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
    pub actor: Net<B>,
    pub critic: Net<B>,
}

impl<B: Backend> Actic<B> {
    pub fn new(
        input_size: usize,
        output_size: usize,
        actor_layers: Vec<usize>,
        critic_layers: Vec<usize>,
        device: &B::Device,
        add_layer_norm: bool,
    ) -> Self {
        Self {
            actor: Net::new(
                input_size,
                output_size,
                actor_layers,
                device,
                add_layer_norm,
            ),
            critic: Net::new(input_size, 1, critic_layers, device, add_layer_norm),
        }
    }
}

impl<B: Backend> Actic<B> {
    pub fn forward(&self, input: Tensor<B, 2>, mask: Option<Tensor<B, 2>>) -> PPOOutput<B> {
        let policies = self.actor.infer(input.clone(), mask);
        let values = self.critic.forward(input);

        PPOOutput::<B>::new(policies, values)
    }
}

use crate::utils::{argmax_actions, sample_actions, to_state_tensor_2d};
use burn::{
    nn::{Initializer, Linear, LinearConfig},
    prelude::*,
    tensor::activation::{relu, softmax},
};

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
    layers: Vec<Linear<B>>,
}

unsafe impl<B: Backend> Sync for Net<B> {}

impl<B: Backend> Net<B> {
    fn new(
        input_size: usize,
        output_size: usize,
        layer_sizes: Vec<usize>,
        device: &B::Device,
    ) -> Self {
        assert_ne!(layer_sizes.len(), 0);

        let initializer = Initializer::XavierUniform { gain: 1.0 };

        let mut layers = Vec::with_capacity(layer_sizes.len() + 2);
        layers.push(
            LinearConfig::new(input_size, layer_sizes[0])
                .with_initializer(initializer.clone())
                .init(device),
        );

        for i in 1..layer_sizes.len() {
            layers.push(
                LinearConfig::new(layer_sizes[i - 1], layer_sizes[i])
                    .with_initializer(initializer.clone())
                    .init(device),
            );
        }

        layers.push(
            LinearConfig::new(layer_sizes[layer_sizes.len() - 1], output_size)
                .with_initializer(initializer.clone())
                .init(device),
        );

        Self { layers }
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut output = input;
        let num_layers = self.layers.len();
        for layer in &self.layers[..num_layers - 1] {
            output = relu(layer.forward(output));
        }

        self.layers[num_layers - 1].forward(output)
    }

    pub fn infer(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        softmax(self.forward(input), 1).clamp(1e-11, 1.0)
    }

    pub fn react_deterministic(&self, state: &[Vec<f32>], device: &B::Device) -> Vec<usize> {
        argmax_actions(self.infer(to_state_tensor_2d(state, device)))
    }

    pub fn react(&self, state: &[Vec<f32>], device: &B::Device) -> (Vec<usize>, Vec<f32>) {
        sample_actions(self.infer(to_state_tensor_2d(state, device)), device)
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
    ) -> Self {
        Self {
            actor: Net::new(input_size, output_size, actor_layers, device),
            critic: Net::new(input_size, 1, critic_layers, device),
        }
    }
}

impl<B: Backend> Actic<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> PPOOutput<B> {
        let policies = self.actor.infer(input.clone());
        let values = self.critic.forward(input);

        PPOOutput::<B>::new(policies, values)
    }
}

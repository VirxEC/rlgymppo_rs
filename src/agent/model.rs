use crate::utils::{argmax_actions, sample_actions, to_state_tensor_2d};
use burn::{
    nn::{Initializer, Linear, LinearConfig},
    prelude::*,
    tensor::activation::{relu, softmax},
};
use rand::Rng;

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
}

pub struct Reaction {
    pub actions: Vec<usize>,
    pub log_probs: Vec<f32>,
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
        let num_actor_layers = self.actor.layers.len();

        let mut actor_input = input.clone();
        for layer in &self.actor.layers[..num_actor_layers - 1] {
            actor_input = relu(layer.forward(actor_input));
        }

        let policies = softmax(
            self.actor.layers[num_actor_layers - 1].forward(actor_input),
            1,
        )
        .clamp(1e-11, 1.0);

        let mut critic_input = input;
        for layer in &self.critic.layers {
            critic_input = layer.forward(critic_input);
        }

        PPOOutput::<B>::new(policies, critic_input)
    }

    pub fn infer(&self, mut input: Tensor<B, 2>) -> Tensor<B, 2> {
        let num_layers = self.actor.layers.len();
        for layer in &self.actor.layers[..num_layers - 1] {
            input = relu(layer.forward(input));
        }

        softmax(self.actor.layers[num_layers - 1].forward(input), 1).clamp(1e-11, 1.0)
    }

    pub fn react_deterministic(&self, state: &[Vec<f32>], device: &B::Device) -> Vec<usize> {
        argmax_actions(self.infer(to_state_tensor_2d(state, device)))
    }

    pub fn react<R: Rng>(&self, state: &[Vec<f32>], rng: &mut R, device: &B::Device) -> Reaction {
        let probs = self.infer(to_state_tensor_2d(state, device));

        let actions = sample_actions(probs.clone(), rng);
        let acts_ten = Tensor::from_data(
            TensorData::new(
                actions.iter().map(|x| *x as i32).collect::<Vec<_>>(),
                [actions.len(), 1],
            ),
            device,
        );
        let log_probs = probs.gather(1, acts_ten).log();

        Reaction {
            actions,
            log_probs: log_probs.into_data().into_vec().unwrap(),
        }
    }
}

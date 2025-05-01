mod avg_tracker;
mod report;

pub use avg_tracker::AvgTracker;
pub use report::{Report, Reportable};

use burn::LearningRate;
use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::rng;
use rand::rngs::ThreadRng;

pub(crate) fn to_state_tensor_2d<B: Backend>(state: &[Vec<f32>]) -> Tensor<B, 2> {
    let mut data: Vec<f32> = Vec::with_capacity(state.len() * state[0].len());
    for elem in state {
        data.extend(elem);
    }

    Tensor::from_data(TensorData::new(data, [state.len(), state[0].len()]), &Default::default())
}

pub(crate) fn to_state_tensor<B: Backend>(state: &[f32]) -> Tensor<B, 1> {
    Tensor::<B, 1>::from_floats(state, &Default::default())
}

pub(crate) fn to_action_tensor<B: Backend>(action: usize) -> Tensor<B, 1, Int> {
    Tensor::<B, 1, Int>::from_ints([action], &Default::default())
}

pub(crate) fn to_reward_tensor<B: Backend>(reward: impl Into<f32> + Clone) -> Tensor<B, 1> {
    Tensor::from_floats([reward.into()], &Default::default())
}

pub(crate) fn to_not_done_tensor<B: Backend>(done: bool) -> Tensor<B, 1> {
    Tensor::from_floats([u8::from(!done)], &Default::default())
}

pub(crate) fn sample_action_from_tensor<B: Backend>(output: Tensor<B, 2>) -> usize {
    let prob = output.to_data().to_vec::<f32>().unwrap();
    let dist = WeightedIndex::new(prob).unwrap();

    let mut rng = rng();
    dist.sample(&mut rng)
}

pub(crate) fn sample_actions_from_tensor<B: Backend>(output: Tensor<B, 2>, rng: &mut ThreadRng) -> Vec<usize> {
    let num_actions = output.shape().dims[0];

    let mut actions = Vec::with_capacity(num_actions);
    for data in output.iter_dim(0) {
        let prob = data.to_data().to_vec::<f32>().unwrap();
        let dist = WeightedIndex::new(prob).unwrap();

        actions.push(dist.sample(rng));
    }

    actions
}

pub(crate) fn get_elem<B: Backend, const D: usize>(i: usize, tensor: &Tensor<B, D>) -> Option<f32> {
    tensor.to_data().as_slice().ok()?.get(i).copied()
}

pub(crate) fn elementwise_min<B: Backend, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
) -> Tensor<B, D> {
    let rhs_lower = rhs.clone().lower(lhs.clone());
    lhs.clone().mask_where(rhs_lower, rhs.clone())
}

pub(crate) fn update_parameters<B: AutodiffBackend, M: AutodiffModule<B>>(
    loss: Tensor<B, 1>,
    module: M,
    optimizer: &mut impl Optimizer<M, B>,
    learning_rate: LearningRate,
) -> M {
    let gradients = loss.backward();
    let gradient_params = GradientsParams::from_grads(gradients, &module);
    optimizer.step(learning_rate, module, gradient_params)
}
